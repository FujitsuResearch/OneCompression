# Copyright 2025-2026 Fujitsu Ltd.

"""A serial, greedy reference backend with isolated, fully accounted probes.

This backend deliberately favors inspectable semantics over serving performance.
Every accepted marker or jump token passes through the normal KV-cache path.
Probes branch from the live KV cache when it can be cropped exactly and restore
it bit-for-bit afterwards; otherwise they replay the full prefix in a new cache.
ASAG attention uses eager attention on the probe forward; its cost is measured.
"""

from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import torch

from onecomp.reasoning import (
    ReasoningCompressionController,
    ReasoningConfig,
    ReasoningFormat,
    RequestState,
)
from onecomp.reasoning.actions import JumpAction, LogitMixtureAction, StopAction
from onecomp.reasoning.signals import (
    asag_confidence,
    attention_statistics,
    deer_confidence,
    logit_observation,
)

DEER_PROBE_CUE = "\n**Final Answer**\n\\boxed"
ASAG_PROBE_CUE = "\n\n Final Answer\n\n \\boxed"
ASAG_JUMP = (
    "Wait, my previous reasoning is not correct. I should adopt a more concise "
    "and different approach to reexamine this problem.\n\n"
)


@dataclass
class ProbeResult:
    token_ids: list[int]
    text: str
    confidence: float
    complete: bool
    model_evaluated_tokens: int
    prefill_tokens: int
    decode_tokens: int
    latency_s: float
    attention_overhead_s: float
    signals: dict[str, Any]
    first_distribution: Any = None
    reused_prefix_tokens: int = 0


@dataclass
class GenerationResult:
    state: RequestState
    text: str
    final_answer: str
    metrics: dict[str, Any]


class _TextStream:
    """Incremental detokenization that holds back incomplete characters.

    Byte-level BPE can split one character across tokens, and decoding such a
    token alone yields U+FFFD. Each push re-decodes from the last emitted chunk
    and returns only text whose characters are complete (empty otherwise).
    A ``final`` push flushes everything held back, so a stream cut off inside
    a character ends with the same U+FFFD as decoding all tokens at once.
    """

    def __init__(self, decode: Callable[[Sequence[int]], str]):
        self._decode = decode
        self._ids: list[int] = []
        self._prefix = 0
        self._read = 0

    def push(self, token_id: int, final: bool = False) -> str:
        self._ids.append(token_id)
        prefix_text = self._decode(self._ids[self._prefix : self._read])
        new_text = self._decode(self._ids[self._prefix :])
        if not final and (len(new_text) <= len(prefix_text) or new_text.endswith("�")):
            return ""
        self._prefix, self._read = self._read, len(self._ids)
        return new_text[len(prefix_text) :]


class TransformersReasoningAdapter:
    """Greedy batch-size-one generation for an already-loaded causal LM.

    Distinct calls own their caches and controller state. A model object itself
    is not promised to support parallel forwards; use independent workers for
    simultaneous reference requests. No serving throughput claim follows from
    this serial runtime. ``on_token(token_id, text)`` is called exactly once per
    visible token, including forced markers, and supports streaming consumers.
    ``text`` holds newly completed characters; it is empty while a multi-byte
    character is still incomplete, and the last token's call flushes any held
    text, so the concatenated texts match the output.

    ``probe_prefix_reuse=False`` forces every probe to re-prefill its whole prefix
    (the conservative replay-cost reference).
    """

    def __init__(self, model: Any, tokenizer: Any, *, probe_prefix_reuse: bool = True):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.probe_prefix_reuse = probe_prefix_reuse
        self._supports_logits_to_keep = (
            "logits_to_keep" in inspect.signature(model.forward).parameters
        )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _eos_ids(self) -> set[int]:
        value = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if value is None:
            value = getattr(self.tokenizer, "eos_token_id", None)
        return set(value if isinstance(value, (tuple, list)) else [value]) - {None}

    def _forward(self, ids: Sequence[int], cache: Any = None, *, attention: bool = False):
        inputs = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        kwargs = {"logits_to_keep": 1} if self._supports_logits_to_keep else {}
        return self.model(
            input_ids=inputs,
            past_key_values=cache,
            use_cache=True,
            output_attentions=attention,
            return_dict=True,
            **kwargs,
        )

    def _require_eager(self) -> None:
        implementation = getattr(getattr(self.model, "config", None), "_attn_implementation", None)
        if implementation != "eager":
            raise ValueError(
                "ASAG reference attention requires attn_implementation='eager' when loading "
                "the model; fused attention does not expose the required weights."
            )

    @staticmethod
    def _can_branch(cache: Any) -> bool:
        """Full-attention per-layer key/value caches can be cropped and restored exactly."""
        layers = getattr(cache, "layers", None)
        return (
            bool(layers)
            and hasattr(cache, "crop")
            and all(
                hasattr(layer, "keys") and not getattr(layer, "is_sliding", False)
                for layer in layers
            )
        )

    @staticmethod
    def _branch(cache: Any, keep: int) -> Callable[[], None]:
        """Crop ``cache`` to ``keep`` tokens; the returned callback restores it exactly."""
        length = cache.get_seq_length()
        saved = [
            (layer.keys[..., keep:length, :].clone(), layer.values[..., keep:length, :].clone())
            for layer in cache.layers
        ]
        cache.crop(keep)

        def restore():
            cache.crop(keep)
            for index, (keys, values) in enumerate(saved):
                if keys.shape[-2]:
                    cache.update(keys, values, index)

        return restore

    def _boxed_content(self, token_ids):
        """Indices of tokens overlapping boxed answer content, excluding delimiters.

        A character split across tokens decodes to U+FFFD until its last token, so
        character offsets are measured only at complete-character prefixes and
        every token of a split character shares that character's span.
        """
        text = self._decode(token_ids)
        start = text.find("{")
        if start < 0:
            return "", [], False
        depth = 0
        end = None
        for position in range(start, len(text)):
            depth += (text[position] == "{") - (text[position] == "}")
            if depth == 0:
                end = position
                break
        if end is None:
            return "", [], False
        selected = []
        pending = []
        left = 0
        for index in range(len(token_ids)):
            pending.append(index)
            prefix = self._decode(token_ids[: index + 1])
            if prefix.endswith("�") and index + 1 < len(token_ids):
                continue
            right = len(prefix)
            if text[max(left, start + 1) : min(right, end)].strip():
                selected.extend(pending)
            pending = []
            left = right
        answer = text[start + 1 : end].strip()
        return answer, selected, bool(answer and selected)

    @torch.inference_mode()
    def probe(
        self,
        prefix_ids: Sequence[int],
        config: ReasoningConfig,
        *,
        previous_span: tuple[int, int] | None = None,
        current_span: tuple[int, int] | None = None,
        collect_attention: bool | None = None,
        answer_probe: bool = True,
        cache: Any = None,
    ) -> ProbeResult:
        """Run a disposable answer probe and count every model-evaluated token.

        DEER confidence is the geometric mean of maximum token probabilities.
        ASAG uses the arithmetic mean of sampled-token probabilities. Greedy
        decoding makes the selected probability equal to the maximum. A probe
        truncated before EOS or a closing boxed-answer brace is not confident.

        ``cache`` is the live request cache holding exactly ``prefix_ids``. When
        given, the probe branches from it and restores it before returning. With
        attention, the cue-sized run of preceding reasoning tokens is re-forwarded
        because ASAG's query window covers those rows too.
        """
        asag = config.policy == "asag_reference" or collect_attention is True
        if collect_attention is None:
            collect_attention = asag and config.asag_mode != "confidence"
        if collect_attention:
            self._require_eager()
        cue = ASAG_PROBE_CUE if asag else DEER_PROBE_CUE
        cue_ids = self._encode(cue)
        if not cue_ids:
            raise ValueError("Probe cue must tokenize to at least one token")
        prefix_ids = list(prefix_ids)
        back = len(cue_ids) if collect_attention else 0
        restore = None
        if (
            self.probe_prefix_reuse
            and cache is not None
            and self._can_branch(cache)
            and cache.get_seq_length() == len(prefix_ids)
            and len(prefix_ids) > back
        ):
            keep = len(prefix_ids) - back
            restore = self._branch(cache, keep)
            inputs = prefix_ids[keep:] + cue_ids
        else:
            keep, cache = 0, None
            inputs = prefix_ids + cue_ids
        self._sync()
        started = perf_counter()
        try:
            output = self._forward(inputs, cache, attention=collect_attention)
            self._sync()
            attention_forward_s = perf_counter() - started if collect_attention else 0.0
            signals: dict[str, Any] = {}
            attention_stat_s = 0.0
            if collect_attention:
                stat_started = perf_counter()
                attentions = getattr(output, "attentions", None)
                if not attentions or any(a is None for a in attentions):
                    raise RuntimeError("Model did not return eager attention weights")
                signals.update(
                    attention_statistics(
                        attentions,
                        query_start=max(0, len(inputs) - 2 * len(cue_ids)),
                        previous_span=previous_span,
                        current_span=current_span,
                    )
                )
                self._sync()
                attention_stat_s = perf_counter() - stat_started
                # Do not hold attention matrices during answer decoding.
                del attentions
                output.attentions = None
                if isinstance(output, dict):
                    # Transformers ModelOutput subclasses forbid pop/del. Assigning
                    # through the mapping clears its retained tensor reference too.
                    output["attentions"] = None
            probe_cache = output.past_key_values
            logits = output.logits[0, -1].float()
            generated: list[int] = []
            probabilities: list[float] = []
            # Next-token distributions, kept only for ASAG's probability mixture.
            distributions: list[torch.Tensor] = []
            decode_evaluations = 0
            complete = False
            eos_ids = self._eos_ids()
            probe_limit = config.probe_max_tokens if answer_probe else 0
            for _ in range(probe_limit):
                distribution = torch.softmax(logits, dim=-1)
                token_id = int(torch.argmax(logits).item())
                generated.append(token_id)
                probabilities.append(float(distribution[token_id].item()))
                if asag:
                    distributions.append(distribution)
                # Cue ends in \\boxed: completion needs the balanced closing brace.
                if "}" in self._decode([token_id]):
                    complete = self._boxed_content(generated)[2]
                if token_id in eos_ids or complete:
                    break
                if len(generated) < probe_limit:
                    output = self._forward([token_id], probe_cache)
                    probe_cache = output.past_key_values
                    logits = output.logits[0, -1].float()
                    decode_evaluations += 1
        finally:
            if restore is not None:
                restore()
        answer, content, complete = self._boxed_content(generated)
        content_probabilities = [probabilities[i] for i in content]
        if not complete:
            confidence = 0.0
        elif asag:
            confidence = asag_confidence(content_probabilities)
        else:
            confidence = deer_confidence(content_probabilities)
        self._sync()
        elapsed = perf_counter() - started
        signals.update(
            {
                "confidence": confidence,
                "answer": answer,
                "probe_complete": complete,
                "probe_cue": cue,
                "attention_forward_s": attention_forward_s,
                "attention_statistics_s": attention_stat_s,
            }
        )
        return ProbeResult(
            token_ids=generated,
            text=self._decode(generated),
            confidence=confidence,
            complete=complete,
            model_evaluated_tokens=len(inputs) + decode_evaluations,
            prefill_tokens=len(inputs),
            decode_tokens=decode_evaluations,
            latency_s=elapsed,
            attention_overhead_s=attention_stat_s,
            signals=signals,
            # The paper's a_r: the distribution of the first answer-content token.
            first_distribution=distributions[content[0]] if content and distributions else None,
            reused_prefix_tokens=keep,
        )

    def _probe_boundary(self, policy, config, prefix_ids, cache, previous_span, current_span):
        """Run the policy's probe(s); return the answer probe, observations and charges."""
        main = self.probe(
            prefix_ids,
            config,
            previous_span=previous_span,
            current_span=current_span,
            cache=cache,
            **policy.probe_options(),
        )
        return main, dict(main.signals), [main]

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        reasoning_format: ReasoningFormat,
        config: ReasoningConfig | None = None,
        request_id: str = "request",
        max_new_tokens: int = 1024,
        seed: int = 0,
        on_token: Callable[[int, str], None] | None = None,
    ) -> GenerationResult:
        """Generate a single deterministic trajectory and measured runtime record.

        ``config=None`` disables the controller entirely. ``policy='none'``
        uses the same generation path and must have identical tokens. ``seed``
        is recorded by the harness but greedy generation consumes no random
        state, so global RNGs are deliberately not modified.
        """
        del seed
        if not prompt_ids or max_new_tokens < 1:
            raise ValueError("A nonempty prompt and positive max_new_tokens are required")
        if not reasoning_format.end_token_ids:
            raise ValueError("An explicit reasoning-end token sequence is required")
        eos_ids = self._eos_ids()
        if eos_ids.intersection(reasoning_format.end_token_ids):
            raise ValueError("Reasoning end tokens must differ from EOS to allow a final answer")
        controller = ReasoningCompressionController(config) if config is not None else None
        policy = controller.policy if controller is not None else None
        observe = controller is not None and config.policy != "none"
        singleton_end = len(reasoning_format.end_token_ids) == 1
        if observe and policy.requires_logits and not singleton_end:
            raise ValueError("Adaptive output-logit signals require a singleton end marker")
        # Recording-only margins are skipped when the end marker spans several tokens.
        uses_logits = observe and policy.uses_logits and singleton_end
        if policy is not None and policy.requires_attention:
            self._require_eager()
        state = RequestState(request_id, reasoning_format)
        generated: list[int] = []
        queued: deque[int] = deque()
        probe_history: list[dict[str, Any]] = []
        candidate_positions = [len(prompt_ids)]
        metrics: dict[str, Any] = {
            "normal_prefill_tokens": len(prompt_ids),
            "normal_decode_model_evaluated_tokens": 0,
            "normal_model_evaluated_tokens": 0,
            "probe_model_evaluated_tokens": 0,
            "probe_prefill_tokens": 0,
            "probe_decode_model_evaluated_tokens": 0,
            "probe_reused_prefix_tokens": 0,
            "probe_tokens": 0,
            "probe_latency_s": 0.0,
            "controller_overhead_s": 0.0,
            "attention_statistics_overhead_s": 0.0,
            "attention_probe_forward_s": 0.0,
            "intervention_overhead_s": 0.0,
            "peak_memory": None,
            "kv_cache_peak_bytes": None,
            "time_to_first_token_s": None,
            "reasoning_latency_s": None,
            "backend": "transformers_reference",
            "serving_throughput": None,
            "synchronization_overhead_s": None,
        }
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sync()
        started = perf_counter()
        output = self._forward(prompt_ids)
        cache = output.past_key_values
        logits = output.logits[0, -1].float()
        text_stream = _TextStream(self._decode)
        last_token_forced = False
        for index in range(max_new_tokens):
            if observe and not queued:
                decision_started = perf_counter()
                next_text = (
                    self._decode([int(torch.argmax(logits).item())])
                    if config.boundary == "transition"
                    else ""
                )
                boundary = controller.is_boundary(
                    state, next_text, last_token_forced=last_token_forced
                )
                metrics["controller_overhead_s"] += perf_counter() - decision_started
                if boundary:
                    logits = self._intervene(
                        controller,
                        state,
                        logits,
                        cache,
                        list(prompt_ids),
                        generated,
                        queued,
                        candidate_positions,
                        probe_history,
                        metrics,
                        uses_logits=uses_logits,
                    )
            last_token_forced = bool(queued)
            token_id = queued.popleft() if last_token_forced else int(torch.argmax(logits).item())
            generated.append(token_id)
            last = token_id in eos_ids or index + 1 == max_new_tokens
            token_text = text_stream.push(token_id, final=last)
            state.append(token_id, token_text)
            if on_token is not None:
                on_token(token_id, token_text)
            if metrics["time_to_first_token_s"] is None:
                self._sync()
                metrics["time_to_first_token_s"] = perf_counter() - started
            if state.phase == "answer" and metrics["reasoning_latency_s"] is None:
                self._sync()
                metrics["reasoning_latency_s"] = perf_counter() - started
            if last:
                break
            output = self._forward([token_id], cache)
            cache = output.past_key_values
            logits = output.logits[0, -1].float()
            metrics["normal_decode_model_evaluated_tokens"] += 1
        self._sync()
        metrics["latency_s"] = perf_counter() - started
        metrics["normal_model_evaluated_tokens"] = (
            metrics["normal_prefill_tokens"] + metrics["normal_decode_model_evaluated_tokens"]
        )
        metrics["model_evaluated_tokens"] = (
            metrics["normal_model_evaluated_tokens"] + metrics["probe_model_evaluated_tokens"]
        )
        if metrics["reasoning_latency_s"] is None:
            metrics["reasoning_latency_s"] = metrics["latency_s"]
        if self.device.type == "cuda":
            metrics["peak_memory"] = torch.cuda.max_memory_allocated(self.device)
        metrics["probe_history"] = probe_history
        metrics["finish_reason"] = "eos" if generated[-1] in eos_ids else "length"
        metrics["total_generated_tokens"] = len(generated)
        metrics["reasoning_tokens"] = state.reasoning_tokens
        metrics["answer_tokens"] = state.answer_tokens
        metrics["marker_tokens"] = state.marker_tokens
        # This is observed attention-probe forward + reduction time, not the
        # unmeasured marginal slowdown versus the same forward with fused attention.
        metrics["attention_latency_s"] = (
            metrics["attention_probe_forward_s"] + metrics["attention_statistics_overhead_s"]
        )
        text = self._decode(generated)
        end_ids = list(reasoning_format.end_token_ids)
        answer_start = None
        for offset in range(len(generated) - len(end_ids) + 1):
            if generated[offset : offset + len(end_ids)] == end_ids:
                answer_start = offset + len(end_ids)
                break
        answer = self._decode(generated[answer_start:]) if answer_start is not None else ""
        state.final_answer = answer
        state.probe_tokens = metrics["probe_tokens"]
        state.probe_model_evaluated_tokens = metrics["probe_model_evaluated_tokens"]
        state.timing.update({key: value for key, value in metrics.items() if key.endswith("_s")})
        return GenerationResult(state=state, text=text, final_answer=answer, metrics=metrics)

    def _intervene(
        self,
        controller,
        state,
        logits,
        cache,
        prompt_ids,
        generated,
        queued,
        candidate_positions,
        probe_history,
        metrics,
        *,
        uses_logits,
    ):
        """Observe one boundary, decide, and apply the action; returns next logits."""
        config, policy = controller.config, controller.policy
        reasoning_format = state.format
        observation: dict[str, Any] = {}
        observe_started = perf_counter()
        if uses_logits:
            observation.update(
                logit_observation(
                    logits, reasoning_format.end_token_ids[0], temperature=config.temperature
                )
            )
        wants_probe = controller.needs_probe(state, observation)
        metrics["controller_overhead_s"] += perf_counter() - observe_started
        position = len(prompt_ids) + len(generated)
        probe_result = None
        if wants_probe:
            current_span = (
                (candidate_positions[-1], position) if position > candidate_positions[-1] else None
            )
            previous_span = (
                (candidate_positions[-2], candidate_positions[-1])
                if len(candidate_positions) >= 2
                else None
            )
            probe_result, probe_signals, charged_probes = self._probe_boundary(
                policy, config, prompt_ids + generated, cache, previous_span, current_span
            )
            observation.update(probe_signals)
            for charged in charged_probes:
                metrics["probe_tokens"] += len(charged.token_ids)
                metrics["probe_model_evaluated_tokens"] += charged.model_evaluated_tokens
                metrics["probe_prefill_tokens"] += charged.prefill_tokens
                metrics["probe_decode_model_evaluated_tokens"] += charged.decode_tokens
                metrics["probe_reused_prefix_tokens"] += charged.reused_prefix_tokens
                metrics["probe_latency_s"] += charged.latency_s
                metrics["attention_statistics_overhead_s"] += charged.attention_overhead_s
                metrics["attention_probe_forward_s"] += charged.signals.get(
                    "attention_forward_s", 0.0
                )
                probe_history.append(
                    {
                        "reasoning_tokens": state.reasoning_tokens,
                        "answer": charged.text,
                        "confidence": charged.confidence,
                        "complete": charged.complete,
                        "model_evaluated_tokens": charged.model_evaluated_tokens,
                        "latency_s": charged.latency_s,
                        "attention_observed": "attention_entropy" in charged.signals,
                    }
                )
        candidate_positions.append(position)
        decision_started = perf_counter()
        decision = controller.decide(state, observation)
        metrics["controller_overhead_s"] += perf_counter() - decision_started
        action_started = perf_counter()
        if decision.kind == "stop":
            StopAction(state.format, state.pending_end_prefix).apply(queued)
        elif decision.kind == "jump":
            JumpAction(self._encode(ASAG_JUMP)).apply(queued)
        elif decision.kind == "bias":
            if probe_result is None or probe_result.first_distribution is None:
                raise RuntimeError("Probability mixture requires an observed answer distribution")
            logits = LogitMixtureAction(
                probe_result.first_distribution, decision.mixture_weight
            ).apply(logits)
        metrics["intervention_overhead_s"] += perf_counter() - action_started
        return logits
