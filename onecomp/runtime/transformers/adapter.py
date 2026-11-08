# Copyright 2025-2026 Fujitsu Ltd.

"""A serial, greedy reference backend with isolated, fully accounted probes.

This backend deliberately favors inspectable semantics over serving performance.
Every forced marker token passes through the normal KV-cache path. Answer probes
replay the full prefix in a new cache, so the request's cache is never touched.
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
from onecomp.reasoning.actions import StopAction
from onecomp.reasoning.signals import deer_confidence, logit_observation

DEER_PROBE_CUE = "\n**Final Answer**\n\\boxed"


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
    signals: dict[str, Any]


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
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model.eval()
        self.tokenizer = tokenizer
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

    def _forward(self, ids: Sequence[int], cache: Any = None):
        inputs = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        kwargs = {"logits_to_keep": 1} if self._supports_logits_to_keep else {}
        return self.model(
            input_ids=inputs,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )

    def _boxed_content(self, token_ids):
        """Indices of tokens overlapping boxed answer content, excluding delimiters."""
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
        left = 0
        for index in range(len(token_ids)):
            right = len(self._decode(token_ids[: index + 1]))
            if text[max(left, start + 1) : min(right, end)].strip():
                selected.append(index)
            left = right
        answer = text[start + 1 : end].strip()
        return answer, selected, bool(answer and selected)

    @torch.inference_mode()
    def probe(self, prefix_ids: Sequence[int], config: ReasoningConfig) -> ProbeResult:
        """Run a disposable answer probe and count every model-evaluated token.

        The probe appends DEER's answer cue to the prefix and decodes greedily, so
        each selected probability is the maximum. Confidence is the geometric mean
        over answer-content tokens. A probe truncated before EOS or a closing
        boxed-answer brace is not confident.
        """
        cue_ids = self._encode(DEER_PROBE_CUE)
        if not cue_ids:
            raise ValueError("Probe cue must tokenize to at least one token")
        inputs = list(prefix_ids) + cue_ids
        self._sync()
        started = perf_counter()
        output = self._forward(inputs)
        probe_cache = output.past_key_values
        logits = output.logits[0, -1].float()
        generated: list[int] = []
        probabilities: list[float] = []
        decode_evaluations = 0
        complete = False
        eos_ids = self._eos_ids()
        for _ in range(config.probe_max_tokens):
            distribution = torch.softmax(logits, dim=-1)
            token_id = int(torch.argmax(logits).item())
            generated.append(token_id)
            probabilities.append(float(distribution[token_id].item()))
            # Cue ends in \\boxed: completion needs the balanced closing brace.
            if "}" in self._decode([token_id]):
                complete = self._boxed_content(generated)[2]
            if token_id in eos_ids or complete:
                break
            if len(generated) < config.probe_max_tokens:
                output = self._forward([token_id], probe_cache)
                probe_cache = output.past_key_values
                logits = output.logits[0, -1].float()
                decode_evaluations += 1
        answer, content, complete = self._boxed_content(generated)
        confidence = deer_confidence([probabilities[i] for i in content]) if complete else 0.0
        self._sync()
        return ProbeResult(
            token_ids=generated,
            text=self._decode(generated),
            confidence=confidence,
            complete=complete,
            model_evaluated_tokens=len(inputs) + decode_evaluations,
            prefill_tokens=len(inputs),
            decode_tokens=decode_evaluations,
            latency_s=perf_counter() - started,
            signals={
                "confidence": confidence,
                "answer": answer,
                "probe_complete": complete,
                "probe_cue": DEER_PROBE_CUE,
            },
        )

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
        state = RequestState(request_id, reasoning_format)
        generated: list[int] = []
        queued: deque[int] = deque()
        probe_history: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {
            "normal_prefill_tokens": len(prompt_ids),
            "normal_decode_model_evaluated_tokens": 0,
            "normal_model_evaluated_tokens": 0,
            "probe_model_evaluated_tokens": 0,
            "probe_prefill_tokens": 0,
            "probe_decode_model_evaluated_tokens": 0,
            "probe_tokens": 0,
            "probe_latency_s": 0.0,
            "controller_overhead_s": 0.0,
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
        for index in range(max_new_tokens):
            if observe and not queued:
                decision_started = perf_counter()
                next_text = (
                    self._decode([int(torch.argmax(logits).item())])
                    if config.boundary == "transition"
                    else ""
                )
                boundary = controller.is_boundary(state, next_text)
                metrics["controller_overhead_s"] += perf_counter() - decision_started
                if boundary:
                    self._intervene(
                        controller,
                        state,
                        logits,
                        list(prompt_ids) + generated,
                        queued,
                        probe_history,
                        metrics,
                        uses_logits=uses_logits,
                    )
            token_id = queued.popleft() if queued else int(torch.argmax(logits).item())
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
        prefix_ids,
        queued,
        probe_history,
        metrics,
        *,
        uses_logits,
    ):
        """Observe one boundary (logits, then an answer probe if needed) and act."""
        observe_started = perf_counter()
        observation = (
            logit_observation(
                logits, state.format.end_token_ids[0], temperature=controller.config.temperature
            )
            if uses_logits
            else {}
        )
        wants_probe = controller.needs_probe(state, observation)
        metrics["controller_overhead_s"] += perf_counter() - observe_started
        if wants_probe:
            probe = self.probe(prefix_ids, controller.config)
            observation.update(probe.signals)
            metrics["probe_tokens"] += len(probe.token_ids)
            metrics["probe_model_evaluated_tokens"] += probe.model_evaluated_tokens
            metrics["probe_prefill_tokens"] += probe.prefill_tokens
            metrics["probe_decode_model_evaluated_tokens"] += probe.decode_tokens
            metrics["probe_latency_s"] += probe.latency_s
            probe_history.append(
                {
                    "reasoning_tokens": state.reasoning_tokens,
                    "answer": probe.text,
                    "confidence": probe.confidence,
                    "complete": probe.complete,
                    "model_evaluated_tokens": probe.model_evaluated_tokens,
                    "latency_s": probe.latency_s,
                }
            )
        decision_started = perf_counter()
        decision = controller.decide(state, observation)
        metrics["controller_overhead_s"] += perf_counter() - decision_started
        action_started = perf_counter()
        if decision.kind == "stop":
            StopAction(state.format, state.pending_end_prefix).apply(queued)
        metrics["intervention_overhead_s"] += perf_counter() - action_started
