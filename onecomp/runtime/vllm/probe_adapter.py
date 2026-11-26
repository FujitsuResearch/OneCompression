# Copyright 2025-2026 Fujitsu Ltd.

"""Offline vLLM generation with independent requests for answer probes."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Sequence

from onecomp.reasoning import (
    ReasoningCompressionController,
    ReasoningConfig,
    ReasoningFormat,
    RequestState,
)
from onecomp.reasoning.actions import JumpAction, LogitMixtureAction, StopAction
from onecomp.reasoning.signals import asag_confidence, deer_confidence

from ..common import (
    ASAG_JUMP,
    ASAG_PROBE_CUE,
    DEER_PROBE_CUE,
    GenerationResult,
    TextStream,
    boxed_content,
)
from .adapter import VLLMReasoningAdapter

if TYPE_CHECKING:
    from vllm import LLM

    from ..transformers import TransformersReasoningAdapter


def _validate_logits_processors(model_config):
    # vLLM loads both engine-configured processors and installed entry points.
    processors = list(getattr(model_config, "logits_processors", None) or ())
    processors.extend(plugin.value for plugin in entry_points(group="vllm.logits_processors"))
    for processor in processors:
        if isinstance(processor, str):
            if processor == VLLMReasoningAdapter.processor_path:
                continue
            name = processor
        else:
            from .logits_processor import ReasoningLogitsProcessor

            if processor is ReasoningLogitsProcessor:
                continue
            name = processor.__qualname__
        raise ValueError(
            f"The vLLM probe adapter cannot use custom logits processor {name!r}: "
            "raw_logprobs are measured before processors can change the sampled token."
        )


@dataclass
class _Work:
    requests: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    sampled_tokens: int = 0

    @property
    def estimated_model_evaluated_tokens(self):
        # The first sample comes from prefill logits, and the last is never forwarded.
        return self.prompt_tokens - self.cached_prompt_tokens + self.sampled_tokens - self.requests


class VLLMProbeReasoningAdapter:
    """Greedy, single-request DEER, Stability and ASAG using an offline vLLM LLM.

    The main trajectory and trial answers use separate `LLM.generate` calls.
    vLLM can reuse matching blocks via prefix caching. ASAG modes that use
    attention require an eager Transformers adapter with matching weights and
    tokenizer to measure attention on the same prefix and cue.
    """

    supported_policies = frozenset({"deer", "stability", "asag_reference"})

    def __init__(
        self,
        llm: LLM,
        *,
        attention_adapter: TransformersReasoningAdapter | None = None,
        lookahead_tokens: int = 32,
    ):
        if type(lookahead_tokens) is not int or lookahead_tokens < 1:
            raise ValueError("lookahead_tokens must be a positive integer")
        if llm.llm_engine.vllm_config.speculative_config is not None:
            raise ValueError("The vLLM probe adapter requires speculative decoding to be disabled")
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        if self.tokenizer is None:
            raise ValueError("The vLLM probe adapter needs a tokenizer")
        model_config = llm.model_config
        if model_config.logprobs_mode != "raw_logprobs":
            raise ValueError("The vLLM probe adapter requires logprobs_mode='raw_logprobs'")
        # This adapter never sends metadata that activates OneComp's own processor.
        _validate_logits_processors(model_config)
        self.attention_adapter = attention_adapter
        self.lookahead_tokens = lookahead_tokens
        self._context_length = model_config.max_model_len
        self._vocab_size = model_config.get_vocab_size()
        eos = model_config.try_get_generation_config().get("eos_token_id")
        self._eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self._eos_ids.add(self.tokenizer.eos_token_id)
        self._eos_ids.discard(None)

    def _encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode(self, ids):
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def _sample(self, prefix_ids, max_tokens, work, *, logprobs=None, stop=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            logprobs=logprobs,
            detokenize=stop is not None,
            stop=stop,
            include_stop_str_in_output=stop is not None,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        [output] = self.llm.generate(
            TokensPrompt(prompt_token_ids=prefix_ids), params, use_tqdm=False
        )
        [completion] = output.outputs
        if not completion.token_ids or completion.finish_reason not in {"stop", "length"}:
            raise RuntimeError(f"Unexpected vLLM completion: {completion.finish_reason!r}")
        work.requests += 1
        work.prompt_tokens += len(prefix_ids)
        work.cached_prompt_tokens += output.num_cached_tokens or 0
        work.sampled_tokens += len(completion.token_ids)
        return completion

    def _probe(self, prefix, spans, config, options, cue, cue_ids, work, attention_probe):
        started = perf_counter()
        probe_prefix = prefix + cue_ids
        room = self._context_length - len(probe_prefix)
        observation = {}
        attention_tokens = 0
        if attention_probe is not None and room >= 0:
            previous_span, current_span = spans
            attention = attention_probe(
                prefix, previous_span=previous_span, current_span=current_span
            )
            observation.update(attention.signals)
            attention_tokens = attention.model_evaluated_tokens

        answer, confidence, complete = "", None, False
        first_content_prefix = None
        ids, rows = [], []
        if options["answer_probe"] and room > 0:
            limit = min(config.probe_max_tokens, room)
            while len(ids) < limit:
                completion = self._sample(
                    probe_prefix + ids, limit - len(ids), work, logprobs=0, stop="}"
                )
                if completion.logprobs is None or len(completion.logprobs) != len(
                    completion.token_ids
                ):
                    raise RuntimeError("vLLM did not return the trial answer's token logprobs")
                ids.extend(completion.token_ids)
                rows.extend(completion.logprobs)
                answer, content, complete = boxed_content(ids, self._decode)
                # A closing brace can end an inner group. Resume from the same token prefix.
                if complete or completion.stop_reason != "}" or ids[-1] in self._eos_ids:
                    break
            if complete:
                probabilities = [math.exp(rows[i][ids[i]].logprob) for i in content]
                if config.policy == "asag_reference":
                    confidence = asag_confidence(probabilities)
                    first_content_prefix = probe_prefix + ids[: content[0]]
                else:
                    confidence = deer_confidence(probabilities)
        observation.update(
            confidence=confidence, answer=answer, probe_complete=complete, probe_cue=cue
        )
        history = {
            "answer": answer,
            "confidence": confidence,
            "complete": complete,
            "sampled_tokens": len(ids),
            "attention_model_evaluated_tokens": attention_tokens,
            "attention_observed": "attention_entropy" in observation,
            "latency_s": perf_counter() - started,
        }
        return observation, first_content_prefix, history

    def _vocab_logprobs(self, prefix, work):
        import torch

        # An explicit top-k of vocab size gives ID-keyed rows through the
        # public completion API. Only ask for this at the two mixture positions.
        completion = self._sample(prefix, 1, work, logprobs=self._vocab_size)
        rows = completion.logprobs
        if not rows or len(rows[0]) != self._vocab_size:
            raise RuntimeError("ASAG bias requires vLLM logprobs for the full vocabulary")
        logprobs = torch.tensor(
            [rows[0][i].logprob for i in range(self._vocab_size)], dtype=torch.float32
        )
        if not torch.isfinite(logprobs.logsumexp(dim=0)):
            raise RuntimeError("vLLM returned an invalid next-token distribution")
        return logprobs

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        reasoning_format: ReasoningFormat,
        config: ReasoningConfig,
        max_new_tokens: int = 1024,
        request_id: str = "request",
        on_token: Callable[[int, str], None] | None = None,
    ) -> GenerationResult:
        """Generate from token IDs and report the accepted trajectory and probe work.

        When set, on_token(token_id, text) reports accepted tokens in order.
        Calls arrive in chunks after synchronous vLLM generation returns. The
        remaining final answer can arrive as one chunk.
        """
        if config.policy not in self.supported_policies:
            raise ValueError(f"{config.policy!r} uses VLLMReasoningAdapter's logits processor")
        prompt_ids = list(prompt_ids)
        if not prompt_ids or type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("A nonempty prompt and positive max_new_tokens are required")
        limit = min(max_new_tokens, self._context_length - len(prompt_ids))
        if limit < 1:
            raise ValueError("The prompt must leave room for at least one output token")
        if self._eos_ids.intersection(reasoning_format.end_token_ids):
            raise ValueError("Reasoning end tokens must differ from EOS to allow a final answer")
        if any(i >= self._vocab_size for i in reasoning_format.end_token_ids):
            raise ValueError("Reasoning end token lies outside the model vocabulary")

        controller = ReasoningCompressionController(config)
        options = controller.policy.probe_options()
        asag = config.policy == "asag_reference"
        cue = ASAG_PROBE_CUE if asag else DEER_PROBE_CUE
        cue_ids = self._encode(cue)
        if not cue_ids:
            raise ValueError("Probe cue must tokenize to at least one token")
        attention_probe = None
        if options["collect_attention"]:
            if self.attention_adapter is None:
                raise ValueError("This ASAG mode needs an eager Transformers attention_adapter")
            if self.attention_adapter._encode(cue) != cue_ids:
                raise ValueError(
                    "The ASAG attention adapter must use the same probe-cue token IDs"
                )
        if asag and config.asag_mode == "full" and config.enable_injection:
            cap = self.llm.model_config.max_logprobs
            if cap != -1 and cap < self._vocab_size:
                raise ValueError("ASAG probability mixing needs LLM(max_logprobs=-1)")
        if options["collect_attention"]:
            attention_probe = self.attention_adapter.start_attention_probe(config)

        state = RequestState(request_id, reasoning_format)
        generated = state.generated_token_ids
        forced: deque[int] = deque()
        candidates: deque[int] = deque()
        last_token_forced = False
        candidate_positions = [len(prompt_ids)]
        stream = TextStream(self._decode)
        main_work, probe_work = _Work(), _Work()
        history = []
        discarded = 0
        finish_reason = "length"

        def refill_candidates():
            remaining = limit - len(generated)
            count = remaining if state.phase == "answer" else min(remaining, self.lookahead_tokens)
            completion = self._sample(prompt_ids + generated, count, main_work)
            candidates.extend(completion.token_ids)

        started = perf_counter()
        while len(generated) < limit:
            if not forced:
                next_text = ""
                # Only transition boundaries need a pending model token.
                if state.phase == "reasoning" and config.boundary == "transition":
                    if not candidates:
                        refill_candidates()
                    next_text = self._decode([candidates[0]])
                if controller.is_boundary(state, next_text, last_token_forced=last_token_forced):
                    prefix = prompt_ids + generated
                    position = len(prefix)
                    previous = (
                        tuple(candidate_positions[-2:]) if len(candidate_positions) > 1 else None
                    )
                    start = candidate_positions[-1]
                    current = (start, position) if start < position else None
                    observation, content_prefix, probe = self._probe(
                        prefix,
                        (previous, current),
                        config,
                        options,
                        cue,
                        cue_ids,
                        probe_work,
                        attention_probe,
                    )
                    history.append({"reasoning_tokens": state.reasoning_tokens, **probe})
                    if start < position:
                        candidate_positions.append(position)
                    decision = controller.decide(state, observation)
                    if decision.kind == "stop":
                        StopAction(state.format, state.pending_end_prefix).apply(forced)
                    elif decision.kind == "jump":
                        jump_ids = self._encode(ASAG_JUMP)
                        if not jump_ids or self._eos_ids.intersection(jump_ids):
                            raise ValueError("ASAG jump needs nonempty tokens without EOS")
                        JumpAction(jump_ids).apply(forced)
                    elif decision.kind == "bias":
                        if content_prefix is None:
                            raise RuntimeError("ASAG bias requires a complete trial answer")
                        answer_dist = self._vocab_logprobs(content_prefix, probe_work).softmax(-1)
                        main_logits = self._vocab_logprobs(prefix, main_work)
                        mixed = LogitMixtureAction(answer_dist, decision.mixture_weight)
                        replacement = int(mixed.apply(main_logits).argmax().item())
                        if not candidates or replacement != candidates[0]:
                            discarded += len(candidates)
                            candidates.clear()
                            candidates.append(replacement)
                    if forced:
                        discarded += len(candidates)
                        candidates.clear()

            last_token_forced = bool(forced)
            if last_token_forced:
                token = forced.popleft()
            else:
                if not candidates:
                    refill_candidates()
                token = candidates.popleft()
            eos = token in self._eos_ids
            text = stream.push(token, final=eos or len(generated) + 1 == limit)
            state.append(token, text)
            if attention_probe is not None and state.phase == "answer":
                attention_probe = None
            if on_token is not None:
                on_token(token, text)
            if eos:
                finish_reason = "eos"
                break

        answer_start = next(
            (i for i, kind in enumerate(state.token_kinds) if kind == "answer"), len(generated)
        )
        state.final_answer = self._decode(generated[answer_start:])
        state.probe_tokens = sum(probe["sampled_tokens"] for probe in history)
        attention_tokens = sum(probe["attention_model_evaluated_tokens"] for probe in history)
        state.probe_model_evaluated_tokens = (
            probe_work.estimated_model_evaluated_tokens + attention_tokens
        )
        state.timing["latency_s"] = perf_counter() - started
        metrics = {
            "backend": "vllm_probe",
            "finish_reason": finish_reason,
            "total_generated_tokens": len(generated),
            "reasoning_tokens": state.reasoning_tokens,
            "answer_tokens": state.answer_tokens,
            "marker_tokens": state.marker_tokens,
            "model_evaluated_tokens": main_work.estimated_model_evaluated_tokens
            + state.probe_model_evaluated_tokens,
            "probe_model_evaluated_tokens": state.probe_model_evaluated_tokens,
            "model_evaluated_tokens_are_estimates": True,
            "probe_tokens": state.probe_tokens,
            "probe_history": history,
            "vllm_main_work": asdict(main_work),
            "vllm_probe_work": asdict(probe_work),
            "vllm_discarded_main_tokens": discarded,
            **state.timing,
        }
        return GenerationResult(state, self._decode(generated), state.final_answer, metrics)
