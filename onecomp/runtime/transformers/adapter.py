# Copyright 2025-2026 Fujitsu Ltd.

"""A serial, greedy reference backend with fully accounted model evaluations.

This backend deliberately favors inspectable semantics over serving performance.
Every forced marker token passes through the normal KV-cache path.
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
        observe = controller is not None and config.policy != "none"
        state = RequestState(request_id, reasoning_format)
        generated: list[int] = []
        queued: deque[int] = deque()
        metrics: dict[str, Any] = {
            "normal_prefill_tokens": len(prompt_ids),
            "normal_decode_model_evaluated_tokens": 0,
            "normal_model_evaluated_tokens": 0,
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
                    self._intervene(controller, state, queued, metrics)
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
        metrics["model_evaluated_tokens"] = metrics["normal_model_evaluated_tokens"]
        if metrics["reasoning_latency_s"] is None:
            metrics["reasoning_latency_s"] = metrics["latency_s"]
        if self.device.type == "cuda":
            metrics["peak_memory"] = torch.cuda.max_memory_allocated(self.device)
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
        state.timing.update({key: value for key, value in metrics.items() if key.endswith("_s")})
        return GenerationResult(state=state, text=text, final_answer=answer, metrics=metrics)

    def _intervene(self, controller, state, queued, metrics):
        """Decide at one boundary and queue the action's tokens."""
        decision_started = perf_counter()
        decision = controller.decide(state, {})
        metrics["controller_overhead_s"] += perf_counter() - decision_started
        action_started = perf_counter()
        if decision.kind == "stop":
            StopAction(state.format, state.pending_end_prefix).apply(queued)
        metrics["intervention_overhead_s"] += perf_counter() - action_started
