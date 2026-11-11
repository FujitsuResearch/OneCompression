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


class TransformersReasoningAdapter:
    """Greedy batch-size-one generation for an already-loaded causal LM.

    Distinct calls own their caches and controller state. A model object itself
    is not promised to support parallel forwards; use independent workers for
    simultaneous reference requests. No serving throughput claim follows from
    this serial runtime. ``on_token`` receives every visible token, including
    forced markers, exactly once and supports streaming consumers.
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
            token_text = self._decode([token_id])
            state.append(token_id, token_text)
            if on_token is not None:
                on_token(token_id, token_text)
            if metrics["time_to_first_token_s"] is None:
                self._sync()
                metrics["time_to_first_token_s"] = perf_counter() - started
            if state.phase == "answer" and metrics["reasoning_latency_s"] is None:
                self._sync()
                metrics["reasoning_latency_s"] = perf_counter() - started
            if token_id in eos_ids:
                break
            if index + 1 < max_new_tokens:
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
            StopAction(state.format).apply(queued)
        metrics["intervention_overhead_s"] += perf_counter() - action_started
