# Copyright 2025-2026 Fujitsu Ltd.

"""vLLM 0.21 continuous-batch logits hook with request-local state.

Register the class in ``LLM(logits_processors=[...])``. Pass options returned by
``VLLMReasoningAdapter.extra_args`` in ``SamplingParams(extra_args=...)``.
Missing or disabled options are exact identity operations. Forced marker tokens
are sampled through vLLM normally, preserving streaming and cache consistency.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from onecomp.reasoning import (
    ReasoningCompressionController,
    ReasoningConfig,
    ReasoningFormat,
    RequestState,
)

from .adapter import VLLMReasoningAdapter


@dataclass
class _Request:
    state: RequestState
    controller: ReasoningCompressionController
    output_ids: list[int]
    seen: int = 0
    forced: deque = field(default_factory=deque)


class ReasoningLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        options = (sampling_params.extra_args or {}).get("onecomp_reasoning")
        if options is None:
            return
        if not isinstance(options, dict) or set(options) - {"config", "format", "request_id"}:
            raise ValueError("Invalid onecomp_reasoning sampling metadata")
        config = ReasoningConfig.from_dict(options.get("config", {}))
        if config.policy == "none":
            return
        if "format" not in options:
            raise ValueError("Active reasoning compression requires an explicit format")
        reasoning_format = ReasoningFormat(**options["format"])
        VLLMReasoningAdapter.extra_args(config, reasoning_format)
        if sampling_params.n != 1:
            raise ValueError("Reasoning integration currently requires n=1 per request")
        # End-of-generation stop IDs cannot also mean transition to final answer.
        stop_ids = set(getattr(sampling_params, "all_stop_token_ids", set()))
        if stop_ids.intersection(reasoning_format.end_token_ids):
            raise ValueError("Reasoning end markers must not be configured as stop tokens")

    def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool):
        del device, is_pin_memory
        VLLMReasoningAdapter.check_version()
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError(
                "Reasoning compression is not validated with speculative decoding"
            )
        self.requests: dict[int, _Request] = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if batch_update is not None:
            # Follow interface.py's documented order, including slot reuse.
            for index in batch_update.removed:
                self.requests.pop(index, None)
            for index, params, _prompt_ids, output_ids in batch_update.added:
                self.validate_params(params)
                self.requests.pop(index, None)
                options = (params.extra_args or {}).get("onecomp_reasoning")
                if options is None:
                    continue
                config = ReasoningConfig.from_dict(options.get("config", {}))
                if config.policy == "none":
                    continue
                state = RequestState(
                    str(options.get("request_id", index)), ReasoningFormat(**options["format"])
                )
                self.requests[index] = _Request(
                    state, ReasoningCompressionController(config), output_ids
                )
            for source, destination, direction in batch_update.moved:
                first = self.requests.pop(source, None)
                second = self.requests.pop(destination, None)
                if first is not None:
                    self.requests[destination] = first
                if direction == MoveDirectionality.SWAP and second is not None:
                    self.requests[source] = second
        for request in self.requests.values():
            if len(request.output_ids) < request.seen:
                raise RuntimeError(
                    "vLLM output history shrank; unsupported cache/trajectory rewind"
                )
            for token_id in request.output_ids[request.seen :]:
                if request.forced:
                    expected = request.forced.popleft()
                    if token_id != expected:
                        raise RuntimeError("A vLLM sampler overrode a forced reasoning marker")
                # Budget policies ignore token text, so outputs are not decoded.
                request.state.append(token_id)
            request.seen = len(request.output_ids)

    def apply(self, logits):
        for index, request in self.requests.items():
            if index >= logits.shape[0]:
                raise RuntimeError("Stale reasoning request slot outside the vLLM batch")
            if not request.forced and request.controller.is_boundary(request.state):
                decision = request.controller.decide(request.state, {})
                if decision.kind == "stop":
                    request.forced.extend(request.state.format.end_token_ids)
                elif decision.kind != "continue":
                    raise RuntimeError("Unsupported intervention in vLLM logits adapter")
            if request.forced:
                token_id = request.forced[0]
                if not 0 <= token_id < logits.shape[-1]:
                    raise ValueError("Reasoning-end token lies outside the model vocabulary")
                logits[index].fill_(-float("inf"))
                logits[index, token_id] = 0.0
        return logits
