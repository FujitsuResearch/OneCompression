# Copyright 2025-2026 Fujitsu Ltd.

"""Translate portable policies into vLLM request sampling metadata.

The installed and inspected API is vLLM 0.21's batched v1 logits processor.
Policies outside ``supported_policies`` fail explicitly.
"""

from dataclasses import asdict
from importlib.metadata import version

from onecomp.reasoning import ReasoningConfig, ReasoningFormat


class VLLMReasoningAdapter:
    processor_path = "onecomp.runtime.vllm.logits_processor:ReasoningLogitsProcessor"
    supported_policies = frozenset({"none", "fixed", "prompt_budget", "thinkbrake"})

    @staticmethod
    def check_version():
        installed = version("vllm")
        if tuple(installed.split(".")[:2]) != ("0", "21"):
            raise RuntimeError(
                f"Reasoning logits integration was inspected for vLLM 0.21.x; found {installed}. "
                "Validate the new BatchUpdate/LogitsProcessor API before enabling it."
            )
        return installed

    @classmethod
    def extra_args(
        cls, config: ReasoningConfig, reasoning_format: ReasoningFormat, request_id="request"
    ):
        if config.policy not in cls.supported_policies:
            raise NotImplementedError(
                f"vLLM logits hooks do not support {config.policy!r}; use the Transformers "
                "reference backend."
            )
        if config.policy == "thinkbrake" and len(reasoning_format.end_token_ids) != 1:
            raise ValueError("ThinkBrake requires a singleton reasoning-end token")
        return {
            "onecomp_reasoning": {
                "config": config.to_dict(),
                "format": asdict(reasoning_format),
                "request_id": str(request_id),
            }
        }
