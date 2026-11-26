"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "calculate_accuracy": (".accuracy", "calculate_accuracy"),
    "capture_input_activations": (".activation_capture", "capture_input_activations"),
    "check_activations": (".activation_check", "check_activations"),
    "expand_kwargs_batch": (".blockwise", "expand_kwargs_batch"),
    "forward_input": (".blockwise", "forward_input"),
    "get_blocks_and_inputs": (".blockwise", "get_blocks_and_inputs"),
    "move_kwargs_to_device": (".blockwise", "move_kwargs_to_device"),
    "cleanup_memory": (".device", "cleanup_memory"),
    "empty_cache": (".device", "empty_cache"),
    "get_default_device": (".device", "get_default_device"),
    "needs_bfloat16": (".dtype", "needs_bfloat16"),
    "add_model_specific_inputs": (".model_inputs", "add_model_specific_inputs"),
    "calculate_perplexity": (".perplexity", "calculate_perplexity"),
    "VRAMBitwidthEstimation": (".vram_estimator", "VRAMBitwidthEstimation"),
    "effective_bits_for_quantizer": (".vram_estimator", "effective_bits_for_quantizer"),
    "effective_bits_per_param": (".vram_estimator", "effective_bits_per_param"),
    "estimate_target_bitwidth": (".vram_estimator", "estimate_target_bitwidth"),
    "estimate_wbits_from_vram": (".vram_estimator", "estimate_wbits_from_vram"),
    "raw_bits_for_quantizer": (".vram_estimator", "raw_bits_for_quantizer"),
    "weight_memory_gb": (".vram_estimator", "weight_memory_gb"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
