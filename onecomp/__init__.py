"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .__version__ import __version__

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "CalibrationConfig": (".calibration", "CalibrationConfig"),
    "ModelConfig": (".model_config", "ModelConfig"),
    "QEPConfig": (".qep", "QEPConfig"),
    "LPCDConfig": (".lpcd", "LPCDConfig"),
    "RotatedModelConfig": (".rotated_model_config", "RotatedModelConfig"),
    "Runner": (".runner", "Runner"),
    "CompactionConfig": (".compaction", "CompactionConfig"),
    "CompactionRunner": (".compaction", "CompactionRunner"),
    "setup_logger": (".log", "setup_logger"),
    "GGUFExportConfig": (".export", "GGUFExportConfig"),
    "export_gguf": (".export", "export_gguf"),
    "generate_model_card": (".export", "generate_model_card"),
    "push_to_hub": (".export", "push_to_hub"),
    "QuantizedModelLoader": (".quantized_model_loader", "QuantizedModelLoader"),
    "StreamingUnsupportedError": (".quantized_model_loader", "StreamingUnsupportedError"),
    "Quantizer": (".quantizer", "Quantizer"),
    "ResultLoader": (".quantizer", "ResultLoader"),
    "AutoBitQuantizer": (".quantizer", "AutoBitQuantizer"),
    "GPTQ": (".quantizer", "GPTQ"),
    "Onebit": (".quantizer", "Onebit"),
    "DBF": (".quantizer", "DBF"),
    "MDBF": (".quantizer", "MDBF"),
    "JointQ": (".quantizer", "JointQ"),
    "RTN": (".quantizer", "RTN"),
    "QUIP": (".quantizer", "QUIP"),
    "CQ": (".quantizer", "CQ"),
    "ARB": (".quantizer", "ARB"),
    "QBB": (".quantizer", "QBB"),
    "calculate_perplexity": (".utils", "calculate_perplexity"),
    "calculate_accuracy": (".utils", "calculate_accuracy"),
    "check_activations": (".utils", "check_activations"),
    "capture_input_activations": (".utils", "capture_input_activations"),
    "estimate_target_bitwidth": (".utils", "estimate_target_bitwidth"),
    "estimate_wbits_from_vram": (".utils", "estimate_wbits_from_vram"),
    "effective_bits_per_param": (".utils", "effective_bits_per_param"),
    "raw_bits_for_quantizer": (".utils", "raw_bits_for_quantizer"),
    "effective_bits_for_quantizer": (".utils", "effective_bits_for_quantizer"),
    "weight_memory_gb": (".utils", "weight_memory_gb"),
    "VRAMBitwidthEstimation": (".utils", "VRAMBitwidthEstimation"),
    "add_model_specific_inputs": (".utils", "add_model_specific_inputs"),
    "get_blocks_and_inputs": (".utils", "get_blocks_and_inputs"),
    "forward_input": (".utils", "forward_input"),
    "move_kwargs_to_device": (".utils", "move_kwargs_to_device"),
    "expand_kwargs_batch": (".utils", "expand_kwargs_batch"),
    "needs_bfloat16": (".utils", "needs_bfloat16"),
    "cleanup_memory": (".utils", "cleanup_memory"),
    "empty_cache": (".utils", "empty_cache"),
    "get_default_device": (".utils", "get_default_device"),
    "PostQuantizationProcess": (".post_process", "PostQuantizationProcess"),
    "BlockWisePTQ": (".post_process", "BlockWisePTQ"),
    "GlobalPTQ": (".post_process", "GlobalPTQ"),
    "GlobalPTQDistributed": (".post_process", "GlobalPTQDistributed"),
    "RouterFineTuning": (".post_process", "RouterFineTuning"),
    "PostProcessLoraSFT": (".post_process", "PostProcessLoraSFT"),
    "PostProcessLoraTeacherOnlySFT": (
        ".post_process",
        "PostProcessLoraTeacherOnlySFT",
    ),
    "PostProcessLoraTeacherSFT": (".post_process", "PostProcessLoraTeacherSFT"),
    "prepare_rotated_model": (".pre_process", "prepare_rotated_model"),
}

_LOADER_METHODS = {
    "load_quantized_model": "load_quantized_model",
    "load_quantized_model_pt": "load_quantized_model_pt",
}

__all__ = ["__version__", *_LAZY_ATTRS, *_LOADER_METHODS]


def __getattr__(name: str) -> Any:
    """Load a public symbol on first access and cache it in this module."""
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module_name, attribute_name = target
        module = import_module(module_name, __name__)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value

    method_name = _LOADER_METHODS.get(name)
    if method_name is not None:
        module = import_module(".quantized_model_loader", __name__)
        value = getattr(module.QuantizedModelLoader, method_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
