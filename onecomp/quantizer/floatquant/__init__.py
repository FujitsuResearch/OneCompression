"""FloatQuant (NVFP4 / MXFP4 / FP8) microscaling quantization module

This module provides fake-quantization functionality for microscaling
floating-point formats: NVFP4 (FP4 E2M1 with FP8 E4M3 block scales and a
per-tensor FP32 scale), MXFP4 (FP4 E2M1 with E8M0 power-of-two block
scales, OCP Microscaling specification), and FP8 E4M3, plus exporters
that write vLLM-native low-precision checkpoints.

Classes:
    FloatQuantResult: Result class for FloatQuant quantization containing codes and scales.
    FloatQuant: FloatQuant quantizer class supporting NVFP4 / MXFP4 / FP8 formats.

Functions:
    save_vllm_native_model: Export FloatQuant results as a vLLM-native
        compressed-tensors checkpoint (nvfp4 / mxfp4 / fp8; nvfp4
        optionally with quantized activations, W4A4).
    collect_input_global_scales: Calibrate per-layer activation global
        scales for NVFP4 W4A4 export.
    diagnose_nvfp4_fused_export_gap: Measure the NVFP4 fused-group export
        gap caused by vLLM global-scale unification.
    select_mixed_formats: Sensitivity-based per-layer NVFP4 / FP8
        assignment under a memory budget.
    save_vllm_mixed_model: Export mixed NVFP4 / FP8 results in the
        compressed-tensors ``mixed-precision`` format.
    save_vllm_fp8_model: Export a model as a plain vLLM FP8 checkpoint
        with per-tensor scales (legacy path).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from ._floatquant import FloatQuant, FloatQuantResult
from .config import (
    DEFAULT_BLOCK_SIZES,
    SUPPORTED_FORMATS,
    SUPPORTED_SCALE_CANDIDATE_STRATEGIES,
    SUPPORTED_SCALE_OBJECTIVES,
    SUPPORTED_SCALE_TIMINGS,
)
from .vllm_export import (
    collect_input_global_scales,
    diagnose_nvfp4_fused_export_gap,
    save_vllm_fp8_model,
    save_vllm_mixed_model,
    save_vllm_native_model,
    select_mixed_formats,
)

__all__ = [
    "DEFAULT_BLOCK_SIZES",
    "FloatQuant",
    "FloatQuantResult",
    "SUPPORTED_FORMATS",
    "SUPPORTED_SCALE_CANDIDATE_STRATEGIES",
    "SUPPORTED_SCALE_OBJECTIVES",
    "SUPPORTED_SCALE_TIMINGS",
    "collect_input_global_scales",
    "diagnose_nvfp4_fused_export_gap",
    "save_vllm_fp8_model",
    "save_vllm_mixed_model",
    "save_vllm_native_model",
    "select_mixed_formats",
]
