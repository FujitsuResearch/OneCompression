"""Auto-detection of quantization / dtype from saved model directories.

Carried over verbatim from the old eval_jp/utils/model_utils.py; only
the prints use the module logger so callers can route them at runtime.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path
from typing import Any

from vllm_plugins.gptq.constants import should_use_gptq_marlin

logger = getLogger(__name__)

QUANT_DTYPE_MAP: dict[str | None, str] = {
    "gptq": "float16",
    "gptq_marlin": "float16",
    "awq": "auto",
    "awq_marlin": "auto",
    "squeezellm": "auto",
    "dbf": "auto",
    None: "auto",
}


def _extract_quant_method(qconfig: dict) -> str | None:
    """Infer quant_method from a quantization_config dict."""
    method = qconfig.get("quant_method")
    if method:
        return method

    fmt = qconfig.get("checkpoint_format", "")
    if fmt in ("gptq", "gptq_v2"):
        return "gptq"
    if fmt in ("marlin",):
        return "gptq_marlin"

    if qconfig.get("bits") and qconfig.get("modules_in_block_to_quantize"):
        return "gptq"

    return None


def detect_model_config(model_path: Path) -> dict:
    """Read config.json / quantize_config.json for quant / dtype info."""
    info: dict = {}
    model_path = Path(model_path)

    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        info["model_type"] = config.get("model_type", "unknown")
        info["architectures"] = config.get("architectures", [])
        info["torch_dtype"] = config.get("torch_dtype")
        info["dtype"] = config.get("dtype") or config.get("torch_dtype")
        qconfig = config.get("quantization_config", {})
        if qconfig:
            info["quant_method"] = _extract_quant_method(qconfig)
            info["bits"] = qconfig.get("bits")
            info["group_size"] = qconfig.get("group_size", qconfig.get("groupsize"))
            info["sym"] = qconfig.get("sym")
            info["desc_act"] = qconfig.get("desc_act", qconfig.get("actorder", False))

    qconfig_file = model_path / "quantize_config.json"
    if qconfig_file.exists():
        with open(qconfig_file) as f:
            qc = json.load(f)
        if not info.get("quant_method"):
            info["quant_method"] = _extract_quant_method(qc)
        if info.get("bits") is None:
            info["bits"] = qc.get("bits")
        if info.get("group_size") is None:
            info["group_size"] = qc.get("group_size", qc.get("groupsize"))
        if info.get("sym") is None:
            info["sym"] = qc.get("sym")
        if info.get("desc_act") is None:
            info["desc_act"] = qc.get("desc_act", qc.get("actorder", False))

    return info


def is_quantized(model_info: dict) -> bool:
    return bool(model_info.get("quant_method"))


def describe_model(model_info: dict) -> str:
    qm = model_info.get("quant_method")
    if qm:
        bits = model_info.get("bits", "?")
        gs = model_info.get("group_size")
        parts = [f"{qm.upper()} {bits}bit"]
        if gs:
            parts.append(f"group_size={gs}")
        return " ".join(parts)
    torch_dtype = model_info.get("torch_dtype", "auto")
    return f"Non-quantized ({torch_dtype})"


def resolve_gptq_backend(
    bits: int | None,
    desc_act: bool = False,
    sym: bool = False,
) -> str:
    """Pick vLLM backend: gptq_marlin when supported, else gptq."""
    if should_use_gptq_marlin(bits=bits, sym=sym, desc_act=desc_act):
        return "gptq_marlin"
    if desc_act:
        logger.info("desc_act=True -> gptq (Marlin incompatible with act_order)")
    elif not sym:
        logger.info("sym is not True -> gptq (Marlin requires symmetric quantization)")
    return "gptq"


def resolve_quantization(
    quant_method: str | None,
    bits: int | None,
    desc_act: bool = False,
    sym: bool = False,
) -> str | None:
    if quant_method == "gptq":
        backend = resolve_gptq_backend(bits, desc_act=desc_act, sym=sym)
        logger.info(
            "GPTQ bits=%s, desc_act=%s, sym=%s -> backend=%s",
            bits,
            desc_act,
            sym,
            backend,
        )
        return backend
    if quant_method == "mixed_gptq":
        logger.info(
            "quantization=mixed_gptq (per-module kernel dispatch via plugin)",
        )
        return "mixed_gptq"
    if quant_method:
        logger.info("quantization=%s", quant_method)
    return quant_method


def resolve_dtype(
    quantization: str | None,
    dtype: str = "auto",
    torch_dtype: str | None = None,
) -> str:
    """GPTQ/Marlin requires float16; transformers passes through otherwise."""
    if dtype != "auto":
        return dtype
    if quantization == "compressed-tensors":
        # Cohere MoE routing_logits require float32/bfloat16; float16 breaks w4a4.
        if torch_dtype in ("bfloat16", "bf16"):
            logger.info("dtype = bfloat16 (required for %s MoE routing)", quantization)
            return "bfloat16"
        logger.info("dtype = auto (%s; follow model config)", quantization)
        return "auto"
    if quantization in QUANT_DTYPE_MAP:
        resolved = QUANT_DTYPE_MAP[quantization]
        if resolved != "auto":
            logger.info("dtype = %s (required for %s)", resolved, quantization)
            return resolved
    if torch_dtype is None and quantization:
        logger.info("dtype = float16 (torch_dtype=None, quantized model)")
        return "float16"
    return dtype


def print_model_summary(model_info: dict) -> None:
    logger.info("Model type  : %s", model_info.get("model_type", "unknown"))
    if is_quantized(model_info):
        logger.info("Mode        : Quantized (%s)", describe_model(model_info))
        logger.info("Quant method: %s", model_info.get("quant_method"))
        logger.info("Bits        : %s", model_info.get("bits", "unknown"))
        if model_info.get("group_size"):
            logger.info("Group size  : %s", model_info["group_size"])
    else:
        torch_dtype = model_info.get("torch_dtype", "auto")
        logger.info("Mode        : Non-quantized (FP16/BF16/FP32)")
        logger.info("torch_dtype : %s", torch_dtype)
