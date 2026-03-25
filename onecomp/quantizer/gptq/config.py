"""
GPTQ-specific helpers for OneComp quantization_config schema.

Resolves per-layer bit-width from quantization_config (e.g. when loading a model).
Delegates the override priority (module_wbits > mlp_wbits > default) to GPTQ.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
from typing import Any

from onecomp.quantizer.gptq._gptq import GPTQ
from onecomp.utils.quant_config import get_quant_param


def _validate_int_bits(name: str, bits: Any) -> int:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise ValueError(f"{name} must be an int in 1..64, got {bits!r}.")
    if not (1 <= bits <= 64):
        raise ValueError(f"{name} must be in 1..64, got {bits}.")
    return bits


def resolve_gptq_layer_wbits(layer_name: str, quant_config: dict[str, Any]) -> int:
    """Resolve GPTQ bit-width for a given layer from quantization_config.

    Priority:
    1) quantization_bits[layer_idx][suffix] (mixed_gptq per-layer table, only in saved config)
    2) module_wbits[layer_name]
    3) mlp_wbits for layers containing "mlp"
    4) bits/wbits default
    """
    default_wbits = quant_config.get("bits", quant_config.get("wbits"))
    if default_wbits is None:
        raise ValueError("Missing bits/wbits in quantization_config for GPTQ model.")

    module_wbits = get_quant_param(quant_config, "module_wbits")
    if module_wbits is not None and not isinstance(module_wbits, dict):
        raise ValueError("module_wbits in quantization_config must be a dict.")

    # Per-layer table (only in saved config)
    quantization_bits_list = quant_config.get("quantization_bits")
    if quantization_bits_list:
        m = re.search(r"\.layers\.(\d+)\.(.*)", layer_name)
        if m:
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            if layer_idx < len(quantization_bits_list):
                layer_cfg = quantization_bits_list[layer_idx]
                if isinstance(layer_cfg, dict):
                    for key, mod_cfg in layer_cfg.items():
                        if key == "_all" or suffix == key or suffix.startswith(key):
                            qb_bits = mod_cfg.get("bits") if isinstance(mod_cfg, dict) else None
                            if qb_bits is not None:
                                return _validate_int_bits("quantization_bits[].bits", int(qb_bits))

    # GPTQ override priority (module > mlp > default), then validate
    bits = GPTQ.resolve_bits(
        layer_name,
        default_wbits,
        get_quant_param(quant_config, "mlp_wbits"),
        module_wbits or {},
    )
    return _validate_int_bits("bits/wbits in quantization_config", bits)


def resolve_gptq_layer_group_size(layer_name: str, quant_config: dict[str, Any]) -> int:
    """Resolve GPTQ group_size for a given layer from quantization_config.

    Priority:
    1) quantization_bits[layer_idx][suffix] per-layer table
    2) mlp_groupsize for layers containing "mlp"
    3) global group_size / groupsize
    """
    default_gs = get_quant_param(quant_config, "group_size", "groupsize", default=-1)

    quantization_bits_list = quant_config.get("quantization_bits")
    if quantization_bits_list:
        m = re.search(r"\.layers\.(\d+)\.(.*)", layer_name)
        if m:
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            if layer_idx < len(quantization_bits_list):
                layer_cfg = quantization_bits_list[layer_idx]
                if isinstance(layer_cfg, dict):
                    for key, mod_cfg in layer_cfg.items():
                        if key == "_all" or suffix == key or suffix.startswith(key):
                            if isinstance(mod_cfg, dict):
                                gs = mod_cfg.get("group_size")
                                if gs is None:
                                    params = mod_cfg.get("params")
                                    if isinstance(params, dict):
                                        gs = params.get("group_size")
                                if gs is not None:
                                    return int(gs)

    mlp_gs = get_quant_param(quant_config, "mlp_groupsize")
    return int(GPTQ.resolve_groupsize(layer_name, default_gs, mlp_gs))
