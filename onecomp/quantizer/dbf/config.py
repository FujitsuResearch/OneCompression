"""
DBF-specific helpers for OneComp quantization_config schema.

Resolves per-layer bit-width from quantization_config (e.g. when loading a model).
Delegates the override priority (module_target_bits > mlp_target_bits > default) to DBF.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
from typing import Any

from onecomp.quantizer.dbf._dbf import DBF
from onecomp.utils.quant_config import get_quant_param


def _validate_bits(name: str, bits: Any) -> float:
    if not isinstance(bits, (int, float)):
        raise ValueError(f"{name} must be a number >= 1.0, got {bits!r}.")
    if bits < 1.0:
        raise ValueError(f"{name} must be >= 1.0, got {bits}.")
    return float(bits)


def resolve_dbf_layer_bits(layer_name: str, quant_config: dict[str, Any]) -> float:
    """Resolve DBF bit-width for a given layer from quantization_config.

    Priority:
    1) quantization_bits[layer_idx][suffix] (per-layer table, only in saved config)
    2) module_target_bits[layer_name]
    3) mlp_target_bits for layers containing "mlp"
    4) bits default
    """
    default_bits = quant_config.get("bits")
    if default_bits is None:
        raise ValueError("Missing bits in quantization_config for DBF model.")

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
                                return _validate_bits("quantization_bits[].bits", qb_bits)

    # DBF override priority (module > mlp > default), then validate
    bits = DBF.resolve_bits(
        layer_name,
        default_bits,
        get_quant_param(quant_config, "mlp_target_bits"),
        get_quant_param(quant_config, "module_target_bits") or {},
    )
    return _validate_bits("bits in quantization_config", bits)
