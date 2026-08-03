"""
MDBF-specific helpers for OneComp quantization_config schema.

Resolves per-layer bit-width and the model-wide path count from
quantization_config (e.g. when loading a model). Delegates the bit-width
override priority (module_target_bits > mlp_target_bits > default) to MDBF.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from __future__ import annotations

import re
from typing import Any

from onecomp.quantizer.mdbf._mdbf import MDBF
from onecomp.utils.quant_config import get_quant_param


def _validate_bits(name: str, bits: Any) -> float:
    """Return *bits* as a float, rejecting non-numeric or non-positive values."""
    if not isinstance(bits, (int, float)):
        raise ValueError(f"{name} must be a number > 0, got {bits!r}.")
    if bits <= 0:
        raise ValueError(f"{name} must be > 0, got {bits}.")
    return float(bits)


def _validate_paths(name: str, paths: Any) -> int:
    """Return *paths* as an int, rejecting anything but a positive integer.

    No upper bound is imposed here: MDBF currently only produces P in {1, 2},
    but that limit lives in ``MDBF.validate_params`` and duplicating it would
    make checkpoints unloadable the day it is relaxed. An implausibly large
    value is caught instead as a path-count mismatch against the checkpoint.
    """
    # bool is an int subclass; a True/False here means a malformed config.
    if not isinstance(paths, int) or isinstance(paths, bool) or paths <= 0:
        raise ValueError(f"{name} must be an integer > 0, got {paths!r}.")
    return paths


def resolve_mdbf_layer_bits(layer_name: str, quant_config: dict[str, Any]) -> float:
    """Resolve MDBF bit-width for a given layer from quantization_config.

    Priority:
    1) quantization_bits[layer_idx][suffix] (per-layer table, only in saved config)
    2) module_target_bits[layer_name]
    3) mlp_target_bits for layers containing "mlp"
    4) bits default
    """
    default_bits = quant_config.get("bits")
    if default_bits is None:
        raise ValueError("Missing bits in quantization_config for MDBF model.")

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

    # MDBF override priority (module > mlp > default), then validate
    bits = MDBF.resolve_bits(
        layer_name,
        default_bits,
        get_quant_param(quant_config, "mlp_target_bits"),
        get_quant_param(quant_config, "module_target_bits") or {},
    )
    return _validate_bits("bits in quantization_config", bits)


def resolve_mdbf_paths(quant_config: dict[str, Any]) -> int | None:
    """Resolve the number of MDBF paths (P) recorded in quantization_config.

    P is model-wide: the per-layer ``quantization_bits`` table echoes the same
    top-level value into every entry, so only the top-level key is read.

    The recorded value is the *requested* path count, which is also the actual
    one: initialization builds exactly P paths and neither ADMM nor the
    gradient refinement adds or drops any.

    Args:
        quant_config: quantization_config dict from config.json.

    Returns:
        The path count, or None for a hand-written or partial config that
        omits P, in which case the caller cannot validate it.

    Raises:
        ValueError: If the recorded value is not a positive integer.
    """
    paths = get_quant_param(quant_config, "P")
    if paths is None:
        return None
    return _validate_paths("P in quantization_config", paths)
