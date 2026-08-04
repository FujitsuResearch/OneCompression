"""
Shared helpers for OneComp quantization_config schema.

OneComp convention:
- quantization_config has all keys at top-level (quant_method, bits, group_size, ...).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Any


def get_quant_param(
    quant_config: dict[str, Any] | None,
    *keys: str,
    default=None,
):
    """Fetch a quantization parameter from quantization_config using alias keys."""
    if not quant_config:
        return default

    for key in keys:
        if key in quant_config:
            return quant_config.get(key)

    return default


def validate_quant_config(quant_config: Any, context: str) -> dict:
    """Validate a quantization_config dict and return it.

    Shared by the save path (in-memory ``model.config.quantization_config``)
    and the load path (``config.json`` ``quantization_config``) so both enforce
    the same required keys and raise the same exception type.

    Args:
        quant_config: The quantization_config object to validate.
        context: Caller label surfaced in the error message.

    Returns:
        The validated ``quant_config`` dict (same object).

    Raises:
        ValueError: If ``quant_config`` is not a dict, or is missing
            ``quant_method`` or ``modules_in_block_to_quantize``.
    """
    if not isinstance(quant_config, dict):
        raise ValueError(f"{context}: quantization_config must be a dict.")
    if not quant_config.get("quant_method"):
        raise ValueError(f"{context}: quantization_config must contain 'quant_method'.")
    if "modules_in_block_to_quantize" not in quant_config:
        raise ValueError(
            f"{context}: quantization_config must contain " "'modules_in_block_to_quantize'."
        )
    return quant_config


def validate_quantized_model_config(model, context: str) -> dict:
    """Validate and return ``model.config.quantization_config``.

    Model-level wrapper around :func:`validate_quant_config`: it pulls
    ``quantization_config`` off ``model.config`` and applies the same dict-level
    schema check.  Shared by the save path (``Runner.save_quantized_model``) and
    the post-process path (``PostQuantizationProcess.run``) so both enforce the
    same required keys and raise the same exception type.

    Args:
        model: A model whose ``config.quantization_config`` should be validated.
        context: Caller label surfaced in the error message.

    Returns:
        The validated ``quant_config`` dict (same object).

    Raises:
        ValueError: If ``model.config`` is missing, or ``quantization_config``
            fails :func:`validate_quant_config`.
    """
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise ValueError(f"{context}: model.config is required.")

    quant_config = getattr(model_config, "quantization_config", None)
    return validate_quant_config(quant_config, context)
