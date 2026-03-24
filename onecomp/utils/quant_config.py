"""
Shared helpers for OneComp quantization_config schema.

OneComp convention:
- quantization_config has all keys at top-level (quant_method, bits, group_size, ...).
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
