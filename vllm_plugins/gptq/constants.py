"""Shared GPTQ kernel dispatch constants for vLLM plugins and eval harness.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

# Bits eligible for GPTQ Marlin (--quantization gptq_marlin / mixed_gptq plugin).
GPTQ_MARLIN_SUPPORTED_BITS: frozenset[int] = frozenset({4, 8})


def should_use_gptq_marlin(
    *,
    bits: int | None,
    sym: bool = False,
    desc_act: bool = False,
    method: str = "gptq",
) -> bool:
    """Return True when GPTQ Marlin should be selected.

    Matches the mixed_gptq plugin dispatch rule:
    method=gptq, bits in {4, 8}, sym=True, and no activation reordering.

    Default sym=False: unknown/missing config must not assume symmetric quantization.
    Callers may pass None (e.g. sym: null in JSON); it is treated as false.
    """
    if method != "gptq":
        return False
    if desc_act or not sym or bits is None:
        return False
    return int(bits) in GPTQ_MARLIN_SUPPORTED_BITS
