# Copyright 2025-2026 Fujitsu Ltd.

"""Probe-answer confidence (standard library only).

Confidence is not a correctness guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _probabilities(values: Sequence[float]) -> list[float]:
    probabilities = [float(value) for value in values]
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in probabilities):
        raise ValueError("Token probabilities must be finite values in [0, 1]")
    return probabilities


def deer_confidence(token_max_probabilities: Sequence[float]) -> float | None:
    """DEER equation 4: geometric mean of per-position maximum probabilities.

    The caller supplies *answer-content* probabilities, excluding the fixed cue
    and delimiters. Empty or incomplete probes must never imply safe stopping.
    """
    probabilities = _probabilities(token_max_probabilities)
    if not probabilities:
        return None
    if 0.0 in probabilities:
        return 0.0
    return math.exp(math.fsum(math.log(p) for p in probabilities) / len(probabilities))
