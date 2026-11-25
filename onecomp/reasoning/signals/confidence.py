# Copyright 2025-2026 Fujitsu Ltd.

"""Probe-answer confidence and relative attention-entropy change (standard library only).

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


def asag_confidence(token_probabilities: Sequence[float]) -> float | None:
    """ASAG equation 5: arithmetic mean of selected answer-token probabilities."""
    probabilities = _probabilities(token_probabilities)
    return math.fsum(probabilities) / len(probabilities) if probabilities else None


def relative_entropy_change(current: float, first: float | None) -> float | None:
    """ASAG compares against the first ATP, never the previous ATP.

    A zero initial entropy has no defined relative change. Return missing data
    rather than inventing convergence or dividing by an arbitrary epsilon.
    """
    if not math.isfinite(current) or current < 0:
        raise ValueError("Entropy must be finite and nonnegative")
    if first is None:
        return None
    if not math.isfinite(first) or first < 0:
        raise ValueError("Initial entropy must be finite and nonnegative")
    return None if first == 0 else (current - first) / first
