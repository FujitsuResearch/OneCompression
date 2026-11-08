# Copyright 2025-2026 Fujitsu Ltd.

"""DEER (Yang et al., arXiv:2504.15895): stop on a confident trial answer."""

from ..actions import Decision
from .base import ReasoningPolicy, is_confident


class DeerPolicy(ReasoningPolicy):
    """Geometric-mean probe confidence (Eq. 4) above the threshold.

    ``relative_confidence`` is an optional non-paper variant that also stops when
    the current confidence is at least every earlier confidence of this request.
    """

    name = "deer"
    uses_probe = True

    def decide(self, state, signals):
        c = self.config
        confidence = signals.get("confidence")
        # The controller already appended the current confidence.
        previous = state.probe_confidences[:-1]
        relative = (
            signals.get("probe_complete", True)
            and c.relative_confidence
            and previous
            and confidence is not None
            and confidence >= max(previous)
        )
        if is_confident(signals, c) or relative:
            return Decision("stop", "probe_confidence")
        return Decision()
