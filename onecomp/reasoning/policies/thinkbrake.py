# Copyright 2025-2026 Fujitsu Ltd.

"""ThinkBrake (Song et al., arXiv:2510.00546): stop when the end marker nearly wins."""

from ..actions import Decision
from .base import ReasoningPolicy, has_cheap_stop


class ThinkBrakePolicy(ReasoningPolicy):
    name = "thinkbrake"
    requires_logits = True

    def decide(self, state, signals):
        return (
            Decision("stop", "end_logit_gap")
            if has_cheap_stop(signals, self.config)
            else Decision()
        )
