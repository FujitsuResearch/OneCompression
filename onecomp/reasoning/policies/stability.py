# Copyright 2025-2026 Fujitsu Ltd.

"""Answer stability inspired by Dynasor/Certaindex; not its multi-path scheduler."""

import re

from ..actions import Decision
from .base import ReasoningPolicy


class StabilityPolicy(ReasoningPolicy):
    """Stop after ``stability_window`` identical, nonempty, non-hesitant probe answers."""

    name = "stability"
    uses_probe = True

    def decide(self, state, signals):
        window = self.config.stability_window
        answers = state.probe_answers[-window:]
        stable = (
            signals.get("probe_complete", True)
            and len(answers) == window
            and bool(answers[0])
            and len(set(answers)) == 1
            and not re.search(r"\b(?:wait|hmm)\b", answers[0], re.IGNORECASE)
        )
        return Decision("stop", "answer_stability") if stable else Decision()
