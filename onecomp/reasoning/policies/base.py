# Copyright 2025-2026 Fujitsu Ltd.

"""Policy interface: decisions consume observations only, never vLLM objects or oracle labels."""

from abc import ABC, abstractmethod

from ..actions import Decision


class ReasoningPolicy(ABC):
    """One stopping method. Class attributes declare what a backend must observe.

    ``uses_logits``: observe the output-logit margin at each boundary (no extra forward).
    ``requires_logits``: ``decide`` consumes that margin, so it must be observable.
    ``uses_probe``: run an isolated answer probe at each boundary.
    ``requires_attention``: some probe may collect eager attention weights.
    """

    name = ""
    uses_logits = True
    requires_logits = False
    uses_probe = False
    requires_attention = False

    def __init__(self, config):
        self.config = config

    def forced_boundary(self, state):
        """Override boundary detection (budgets); None defers to the detector."""
        del state
        return None

    def needs_probe(self, state, observation):
        del state, observation
        return self.uses_probe

    def probe_options(self):
        """Keyword arguments for the backend's answer probe."""
        return {"collect_attention": False, "answer_probe": True}

    def needs_attention_followup(self, probe):
        """Whether a completed answer probe should be followed by an attention probe."""
        del probe
        return False

    @abstractmethod
    def decide(self, state, signals) -> Decision:
        pass


def is_confident(signals, config):
    """Complete probe whose confidence strictly exceeds the configured threshold."""
    confidence = signals.get("confidence")
    return (
        signals.get("probe_complete", True)
        and confidence is not None
        and confidence > config.confidence_threshold
    )


def has_cheap_stop(signals, config):
    """ThinkBrake margin test: best non-end minus end log-probability <= threshold."""
    margin = signals.get("logit_margin")
    return margin is not None and margin <= config.threshold


class NonePolicy(ReasoningPolicy):
    """Identity policy: never intervenes and needs no observation."""

    name = "none"
    uses_logits = False

    def decide(self, state, signals):
        return Decision()
