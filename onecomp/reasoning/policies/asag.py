# Copyright 2025-2026 Fujitsu Ltd.

"""ASAG reference interpretation (Li et al., arXiv:2606.15070, Algorithm 1)."""

from ..actions import Decision
from .base import ReasoningPolicy, is_confident


def backward_dominates(signals):
    """Eq. 8 trap test: previous thought receives more attention per token than the current."""
    previous = signals.get("previous_attention")
    current = signals.get("current_attention")
    if previous is not None and current is not None:
        return previous > current
    ratio = signals.get("backward_ratio")
    return ratio is not None and ratio > 1


class AsagPolicy(ReasoningPolicy):
    """Confidence + first-boundary-relative attention entropy, mixture and jump.

    ``asag_mode``: ``full`` (paper), ``confidence``/``attention`` (single-signal
    ablations) or ``both`` (exit rule only, without mixture/trap actions).
    """

    name = "asag_reference"
    uses_probe = True

    @property
    def requires_attention(self):
        return self.config.asag_mode != "confidence"

    def probe_options(self):
        return {
            "collect_attention": self.requires_attention,
            "answer_probe": self.config.asag_mode != "attention",
        }

    def decide(self, state, signals):
        c = self.config
        confident = is_confident(signals, c)
        relative_entropy = signals.get("relative_entropy")
        if c.asag_mode == "confidence":
            return Decision("stop", "confidence_ablation") if confident else Decision()
        converged = relative_entropy is not None and relative_entropy < c.entropy_threshold
        if c.asag_mode == "attention":
            return Decision("stop", "attention_ablation") if converged else Decision()
        if signals.get("first_attention_boundary", False) and confident:
            return Decision("stop", "asag_first_confidence")
        if confident and converged:
            return Decision("stop", "asag_confidence_and_convergence")
        if c.asag_mode == "both":
            return Decision()
        if (
            converged
            and signals.get("probe_complete", True)
            and signals.get("confidence") is not None
            and c.enable_injection
        ):
            return Decision("bias", "asag_probability_mixture")
        if (
            relative_entropy is not None
            and relative_entropy > c.entropy_threshold
            and backward_dominates(signals)
            and c.enable_trap
        ):
            if state.jump_count < c.max_jumps:
                return Decision("jump", "asag_backward_attention")
            return Decision("stop", "asag_jump_limit")
        return Decision()
