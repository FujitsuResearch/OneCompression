# Copyright 2025-2026 Fujitsu Ltd.

"""Reasoning-token budgets; they stop at the budget without waiting for a boundary."""

from ..actions import Decision
from .base import ReasoningPolicy


class BudgetPolicy(ReasoningPolicy):
    """``fixed`` uses ``config.budget``; ``prompt_budget`` gets a per-prompt budget.

    The prompt budget (``estimate_prompt_budget``) is an untrained TALE-like
    heuristic; the caller substitutes it into the config for each prompt.
    """

    name = "fixed"
    uses_logits = False

    def forced_boundary(self, state):
        return state.reasoning_tokens >= self.config.budget

    def decide(self, state, signals):
        if state.reasoning_tokens >= self.config.budget:
            return Decision("stop", "reasoning_budget")
        return Decision()
