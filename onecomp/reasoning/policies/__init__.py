# Copyright 2025-2026 Fujitsu Ltd.

"""One policy class per stopping method, selected by ``ReasoningConfig.policy``."""

from .base import NonePolicy, ReasoningPolicy
from .budget import BudgetPolicy
from .thinkbrake import ThinkBrakePolicy

POLICIES = {
    "none": NonePolicy,
    "fixed": BudgetPolicy,
    "prompt_budget": BudgetPolicy,
    "thinkbrake": ThinkBrakePolicy,
}


def make_policy(config) -> ReasoningPolicy:
    return POLICIES[config.policy](config)


__all__ = [
    "POLICIES",
    "BudgetPolicy",
    "NonePolicy",
    "ReasoningPolicy",
    "ThinkBrakePolicy",
    "make_policy",
]
