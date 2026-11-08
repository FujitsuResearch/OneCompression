# Copyright 2025-2026 Fujitsu Ltd.

"""One policy class per stopping method, selected by ``ReasoningConfig.policy``."""

from .asag import AsagPolicy
from .base import NonePolicy, ReasoningPolicy
from .budget import BudgetPolicy
from .deer import DeerPolicy
from .stability import StabilityPolicy
from .thinkbrake import ThinkBrakePolicy

POLICIES = {
    "none": NonePolicy,
    "fixed": BudgetPolicy,
    "prompt_budget": BudgetPolicy,
    "thinkbrake": ThinkBrakePolicy,
    "deer": DeerPolicy,
    "stability": StabilityPolicy,
    "asag_reference": AsagPolicy,
}


def make_policy(config) -> ReasoningPolicy:
    return POLICIES[config.policy](config)


__all__ = [
    "POLICIES",
    "AsagPolicy",
    "BudgetPolicy",
    "DeerPolicy",
    "NonePolicy",
    "ReasoningPolicy",
    "StabilityPolicy",
    "ThinkBrakePolicy",
    "make_policy",
]
