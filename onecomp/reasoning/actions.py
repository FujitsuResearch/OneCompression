# Copyright 2025-2026 Fujitsu Ltd.

"""Actions describe interventions; adapters apply them to their own runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    kind: str = "continue"
    reason: str = ""
    mixture_weight: float = 0.05


class ReasoningAction(ABC):
    @abstractmethod
    def apply(self, target):
        """Apply to a backend-owned target; no global request state."""


class StopAction(ReasoningAction):
    def __init__(self, reasoning_format):
        self.token_ids = reasoning_format.end_token_ids

    def apply(self, pending_tokens):
        pending_tokens.extend(self.token_ids)
