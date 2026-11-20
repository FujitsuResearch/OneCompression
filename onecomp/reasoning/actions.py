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
    def __init__(self, reasoning_format, emitted=0):
        # ``emitted`` end-marker tokens are already in the output; force only the rest.
        self.token_ids = reasoning_format.end_token_ids[emitted:]

    def apply(self, pending_tokens):
        pending_tokens.extend(self.token_ids)


class LogitMixtureAction(ReasoningAction):
    """ASAG Eq. 7 probability mixture, NOT modification of attention weights."""

    def __init__(self, probe_probabilities, weight=0.05):
        if not 0 <= weight <= 1:
            raise ValueError("weight must lie in [0, 1]")
        self.probe_probabilities = probe_probabilities
        self.weight = weight

    def apply(self, logits):
        import torch  # pylint: disable=import-outside-toplevel

        probs = self.probe_probabilities.to(device=logits.device, dtype=torch.float32)
        if probs.shape != logits.shape or not torch.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("Probe distribution must match logits and be finite/nonnegative")
        if probs.sum() <= 0:
            raise ValueError("Probe distribution must have positive mass")
        probs = probs / probs.sum()
        mixed = (1 - self.weight) * logits.float().softmax(-1) + self.weight * probs
        return mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log().to(logits.dtype)


class JumpAction(ReasoningAction):
    def __init__(self, token_ids):
        self.token_ids = tuple(token_ids)

    def apply(self, pending_tokens):
        pending_tokens.extend(self.token_ids)
