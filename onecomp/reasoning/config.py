# Copyright 2025-2026 Fujitsu Ltd.

"""Validated, backend-independent reasoning policy configuration."""

import math
from dataclasses import asdict, dataclass, fields

from .policies import POLICIES


@dataclass(frozen=True)
class ReasoningConfig:
    """Compression is opt-in: the default ``policy="none"`` never intervenes.

    ``temperature`` scales the ThinkBrake logit gap; it is distinct from decoding
    temperature (the reference runner is greedy). ``asag_reference`` implements
    the documented reference interpretation, not a completed paper reproduction.
    """

    policy: str = "none"
    budget: int = 2048
    threshold: float = 0.1
    confidence_threshold: float = 0.95
    boundary: str = "sentence"
    interval: int = 128
    min_reasoning_tokens: int = 0
    temperature: float = 1.0
    probe_max_tokens: int = 128
    entropy_threshold: float = -0.1
    max_jumps: int = 1
    stability_window: int = 3
    enable_injection: bool = True
    enable_trap: bool = True
    asag_mode: str = "full"
    relative_confidence: bool = False

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"Unknown reasoning policy: {self.policy}")
        if self.boundary not in {"sentence", "tokens", "transition"}:
            raise ValueError(f"Unknown boundary: {self.boundary}")
        if self.asag_mode not in {"full", "confidence", "attention", "both"}:
            raise ValueError("asag_mode must be full, confidence, attention, or both")
        for name in ("budget", "min_reasoning_tokens", "max_jumps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("interval", "probe_max_tokens", "stability_window"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("threshold", "confidence_threshold", "temperature", "entropy_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        for name in ("enable_injection", "enable_trap", "relative_confidence"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown reasoning options: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)


def estimate_prompt_budget(prompt: str) -> int:
    """Untrained, deterministic TALE-like baseline; not TALE reproduction."""
    words = len(prompt.split())
    operations = sum(prompt.count(c) for c in "+-*/=^")
    if words <= 60 and operations <= 4:
        return 512
    if words <= 180 and operations <= 12:
        return 2048
    return 8192
