# Copyright 2025-2026 Fujitsu Ltd.

"""Boundary detection does not import a model runtime."""

import re
from abc import ABC, abstractmethod


class ReasoningBoundaryDetector(ABC):
    @abstractmethod
    def detect(self, state, next_token_text="") -> bool:
        """Whether to inspect the prefix before sampling its next token."""


class TokenBoundaryDetector(ReasoningBoundaryDetector):
    def __init__(self, interval=128):
        if interval < 1:
            raise ValueError("interval must be positive")
        self.interval = interval

    def detect(self, state, next_token_text=""):
        return state.reasoning_tokens > 0 and state.reasoning_tokens % self.interval == 0


class SentenceBoundaryDetector(ReasoningBoundaryDetector):
    def detect(self, state, next_token_text=""):
        # Token-fragment approximation, documented separately from paper boundaries.
        return bool(re.search(r"(?:[.!?][\s\"')\]]*|\n\s*\n)$", state.last_text))


class TransitionBoundaryDetector(ReasoningBoundaryDetector):
    def detect(self, state, next_token_text=""):
        # ATP occurs BEFORE the transition token; never consume it for a probe.
        return next_token_text.strip().lower() in {"wait", "alternatively"}


def make_boundary(config):
    return {
        "tokens": lambda: TokenBoundaryDetector(config.interval),
        "sentence": SentenceBoundaryDetector,
        "transition": TransitionBoundaryDetector,
    }[config.boundary]()
