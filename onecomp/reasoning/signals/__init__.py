# Copyright 2025-2026 Fujitsu Ltd.

"""Backend-independent observations for runtime reasoning compression.

Scalar functions use only the standard library; torch is imported only when
logit or attention tensors are supplied.
"""

from .attention import attention_statistics
from .base import ReasoningSignal
from .confidence import asag_confidence, deer_confidence, relative_entropy_change
from .logits import logit_observation

__all__ = [
    "ReasoningSignal",
    "asag_confidence",
    "attention_statistics",
    "deer_confidence",
    "logit_observation",
    "relative_entropy_change",
]
