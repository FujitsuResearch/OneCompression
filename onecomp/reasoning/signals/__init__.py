# Copyright 2025-2026 Fujitsu Ltd.

"""Backend-independent observations for runtime reasoning compression.

Scalar functions use only the standard library; torch is imported only when
logit tensors are supplied.
"""

from .base import ReasoningSignal
from .logits import logit_observation

__all__ = ["ReasoningSignal", "logit_observation"]
