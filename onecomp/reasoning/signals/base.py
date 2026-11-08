# Copyright 2025-2026 Fujitsu Ltd.

"""Signal protocol: observe a request without deciding or modifying generation."""

from __future__ import annotations

from typing import Any, Protocol


class ReasoningSignal(Protocol):
    """A signal observes a request without deciding or modifying generation."""

    def observe(self, state: Any, observation: Any) -> dict[str, Any]:
        """Return named observations; unavailable measurements should be None."""
