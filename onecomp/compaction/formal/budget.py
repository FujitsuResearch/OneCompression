"""Strict provider-boundary context budget enforcement for formal ARC.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any, Awaitable, Callable, TypeVar

from .errors import BudgetExceeded
from .models import BudgetSnapshot, InvocationAudit

RequestCounter = Callable[[Any], int]
ResultT = TypeVar("ResultT")

_RESERVED_COMPONENTS = {
    "context_window",
    "completion_reserve",
    "baseline",
    "memory",
    "recall",
    "protected_history",
    "query",
    "serialization_margin",
}


def validate_budget(snapshot: BudgetSnapshot) -> BudgetSnapshot:
    """Validate all components and fail closed when the envelope is exceeded."""
    if not isinstance(snapshot, BudgetSnapshot):
        raise TypeError("snapshot must be a BudgetSnapshot")
    labels = dict(snapshot.labels)
    for name in labels:
        if not isinstance(name, str) or not name:
            raise ValueError("budget labels must be non-empty strings")
        if name in _RESERVED_COMPONENTS:
            raise ValueError(f"budget label {name!r} conflicts with a built-in field")
    fields = {
        "context_window": snapshot.context_window,
        "completion_reserve": snapshot.completion_reserve,
        "baseline": snapshot.baseline,
        "memory": snapshot.memory,
        "recall": snapshot.recall,
        "protected_history": snapshot.protected_history,
        "query": snapshot.query,
        "serialization_margin": snapshot.serialization_margin,
        **labels,
    }
    for name, value in fields.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"budget component {name!r} must be a non-negative integer")
    if snapshot.context_window <= 0:
        raise ValueError("context_window must be positive")
    if snapshot.completion_reserve >= snapshot.context_window:
        raise ValueError("completion_reserve must be smaller than context_window")
    if snapshot.remaining < 0:
        raise BudgetExceeded(
            "formal ARC context envelope exceeded: "
            f"input_used={snapshot.input_used}, "
            f"input_capacity={snapshot.input_capacity}, "
            f"over_by={-snapshot.remaining}"
        )
    return snapshot


class InvocationBudgetGuard:
    """Count the final provider request immediately before dispatch.

    Formal prompt-bound guarantees are host-boundary properties. Callers must
    pass the final provider request, including role wrappers, tool schemas,
    images, and provider serialization overhead, to ``dispatch`` or
    ``dispatch_async``. Using ``check`` alone does not prove dispatch coverage;
    the audit counters make that distinction observable.
    """

    def __init__(
        self,
        *,
        context_window: int,
        completion_reserve: int,
        count_request: RequestCounter,
        serialization_margin: int = 0,
    ) -> None:
        if not callable(count_request):
            raise TypeError("count_request must be callable")
        validate_budget(
            BudgetSnapshot(
                context_window=context_window,
                completion_reserve=completion_reserve,
                serialization_margin=serialization_margin,
            )
        )
        self.context_window = context_window
        self.completion_reserve = completion_reserve
        self.count_request = count_request
        self.serialization_margin = serialization_margin
        self._lock = threading.Lock()
        self._checked_requests = 0
        self._dispatched_requests = 0
        self._dispatch_successes = 0
        self._dispatch_failures = 0
        self._rejected_requests = 0
        self._last_checked_units: int | None = None

    @property
    def input_capacity(self) -> int:
        return self.context_window - self.completion_reserve

    def check(self, provider_request: Any) -> int:
        """Return total input units or reject before provider invocation."""
        used = self.count_request(provider_request)
        if not isinstance(used, int) or isinstance(used, bool) or used < 0:
            raise TypeError("count_request must return a non-negative integer")
        total = used + self.serialization_margin
        with self._lock:
            self._checked_requests += 1
            self._last_checked_units = total
            if total > self.input_capacity:
                self._rejected_requests += 1
        if total > self.input_capacity:
            raise BudgetExceeded(
                "model invocation rejected before dispatch: "
                f"request_units={used}, margin={self.serialization_margin}, "
                f"input_capacity={self.input_capacity}, "
                f"over_by={total - self.input_capacity}"
            )
        return total

    def dispatch(
        self,
        provider_request: Any,
        invoke: Callable[[Any], ResultT],
        *,
        on_checked: Callable[[], None] | None = None,
    ) -> ResultT:
        """Check and synchronously invoke using the same request object."""
        if not callable(invoke):
            raise TypeError("invoke must be callable")
        if inspect.iscoroutinefunction(invoke):
            raise TypeError("invoke is asynchronous; use dispatch_async")
        if on_checked is not None and not callable(on_checked):
            raise TypeError("on_checked must be callable or None")
        self.check(provider_request)
        if on_checked is not None:
            on_checked()
        with self._lock:
            self._dispatched_requests += 1
        try:
            result = invoke(provider_request)
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                raise TypeError("invoke returned an awaitable; use dispatch_async")
        # Provider cancellation and interrupts are failed dispatch attempts too.
        except BaseException:
            self._record_dispatch_outcome(succeeded=False)
            raise
        self._record_dispatch_outcome(succeeded=True)
        return result

    async def dispatch_async(
        self,
        provider_request: Any,
        invoke: Callable[[Any], Awaitable[ResultT]],
        *,
        on_checked: Callable[[], None] | None = None,
    ) -> ResultT:
        """Check and asynchronously invoke using the same request object."""
        if not callable(invoke):
            raise TypeError("invoke must be callable")
        if on_checked is not None and not callable(on_checked):
            raise TypeError("on_checked must be callable or None")
        self.check(provider_request)
        if on_checked is not None:
            on_checked()
        with self._lock:
            self._dispatched_requests += 1
        try:
            result = invoke(provider_request)
            if not inspect.isawaitable(result):
                raise TypeError("invoke must return an awaitable")
            response = await result
        # Preserve cancellation while keeping attempt telemetry truthful.
        except BaseException:
            self._record_dispatch_outcome(succeeded=False)
            raise
        self._record_dispatch_outcome(succeeded=True)
        return response

    def _record_dispatch_outcome(self, *, succeeded: bool) -> None:
        with self._lock:
            if succeeded:
                self._dispatch_successes += 1
            else:
                self._dispatch_failures += 1

    def audit(self) -> InvocationAudit:
        """Return immutable coverage counters for diagnostics and tests."""
        with self._lock:
            return InvocationAudit(
                checked_requests=self._checked_requests,
                dispatched_requests=self._dispatched_requests,
                rejected_requests=self._rejected_requests,
                last_checked_units=self._last_checked_units,
                dispatch_successes=self._dispatch_successes,
                dispatch_failures=self._dispatch_failures,
            )

    def with_margin(self, serialization_margin: int) -> "InvocationBudgetGuard":
        """Build an independent guard with the same authoritative counter."""
        return InvocationBudgetGuard(
            context_window=self.context_window,
            completion_reserve=self.completion_reserve,
            count_request=self.count_request,
            serialization_margin=serialization_margin,
        )
