"""

Copyright 2025-2026 Fujitsu Ltd.

"""

from __future__ import annotations

import threading
import time
from logging import Logger
from typing import Optional


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    sec = max(0, int(round(seconds)))
    hours, rem = divmod(sec, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


class QuantizationProgressTracker:
    """Log coarse progress (done/total, elapsed, linear ETA) during long quantization."""

    def __init__(
        self,
        logger: Logger,
        total_steps: int,
        label: str,
        *,
        thread_safe: bool = False,
    ):
        self._logger = logger
        self._total = int(total_steps)
        self._label = label
        self._done = 0
        self._start = time.monotonic()
        self._lock = threading.Lock() if thread_safe else None

    @property
    def done(self) -> int:
        if self._lock:
            with self._lock:
                return self._done
        return self._done

    def step_complete(self, detail: Optional[str] = None) -> None:
        """Record one completed step and emit a single INFO line with ETA."""

        if self._lock:
            with self._lock:
                self._step_complete_unlocked(detail)
        else:
            self._step_complete_unlocked(detail)

    def _step_complete_unlocked(self, detail: Optional[str]) -> None:
        if self._total <= 0:
            return

        now = time.monotonic()
        self._done += 1
        done = min(self._done, self._total)
        elapsed = now - self._start

        if done < self._total:
            eta_sec = (elapsed / done) * (self._total - done)
            eta_str = _format_duration(eta_sec)
        else:
            eta_str = "0s"

        pct = 100.0 * done / self._total
        suffix = f" ({detail})" if detail else ""
        self._logger.info(
            "[progress] %s: %d/%d (%.1f%%) elapsed=%s ETA=%s%s",
            self._label,
            done,
            self._total,
            pct,
            _format_duration(elapsed),
            eta_str,
            suffix,
        )
