"""Lightweight progress logger for long-running quantization workflows.

This module exposes :class:`QuantizationProgressTracker`, a small helper
used by :class:`onecomp.runner.Runner` and the underlying quantization
entry points (calibration / chunked / multi-GPU / QEP) to emit a single
``[progress]`` INFO line per completed step with done/total counts,
percentage, wall-clock elapsed time, and a linear ETA estimate.

The tracker is intentionally minimal:

- Standard-library only (``threading``, ``time``, ``logging``); no torch
  or other heavy dependencies.
- ETA is a simple ``elapsed / done * remaining`` linear extrapolation;
  it can be off when per-step cost varies (e.g. MoE expert vs regular
  layers), but is sufficient for "is this going to finish before lunch"
  scale planning.
- Output is via the caller-supplied ``logging.Logger`` at ``INFO``
  level, so existing log handlers / filters apply transparently.

Copyright 2025-2026 Fujitsu Ltd.

"""

from __future__ import annotations

import threading
import time
from logging import Logger
from typing import Optional


def _format_duration(seconds: Optional[float]) -> str:
    """Format a number of seconds as a short human-readable string.

    Args:
        seconds (float or None): Duration in seconds. ``None`` is treated
            as "unknown" (currently unused by the tracker itself but kept
            so the helper can be reused in future call sites).

    Returns:
        str: ``"unknown"`` when ``seconds`` is ``None``; otherwise a
        compact string such as ``"45s"``, ``"3m12s"``, or ``"1h05m"``.
        Negative or sub-second values clamp to ``"0s"``.
    """
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
    """Emit one INFO log line per completed step with done/total and ETA.

    Each ``step_complete()`` call emits a single line of the form::

        [progress] <label>: <done>/<total> (<pct>%) elapsed=<elapsed> ETA=<eta> (<detail>)

    The trailing ``(<detail>)`` portion is omitted when no detail string
    is supplied.

    Timing starts when the tracker is constructed (``time.monotonic()``),
    not on the first ``step_complete`` call, so a tracker should be
    instantiated as close as possible to the work it measures.

    ETA estimation:
        ``ETA = (elapsed / done) * (total - done)`` (simple linear
        extrapolation). The final step prints ``ETA=0s``.

    Overflow behaviour:
        If ``step_complete`` is invoked more times than ``total_steps``
        (e.g. because a forward hook fires for shared-weight layers),
        the internal counter still increments so :attr:`done` reflects
        the true call count, but no additional log lines are emitted
        after the ``done == total`` line.

    Thread safety:
        With ``thread_safe=True`` an internal :class:`threading.Lock`
        guards both the counter update and the log emission, so multiple
        worker threads (e.g. multi-GPU quantization workers) can call
        ``step_complete`` concurrently without producing torn counts or
        interleaved log lines. The :attr:`done` property also takes the
        lock when present. With ``thread_safe=False`` (default) no
        locking is performed and callers must serialise their access.

    Example:
        >>> import logging
        >>> logger = logging.getLogger("example")
        >>> tracker = QuantizationProgressTracker(
        ...     logger, total_steps=len(layers), label="GPTQ layers"
        ... )
        >>> for module, name in layers:
        ...     quantize(module)
        ...     tracker.step_complete(name)
    """

    def __init__(
        self,
        logger: Logger,
        total_steps: int,
        label: str,
        *,
        thread_safe: bool = False,
    ):
        """Initialise the tracker and start the wall-clock timer.

        Args:
            logger (logging.Logger): Destination logger. All output is
                written at ``INFO`` level.
            total_steps (int): Expected total number of ``step_complete``
                calls. Values ``<= 0`` disable logging entirely (the
                tracker becomes a no-op).
            label (str): Short human-readable label that appears at the
                start of every log line (e.g. ``"GPTQ layers"`` or
                ``"Multi-GPU layer quantization"``).
            thread_safe (bool): If ``True``, guard the counter and log
                emission with a :class:`threading.Lock` so the tracker
                can be safely shared across worker threads. Default
                ``False``.
        """
        self._logger = logger
        self._total = int(total_steps)
        self._label = label
        self._done = 0
        self._start = time.monotonic()
        self._lock = threading.Lock() if thread_safe else None

    @property
    def done(self) -> int:
        """Number of ``step_complete`` calls observed so far.

        May exceed ``total_steps`` if callers over-invoke ``step_complete``;
        see the class-level "Overflow behaviour" note.
        """
        if self._lock is not None:
            with self._lock:
                return self._done
        return self._done

    def step_complete(self, detail: Optional[str] = None) -> None:
        """Record one completed step and emit a ``[progress]`` log line.

        Args:
            detail (str or None): Optional context appended in parentheses
                at the end of the log line (e.g. the name of the layer
                that was just processed). Pass ``None`` (default) to omit
                the suffix entirely.
        """

        if self._lock is not None:
            with self._lock:
                self._step_complete_unlocked(detail)
        else:
            self._step_complete_unlocked(detail)

    def _step_complete_unlocked(self, detail: Optional[str]) -> None:
        """Counter update + log emission body (no locking)."""
        if self._total <= 0:
            return

        now = time.monotonic()
        self._done += 1
        # Suppress further log lines once we've already announced 100%
        # completion. The final ``done == _total`` line is still emitted.
        if self._done > self._total:
            return

        done = self._done
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
