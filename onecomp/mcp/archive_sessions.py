"""Process- and host-local leases for persistent MCP compaction archives.

SQLite serializes writes, but two live agent sessions must not share one
archive because their recall budgets and message histories are independent.
This module combines an in-process registry with a non-blocking OS file lock
and releases both when the corresponding compaction resource closes.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import errno
import fcntl
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from ..compaction import ArcEngine, CompactionRunner

_ACTIVE_ARCHIVE_LOCK = threading.RLock()
_ACTIVE_ARCHIVES: Dict[str, "ArchiveLease"] = {}


class ArchiveLease:
    """One idempotently releasable archive lease."""

    def __init__(self, path: str, descriptor: int) -> None:
        self.path = path
        self._descriptor = descriptor
        self._released = False

    def release(self) -> None:
        """Release this lease once, even when close paths race."""
        with _ACTIVE_ARCHIVE_LOCK:
            if self._released:
                return
            if _ACTIVE_ARCHIVES.get(self.path) is self:
                del _ACTIVE_ARCHIVES[self.path]
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)
            self._released = True


class _LeasedArchiveMixin:
    _archive_lease: Optional[ArchiveLease] = None

    def attach_archive_lease(self, lease: Optional[ArchiveLease]) -> None:
        """Attach the lease acquired before this resource was constructed."""
        self._archive_lease = lease

    def close(self) -> None:
        """Close the resource and always release its process-local lease."""
        try:
            super().close()
        finally:
            if self._archive_lease is not None:
                self._archive_lease.release()


class ManagedCompactionRunner(_LeasedArchiveMixin, CompactionRunner):
    """Formal session runner that releases its archive lease on close."""


class ManagedArcEngine(_LeasedArchiveMixin, ArcEngine):
    """Compatibility engine that releases its archive lease on close."""


def reserve_archive(path: Optional[str]) -> Optional[ArchiveLease]:
    """Reserve ``path`` for one active MCP session across local processes."""
    if path is None:
        return None
    with _ACTIVE_ARCHIVE_LOCK:
        if path in _ACTIVE_ARCHIVES:
            raise ValueError(
                "wal_path is already active in another compaction session; "
                "close that session before resuming it"
            )
        lock_path = Path(path + ".onecomp.lock")
        lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ValueError(
                    "wal_path is already active in another compaction process; "
                    "close that session before resuming it"
                ) from exc
            raise
        lease = ArchiveLease(path, descriptor)
        _ACTIVE_ARCHIVES[path] = lease
        return lease
