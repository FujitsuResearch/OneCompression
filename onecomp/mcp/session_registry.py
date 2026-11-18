"""Thread-safe, tenant-bound, expiring registry for state-referenced MCP tools.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generic, Iterator, Optional, TypeVar
from uuid import uuid4

ResourceT = TypeVar("ResourceT")


@dataclass
class _SessionMetadata:
    tenant_id: str
    token_hash: str
    auth_required: bool
    created_monotonic: float
    expires_monotonic: float
    ttl_seconds: int
    last_access_monotonic: float
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass(frozen=True)
class SessionHandle:
    session_id: str
    session_token: str
    tenant_id: str
    expires_in_seconds: int
    auth_required: bool


class SessionRegistry(Generic[ResourceT]):
    """Own resources with capacity, TTL, tenant, token, and close serialization."""

    def __init__(self, *, max_sessions: int = 256) -> None:
        if (
            not isinstance(max_sessions, int)
            or isinstance(max_sessions, bool)
            or max_sessions <= 0
        ):
            raise ValueError("max_sessions must be a positive integer")
        self.max_sessions = max_sessions
        self.resources: Dict[str, ResourceT] = {}
        self._metadata: Dict[str, _SessionMetadata] = {}
        self._lock = threading.RLock()

    def create(
        self,
        resource: ResourceT,
        *,
        tenant_id: str = "default",
        ttl_seconds: int = 3600,
        require_token: bool = True,
    ) -> SessionHandle:
        _validate_tenant(tenant_id)
        if (
            not isinstance(ttl_seconds, int)
            or isinstance(ttl_seconds, bool)
            or not 1 <= ttl_seconds <= 7 * 24 * 3600
        ):
            raise ValueError("ttl_seconds must be an integer between 1 and 604800")
        if not isinstance(require_token, bool):
            raise TypeError("require_token must be a boolean")
        raw_token = secrets.token_urlsafe(32)
        self.cleanup_expired()
        now = time.monotonic()
        with self._lock:
            if len(self.resources) >= self.max_sessions:
                raise ValueError(
                    f"Session limit reached ({self.max_sessions}). Close an unused session."
                )
            session_id = uuid4().hex[:12]
            while session_id in self.resources:
                session_id = uuid4().hex[:12]
            self.resources[session_id] = resource
            self._metadata[session_id] = _SessionMetadata(
                tenant_id=tenant_id,
                token_hash=_hash_token(raw_token),
                auth_required=require_token,
                created_monotonic=now,
                expires_monotonic=now + ttl_seconds,
                ttl_seconds=ttl_seconds,
                last_access_monotonic=now,
            )
        return SessionHandle(
            session_id=session_id,
            session_token=raw_token,
            tenant_id=tenant_id,
            expires_in_seconds=ttl_seconds,
            auth_required=require_token,
        )

    @contextmanager
    def lease(
        self,
        session_id: str,
        *,
        session_token: Optional[str] = None,
        tenant_id: str = "default",
    ) -> Iterator[ResourceT]:
        """Serialize one operation against close and refresh idle expiry."""
        metadata, resource = self._acquire(
            session_id,
            session_token=session_token,
            tenant_id=tenant_id,
        )
        try:
            yield resource
        finally:
            metadata.lock.release()

    def get(
        self,
        session_id: str,
        *,
        session_token: Optional[str] = None,
        tenant_id: str = "default",
    ) -> ResourceT:
        """Compatibility lookup. Prefer ``lease`` for multi-step operations."""
        metadata, resource = self._acquire(
            session_id,
            session_token=session_token,
            tenant_id=tenant_id,
        )
        metadata.lock.release()
        return resource

    def close(
        self,
        session_id: str,
        *,
        session_token: Optional[str] = None,
        tenant_id: str = "default",
    ) -> ResourceT:
        """Remove a resource atomically and wait for an active lease."""
        _validate_session_id(session_id)
        _validate_tenant(tenant_id)
        with self._lock:
            metadata = self._metadata.get(session_id)
            resource = self.resources.get(session_id)
            self._authorize(metadata, session_token, tenant_id)
            assert metadata is not None and resource is not None
            del self._metadata[session_id]
            del self.resources[session_id]
        metadata.lock.acquire()
        try:
            _close_resource(resource)
        finally:
            metadata.lock.release()
        return resource

    def close_all(self) -> None:
        with self._lock:
            detached = self._detach_all_locked()
        self._close_detached(detached)

    def cleanup_expired(self) -> int:
        with self._lock:
            detached = self._detach_expired_locked(time.monotonic())
        self._close_detached(detached)
        return len(detached)

    def _acquire(
        self,
        session_id: str,
        *,
        session_token: Optional[str],
        tenant_id: str,
    ) -> tuple[_SessionMetadata, ResourceT]:
        _validate_session_id(session_id)
        _validate_tenant(tenant_id)
        self.cleanup_expired()
        with self._lock:
            metadata = self._metadata.get(session_id)
            resource = self.resources.get(session_id)
            self._authorize(metadata, session_token, tenant_id)
            assert metadata is not None and resource is not None
        metadata.lock.acquire()
        with self._lock:
            if (
                self._metadata.get(session_id) is not metadata
                or self.resources.get(session_id) is not resource
            ):
                metadata.lock.release()
                raise ValueError("Unknown session_id or invalid session credentials")
            now = time.monotonic()
            metadata.last_access_monotonic = now
            metadata.expires_monotonic = now + metadata.ttl_seconds
            return metadata, resource

    @staticmethod
    def _authorize(
        metadata: Optional[_SessionMetadata],
        session_token: Optional[str],
        tenant_id: str,
    ) -> None:
        valid = metadata is not None and metadata.tenant_id == tenant_id
        if valid and metadata.auth_required:
            supplied_hash = _hash_token(session_token or "")
            valid = hmac.compare_digest(metadata.token_hash, supplied_hash)
        if not valid:
            raise ValueError("Unknown session_id or invalid session credentials")

    def _detach_expired_locked(
        self,
        now: float,
    ) -> list[tuple[_SessionMetadata, ResourceT]]:
        expired_ids = [
            session_id
            for session_id, metadata in self._metadata.items()
            if now >= metadata.expires_monotonic
        ]
        return self._detach_locked(expired_ids)

    def _detach_all_locked(self) -> list[tuple[_SessionMetadata, ResourceT]]:
        return self._detach_locked(list(self._metadata))

    def _detach_locked(
        self,
        session_ids: list[str],
    ) -> list[tuple[_SessionMetadata, ResourceT]]:
        detached: list[tuple[_SessionMetadata, ResourceT]] = []
        for session_id in session_ids:
            metadata = self._metadata.pop(session_id, None)
            resource = self.resources.pop(session_id, None)
            if metadata is not None and resource is not None:
                detached.append((metadata, resource))
        return detached

    @staticmethod
    def _close_detached(detached: list[tuple[_SessionMetadata, ResourceT]]) -> None:
        first_error: BaseException | None = None
        for metadata, resource in detached:
            metadata.lock.acquire()
            try:
                _close_resource(resource)
            # Deliberately include cancellation and interrupts: every detached
            # resource must receive close before the first failure is re-raised.
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                if first_error is None:
                    first_error = exc
            finally:
                metadata.lock.release()
        if first_error is not None:
            raise first_error


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_session_id(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 12
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Unknown session_id or invalid session credentials")


def _validate_tenant(value: object) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("tenant_id must be a non-empty string up to 256 characters")


def _close_resource(resource: object) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


__all__ = ["SessionHandle", "SessionRegistry"]
