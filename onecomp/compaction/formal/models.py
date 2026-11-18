"""Immutable models shared by the formal ARC profile.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


class _FrozenDict(dict):
    """JSON-compatible mapping that rejects mutation after construction."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("formal ARC metadata is immutable")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class ObservationRecord:
    """One exact tool-result occurrence."""

    sequence: int
    obs_id: str
    source_key: Optional[str]
    cmd_signature: str
    full_content: str
    returncode: int
    created_step: int
    extra: Mapping[str, Any]
    created_ns: int
    content_sha256: str
    char_count: int
    line_count: int
    byte_count: int
    tags: Tuple[str, ...] = ()
    head: str = ""
    tail: str = ""
    last_recalled: int = 0
    n_recalls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_json(self.extra))

    @property
    def command_signature(self) -> str:
        """Framework-neutral spelling used by the standalone service."""
        return self.cmd_signature


@dataclass(frozen=True)
class ObservationMetadata:
    """Catalog row that intentionally excludes the archived body."""

    sequence: int
    obs_id: str
    source_key: Optional[str]
    cmd_signature: str
    returncode: int
    created_step: int
    created_ns: int
    content_sha256: str
    char_count: int
    line_count: int
    byte_count: int
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogPage:
    """Stable sequence-cursor page from the complete catalog."""

    items: Tuple[ObservationMetadata, ...]
    next_cursor: Optional[int]
    has_more: bool


@dataclass(frozen=True)
class TextChunk:
    """Exact, adjacent slice of a Unicode observation."""

    obs_id: str
    index: int
    total_chunks: int
    char_start: int
    char_end: int
    text: str
    measured_units: int
    content_sha256: str = ""


@dataclass(frozen=True)
class TokenIdChunk:
    """Exact, adjacent slice of an encoded token sequence."""

    obs_id: str
    index: int
    total_chunks: int
    token_start: int
    token_end: int
    token_ids: Tuple[int, ...]
    text: str
    content_sha256: str = ""


@dataclass(frozen=True)
class BudgetSnapshot:
    """Auditable decomposition of one model input envelope."""

    context_window: int
    completion_reserve: int
    baseline: int = 0
    memory: int = 0
    recall: int = 0
    protected_history: int = 0
    query: int = 0
    serialization_margin: int = 0
    labels: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _freeze_json(self.labels))

    @property
    def input_capacity(self) -> int:
        return self.context_window - self.completion_reserve

    @property
    def input_used(self) -> int:
        return (
            self.baseline
            + self.memory
            + self.recall
            + self.protected_history
            + self.query
            + self.serialization_margin
            + sum(self.labels.values())
        )

    @property
    def remaining(self) -> int:
        return self.input_capacity - self.input_used


@dataclass(frozen=True)
class ArchiveStats:
    """Body-free archive accounting for status and telemetry."""

    observations: int
    body_bytes: int
    recalled_observations: int
    recall_count: int


@dataclass(frozen=True)
class InvocationAudit:
    """Thread-safe counter snapshot from a pre-dispatch budget guard."""

    checked_requests: int
    dispatched_requests: int
    rejected_requests: int
    last_checked_units: Optional[int]
    dispatch_successes: int = 0
    dispatch_failures: int = 0


@dataclass(frozen=True)
class MigrationReport:
    """Deterministic result of importing one experimental JSONL WAL."""

    source: str
    destination: str
    loaded: int
    skipped: int
    aliases_registered: int
    errors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionResponse:
    """Host-neutral action result with exact body separated from metadata."""

    kind: str
    body: str
    obs_id: Optional[str] = None
    exact: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
