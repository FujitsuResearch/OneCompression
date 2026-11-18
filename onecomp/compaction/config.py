"""User-facing configuration for standalone and host-integrated compaction.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .compactor import DEFAULT_ACTION_REGEX, DEFAULT_OBSERVATION_REGEX

CompactionProfile = Literal["formal-v1", "experimental-v1"]
CompactionDurability = Literal["full", "normal"]


@dataclass(frozen=True)
class CompactionConfig:
    """OneComp-style recipe for Addressable Recall Compaction.

    ``formal-v1`` is the production profile. It uses occurrence-addressed,
    transactional storage and bounded exact recovery. ``experimental-v1`` is
    retained only for behavior compatibility with existing integrations.

    The default archive is in-memory to avoid surprising filesystem writes.
    Set ``archive_path`` for resumable runs, or use :meth:`persistent`.
    """

    profile: CompactionProfile = "formal-v1"
    archive_path: Optional[str | Path] = None
    keep_recent: int = 1
    thought_max_chars: int = 300
    result_max_chars: int = 500
    citation_head_chars: int = 200
    citation_tail_chars: int = 200
    summary_max_chars: int = 24_000
    max_recall_units: int = 4096
    recalls_per_step: int = 2
    max_archive_bytes: int = 0
    max_occurrences: int = 1 << 48
    fingerprint_bits: int = 64
    verify_archive_reads: bool = True
    durability: CompactionDurability = "full"
    action_regex: str = DEFAULT_ACTION_REGEX
    observation_regex: str = DEFAULT_OBSERVATION_REGEX
    context_window: Optional[int] = None
    completion_reserve: int = 0
    serialization_margin: int = 0
    max_telemetry_events: int = 10_000

    def __post_init__(self) -> None:
        if self.profile not in ("formal-v1", "experimental-v1"):
            raise ValueError("profile must be 'formal-v1' or 'experimental-v1'")
        if self.archive_path is not None and (
            not isinstance(self.archive_path, (str, Path)) or not str(self.archive_path).strip()
        ):
            raise ValueError("archive_path must be a non-empty path or None")
        non_negative = {
            "keep_recent": self.keep_recent,
            "thought_max_chars": self.thought_max_chars,
            "result_max_chars": self.result_max_chars,
            "citation_head_chars": self.citation_head_chars,
            "citation_tail_chars": self.citation_tail_chars,
            "summary_max_chars": self.summary_max_chars,
            "recalls_per_step": self.recalls_per_step,
            "max_archive_bytes": self.max_archive_bytes,
            "completion_reserve": self.completion_reserve,
            "serialization_margin": self.serialization_margin,
            "max_telemetry_events": self.max_telemetry_events,
        }
        for name, value in non_negative.items():
            _require_integer(name, value, minimum=0)
        _require_integer("max_recall_units", self.max_recall_units, minimum=1)
        _require_integer("max_occurrences", self.max_occurrences, minimum=1)
        _require_integer("fingerprint_bits", self.fingerprint_bits, minimum=16)
        if self.max_occurrences > 1 << 63:
            raise ValueError("max_occurrences must not exceed 2**63")
        if self.fingerprint_bits > 256 or self.fingerprint_bits % 8:
            raise ValueError("fingerprint_bits must be a multiple of 8 between 16 and 256")
        if not isinstance(self.verify_archive_reads, bool):
            raise TypeError("verify_archive_reads must be a boolean")
        if self.durability not in ("full", "normal"):
            raise ValueError("durability must be 'full' or 'normal'")
        for name, pattern in (
            ("action_regex", self.action_regex),
            ("observation_regex", self.observation_regex),
        ):
            if not isinstance(pattern, str):
                raise TypeError(f"{name} must be a string")
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{name} is not a valid regular expression") from exc
            if compiled.groups != 1:
                raise ValueError(f"{name} must contain exactly one capture group")
        if self.context_window is None:
            if self.completion_reserve or self.serialization_margin:
                raise ValueError(
                    "context_window is required when provider budget reserves are set"
                )
        else:
            _require_integer("context_window", self.context_window, minimum=1)
            if self.completion_reserve >= self.context_window:
                raise ValueError("completion_reserve must be smaller than context_window")

    @classmethod
    def persistent(
        cls,
        root: str | Path = ".onecomp/compaction",
        **overrides: Any,
    ) -> "CompactionConfig":
        """Create a formal recipe with a conventional resumable archive path."""
        if "profile" in overrides or "archive_path" in overrides:
            raise TypeError("persistent() owns profile and archive_path")
        return cls(
            profile="formal-v1",
            archive_path=Path(root) / "archive.sqlite3",
            **overrides,
        )

    @classmethod
    def experimental(
        cls,
        wal_path: Optional[str | Path] = None,
        **overrides: Any,
    ) -> "CompactionConfig":
        """Create an explicitly compatibility-only experimental recipe."""
        if "profile" in overrides or "archive_path" in overrides:
            raise TypeError("experimental() owns profile and archive_path")
        return cls(
            profile="experimental-v1",
            archive_path=wal_path,
            **overrides,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly recipe without creating the archive."""
        value = asdict(self)
        if self.archive_path is not None:
            value["archive_path"] = str(self.archive_path)
        return value


def _require_integer(name: str, value: object, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
