"""Transactional, append-only observation archive for formal ARC.

The archive is the source of truth for every observation occurrence. SQLite
allocates the occurrence sequence, writes the exact body and advances the
sequence in one transaction. Canonical citations never depend on process-local
state or on replaying an ID algorithm.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Optional

from .blocks import SQLiteBodyBlockIndex, build_body_layout
from .errors import (
    AmbiguousObservation,
    ArchiveCapacityExceeded,
    ArchiveConfigurationMismatch,
    ArchiveIntegrityError,
    SourceOccurrenceConflict,
    UnknownObservation,
)
from .ids import OccurrenceIdConfig, OccurrenceIdFactory
from .models import ArchiveStats, CatalogPage, ObservationMetadata, ObservationRecord

_SCHEMA_VERSION = 2
_MAX_COMMAND_CHARS = 16_384
_MAX_SOURCE_KEY_CHARS = 1_024
_MAX_METADATA_BYTES = 65_536
_MAX_TAGS = 64
_MAX_TAG_CHARS = 256


class SQLiteObservationArchive:
    """Process-safe, exact observation storage using Python's stdlib SQLite.

    Args:
        path: SQLite path or ``":memory:"``.
        id_config: Permanent fixed-width ID configuration. Reopening with a
            different configuration fails closed.
        timeout_seconds: SQLite busy timeout.
        max_archive_bytes: Optional exact UTF-8 body quota. ``0`` disables the
            quota. The value is persisted and cannot silently change.
        preview_chars: Head and tail characters exposed in bounded citations.
        verify_on_read: Recompute SHA-256 and the occurrence fingerprint for
            full-body reads. Range reads verify only the fixed body blocks they
            return against hashes committed with the occurrence.
        read_only: Open an existing archive without chmod, schema updates,
            telemetry writes, or partial-index backfills. This supports recall
            commands running inside a host sandbox that can read but not write
            the hook-owned archive.
        durability: ``"full"`` fsyncs each body transaction. ``"normal"``
            trades power-loss durability for higher write throughput while
            retaining SQLite transactional and process-crash recovery.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        id_config: OccurrenceIdConfig | None = None,
        timeout_seconds: float = 30.0,
        max_archive_bytes: int = 0,
        preview_chars: int = 200,
        verify_on_read: bool = True,
        read_only: bool = False,
        durability: Literal["full", "normal"] = "full",
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name, value in (
            ("max_archive_bytes", max_archive_bytes),
            ("preview_chars", preview_chars),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(verify_on_read, bool):
            raise ValueError("verify_on_read must be a boolean")
        if not isinstance(read_only, bool):
            raise ValueError("read_only must be a boolean")
        if durability not in {"full", "normal"}:
            raise ValueError("durability must be 'full' or 'normal'")

        self.path = str(path)
        self.preview_chars = preview_chars
        self.verify_on_read = verify_on_read
        self.read_only = read_only
        self.durability = durability
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._pending_recalls: dict[str, tuple[int, int]] = {}
        self._data_version = 0
        self.body_search_index_enabled = False
        self._closed = False
        if read_only:
            if self.path == ":memory:":
                raise ValueError("read_only archives require an existing file path")
            path = Path(self.path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"archive does not exist: {path}")
            self.path = str(path)
            # ``immutable=1`` prevents SQLite from taking write locks or
            # opening mutable WAL shared memory in a restricted agent
            # sandbox. Hook writers checkpoint before publishing a citation,
            # so every cited occurrence is already present in the main file.
            connection_path = path.as_uri() + "?mode=ro&immutable=1"
        else:
            self._prepare_path()
            connection_path = self.path
        self._conn = sqlite3.connect(
            connection_path,
            timeout=float(timeout_seconds),
            isolation_level=None,
            check_same_thread=False,
            uri=read_only,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.create_function("arc_casefold", 1, _sqlite_casefold, deterministic=True)
        self._block_index = SQLiteBodyBlockIndex(self._conn, verify=verify_on_read)
        try:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
            if read_only:
                self._conn.execute("PRAGMA query_only = ON")
                self.id_config, self.max_archive_bytes = self._load_existing_configuration(
                    id_config,
                    max_archive_bytes,
                    repair_missing=False,
                )
                self.body_search_index_enabled = self._body_search_index_available()
            else:
                if self.path != ":memory:":
                    row = self._conn.execute("PRAGMA journal_mode = WAL").fetchone()
                    actual_mode = "" if row is None else str(row[0]).casefold()
                    if actual_mode != "wal":
                        raise ArchiveConfigurationMismatch(
                            "formal archive requires SQLite WAL journal mode; "
                            f"storage returned {actual_mode or 'no mode'}"
                        )
                    self._conn.execute(f"PRAGMA synchronous = {durability.upper()}")
                self.id_config, self.max_archive_bytes = self._initialize(
                    id_config, max_archive_bytes
                )
                self._secure_file()
            self.id_factory = OccurrenceIdFactory(self.id_config)
            self._data_version = self._current_data_version()
        except sqlite3.OperationalError as exc:
            self._conn.close()
            if read_only:
                raise ArchiveConfigurationMismatch(
                    "read-only archive is missing or has an incompatible schema"
                ) from exc
            raise
        except BaseException:
            self._conn.close()
            raise

    def _prepare_path(self) -> None:
        if self.path == ":memory:":
            return
        path = Path(self.path).expanduser()
        _create_private_directories(path.parent)
        path = path.resolve()
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError as exc:
            if not path.is_file():
                raise ValueError(f"archive path is not a regular file: {path}") from exc
        else:
            os.close(descriptor)
        os.chmod(path, 0o600)
        self.path = str(path)
        self._secure_file()

    def _secure_file(self) -> None:
        if self.path == ":memory:":
            return
        # SQLite may keep the WAL and shared-memory sidecars alive while the
        # archive is open. Protect all existing database artifacts, not only
        # the main file; never create a missing sidecar merely to chmod it.
        for suffix in ("", "-wal", "-shm", "-journal"):
            candidate = self.path + suffix
            if not os.path.exists(candidate):
                continue
            try:
                os.chmod(candidate, 0o600)
            except FileNotFoundError:
                continue

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._require_open()
        self._require_writable()
        if self._transaction_depth:
            savepoint = f"arc_nested_{self._transaction_depth}"
            self._conn.execute(f"SAVEPOINT {savepoint}")
            self._transaction_depth += 1
            try:
                yield
            except BaseException:
                self._conn.execute(f"ROLLBACK TO {savepoint}")
                self._conn.execute(f"RELEASE {savepoint}")
                raise
            else:
                self._conn.execute(f"RELEASE {savepoint}")
            finally:
                self._transaction_depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._transaction_depth = 1
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")
        finally:
            self._transaction_depth = 0

    @contextmanager
    def batch(self) -> Iterator["SQLiteObservationArchive"]:
        """Commit multiple archive operations in one atomic transaction."""
        with self._lock, self._transaction():
            yield self

    @contextmanager
    def _read_snapshot(self) -> Iterator[None]:
        """Hold one SQLite snapshot across cache validation and range read."""
        if self._transaction_depth:
            yield
            return
        self._conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _initialize(
        self,
        requested_id: OccurrenceIdConfig | None,
        requested_quota: int,
    ) -> tuple[OccurrenceIdConfig, int]:
        with self._lock, self._transaction():
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS arc_meta ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    sequence INTEGER PRIMARY KEY,
                    obs_id TEXT NOT NULL UNIQUE,
                    source_key TEXT,
                    cmd_signature TEXT NOT NULL,
                    full_content TEXT NOT NULL,
                    returncode INTEGER NOT NULL,
                    created_step INTEGER NOT NULL,
                    extra_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_ns INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    line_count INTEGER NOT NULL,
                    byte_count INTEGER NOT NULL,
                    record_sha256 TEXT NOT NULL,
                    last_recalled INTEGER NOT NULL DEFAULT 0,
                    n_recalls INTEGER NOT NULL DEFAULT 0
                )
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS observation_aliases (
                    alias TEXT PRIMARY KEY,
                    obs_id TEXT NOT NULL REFERENCES observations(obs_id)
                )
                """)
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_observations_source_key "
                "ON observations(source_key) WHERE source_key IS NOT NULL"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_step "
                "ON observations(created_step, sequence)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_created "
                "ON observations(created_ns, sequence)"
            )
            self._block_index.initialize()
            self._initialize_body_search_index()

            existing = self._meta_get("id_config")
            if existing is None:
                config = requested_id or OccurrenceIdConfig()
                self._meta_set("schema_version", str(_SCHEMA_VERSION))
                self._meta_set("id_config", json.dumps(config.to_dict(), sort_keys=True))
                self._meta_set("next_sequence", "0")
                self._meta_set("body_bytes", "0")
                self._meta_set("max_archive_bytes", str(requested_quota))
                return config, requested_quota

            return self._load_existing_configuration(
                requested_id,
                requested_quota,
                repair_missing=True,
            )

    def _load_existing_configuration(
        self,
        requested_id: OccurrenceIdConfig | None,
        requested_quota: int,
        *,
        repair_missing: bool,
    ) -> tuple[OccurrenceIdConfig, int]:
        existing = self._meta_get("id_config")
        if existing is None:
            raise ArchiveConfigurationMismatch("formal archive identity metadata is missing")
        config = OccurrenceIdConfig.from_dict(json.loads(existing))
        version = int(self._meta_get("schema_version") or "0")
        if version == 1 and repair_missing:
            self._upgrade_v1_integrity_metadata(config)
            version = _SCHEMA_VERSION
        if version != _SCHEMA_VERSION:
            raise ArchiveConfigurationMismatch(
                f"unsupported formal archive schema version {version}"
            )
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(observations)").fetchall()
        }
        if "record_sha256" not in columns:
            raise ArchiveConfigurationMismatch("archive record integrity column is missing")
        missing_digest = self._conn.execute(
            "SELECT 1 FROM observations WHERE record_sha256 IS NULL LIMIT 1"
        ).fetchone()
        if missing_digest is not None:
            raise ArchiveIntegrityError("archive record integrity metadata is incomplete")
        if requested_id is not None and requested_id != config:
            raise ArchiveConfigurationMismatch(
                "existing archive ID configuration differs from the request"
            )
        quota = int(self._meta_get("max_archive_bytes") or "0")
        if requested_quota not in (0, quota) and requested_quota != quota:
            raise ArchiveConfigurationMismatch(
                "existing archive byte quota differs from the request"
            )
        self._verify_archive_counters(repair_missing=repair_missing)
        return config, quota

    def _upgrade_v1_integrity_metadata(self, config: OccurrenceIdConfig) -> None:
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(observations)").fetchall()
        }
        if "record_sha256" not in columns:
            self._conn.execute("ALTER TABLE observations ADD COLUMN record_sha256 TEXT")
        id_factory = OccurrenceIdFactory(config)
        rows = self._conn.execute("SELECT * FROM observations ORDER BY sequence")
        for row in rows:
            _verify_v1_record(row, id_factory)
            self._conn.execute(
                "UPDATE observations SET record_sha256 = ? WHERE sequence = ?",
                (_record_sha256_from_row(row), int(row["sequence"])),
            )
        self._meta_set("schema_version", str(_SCHEMA_VERSION))

    def _verify_archive_counters(self, *, repair_missing: bool) -> None:
        row = self._conn.execute(
            "SELECT COUNT(*) AS count, COALESCE(MAX(sequence), -1) + 1 AS next_sequence, "
            "COALESCE(SUM(byte_count), 0) AS body_bytes FROM observations"
        ).fetchone()
        actual_next = int(row["next_sequence"])
        actual_bytes = int(row["body_bytes"])
        if int(row["count"]) != actual_next:
            raise ArchiveIntegrityError("formal archive occurrence sequence contains a gap")
        for key, actual in (("next_sequence", actual_next), ("body_bytes", actual_bytes)):
            stored = self._meta_get(key)
            if stored is None:
                if not repair_missing:
                    raise ArchiveConfigurationMismatch(
                        f"formal archive {key.replace('_', ' ')} metadata is missing"
                    )
                self._meta_set(key, str(actual))
                continue
            try:
                parsed = int(stored)
            except ValueError as exc:
                raise ArchiveIntegrityError(
                    f"formal archive {key.replace('_', ' ')} metadata is invalid"
                ) from exc
            if parsed != actual:
                raise ArchiveIntegrityError(
                    f"formal archive {key.replace('_', ' ')} metadata does not match records"
                )

    def _body_search_index_available(self) -> bool:
        query = "SELECT 1 FROM sqlite_master " + (
            "WHERE type = 'table' AND name = 'observation_body_fts'"
        )
        row = self._conn.execute(query).fetchone()
        return row is not None

    def _initialize_body_search_index(self) -> None:
        """Enable an exact-filtered trigram index when SQLite provides FTS5."""
        self._conn.execute("SAVEPOINT arc_body_search_setup")
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS observation_body_fts USING fts5("
                "full_content, content='observations', content_rowid='sequence', "
                "tokenize='trigram')"
            )
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS observation_body_fts_insert "
                "AFTER INSERT ON observations BEGIN "
                "INSERT INTO observation_body_fts(rowid, full_content) "
                "VALUES (new.sequence, new.full_content); END"
            )
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS observation_body_fts_delete "
                "AFTER DELETE ON observations BEGIN "
                "INSERT INTO observation_body_fts(observation_body_fts, rowid, full_content) "
                "VALUES ('delete', old.sequence, old.full_content); END"
            )
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS observation_body_fts_update "
                "AFTER UPDATE OF full_content ON observations BEGIN "
                "INSERT INTO observation_body_fts(observation_body_fts, rowid, full_content) "
                "VALUES ('delete', old.sequence, old.full_content); "
                "INSERT INTO observation_body_fts(rowid, full_content) "
                "VALUES (new.sequence, new.full_content); END"
            )
            if self._meta_get("body_search_index") != "trigram-v1":
                self._conn.execute(
                    "INSERT INTO observation_body_fts(observation_body_fts) VALUES ('rebuild')"
                )
                self._meta_set("body_search_index", "trigram-v1")
        except sqlite3.OperationalError:
            # Some minimal Python/SQLite builds omit FTS5 or its trigram
            # tokenizer. Roll back partial optional objects so they cannot
            # interfere with later core inserts. Literal ``instr`` remains a
            # correct fallback.
            self._conn.execute("ROLLBACK TO arc_body_search_setup")
            self._conn.execute("RELEASE arc_body_search_setup")
            self.body_search_index_enabled = False
        else:
            self._conn.execute("RELEASE arc_body_search_setup")
            self.body_search_index_enabled = True

    def _meta_get(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM arc_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO arc_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def append(
        self,
        cmd_signature: str,
        full_content: str,
        *,
        source_key: str | None = None,
        returncode: int = 0,
        created_step: int = 0,
        extra: Mapping[str, Any] | None = None,
        tags: tuple[str, ...] | list[str] = (),
        created_ns: int | None = None,
    ) -> ObservationRecord:
        """Atomically commit one occurrence and return its canonical record."""
        self._require_writable()
        _validate_occurrence(
            cmd_signature,
            full_content,
            source_key,
            returncode,
            created_step,
            created_ns,
        )
        payload = dict(extra or {})
        tag_tuple = _normalize_tags(tags)
        extra_json = _json_object("extra", payload)
        tags_json = json.dumps(tag_tuple, ensure_ascii=False, separators=(",", ":"))
        metadata_bytes = len(extra_json.encode("utf-8")) + len(tags_json.encode("utf-8"))
        if metadata_bytes > _MAX_METADATA_BYTES:
            raise ValueError(f"extra and tags must fit within {_MAX_METADATA_BYTES} UTF-8 bytes")
        created_ns = time.time_ns() if created_ns is None else created_ns
        body_layout = build_body_layout(full_content)
        checksum = body_layout.content_sha256
        byte_count = body_layout.byte_count
        char_count = len(full_content)
        line_count = (
            0
            if not full_content
            else full_content.count("\n") + (0 if full_content.endswith("\n") else 1)
        )

        with self._lock, self._transaction():
            if source_key is not None:
                prior = self._conn.execute(
                    "SELECT * FROM observations WHERE source_key = ?", (source_key,)
                ).fetchone()
                if prior is not None:
                    existing = self._verified_record_from_row(prior)
                    if not _same_source_occurrence(
                        existing,
                        cmd_signature,
                        full_content,
                        returncode,
                        extra_json,
                        tag_tuple,
                    ):
                        raise SourceOccurrenceConflict(
                            "source_key was already committed with different occurrence data"
                        )
                    return existing

            sequence = int(self._meta_get("next_sequence") or "0")
            if sequence >= self.id_config.max_occurrences:
                raise ArchiveCapacityExceeded(
                    "archive exhausted its declared occurrence namespace"
                )
            if self.max_archive_bytes:
                used = int(self._meta_get("body_bytes") or "0")
                if used + byte_count > self.max_archive_bytes:
                    raise ArchiveCapacityExceeded(
                        "archive body quota exceeded: "
                        f"used={used}, new={byte_count}, quota={self.max_archive_bytes}"
                    )
            obs_id = self.id_factory.mint(sequence, cmd_signature, full_content)
            record_sha256 = _record_sha256(
                sequence=sequence,
                obs_id=obs_id,
                source_key=source_key,
                cmd_signature=cmd_signature,
                returncode=returncode,
                created_step=created_step,
                extra_json=extra_json,
                tags_json=tags_json,
                created_ns=created_ns,
                content_sha256=checksum,
                char_count=char_count,
                line_count=line_count,
                byte_count=byte_count,
            )
            self._conn.execute(
                """
                INSERT INTO observations(
                    sequence, obs_id, source_key, cmd_signature, full_content,
                    returncode, created_step, extra_json, tags_json, created_ns,
                    content_sha256, char_count, line_count, byte_count,
                    record_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence,
                    obs_id,
                    source_key,
                    cmd_signature,
                    full_content,
                    returncode,
                    created_step,
                    extra_json,
                    tags_json,
                    created_ns,
                    checksum,
                    char_count,
                    line_count,
                    byte_count,
                    record_sha256,
                ),
            )
            self._meta_set("next_sequence", str(sequence + 1))
            self._meta_set(
                "body_bytes",
                str(int(self._meta_get("body_bytes") or "0") + byte_count),
            )
            self._block_index.write(sequence, checksum, body_layout.blocks)
            record = ObservationRecord(
                sequence=sequence,
                obs_id=obs_id,
                source_key=source_key,
                cmd_signature=cmd_signature,
                full_content=full_content,
                returncode=returncode,
                created_step=created_step,
                extra=payload,
                created_ns=created_ns,
                content_sha256=checksum,
                char_count=char_count,
                line_count=line_count,
                byte_count=byte_count,
                tags=tag_tuple,
                head=full_content[: self.preview_chars],
                tail=(
                    full_content[-self.preview_chars :]
                    if self.preview_chars and len(full_content) > self.preview_chars
                    else ""
                ),
            )

        # The object was built from the values committed above, so selecting
        # and hashing the potentially large body again would add no assurance.
        return record

    def put(
        self,
        *,
        cmd_signature: str,
        full_content: str,
        returncode: int = 0,
        created_step: int = 0,
        head_chars: int = 200,
        tail_chars: int = 200,
        extra: Mapping[str, Any] | None = None,
        source_key: str | None = None,
    ) -> ObservationRecord:
        """Compatibility surface used by :class:`ArcEngine`."""
        del head_chars, tail_chars
        return self.append(
            cmd_signature,
            full_content,
            source_key=source_key,
            returncode=returncode,
            created_step=created_step,
            extra=extra,
        )

    def register_alias(self, alias: str, obs_id: str) -> None:
        """Register a legacy citation without changing the canonical ID."""
        self._require_writable()
        if (
            not isinstance(alias, str)
            or not 4 <= len(alias) <= 128
            or any(character not in "0123456789abcdefABCDEF" for character in alias)
        ):
            raise ValueError("legacy alias must be 4-128 hexadecimal characters")
        alias = alias.lower()
        canonical = self._canonical_id(obs_id)
        with self._lock, self._transaction():
            row = self._conn.execute(
                "SELECT obs_id FROM observation_aliases WHERE alias = ?", (alias,)
            ).fetchone()
            if row is not None and str(row["obs_id"]) != canonical:
                raise SourceOccurrenceConflict("legacy alias maps to another observation")
            self._conn.execute(
                "INSERT OR IGNORE INTO observation_aliases(alias, obs_id) VALUES (?, ?)",
                (alias, canonical),
            )

    def bind_owner(self, owner: str) -> None:
        """Permanently bind a persistent archive to one opaque owner.

        The archive stores only a SHA-256 digest. This is primarily used by
        multi-tenant control planes so a user cannot resume another tenant's
        otherwise guessable WAL path.
        """
        self._require_writable()
        if not isinstance(owner, str) or not owner or len(owner) > 1024:
            raise ValueError("owner must be a non-empty string up to 1024 characters")
        digest = hashlib.sha256(owner.encode("utf-8")).hexdigest()
        with self._lock, self._transaction():
            existing = self._meta_get("owner_sha256")
            if existing is None:
                self._meta_set("owner_sha256", digest)
            elif not hmac.compare_digest(existing, digest):
                raise ArchiveConfigurationMismatch("archive is bound to a different owner")

    def _canonical_id(self, obs_id: str) -> str:
        self._require_open()
        if not isinstance(obs_id, str) or not obs_id:
            raise UnknownObservation("observation id must be a non-empty string")
        normalized = obs_id.lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT obs_id FROM observations WHERE obs_id = ?", (normalized,)
            ).fetchone()
            if row is None:
                row = self._conn.execute(
                    "SELECT obs_id FROM observation_aliases WHERE alias = ?",
                    (normalized,),
                ).fetchone()
        if row is None:
            raise UnknownObservation(f"unknown observation id: §{obs_id}")
        return str(row["obs_id"])

    def has(self, obs_id: str) -> bool:
        try:
            self._canonical_id(obs_id)
            return True
        except UnknownObservation:
            return False

    def get(self, obs_id: str) -> ObservationRecord:
        canonical = self._canonical_id(obs_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM observations WHERE obs_id = ?", (canonical,)
            ).fetchone()
        if row is None:  # pragma: no cover - protected by the canonical lookup
            raise UnknownObservation(f"unknown observation id: §{obs_id}")
        return self._verified_record_from_row(row)

    def get_metadata(self, obs_id: str) -> ObservationMetadata:
        """Read metadata without selecting the potentially large body."""
        canonical = self._canonical_id(obs_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT sequence, obs_id, source_key, cmd_signature, returncode, "
                "created_step, extra_json, tags_json, created_ns, content_sha256, "
                "char_count, line_count, byte_count, record_sha256 "
                "FROM observations WHERE obs_id = ?",
                (canonical,),
            ).fetchone()
        if row is None:  # pragma: no cover - protected by canonical lookup
            raise UnknownObservation(f"unknown observation id: §{obs_id}")
        return self._metadata_from_row(row)

    def resolve_prefix(self, prefix: str, *, minimum_chars: int = 8) -> str:
        if (
            not isinstance(prefix, str)
            or len(prefix) < minimum_chars
            or any(character not in "0123456789abcdef" for character in prefix)
        ):
            raise UnknownObservation(
                f"prefix must contain at least {minimum_chars} lowercase hex characters"
            )
        with self._lock:
            rows = self._conn.execute(
                "SELECT obs_id FROM observations WHERE obs_id LIKE ? ORDER BY sequence LIMIT 2",
                (prefix + "%",),
            ).fetchall()
        if not rows:
            raise UnknownObservation(f"unknown observation prefix: §{prefix}")
        if len(rows) > 1:
            raise AmbiguousObservation(f"ambiguous observation prefix: §{prefix}")
        return str(rows[0]["obs_id"])

    def read_range(self, obs_id: str, start: int, end: int) -> str:
        """Read an exact character range without repeatedly loading the body.

        Fixed character blocks map to byte offsets in the canonical SQLite
        value. Only overlapping blocks are read and verified, so sequential
        chunk recovery is linear in returned data rather than body size.
        Existing archives without block metadata are fully verified and
        backfilled on their first range access.
        """
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
        ):
            raise ValueError("range must satisfy 0 <= start <= end")
        canonical = self._canonical_id(obs_id)
        fallback_record: ObservationRecord | None = None
        with self._lock, self._read_snapshot():
            self._refresh_verification_cache()
            metadata = self._conn.execute(
                "SELECT sequence, obs_id, source_key, cmd_signature, returncode, "
                "created_step, extra_json, tags_json, created_ns, content_sha256, "
                "char_count, line_count, byte_count, record_sha256 "
                "FROM observations WHERE obs_id = ?",
                (canonical,),
            ).fetchone()
            if metadata is None:  # pragma: no cover - canonical lookup guards this
                raise UnknownObservation(f"unknown observation id: §{obs_id}")
            self._verify_metadata_row(metadata)
            if end > int(metadata["char_count"]):
                raise ValueError("range end exceeds observation length")
            if start == end:
                return ""
            try:
                indexed = self._block_index.read(canonical, metadata, start, end)
            except ArchiveIntegrityError:
                # Block metadata is a rebuildable accelerator, not the source
                # of truth. A full canonical verification below distinguishes
                # index damage from body corruption before replacing it.
                indexed = None
            if indexed is not None:
                return indexed
            row = self._conn.execute(
                "SELECT * FROM observations WHERE obs_id = ?", (canonical,)
            ).fetchone()
            if row is None:  # pragma: no cover - metadata lookup guards this
                raise UnknownObservation(f"unknown observation id: §{obs_id}")
            fallback_record = self._verified_record_from_row(row)
            result = fallback_record.full_content[start:end]
        assert fallback_record is not None
        if self.read_only:
            return result
        layout = build_body_layout(fallback_record.full_content)
        with self._lock, self._transaction():
            current = self._conn.execute(
                "SELECT content_sha256 FROM observations WHERE sequence = ?",
                (fallback_record.sequence,),
            ).fetchone()
            if current is not None and str(current["content_sha256"]) == layout.content_sha256:
                self._block_index.write(
                    fallback_record.sequence,
                    layout.content_sha256,
                    layout.blocks,
                )
        return result

    def catalog_page(
        self,
        *,
        after_sequence: int | None = None,
        limit: int = 50,
        query: str = "",
        created_step: int | None = None,
        returncode: int | None = None,
        tag: str | None = None,
    ) -> CatalogPage:
        """Scan the complete body-free catalog using a stable cursor."""
        cursor = -1 if after_sequence is None else after_sequence
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < -1:
            raise ValueError("after_sequence must be None or an integer >= -1")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer between 1 and 500")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        conditions = ["sequence > ?"]
        params: list[object] = [cursor]
        if query:
            conditions.append(
                "(instr(arc_casefold(cmd_signature), arc_casefold(?)) > 0 "
                "OR instr(arc_casefold(obs_id), arc_casefold(?)) > 0 "
                "OR instr(arc_casefold(tags_json), arc_casefold(?)) > 0)"
            )
            params.extend([query, query, query])
        if created_step is not None:
            _validate_optional_int("created_step", created_step)
            conditions.append("created_step = ?")
            params.append(created_step)
        if returncode is not None:
            _validate_optional_int("returncode", returncode, non_negative=False)
            conditions.append("returncode = ?")
            params.append(returncode)
        if tag is not None:
            if not isinstance(tag, str) or not tag:
                raise ValueError("tag must be a non-empty string")
            conditions.append("instr(tags_json, ?) > 0")
            params.append(json.dumps(tag, ensure_ascii=False))
        params.append(limit + 1)
        # All entries in `conditions` are fixed string literals chosen from the
        # branches above; only `params` carries caller-supplied values, bound
        # positionally below. The dynamic join is not an injection vector.
        sql = (
            "SELECT sequence, obs_id, source_key, cmd_signature, returncode, "
            "created_step, extra_json, tags_json, created_ns, content_sha256, "
            "char_count, line_count, byte_count, record_sha256 FROM observations WHERE "
            + " AND ".join(conditions)
            + " ORDER BY sequence LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()  # nosemgrep
        has_more = len(rows) > limit
        items = tuple(self._metadata_from_row(row) for row in rows[:limit])
        return CatalogPage(
            items=items,
            next_cursor=items[-1].sequence if has_more and items else None,
            has_more=has_more,
        )

    def recent_catalog_page(
        self,
        *,
        before_sequence: int | None = None,
        limit: int = 50,
        query: str = "",
    ) -> CatalogPage:
        """Return a newest-first body-free page with a stable sequence cursor."""
        if before_sequence is not None and (
            not isinstance(before_sequence, int)
            or isinstance(before_sequence, bool)
            or before_sequence < 0
        ):
            raise ValueError("before_sequence must be None or a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer between 1 and 500")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        conditions: list[str] = []
        params: list[object] = []
        if before_sequence is not None:
            conditions.append("sequence < ?")
            params.append(before_sequence)
        if query:
            conditions.append(
                "(instr(arc_casefold(cmd_signature), arc_casefold(?)) > 0 "
                "OR instr(arc_casefold(obs_id), arc_casefold(?)) > 0 "
                "OR instr(arc_casefold(tags_json), arc_casefold(?)) > 0)"
            )
            params.extend([query, query, query])
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit + 1)
        with self._lock:
            rows = self._conn.execute(
                "SELECT sequence, obs_id, source_key, cmd_signature, returncode, "
                "created_step, extra_json, tags_json, created_ns, content_sha256, "
                f"char_count, line_count, byte_count, record_sha256 FROM observations{where} "
                "ORDER BY sequence DESC LIMIT ?",
                params,
            ).fetchall()
        has_more = len(rows) > limit
        items = tuple(self._metadata_from_row(row) for row in rows[:limit])
        return CatalogPage(
            items=items,
            next_cursor=items[-1].sequence if has_more and items else None,
            has_more=has_more,
        )

    def known_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT obs_id FROM observations ORDER BY sequence"
            ).fetchall()
        return [str(row["obs_id"]) for row in rows]

    def recent_ids(self, limit: int = 100) -> list[str]:
        """Return at most ``limit`` newest IDs without selecting bodies."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if limit == 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT obs_id FROM observations ORDER BY sequence DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [str(row["obs_id"]) for row in reversed(rows)]

    def catalog_count(self, *, query: str = "") -> int:
        """Count body-free catalog rows, optionally by command/id/tag query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if query:
            sql = (
                "SELECT COUNT(*) AS value FROM observations WHERE "
                "instr(arc_casefold(cmd_signature), arc_casefold(?)) > 0 "
                "OR instr(arc_casefold(obs_id), arc_casefold(?)) > 0 "
                "OR instr(arc_casefold(tags_json), arc_casefold(?)) > 0"
            )
            params: tuple[object, ...] = (query, query, query)
        else:
            sql = "SELECT COUNT(*) AS value FROM observations"
            params = ()
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["value"])

    def find_by_signature(self, cmd_signature: str) -> list[ObservationRecord]:
        """Return verified records matching one exact command signature."""
        if not isinstance(cmd_signature, str):
            raise TypeError("cmd_signature must be a string")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM observations WHERE cmd_signature = ? ORDER BY sequence",
                (cmd_signature,),
            ).fetchall()
        return [self._verified_record_from_row(row) for row in rows]

    def contains_value(self, value: str) -> bool:
        """Check a literal body substring, verifying a matched record on read."""
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        if not value:
            return False
        with self._lock:
            row = self._find_containing_rows(value, limit=1)[0:1]
        if not row:
            return False
        self._verified_record_from_row(row[0])
        return True

    def contains_value_with_prefix(self, prefix: str, value: str) -> bool:
        if not isinstance(prefix, str) or not isinstance(value, str):
            raise TypeError("prefix and value must be strings")
        return bool(prefix and value and self.contains_value(prefix + value))

    def contains_exact_line(self, value: str) -> bool:
        """Check a complete logical line, verifying every candidate record."""
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        if not value or "\n" in value or "\r" in value:
            return False
        with self._lock:
            rows = self._find_containing_rows(value, limit=max(1, self.count))
        for row in rows:
            record = self._verified_record_from_row(row)
            if value in record.full_content.splitlines():
                return True
        return False

    def find_records_containing(self, value: str, limit: int = 5) -> list[ObservationRecord]:
        """Return newest verified records containing a literal substring."""
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        if not value:
            return []
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            rows = self._find_containing_rows(value, limit=limit)
        return [self._verified_record_from_row(row) for row in rows]

    def _find_containing_rows(self, value: str, *, limit: int) -> list[sqlite3.Row]:
        if self.body_search_index_enabled and len(value) >= 3:
            phrase = '"' + value.replace('"', '""') + '"'
            try:
                return self._conn.execute(
                    "SELECT observations.* FROM observation_body_fts "
                    "JOIN observations ON observations.sequence = observation_body_fts.rowid "
                    "WHERE observation_body_fts MATCH ? "
                    "AND instr(observations.full_content, ?) > 0 "
                    "ORDER BY observations.sequence DESC LIMIT ?",
                    (phrase, value, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                # Control characters or a vendor tokenizer difference must not
                # turn an exact progress check into a false negative.
                pass
        return self._conn.execute(
            "SELECT * FROM observations WHERE instr(full_content, ?) > 0 "
            "ORDER BY sequence DESC LIMIT ?",
            (value, limit),
        ).fetchall()

    def record_recall(self, obs_id: str, step: int) -> None:
        _validate_optional_int("step", step)
        canonical = self._canonical_id(obs_id)
        if self.read_only:
            return
        with self._lock:
            prior_step, prior_count = self._pending_recalls.get(canonical, (0, 0))
            self._pending_recalls[canonical] = (max(step, prior_step), prior_count + 1)

    def flush_recall_metrics(self) -> None:
        """Persist buffered recall telemetry without weakening body durability."""
        if self.read_only:
            return
        with self._lock:
            if not self._pending_recalls:
                return
            pending = dict(self._pending_recalls)
            with self._transaction():
                for obs_id, (last_step, count) in pending.items():
                    self._conn.execute(
                        "UPDATE observations SET last_recalled = MAX(last_recalled, ?), "
                        "n_recalls = n_recalls + ? WHERE obs_id = ?",
                        (last_step, count, obs_id),
                    )
            for obs_id, value in pending.items():
                if self._pending_recalls.get(obs_id) == value:
                    self._pending_recalls.pop(obs_id, None)

    def verify_integrity(self, obs_id: str) -> bool:
        return self._verify_record(self.get(obs_id))

    def stats(self) -> ArchiveStats:
        self.flush_recall_metrics()
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS observations, COALESCE(SUM(byte_count), 0) AS bytes, "
                "SUM(CASE WHEN n_recalls > 0 THEN 1 ELSE 0 END) AS recalled, "
                "COALESCE(SUM(n_recalls), 0) AS recalls FROM observations"
            ).fetchone()
        return ArchiveStats(
            observations=int(row["observations"]),
            body_bytes=int(row["bytes"]),
            recalled_observations=int(row["recalled"] or 0),
            recall_count=int(row["recalls"]),
        )

    @property
    def n_records(self) -> int:
        return self.stats().observations

    @property
    def count(self) -> int:
        return self.n_records

    @property
    def max_created_step(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(created_step), 0) AS value FROM observations"
            ).fetchone()
        return int(row["value"])

    def iter_serializable(
        self,
        *,
        include_bodies: bool = True,
        batch_size: int = 500,
    ) -> Iterator[dict[str, Any]]:
        """Yield a bounded-memory, high-watermark view in sequence order."""
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= 10_000
        ):
            raise ValueError("batch_size must be an integer between 1 and 10000")
        self.flush_recall_metrics()
        select = (
            "SELECT * FROM observations WHERE sequence > ? AND sequence <= ? "
            "ORDER BY sequence LIMIT ?"
            if include_bodies
            else (
                "SELECT sequence, obs_id, source_key, cmd_signature, returncode, "
                "created_step, extra_json, tags_json, created_ns, content_sha256, "
                "char_count, line_count, byte_count, record_sha256, last_recalled, "
                "n_recalls FROM observations WHERE sequence > ? AND sequence <= ? "
                "ORDER BY sequence LIMIT ?"
            )
        )
        cursor = -1
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(sequence), -1) AS value FROM observations"
            ).fetchone()
            high_watermark = int(row["value"])
        while True:
            with self._lock:
                rows = self._conn.execute(
                    select,
                    (cursor, high_watermark, batch_size),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                if include_bodies:
                    self._verified_record_from_row(row)
                else:
                    self._verify_metadata_row(row)
                item = dict(row)
                item["tags"] = json.loads(item.pop("tags_json"))
                if include_bodies:
                    item["extra"] = json.loads(item.pop("extra_json"))
                else:
                    item.pop("extra_json", None)
                    item["body_redacted"] = True
                    item["extra_redacted"] = True
                yield item
            cursor = int(rows[-1]["sequence"])

    def to_serializable(self, *, include_bodies: bool = True) -> list[dict[str, Any]]:
        """Return a JSON-friendly snapshot, loading all selected rows in memory."""
        return list(self.iter_serializable(include_bodies=include_bodies))

    def verify_all(self, *, batch_size: int = 500) -> int:
        """Verify every body in bounded batches and return the checked count."""
        checked = 0
        for _row in self.iter_serializable(include_bodies=True, batch_size=batch_size):
            checked += 1
        return checked

    def checkpoint(self, *, truncate: bool = False) -> None:
        """Checkpoint committed WAL frames before publishing an external reader.

        A successful full checkpoint makes the current archive snapshot
        available to lock-free immutable readers. ``truncate=True`` also
        releases WAL storage. A busy result fails closed because publishing a
        citation before its occurrence reaches the main file would make
        sandboxed recall race-dependent.
        """
        self._require_open()
        self._require_writable()
        if self.path == ":memory:":
            return
        mode = "TRUNCATE" if truncate else "FULL"
        with self._lock:
            self.flush_recall_metrics()
            row = self._conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
            if row is None or int(row[0]) != 0:
                raise ArchiveConfigurationMismatch(
                    "archive checkpoint remained busy; external recall was not published"
                )
            self._secure_file()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                if not self.read_only:
                    self.flush_recall_metrics()
                self._conn.close()
                self._block_index.clear_cache()
                self._pending_recalls.clear()
                self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("formal archive is closed")

    def _require_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("formal archive is read-only")

    def _current_data_version(self) -> int:
        row = self._conn.execute("PRAGMA data_version").fetchone()
        return int(row[0])

    def _refresh_verification_cache(self) -> None:
        version = self._current_data_version()
        if version != self._data_version:
            self._block_index.clear_cache()
            self._data_version = version

    def _verify_record(self, record: ObservationRecord) -> bool:
        digest = hashlib.sha256(record.full_content.encode("utf-8")).hexdigest()
        try:
            sequence_matches = self.id_factory.sequence_from_id(record.obs_id) == record.sequence
        except ValueError:
            sequence_matches = False
        return (
            digest == record.content_sha256
            and sequence_matches
            and len(record.full_content) == record.char_count
            and len(record.full_content.encode("utf-8")) == record.byte_count
            and (
                0
                if not record.full_content
                else record.full_content.count("\n")
                + (0 if record.full_content.endswith("\n") else 1)
            )
            == record.line_count
            and self.id_factory.verify(record.obs_id, record.cmd_signature, record.full_content)
        )

    def _verified_record_from_row(self, row: sqlite3.Row) -> ObservationRecord:
        self._verify_metadata_row(row)
        record = self._record_from_row(row)
        if self.verify_on_read and not self._verify_record(record):
            raise ArchiveIntegrityError(f"integrity verification failed for §{record.obs_id}")
        pending = self._pending_recalls.get(record.obs_id)
        if pending is not None:
            last_step, count = pending
            record = replace(
                record,
                last_recalled=max(record.last_recalled, last_step),
                n_recalls=record.n_recalls + count,
            )
        return record

    def _record_from_row(self, row: sqlite3.Row) -> ObservationRecord:
        content = str(row["full_content"])
        preview = self.preview_chars
        extra = _decode_extra(row["extra_json"])
        tags = _decode_tags(row["tags_json"])
        return ObservationRecord(
            sequence=int(row["sequence"]),
            obs_id=str(row["obs_id"]),
            source_key=None if row["source_key"] is None else str(row["source_key"]),
            cmd_signature=str(row["cmd_signature"]),
            full_content=content,
            returncode=int(row["returncode"]),
            created_step=int(row["created_step"]),
            extra=extra,
            created_ns=int(row["created_ns"]),
            content_sha256=str(row["content_sha256"]),
            char_count=int(row["char_count"]),
            line_count=int(row["line_count"]),
            byte_count=int(row["byte_count"]),
            tags=tags,
            head=content[:preview],
            tail=content[-preview:] if preview and len(content) > preview else "",
            last_recalled=int(row["last_recalled"]),
            n_recalls=int(row["n_recalls"]),
        )

    def _verify_metadata_row(self, row: sqlite3.Row) -> None:
        if not self.verify_on_read:
            return
        try:
            stored = str(row["record_sha256"])
        except (IndexError, KeyError) as exc:
            raise ArchiveIntegrityError("archive record integrity metadata is missing") from exc
        expected = _record_sha256_from_row(row)
        if not hmac.compare_digest(stored, expected):
            obs_id = str(row["obs_id"])
            raise ArchiveIntegrityError(f"metadata integrity verification failed for §{obs_id}")

    def _metadata_from_row(self, row: sqlite3.Row) -> ObservationMetadata:
        self._verify_metadata_row(row)
        return ObservationMetadata(
            sequence=int(row["sequence"]),
            obs_id=str(row["obs_id"]),
            source_key=None if row["source_key"] is None else str(row["source_key"]),
            cmd_signature=str(row["cmd_signature"]),
            returncode=int(row["returncode"]),
            created_step=int(row["created_step"]),
            created_ns=int(row["created_ns"]),
            content_sha256=str(row["content_sha256"]),
            char_count=int(row["char_count"]),
            line_count=int(row["line_count"]),
            byte_count=int(row["byte_count"]),
            tags=_decode_tags(row["tags_json"]),
        )

    def __len__(self) -> int:
        return self.n_records

    def __contains__(self, obs_id: str) -> bool:
        return self.has(obs_id)

    def __enter__(self) -> "SQLiteObservationArchive":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


def _create_private_directories(directory: Path) -> None:
    """Create missing ancestors privately without chmodding existing paths."""
    missing: list[Path] = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent

    for candidate in reversed(missing):
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            if not candidate.is_dir():
                raise
        else:
            try:
                candidate.chmod(0o700)
            except OSError:
                pass


def _validate_occurrence(
    cmd_signature: object,
    full_content: object,
    source_key: object,
    returncode: object,
    created_step: object,
    created_ns: object,
) -> None:
    if (
        not isinstance(cmd_signature, str)
        or not cmd_signature
        or len(cmd_signature) > _MAX_COMMAND_CHARS
    ):
        raise ValueError(
            f"cmd_signature must be a non-empty string up to {_MAX_COMMAND_CHARS} characters"
        )
    if not isinstance(full_content, str):
        raise TypeError("full_content must be a string")
    _validate_unicode("cmd_signature", cmd_signature)
    _validate_unicode("full_content", full_content)
    if source_key is not None and (
        not isinstance(source_key, str)
        or not source_key
        or len(source_key) > _MAX_SOURCE_KEY_CHARS
    ):
        raise ValueError(
            f"source_key must be None or a non-empty string up to "
            f"{_MAX_SOURCE_KEY_CHARS} characters"
        )
    if source_key is not None:
        _validate_unicode("source_key", source_key)
    _validate_optional_int("returncode", returncode, non_negative=False)
    _validate_optional_int("created_step", created_step)
    if created_ns is not None:
        _validate_optional_int("created_ns", created_ns)


def _validate_optional_int(name: str, value: object, *, non_negative: bool = True) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f"{name} must fit a signed 64-bit SQLite INTEGER")


def _validate_unicode(name: str, value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc


def _json_object(name: str, value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        encoded.encode("utf-8")
        return encoded
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{name} must be JSON-serializable") from exc


def _normalize_tags(tags: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(tags, (tuple, list)) or any(
        not isinstance(tag, str) or not tag or len(tag) > _MAX_TAG_CHARS for tag in tags
    ):
        raise ValueError(f"tags must be non-empty strings up to {_MAX_TAG_CHARS} characters")
    normalized = tuple(dict.fromkeys(tags))
    for tag in normalized:
        _validate_unicode("tag", tag)
    if len(normalized) > _MAX_TAGS:
        raise ValueError(f"at most {_MAX_TAGS} unique tags are allowed")
    return normalized


def _same_source_occurrence(
    record: ObservationRecord,
    cmd_signature: str,
    full_content: str,
    returncode: int,
    extra_json: str,
    tags: tuple[str, ...],
) -> bool:
    return (
        record.cmd_signature == cmd_signature
        and record.full_content == full_content
        and record.returncode == returncode
        and _json_object("extra", record.extra) == extra_json
        and record.tags == tags
    )


def _record_sha256(
    *,
    sequence: int,
    obs_id: str,
    source_key: Optional[str],
    cmd_signature: str,
    returncode: int,
    created_step: int,
    extra_json: str,
    tags_json: str,
    created_ns: int,
    content_sha256: str,
    char_count: int,
    line_count: int,
    byte_count: int,
) -> str:
    payload = json.dumps(
        [
            sequence,
            obs_id,
            source_key,
            cmd_signature,
            returncode,
            created_step,
            extra_json,
            tags_json,
            created_ns,
            content_sha256,
            char_count,
            line_count,
            byte_count,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record_sha256_from_row(row: sqlite3.Row) -> str:
    return _record_sha256(
        sequence=int(row["sequence"]),
        obs_id=str(row["obs_id"]),
        source_key=None if row["source_key"] is None else str(row["source_key"]),
        cmd_signature=str(row["cmd_signature"]),
        returncode=int(row["returncode"]),
        created_step=int(row["created_step"]),
        extra_json=str(row["extra_json"]),
        tags_json=str(row["tags_json"]),
        created_ns=int(row["created_ns"]),
        content_sha256=str(row["content_sha256"]),
        char_count=int(row["char_count"]),
        line_count=int(row["line_count"]),
        byte_count=int(row["byte_count"]),
    )


def _verify_v1_record(row: sqlite3.Row, id_factory: OccurrenceIdFactory) -> None:
    """Validate legacy immutable fields before creating a v2 metadata digest."""
    content = str(row["full_content"])
    content_bytes = content.encode("utf-8")
    expected_lines = (
        0 if not content else content.count("\n") + (0 if content.endswith("\n") else 1)
    )
    obs_id = str(row["obs_id"])
    try:
        sequence_matches = id_factory.sequence_from_id(obs_id) == int(row["sequence"])
    except ValueError:
        sequence_matches = False
    if not (
        hmac.compare_digest(
            str(row["content_sha256"]),
            hashlib.sha256(content_bytes).hexdigest(),
        )
        and sequence_matches
        and id_factory.verify(obs_id, str(row["cmd_signature"]), content)
        and int(row["char_count"]) == len(content)
        and int(row["byte_count"]) == len(content_bytes)
        and int(row["line_count"]) == expected_lines
    ):
        raise ArchiveIntegrityError(f"v1 integrity verification failed for §{obs_id}")
    _decode_extra(row["extra_json"])
    _decode_tags(row["tags_json"])


def _decode_extra(value: object) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value), parse_constant=_reject_json_constant)
    except (TypeError, ValueError) as exc:
        raise ArchiveIntegrityError("archive contains invalid extra metadata JSON") from exc
    if not isinstance(decoded, dict):
        raise ArchiveIntegrityError("archive extra metadata must be a JSON object")
    return decoded


def _decode_tags(value: object) -> tuple[str, ...]:
    try:
        decoded = json.loads(str(value), parse_constant=_reject_json_constant)
    except (TypeError, ValueError) as exc:
        raise ArchiveIntegrityError("archive contains invalid tag metadata JSON") from exc
    if not isinstance(decoded, list):
        raise ArchiveIntegrityError("archive tags must be a JSON string array")
    try:
        normalized = _normalize_tags(decoded)
    except ValueError as exc:
        raise ArchiveIntegrityError("archive tags violate metadata bounds") from exc
    if normalized != tuple(decoded):
        raise ArchiveIntegrityError("archive tags must be unique and canonical")
    return normalized


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _sqlite_casefold(value: object) -> str:
    return str(value).casefold()
