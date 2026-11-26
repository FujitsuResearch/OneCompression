"""Explicit migration from the experimental JSONL observation WAL.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Protocol

from .archive import SQLiteObservationArchive
from .errors import LegacyMigrationError
from .models import MigrationReport


class _Digest(Protocol):
    def update(self, value: bytes) -> object:
        """Consume another byte block."""


def migrate_experimental_wal(
    source: str | Path,
    archive: SQLiteObservationArchive,
    *,
    strict: bool = True,
    dry_run: bool = False,
    migration_id: Optional[str] = None,
) -> MigrationReport:
    """Import every valid JSONL row and preserve old citations as aliases.

    Legacy stores content-addressed observations, so invocation multiplicity
    that was collapsed before WAL creation cannot be reconstructed. Distinct
    WAL rows are nevertheless retained as distinct formal occurrences. A
    deterministic source key makes interrupted imports safely resumable.
    ``migration_id`` supplies a stable source identity when a WAL may move or
    grow; otherwise a digest of its canonical path is used.
    """
    if not isinstance(archive, SQLiteObservationArchive):
        raise TypeError("archive must be a SQLiteObservationArchive")
    if not isinstance(strict, bool) or not isinstance(dry_run, bool):
        raise TypeError("strict and dry_run must be booleans")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"experimental WAL not found: {source_path}")
    if migration_id is not None and (
        not isinstance(migration_id, str)
        or not migration_id
        or len(migration_id) > 128
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in migration_id
        )
    ):
        raise ValueError("migration_id must use 1-128 ASCII letters, digits, '.', '_', or '-'")

    source_digest = _file_sha256(source_path)
    source_identity = (
        migration_id or hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:32]
    )
    errors: list[str] = []
    skipped = 0
    if strict or dry_run:
        valid_rows = 0
        aliases: set[str] = set()
        for line_number, row, error in _iter_legacy_rows(source_path):
            if error is not None:
                if strict:
                    raise LegacyMigrationError(error)
                errors.append(error)
                skipped += 1
                continue
            assert row is not None
            valid_rows += 1
            if row.get("obs_id"):
                aliases.add(str(row["obs_id"]).lower())
        if dry_run:
            return MigrationReport(
                source=str(source_path),
                destination=archive.path,
                loaded=valid_rows,
                skipped=skipped,
                aliases_registered=len(aliases),
                errors=tuple(errors),
            )

    if dry_run:  # pragma: no cover - handled by the scan above
        return MigrationReport(
            source=str(source_path),
            destination=archive.path,
            loaded=0,
            skipped=skipped,
            aliases_registered=0,
            errors=tuple(errors),
        )

    loaded = 0
    aliases_registered = 0
    seen_aliases: set[str] = set()
    imported_digest = hashlib.sha256()
    # One outer transaction keeps strict imports atomic and avoids one durable
    # SQLite commit per line. The source is streamed, so memory stays bounded by
    # one legacy row plus one archived occurrence.
    with archive.batch():
        for line_number, row, parse_error in _iter_legacy_rows(
            source_path,
            digest=imported_digest,
        ):
            if parse_error is not None:
                if strict:
                    raise LegacyMigrationError(parse_error)
                errors.append(parse_error)
                skipped += 1
                continue
            assert row is not None
            source_key = f"legacy-jsonl:{source_identity}:{line_number}"
            try:
                registered_alias: Optional[str] = None
                # A nested savepoint makes lenient recovery truly row-local:
                # an alias conflict or late SQLite error cannot leave a body,
                # sequence advance, or derived index from the rejected row.
                with archive.batch():
                    record = archive.append(
                        row["cmd_signature"],
                        row["full_content"],
                        source_key=source_key,
                        returncode=row["returncode"],
                        created_step=row["created_step"],
                        extra={
                            **row["extra"],
                            "migration": {
                                "format": "experimental-jsonl-v1",
                                "source_identity": source_identity,
                                "line": line_number,
                                "row_sha256": _row_sha256(row),
                            },
                        },
                    )
                    legacy_id = row.get("obs_id")
                    if legacy_id:
                        alias = str(legacy_id).lower()
                        if alias not in seen_aliases:
                            archive.register_alias(alias, record.obs_id)
                            registered_alias = alias
                loaded += 1
                if registered_alias is not None:
                    seen_aliases.add(registered_alias)
                    aliases_registered += 1
            # Lenient migration is an explicit file-boundary recovery mode: one
            # malformed or conflicting row must not prevent later independent rows.
            except Exception as exc:  # pylint: disable=broad-exception-caught
                message = f"line {line_number}: {type(exc).__name__}: {exc}"
                if strict:
                    raise LegacyMigrationError(message) from exc
                errors.append(message)
                skipped += 1
        if imported_digest.hexdigest() != source_digest:
            raise LegacyMigrationError("experimental WAL changed while migration was running")

    return MigrationReport(
        source=str(source_path),
        destination=archive.path,
        loaded=loaded,
        skipped=skipped,
        aliases_registered=aliases_registered,
        errors=tuple(errors),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_legacy_rows(
    path: Path,
    *,
    digest: Optional[_Digest] = None,
) -> Iterator[tuple[int, Optional[dict[str, Any]], Optional[str]]]:
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            if digest is not None:
                digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
                row = _validate_legacy_row(value)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                yield line_number, None, f"line {line_number}: {exc}"
            else:
                yield line_number, row, None


def _validate_legacy_row(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("entry must be a JSON object")
    command = value.get("cmd_signature")
    content = value.get("full_content")
    if not isinstance(command, str) or not command:
        raise ValueError("cmd_signature must be a non-empty string")
    if not isinstance(content, str):
        raise TypeError("full_content must be a string")
    returncode = value.get("returncode", 0) or 0
    created_step = value.get("created_step", 0) or 0
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise TypeError("returncode must be an integer")
    if not isinstance(created_step, int) or isinstance(created_step, bool) or created_step < 0:
        raise ValueError("created_step must be a non-negative integer")
    extra = value.get("extra") or {}
    if not isinstance(extra, Mapping):
        raise TypeError("extra must be a JSON object")
    try:
        normalized_extra = json.loads(
            json.dumps(
                dict(extra),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("extra must be JSON-serializable") from exc
    legacy_id = value.get("obs_id")
    if legacy_id is not None and (
        not isinstance(legacy_id, str)
        or not 4 <= len(legacy_id) <= 128
        or any(character not in "0123456789abcdefABCDEF" for character in legacy_id)
    ):
        raise ValueError("obs_id must be 4-128 hexadecimal characters when present")
    return {
        "obs_id": legacy_id,
        "cmd_signature": command,
        "full_content": content,
        "returncode": returncode,
        "created_step": created_step,
        "extra": normalized_extra,
    }


def _row_sha256(row: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
