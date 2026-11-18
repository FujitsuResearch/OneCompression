"""Sparse byte-addressing metadata for exact bounded body reads.

SQLite stores the canonical body once in ``observations.full_content``. These
helpers derive fixed character blocks with byte offsets and hashes, allowing
the archive to use incremental BLOB reads without duplicating body content.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Optional

from .errors import ArchiveIntegrityError

BODY_BLOCK_CHARS = 16_384


@dataclass(frozen=True)
class BodyBlock:
    """One character-aligned UTF-8 block and its integrity metadata."""

    index: int
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    content_sha256: str
    payload: bytes


@dataclass(frozen=True)
class BodyLayout:
    """Whole-body digest plus sparse block layout."""

    content_sha256: str
    byte_count: int
    blocks: tuple[BodyBlock, ...]


def build_body_layout(text: str) -> BodyLayout:
    """Encode ``text`` once into bounded blocks and derive exact offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    digest = hashlib.sha256()
    blocks: list[BodyBlock] = []
    byte_start = 0
    for index, char_start in enumerate(range(0, len(text), BODY_BLOCK_CHARS)):
        char_end = min(len(text), char_start + BODY_BLOCK_CHARS)
        payload = text[char_start:char_end].encode("utf-8")
        digest.update(payload)
        byte_end = byte_start + len(payload)
        blocks.append(
            BodyBlock(
                index=index,
                char_start=char_start,
                char_end=char_end,
                byte_start=byte_start,
                byte_end=byte_end,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                payload=payload,
            )
        )
        byte_start = byte_end
    return BodyLayout(
        content_sha256=digest.hexdigest(),
        byte_count=byte_start,
        blocks=tuple(blocks),
    )


def decode_body_block(
    payload: bytes,
    *,
    char_count: int,
    byte_count: int,
    content_sha256: str,
    verify: bool,
) -> str:
    """Decode one block and fail closed on layout or checksum corruption."""
    if len(payload) != byte_count:
        raise ValueError("body block byte count does not match its index")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("body block is not valid UTF-8") from exc
    if len(text) != char_count:
        raise ValueError("body block character count does not match its index")
    if verify and hashlib.sha256(payload).hexdigest() != content_sha256:
        raise ValueError("body block checksum does not match its index")
    return text


class SQLiteBodyBlockIndex:
    """Rebuildable sparse index over canonical SQLite body values."""

    def __init__(self, connection: sqlite3.Connection, *, verify: bool) -> None:
        self._connection = connection
        self._verify = verify
        self._cache: dict[tuple[str, int, str], str] = {}

    def initialize(self) -> None:
        """Create additive index tables without changing archive identity."""
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS observation_body_block_meta (
                sequence INTEGER PRIMARY KEY REFERENCES observations(sequence),
                content_sha256 TEXT NOT NULL,
                block_chars INTEGER NOT NULL,
                block_count INTEGER NOT NULL
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS observation_body_blocks (
                sequence INTEGER NOT NULL REFERENCES observations(sequence),
                block_index INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                PRIMARY KEY(sequence, block_index)
            )
            """)

    def write(self, sequence: int, checksum: str, blocks: tuple[BodyBlock, ...]) -> None:
        """Replace one occurrence's derived metadata in the active transaction."""
        self._connection.execute(
            "INSERT OR REPLACE INTO observation_body_block_meta("
            "sequence, content_sha256, block_chars, block_count) VALUES (?, ?, ?, ?)",
            (sequence, checksum, BODY_BLOCK_CHARS, len(blocks)),
        )
        self._connection.execute(
            "DELETE FROM observation_body_blocks WHERE sequence = ?",
            (sequence,),
        )
        self._connection.executemany(
            "INSERT INTO observation_body_blocks("
            "sequence, block_index, char_start, char_end, byte_start, byte_end, "
            "content_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    sequence,
                    block.index,
                    block.char_start,
                    block.char_end,
                    block.byte_start,
                    block.byte_end,
                    block.content_sha256,
                )
                for block in blocks
            ),
        )

    def read(
        self,
        obs_id: str,
        metadata: sqlite3.Row,
        start: int,
        end: int,
    ) -> Optional[str]:
        """Read and verify overlapping blocks, or return None for a legacy row."""
        sequence = int(metadata["sequence"])
        checksum = str(metadata["content_sha256"])
        try:
            meta = self._connection.execute(
                "SELECT content_sha256, block_chars, block_count "
                "FROM observation_body_block_meta WHERE sequence = ?",
                (sequence,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if meta is None:
            return None
        expected_blocks = (int(metadata["char_count"]) + BODY_BLOCK_CHARS - 1) // BODY_BLOCK_CHARS
        if (
            str(meta["content_sha256"]) != checksum
            or int(meta["block_chars"]) != BODY_BLOCK_CHARS
            or int(meta["block_count"]) != expected_blocks
        ):
            raise ArchiveIntegrityError(f"body block index is stale for §{obs_id}")
        first_index = start // BODY_BLOCK_CHARS
        last_index = (end - 1) // BODY_BLOCK_CHARS
        rows = self._connection.execute(
            "SELECT block_index, char_start, char_end, byte_start, byte_end, "
            "content_sha256 FROM observation_body_blocks "
            "WHERE sequence = ? AND block_index BETWEEN ? AND ? "
            "ORDER BY block_index",
            (sequence, first_index, last_index),
        ).fetchall()
        self._validate_rows(obs_id, metadata, rows, first_index, last_index, start, end)
        try:
            blob = self._connection.blobopen(
                "observations",
                "full_content",
                sequence,
                readonly=True,
            )
        except (AttributeError, sqlite3.OperationalError):
            return None
        parts: list[str] = []
        try:
            if len(blob) != int(metadata["byte_count"]):
                raise ArchiveIntegrityError(f"body byte count changed for §{obs_id}")
            for row in rows:
                parts.append(self._read_block(blob, obs_id, row))
        finally:
            blob.close()
        combined = "".join(parts)
        first_start = int(rows[0]["char_start"])
        return combined[start - first_start : end - first_start]

    def clear_cache(self) -> None:
        """Discard process-local verified blocks after an external commit."""
        self._cache.clear()

    @staticmethod
    def _validate_rows(
        obs_id: str,
        metadata: sqlite3.Row,
        rows: list[sqlite3.Row],
        first_index: int,
        last_index: int,
        start: int,
        end: int,
    ) -> None:
        if (
            len(rows) != last_index - first_index + 1
            or int(rows[0]["block_index"]) != first_index
            or int(rows[-1]["block_index"]) != last_index
            or int(rows[0]["char_start"]) > start
            or int(rows[-1]["char_end"]) < end
        ):
            raise ArchiveIntegrityError(f"body block index has incomplete coverage for §{obs_id}")
        for offset, row in enumerate(rows):
            block_index = first_index + offset
            if (
                int(row["block_index"]) != block_index
                or int(row["char_start"]) != block_index * BODY_BLOCK_CHARS
                or int(row["char_end"])
                != min(int(metadata["char_count"]), (block_index + 1) * BODY_BLOCK_CHARS)
            ):
                raise ArchiveIntegrityError(f"body block index has invalid bounds for §{obs_id}")
        for prior, current in zip(rows, rows[1:]):
            if int(prior["char_end"]) != int(current["char_start"]) or int(
                prior["byte_end"]
            ) != int(current["byte_start"]):
                raise ArchiveIntegrityError(f"body block index is not contiguous for §{obs_id}")

    def _read_block(self, blob: sqlite3.Blob, obs_id: str, row: sqlite3.Row) -> str:
        block_index = int(row["block_index"])
        block_checksum = str(row["content_sha256"])
        cache_key = (obs_id, block_index, block_checksum)
        text = self._cache.get(cache_key)
        if text is not None:
            return text
        byte_start = int(row["byte_start"])
        byte_count = int(row["byte_end"]) - byte_start
        blob.seek(byte_start)
        payload = blob.read(byte_count)
        try:
            text = decode_body_block(
                payload,
                char_count=int(row["char_end"]) - int(row["char_start"]),
                byte_count=byte_count,
                content_sha256=block_checksum,
                verify=self._verify,
            )
        except ValueError as exc:
            raise ArchiveIntegrityError(f"body block verification failed for §{obs_id}") from exc
        if len(self._cache) >= 128:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = text
        return text


__all__ = [
    "BODY_BLOCK_CHARS",
    "BodyBlock",
    "BodyLayout",
    "SQLiteBodyBlockIndex",
    "build_body_layout",
    "decode_body_block",
]
