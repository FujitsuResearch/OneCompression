"""Exact, non-overlapping recall chunk primitives for formal ARC.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable, Protocol, Sequence, Tuple

from .errors import ExactRecoveryError
from .models import TextChunk, TokenIdChunk

UnitCounter = Callable[[str], int]


class TokenCodec(Protocol):
    """Minimal reversible tokenizer contract used by token-ID chunks."""

    def encode(self, text: str) -> Sequence[int]:
        """Encode text without adding model-special tokens."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode the complete token sequence."""


def exact_text_chunks(
    obs_id: str,
    text: str,
    *,
    max_units: int,
    measure: UnitCounter,
) -> Tuple[TextChunk, ...]:
    """Split text into adjacent, lossless slices bounded by ``measure``.

    Every chosen boundary is measured explicitly. The search may return a
    smaller chunk for a non-monotonic tokenizer, but it never returns an
    over-budget chunk or omits characters. Token-ID chunks are preferable
    when tokenizer-boundary exactness is required.
    """
    _validate_chunk_request(obs_id, text, max_units)
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if text == "":
        measured = _measure(measure, "")
        if measured > max_units:
            raise ExactRecoveryError("empty text exceeds the declared unit budget")
        return (
            TextChunk(
                obs_id=obs_id,
                index=0,
                total_chunks=1,
                char_start=0,
                char_end=0,
                text="",
                measured_units=measured,
                content_sha256=checksum,
            ),
        )

    if measure is len:
        total_chunks = (len(text) + max_units - 1) // max_units
        return tuple(
            TextChunk(
                obs_id=obs_id,
                index=index,
                total_chunks=total_chunks,
                char_start=index * max_units,
                char_end=min(len(text), (index + 1) * max_units),
                text=text[index * max_units : (index + 1) * max_units],
                measured_units=min(max_units, len(text) - index * max_units),
                content_sha256=checksum,
            )
            for index in range(total_chunks)
        )

    spans: list[tuple[int, int, int]] = []
    start = 0
    while start < len(text):
        one_end = start + 1
        one_units = _measure(measure, text[start:one_end])
        if one_units > max_units:
            raise ExactRecoveryError(
                "one Unicode code point exceeds max_units under the configured counter; "
                "text and range recall cannot satisfy this response envelope. Increase "
                "the hard limit or use the host-level token-ID API with a reversible codec"
            )

        best_end = one_end
        best_units = one_units
        candidate_width = 2
        first_bad = len(text) + 1
        while start + candidate_width <= len(text):
            candidate_end = start + candidate_width
            candidate_units = _measure(measure, text[start:candidate_end])
            if candidate_units <= max_units:
                best_end = candidate_end
                best_units = candidate_units
                candidate_width *= 2
            else:
                first_bad = candidate_end
                break

        if first_bad == len(text) + 1:
            candidate_end = len(text)
            candidate_units = _measure(measure, text[start:candidate_end])
            if candidate_units <= max_units:
                best_end = candidate_end
                best_units = candidate_units
                spans.append((start, best_end, best_units))
                start = best_end
                continue
            first_bad = candidate_end

        low = best_end + 1
        high = first_bad - 1
        while low <= high:
            middle = (low + high) // 2
            candidate_units = _measure(measure, text[start:middle])
            if candidate_units <= max_units:
                best_end = middle
                best_units = candidate_units
                low = middle + 1
            else:
                high = middle - 1

        spans.append((start, best_end, best_units))
        start = best_end

    total_chunks = len(spans)
    chunks = tuple(
        TextChunk(
            obs_id=obs_id,
            index=index,
            total_chunks=total_chunks,
            char_start=start,
            char_end=end,
            text=text[start:end],
            measured_units=units,
            content_sha256=checksum,
        )
        for index, (start, end, units) in enumerate(spans)
    )
    # Every payload is a direct source slice and ``start`` advances to the
    # preceding end. Rejoining the complete body here only duplicates memory
    # and work; use ``verify_text_chunks`` for externally transported sets.
    return chunks


def verify_text_chunks(chunks: Iterable[TextChunk], expected: str) -> bool:
    """Return whether a complete text chunk set reconstructs ``expected``."""
    ordered = sorted(chunks, key=lambda chunk: chunk.index)
    if not ordered:
        return expected == ""
    total = ordered[0].total_chunks
    checksum = hashlib.sha256(expected.encode("utf-8")).hexdigest()
    if (
        total != len(ordered)
        or [chunk.index for chunk in ordered] != list(range(total))
        or any(chunk.total_chunks != total for chunk in ordered)
        or any(chunk.content_sha256 != checksum for chunk in ordered)
        or ordered[0].char_start != 0
        or ordered[-1].char_end != len(expected)
    ):
        return False
    for left, right in zip(ordered, ordered[1:]):
        if left.char_end != right.char_start:
            return False
    return "".join(chunk.text for chunk in ordered) == expected


def exact_token_id_chunks(
    obs_id: str,
    text: str,
    *,
    max_tokens: int,
    codec: TokenCodec,
) -> Tuple[TokenIdChunk, ...]:
    """Partition a reversible token sequence into exact adjacent chunks."""
    _validate_chunk_request(obs_id, text, max_tokens)
    try:
        raw_token_ids = tuple(codec.encode(text))
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in raw_token_ids
        ):
            raise TypeError("token IDs must be non-negative integers")
        token_ids = raw_token_ids
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExactRecoveryError("codec returned invalid token IDs") from exc
    if codec.decode(token_ids) != text:
        raise ExactRecoveryError("codec.decode(codec.encode(text)) is not exact")

    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not token_ids:
        return (
            TokenIdChunk(
                obs_id=obs_id,
                index=0,
                total_chunks=1,
                token_start=0,
                token_end=0,
                token_ids=(),
                text=codec.decode(()),
                content_sha256=checksum,
            ),
        )

    total_chunks = (len(token_ids) + max_tokens - 1) // max_tokens
    chunks = []
    for index in range(total_chunks):
        start = index * max_tokens
        end = min(len(token_ids), start + max_tokens)
        ids = token_ids[start:end]
        chunks.append(
            TokenIdChunk(
                obs_id=obs_id,
                index=index,
                total_chunks=total_chunks,
                token_start=start,
                token_end=end,
                token_ids=ids,
                text=codec.decode(ids),
                content_sha256=checksum,
            )
        )
    if reconstruct_token_chunks(chunks, codec) != text:
        raise ExactRecoveryError("token chunks do not reconstruct exactly")
    return tuple(chunks)


def reconstruct_token_chunks(chunks: Iterable[TokenIdChunk], codec: TokenCodec) -> str:
    """Reconstruct text only from a complete, consistent token chunk set."""
    ordered = sorted(chunks, key=lambda chunk: chunk.index)
    if not ordered:
        return codec.decode(())
    total = ordered[0].total_chunks
    checksum = ordered[0].content_sha256
    if (
        len(ordered) != total
        or [chunk.index for chunk in ordered] != list(range(total))
        or any(chunk.total_chunks != total for chunk in ordered)
        or any(chunk.obs_id != ordered[0].obs_id for chunk in ordered)
        or any(chunk.content_sha256 != checksum for chunk in ordered)
        or ordered[0].token_start != 0
    ):
        raise ExactRecoveryError("token chunk set is incomplete or inconsistent")
    for left, right in zip(ordered, ordered[1:]):
        if left.token_end != right.token_start:
            raise ExactRecoveryError("token chunk spans are not adjacent")
    token_ids = tuple(token for chunk in ordered for token in chunk.token_ids)
    text = codec.decode(token_ids)
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != checksum:
        raise ExactRecoveryError("token chunk checksum does not match reconstructed text")
    return text


def _validate_chunk_request(obs_id: object, text: object, max_units: object) -> None:
    if not isinstance(obs_id, str) or not obs_id:
        raise ValueError("obs_id must be a non-empty string")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(max_units, int) or isinstance(max_units, bool) or max_units <= 0:
        raise ValueError("chunk limit must be a positive integer")


def _measure(measure: UnitCounter, text: str) -> int:
    value = measure(text)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError("measure must return a non-negative integer")
    return value
