"""Framework-neutral application service for formal ARC.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .archive import SQLiteObservationArchive
from .chunks import TokenCodec, UnitCounter, exact_text_chunks, exact_token_id_chunks
from .errors import InvalidAction, RecallRequiresChunking
from .models import (
    ActionResponse,
    CatalogPage,
    ObservationMetadata,
    ObservationRecord,
    TextChunk,
    TokenIdChunk,
)
from .protocol import (
    CatalogAction,
    RecallAction,
    RecallChunkAction,
    RecallFullAction,
    RecallMetadataAction,
    RecallRangeAction,
    format_catalog,
    format_metadata,
    parse_action,
)


class ArcCoreService:
    """Compose occurrence storage, discovery, and budgeted exact recovery.

    ``max_recall_units`` is a hard response-body limit for model-facing full,
    chunk, and range recalls. ``measure`` should be replaced by the host's
    authoritative tokenizer counter; character count is the dependency-free
    conservative default for ordinary text integrations.
    """

    def __init__(
        self,
        archive: SQLiteObservationArchive,
        *,
        max_recall_units: int = 4096,
        measure: UnitCounter = len,
    ) -> None:
        if not isinstance(archive, SQLiteObservationArchive):
            raise TypeError("archive must be a SQLiteObservationArchive")
        _validate_unit_limit(max_recall_units)
        if not callable(measure):
            raise TypeError("measure must be callable")
        self.archive = archive
        self.max_recall_units = max_recall_units
        self.measure = measure
        self._chunk_spans: dict[tuple[str, str, int], tuple[tuple[int, int, int], ...]] = {}

    def record(
        self,
        command_signature: str,
        full_content: str,
        *,
        source_key: Optional[str] = None,
        returncode: int = 0,
        created_step: int = 0,
        extra: Optional[Mapping[str, Any]] = None,
        tags: tuple[str, ...] | list[str] = (),
        created_ns: Optional[int] = None,
    ) -> ObservationRecord:
        """Atomically preserve one host occurrence before compaction."""
        return self.archive.append(
            command_signature,
            full_content,
            source_key=source_key,
            returncode=returncode,
            created_step=created_step,
            extra=extra,
            tags=tags,
            created_ns=created_ns,
        )

    def metadata(self, obs_id: str) -> ObservationMetadata:
        return self.archive.get_metadata(obs_id)

    def catalog(
        self,
        *,
        after_sequence: Optional[int] = None,
        limit: int = 50,
        query: str = "",
        created_step: Optional[int] = None,
        returncode: Optional[int] = None,
        tag: Optional[str] = None,
    ) -> CatalogPage:
        """Return a stable page from the complete body-free catalog."""
        return self.archive.catalog_page(
            after_sequence=after_sequence,
            limit=limit,
            query=query,
            created_step=created_step,
            returncode=returncode,
            tag=tag,
        )

    def recall_full(
        self,
        obs_id: str,
        *,
        max_units: Optional[int] = None,
        recall_step: Optional[int] = None,
    ) -> str:
        """Return the complete body or require exact chunk/range recovery."""
        limit = self._effective_limit(max_units)
        record = self.archive.get(obs_id)
        measured = _measure(self.measure, record.full_content)
        if measured > limit:
            raise RecallRequiresChunking(
                f"§{record.obs_id} measures {measured} units; full recall limit is "
                f"{limit}. Use exact chunk or range recall."
            )
        self._record_recall(record.obs_id, recall_step)
        return record.full_content

    def recall_range(
        self,
        obs_id: str,
        start: int,
        end: int,
        *,
        max_units: Optional[int] = None,
        recall_step: Optional[int] = None,
    ) -> str:
        """Return one exact half-open character range within the hard limit."""
        body = self.archive.read_range(obs_id, start, end)
        self._require_fits(body, self._effective_limit(max_units), "range recall")
        self._record_recall(obs_id, recall_step)
        return body

    def text_chunks(
        self,
        obs_id: str,
        *,
        max_units: Optional[int] = None,
    ) -> tuple[TextChunk, ...]:
        """Create a deterministic lossless text chunk set for one body."""
        record = self.archive.get(obs_id)
        return exact_text_chunks(
            record.obs_id,
            record.full_content,
            max_units=self._effective_limit(max_units),
            measure=self.measure,
        )

    def recall_text_chunk(
        self,
        obs_id: str,
        index: int,
        *,
        max_units: Optional[int] = None,
        recall_step: Optional[int] = None,
    ) -> TextChunk:
        """Return one exact chunk and reject an invalid or stale index."""
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("chunk index must be a non-negative integer")
        limit = self._effective_limit(max_units)
        metadata = self.archive.get_metadata(obs_id)

        if self.measure is len:
            total = max(1, (metadata.char_count + limit - 1) // limit)
            if index >= total:
                raise InvalidAction(f"chunk index {index} is outside [0, {total - 1}]")
            start = index * limit
            end = min(metadata.char_count, start + limit)
            text = self.archive.read_range(metadata.obs_id, start, end)
            chunk = TextChunk(
                obs_id=metadata.obs_id,
                index=index,
                total_chunks=total,
                char_start=start,
                char_end=end,
                text=text,
                measured_units=len(text),
                content_sha256=metadata.content_sha256,
            )
        else:
            cache_key = (metadata.obs_id, metadata.content_sha256, limit)
            spans = self._chunk_spans.get(cache_key)
            if spans is None:
                chunks = self.text_chunks(metadata.obs_id, max_units=limit)
                spans = tuple(
                    (chunk.char_start, chunk.char_end, chunk.measured_units) for chunk in chunks
                )
                if len(self._chunk_spans) >= 128:
                    self._chunk_spans.pop(next(iter(self._chunk_spans)))
                self._chunk_spans[cache_key] = spans
                if index >= len(chunks):
                    raise InvalidAction(f"chunk index {index} is outside [0, {len(chunks) - 1}]")
                chunk = chunks[index]
            else:
                if index >= len(spans):
                    raise InvalidAction(f"chunk index {index} is outside [0, {len(spans) - 1}]")
                start, end, units = spans[index]
                chunk = TextChunk(
                    obs_id=metadata.obs_id,
                    index=index,
                    total_chunks=len(spans),
                    char_start=start,
                    char_end=end,
                    text=self.archive.read_range(metadata.obs_id, start, end),
                    measured_units=units,
                    content_sha256=metadata.content_sha256,
                )
        self._record_recall(metadata.obs_id, recall_step)
        return chunk

    def token_id_chunks(
        self,
        obs_id: str,
        *,
        max_tokens: int,
        codec: TokenCodec,
    ) -> tuple[TokenIdChunk, ...]:
        """Create exact chunks over the codec's original token sequence."""
        record = self.archive.get(obs_id)
        return exact_token_id_chunks(
            record.obs_id,
            record.full_content,
            max_tokens=max_tokens,
            codec=codec,
        )

    def execute_action(
        self,
        action_text: str,
        *,
        recall_step: Optional[int] = None,
    ) -> ActionResponse:
        """Execute the strict grammar while keeping exact bytes out of markup."""
        return self.execute(parse_action(action_text), recall_step=recall_step)

    def execute(
        self,
        action: RecallAction,
        *,
        recall_step: Optional[int] = None,
    ) -> ActionResponse:
        """Execute one already parsed action without repeating grammar work."""
        if isinstance(action, RecallFullAction):
            body = self.recall_full(action.obs_id, recall_step=recall_step)
            metadata = self.metadata(action.obs_id)
            return ActionResponse(
                kind="recall_full",
                body=body,
                obs_id=metadata.obs_id,
                exact=True,
                metadata=_metadata_dict(metadata),
            )
        if isinstance(action, RecallMetadataAction):
            metadata = self.metadata(action.obs_id)
            body = format_metadata(metadata)
            self._require_fits(body, self.max_recall_units, "metadata recall")
            return ActionResponse(
                kind="recall_metadata",
                body=body,
                obs_id=metadata.obs_id,
                metadata=_metadata_dict(metadata),
            )
        if isinstance(action, RecallChunkAction):
            chunk = self.recall_text_chunk(
                action.obs_id,
                action.index,
                recall_step=recall_step,
            )
            return ActionResponse(
                kind="recall_chunk",
                body=chunk.text,
                obs_id=chunk.obs_id,
                exact=True,
                metadata={
                    "index": chunk.index,
                    "total_chunks": chunk.total_chunks,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "measured_units": chunk.measured_units,
                    "content_sha256": chunk.content_sha256,
                },
            )
        if isinstance(action, RecallRangeAction):
            body = self.recall_range(
                action.obs_id,
                action.start,
                action.end,
                recall_step=recall_step,
            )
            metadata = self.metadata(action.obs_id)
            return ActionResponse(
                kind="recall_range",
                body=body,
                obs_id=metadata.obs_id,
                exact=True,
                metadata={
                    "char_start": action.start,
                    "char_end": action.end,
                    "content_sha256": metadata.content_sha256,
                },
            )
        if isinstance(action, CatalogAction):
            page = self.catalog(
                after_sequence=action.after_sequence,
                limit=action.limit,
                query=action.query,
            )
            page, body = self._fit_catalog(page)
            return ActionResponse(
                kind="catalog",
                body=body,
                metadata={
                    "has_more": page.has_more,
                    "next_cursor": page.next_cursor,
                    "count": len(page.items),
                },
            )
        raise AssertionError("unreachable formal ARC action")

    def _effective_limit(self, requested: Optional[int]) -> int:
        if requested is None:
            return self.max_recall_units
        _validate_unit_limit(requested)
        if requested > self.max_recall_units:
            raise ValueError("per-call max_units cannot exceed the service hard recall limit")
        return requested

    def _require_fits(self, body: str, limit: int, operation: str) -> None:
        measured = _measure(self.measure, body)
        if measured > limit:
            raise RecallRequiresChunking(
                f"{operation} measures {measured} units and exceeds limit {limit}"
            )

    def _record_recall(self, obs_id: str, recall_step: Optional[int]) -> None:
        self.archive.record_recall(obs_id, 0 if recall_step is None else recall_step)

    def _fit_catalog(self, page: CatalogPage) -> tuple[CatalogPage, str]:
        body = format_catalog(page)
        if _measure(self.measure, body) <= self.max_recall_units:
            return page, body
        visible = []
        for item in page.items:
            candidate_items = tuple((*visible, item))
            candidate = CatalogPage(
                items=candidate_items,
                next_cursor=item.sequence,
                has_more=True,
            )
            candidate_body = format_catalog(candidate)
            if _measure(self.measure, candidate_body) > self.max_recall_units:
                break
            visible.append(item)
            body = candidate_body
        if not visible:
            raise RecallRequiresChunking(
                "one bounded catalog row exceeds the configured response envelope"
            )
        bounded = CatalogPage(
            items=tuple(visible),
            next_cursor=visible[-1].sequence,
            has_more=True,
        )
        return bounded, format_catalog(bounded)


def _validate_unit_limit(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max recall units must be a positive integer")


def _measure(counter: UnitCounter, value: str) -> int:
    measured = counter(value)
    if not isinstance(measured, int) or isinstance(measured, bool) or measured < 0:
        raise TypeError("measure must return a non-negative integer")
    return measured


def _metadata_dict(metadata: ObservationMetadata) -> dict[str, Any]:
    return {
        "sequence": metadata.sequence,
        "returncode": metadata.returncode,
        "created_step": metadata.created_step,
        "created_ns": metadata.created_ns,
        "content_sha256": metadata.content_sha256,
        "char_count": metadata.char_count,
        "line_count": metadata.line_count,
        "byte_count": metadata.byte_count,
        "tags": list(metadata.tags),
    }
