"""Optional OpenHands SDK adapter for ARC compaction and recall.

This module is intentionally not imported by :mod:`onecomp.compaction.adapters`:
OneComp's core has no OpenHands dependency.  Import it explicitly in an
OpenHands environment (``pip install openhands-sdk``), then pass the condenser
to ``Agent`` and add ``OpenHandsArcRecallTool`` and ``OpenHandsArcListTool``
using the same ``runtime_id``.

The condenser emits a native OpenHands :class:`Condensation`; consequently the
forgotten events are genuinely removed from the model view.  The paired recall
tool shares the condenser's in-process lossless store, avoiding the common but
incorrect pattern where an MCP tool returns a shorter copy while the host keeps
its original private history unchanged.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Sequence
from typing import ClassVar, Literal, Optional
from uuid import uuid4

try:
    from openhands.sdk.context.condenser import RollingCondenser
    from openhands.sdk.context.condenser.base import (
        CondensationRequirement,
        NoCondensationAvailableException,
    )
    from openhands.sdk.context.condenser.utils import (
        get_suffix_length_for_token_reduction,
        get_total_token_count,
    )
    from openhands.sdk.context.view import View
    from openhands.sdk.event import Condensation
    from openhands.sdk.llm import LLM
    from openhands.sdk.tool import (
        Action,
        Observation,
        ToolAnnotations,
        ToolDefinition,
        ToolExecutor,
        register_tool,
    )
    from pydantic import Field, PrivateAttr, model_validator
except ImportError as exc:  # pragma: no cover - exercised only without the optional SDK
    raise ImportError(
        "onecomp.compaction.adapters.openhands requires the optional 'openhands-sdk' package"
    ) from exc

from ..compactor import SUMMARY_HEADER
from ..config import CompactionConfig
from ..formal import FormalArcError
from ..formal.protocol import CITATION_RECOVERY_TEXT
from ..runner import CompactionRunner
from .messages import message_role, message_text

_OPENHANDS_RECOVERY_TEXT = (
    "Call arc_recall with full only when the item fits; otherwise use "
    "chunk or range. Call arc_list_observations to rediscover every occurrence."
)

_RUNTIMES: dict[str, "OpenHandsArcCondenser"] = {}
_RUNTIMES_LOCK = threading.RLock()
_TEXT_RECALL_HINT_RE = re.compile(
    r"\[to recall full output: issue `_recall §([0-9a-fA-F]{4,128})` "
    r"as your ONLY action — no output follows\]"
)


class OpenHandsArcCondenser(RollingCondenser):
    """Deterministic ARC condenser for OpenHands SDK 1.36+.

    Unlike an LLM summarizing condenser, this stores tool observations verbatim
    and emits bounded citations without making a second model call.  ``max_tokens``
    uses OpenHands' own provider-aware counter when an ``agent_llm`` is supplied.
    """

    runtime_id: str = Field(default_factory=lambda: uuid4().hex)
    max_size: int = Field(default=240, gt=0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    keep_first: int = Field(default=2, ge=0)
    minimum_progress: float = Field(default=0.1, gt=0.0, lt=1.0)
    summary_max_chars: int = Field(default=24_000, ge=0)
    result_max_chars: int = Field(default=500, ge=0)
    recall_per_step: int = Field(default=2, ge=0)
    recall_max_chars: int = Field(default=0, ge=0)
    wal_path: Optional[str] = None

    _engine: CompactionRunner = PrivateAttr()
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @model_validator(mode="after")
    def _validate_window(self):
        if self.keep_first >= self.max_size // 2:
            raise ValueError("keep_first must be less than max_size // 2")
        return self

    def model_post_init(self, __context: object) -> None:
        del __context
        config = CompactionConfig(
            profile="formal-v1",
            archive_path=self.wal_path,
            keep_recent=0,
            result_max_chars=self.result_max_chars,
            summary_max_chars=self.summary_max_chars,
            recalls_per_step=self.recall_per_step,
            max_recall_units=self.recall_max_chars or 4096,
        )
        self._engine = CompactionRunner(config)
        with _RUNTIMES_LOCK:
            if self.runtime_id in _RUNTIMES:
                self._engine.close()
                raise ValueError(f"OpenHands ARC runtime_id is already active: {self.runtime_id}")
            _RUNTIMES[self.runtime_id] = self

    @property
    def engine(self) -> CompactionRunner:
        """The shared engine, useful for telemetry and integration tests."""
        return self._engine

    def close(self) -> None:
        with _RUNTIMES_LOCK:
            if _RUNTIMES.get(self.runtime_id) is self:
                _RUNTIMES.pop(self.runtime_id, None)
        with self._lock:
            self._engine.close()

    def handles_condensation_requests(self) -> bool:
        return True

    def condense(self, view: View, agent_llm: Optional[LLM] = None):
        # OpenHands calls its condenser once before each model response.  This
        # gives multiple parallel arc_recall calls from that response one shared
        # recall_per_step budget.
        with self._lock:
            self._engine.begin_step()
            return super().condense(view, agent_llm=agent_llm)

    async def acondense(self, view: View, agent_llm: Optional[LLM] = None):
        return await asyncio.to_thread(self.condense, view, agent_llm)

    def condensation_requirement(
        self, view: View, agent_llm: Optional[LLM] = None
    ) -> Optional[CondensationRequirement]:
        if view.unhandled_condensation_request:
            return CondensationRequirement.HARD
        if self.max_tokens is not None and agent_llm is not None:
            if get_total_token_count(view.events, agent_llm) > self.max_tokens:
                return CondensationRequirement.HARD
        if len(view) > self.max_size:
            return CondensationRequirement.SOFT
        return None

    def get_condensation(self, view: View, agent_llm: Optional[LLM] = None) -> Condensation:
        forgotten, offset = self._get_forgotten_events(view, agent_llm)
        if not forgotten:
            raise NoCondensationAvailableException("ARC found no manipulation-safe events")
        if len(forgotten) < len(view) * self.minimum_progress:
            raise NoCondensationAvailableException(
                "ARC condensation would make less than minimum_progress"
            )
        return self._build_condensation(view, forgotten, offset, agent_llm)

    def hard_context_reset(
        self, view: View, agent_llm: Optional[LLM] = None
    ) -> Optional[Condensation]:
        if not view.events:
            return None
        return self._build_condensation(view, view.events, 0, agent_llm)

    def recall(
        self,
        obs_id: str,
        recall_kind: Literal["full", "meta", "chunk", "range"] = "full",
        *,
        chunk_index: Optional[int] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
    ) -> str:
        """Serve one recall for :class:`OpenHandsArcRecallTool`."""
        if recall_kind == "full":
            action = f"_recall §{obs_id}"
        elif recall_kind == "meta":
            action = f"_recall_meta §{obs_id}"
        elif recall_kind == "chunk":
            if chunk_index is None:
                raise ValueError("chunk recall requires chunk_index")
            action = f"_recall_chunk §{obs_id} {chunk_index}"
        elif recall_kind == "range":
            if range_start is None or range_end is None:
                raise ValueError("range recall requires range_start and range_end")
            action = f"_recall_range §{obs_id} {range_start}:{range_end}"
        else:
            raise ValueError("unsupported recall_kind")
        scratch: list[dict] = []
        with self._lock:
            outcome = self._engine.handle_action(action, scratch)
        if outcome is None:  # pragma: no cover - constructed command is always a recall
            raise ValueError("ARC recall was not handled")
        return outcome.body

    def list_observations(self, query: str = "", cursor: int = 0, limit: int = 20) -> str:
        """Return a bounded command/id index for handles absent from summary."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        with self._lock:
            assert self._engine.archive is not None
            store = self._engine.archive
            page = store.recent_catalog_page(
                before_sequence=(None if cursor == 0 else cursor),
                limit=limit,
                query=query,
            )
            observations = [
                {
                    "obs_id": record.obs_id,
                    "cmd_signature": record.cmd_signature[:200],
                    "returncode": record.returncode,
                    "line_count": record.line_count,
                    "char_count": record.char_count,
                }
                for record in page.items
            ]
            return json.dumps(
                {
                    "observations": observations,
                    "next_cursor": page.next_cursor,
                    "total": store.catalog_count(query=query),
                },
                ensure_ascii=False,
            )

    def _get_forgotten_events(self, view: View, agent_llm: Optional[LLM]) -> tuple[Sequence, int]:
        suffix_lengths: list[int] = []
        if view.unhandled_condensation_request:
            suffix_lengths.append(len(view) // 2 - self.keep_first - 1)
        if len(view) > self.max_size:
            suffix_lengths.append(self.max_size // 2 - self.keep_first - 1)
        if self.max_tokens is not None and agent_llm is not None:
            total = get_total_token_count(view.events, agent_llm)
            if total > self.max_tokens:
                suffix_lengths.append(
                    get_suffix_length_for_token_reduction(
                        events=view.events[self.keep_first :],
                        llm=agent_llm,
                        token_reduction=total - self.max_tokens // 2,
                        base_events=view.events[: self.keep_first],
                    )
                )
        if not suffix_lengths:
            raise ValueError("no condensation reason")
        events_from_tail = max(0, min(suffix_lengths))
        naive_end = len(view) - events_from_tail
        start = view.manipulation_indices.find_next(self.keep_first)
        end = view.manipulation_indices.find_next(naive_end)
        return view[start:end], start

    def _build_condensation(
        self,
        view: View,
        forgotten_events: Sequence,
        summary_offset: int,
        agent_llm: Optional[LLM] = None,
    ) -> Condensation:
        event_ids = [str(getattr(event, "id", "")) for event in forgotten_events]
        if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
            raise NoCondensationAvailableException(
                "ARC requires unique non-empty IDs for every forgotten event"
            )
        by_id = {getattr(event, "id", ""): event for event in view.events}
        messages: list[dict] = [
            {"role": "system", "content": "OpenHands ARC condenser"},
            {"role": "user", "content": "Preserve the forgotten trajectory"},
        ]
        converted_event_ids: set[str] = set()
        for event in forgotten_events:
            message = self._event_to_arc_message(event, by_id)
            if message is not None:
                messages.append(message)
                converted_event_ids.add(str(getattr(event, "id", "")))

        forgotten_event_ids = set(event_ids)
        if converted_event_ids != forgotten_event_ids:
            missing = len(forgotten_event_ids - converted_event_ids)
            raise NoCondensationAvailableException(
                f"ARC could not safely convert {missing} forgotten event(s)"
            )

        compacted = self._engine.compact(messages)
        summary = next(
            (
                str(message.get("content", ""))
                for message in compacted
                if isinstance(message.get("content"), str)
                and message["content"].startswith(SUMMARY_HEADER)
            ),
            "",
        )
        if not summary:
            raise NoCondensationAvailableException("ARC did not produce a valid summary")
        summary = _TEXT_RECALL_HINT_RE.sub(
            lambda match: (
                "[to recall full output: call the `arc_recall` tool with "
                f'obs_id="{match.group(1)}" and recall_kind="full"]'
            ),
            summary,
        )
        summary = summary.replace(CITATION_RECOVERY_TEXT, _OPENHANDS_RECOVERY_TEXT)
        digest = hashlib.sha1("\x1f".join(event_ids).encode()).hexdigest()[:16]
        condensation = Condensation(
            forgotten_event_ids=forgotten_event_ids,
            summary=summary,
            summary_offset=summary_offset,
            llm_response_id=f"arc-{digest}",
        )
        if self.max_tokens is not None and agent_llm is not None:
            final_events = condensation.apply(list(view.events))
            if get_total_token_count(final_events, agent_llm) > self.max_tokens:
                raise NoCondensationAvailableException(
                    "ARC condensation still exceeds max_tokens after final SDK counting"
                )
        return condensation

    def _event_to_arc_message(self, event, by_id: dict[str, object]) -> Optional[dict]:
        try:
            llm_message = event.to_llm_message()
        except (AttributeError, NotImplementedError):
            return None
        payload = llm_message.model_dump(mode="json", exclude_none=True)
        role = message_role(payload)
        # Archival ingestion must not use the bounded summary projection.
        # Store the complete SDK-visible observation before replacing it.
        text = message_text(payload, max_chars=None)
        if not role:
            return None

        if role == "assistant":
            command = _event_command(event)
            if command:
                payload["content"] = f"{text}\n\n```bash\n{command}\n```"
            return payload

        if role == "tool":
            assert self._engine.archive is not None
            parent = by_id.get(str(getattr(event, "action_id", "")))
            parent_action = getattr(parent, "action", parent)
            observation = getattr(event, "observation", None)
            is_error = bool(getattr(observation, "is_error", False))
            if (
                not is_error
                and isinstance(parent_action, OpenHandsArcRecallAction)
                and self._engine.archive.has(parent_action.obs_id)
            ):
                # A recalled body is already archived under its original ID.
                # Reusing that handle prevents recursive duplicate observations
                # each time a native recall result is later condensed.
                payload["_obs_id"] = parent_action.obs_id.lower()
                return payload
            command = _event_command(parent) or _bounded_command(
                str(getattr(event, "tool_name", "tool"))
            )
            returncode = 1 if is_error else 0
            record = self._engine.record_observation(
                command=command,
                content=text,
                source_key=(
                    _openhands_source_key(str(getattr(event, "id", "")))
                    if getattr(event, "id", None)
                    else None
                ),
                returncode=returncode,
                extra={
                    "openhands_event_id": str(getattr(event, "id", ""))[:1024],
                    "tool_name": str(getattr(event, "tool_name", ""))[:1024],
                },
            )
            payload["_obs_id"] = record.obs_id
        return payload


def _event_command(event: object) -> str:
    if event is None:
        return ""
    tool_call = getattr(event, "tool_call", None)
    if tool_call is not None:
        name = str(getattr(tool_call, "name", getattr(event, "tool_name", "tool")))
        arguments = getattr(tool_call, "arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return _bounded_command(f"{name}({arguments})")
    return _bounded_command(str(getattr(event, "tool_name", "")))


def _bounded_command(value: str) -> str:
    if len(value) <= 16_384:
        _utf8_bytes("OpenHands command", value)
        return value
    digest = hashlib.sha256(_utf8_bytes("OpenHands command", value)).hexdigest()
    suffix = f"… [sha256={digest}]"
    return value[: 16_384 - len(suffix)] + suffix


def _openhands_source_key(event_id: str) -> str:
    digest = hashlib.sha256(_utf8_bytes("OpenHands event ID", event_id)).hexdigest()
    return f"openhands:{digest}"


def _utf8_bytes(name: str, value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc


def _get_runtime(runtime_id: str) -> Optional["OpenHandsArcCondenser"]:
    with _RUNTIMES_LOCK:
        return _RUNTIMES.get(runtime_id)


class OpenHandsArcRecallAction(Action):
    obs_id: str = Field(description="ARC observation id, without the section sign")
    recall_kind: Literal["full", "meta", "chunk", "range"] = "full"
    chunk_index: Optional[int] = Field(default=None, ge=0)
    range_start: Optional[int] = Field(default=None, ge=0)
    range_end: Optional[int] = Field(default=None, ge=0)


class OpenHandsArcRecallObservation(Observation):
    pass


class _OpenHandsArcRecallExecutor(
    ToolExecutor[OpenHandsArcRecallAction, OpenHandsArcRecallObservation]
):
    def __init__(self, runtime_id: str):
        self.runtime_id = runtime_id

    def __call__(self, action: OpenHandsArcRecallAction, conversation=None):  # noqa: ARG002
        del conversation
        runtime = _get_runtime(self.runtime_id)
        if runtime is None:
            return OpenHandsArcRecallObservation.from_text(
                "ARC runtime is unavailable; resume the condenser with its WAL first.",
                is_error=True,
            )
        try:
            content = runtime.recall(
                action.obs_id,
                action.recall_kind,
                chunk_index=action.chunk_index,
                range_start=action.range_start,
                range_end=action.range_end,
            )
        except (FormalArcError, ValueError) as exc:
            return OpenHandsArcRecallObservation.from_text(str(exc), is_error=True)
        return OpenHandsArcRecallObservation.from_text(content)


class OpenHandsArcRecallTool(
    ToolDefinition[OpenHandsArcRecallAction, OpenHandsArcRecallObservation]
):
    """OpenHands tool sharing the matching condenser's ARC store."""

    name: ClassVar[str] = "arc_recall"

    @classmethod
    def create(cls, conv_state=None, **params):  # noqa: ARG003
        del conv_state
        runtime_id = params.pop("runtime_id", None)
        if params:
            raise ValueError(f"Unsupported OpenHandsArcRecallTool params: {sorted(params)}")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("OpenHandsArcRecallTool requires runtime_id")
        return [
            cls(
                description=(
                    "Recall a tool observation preserved by the ARC condenser. Use an "
                    "obs_id shown in a § citation; choose meta for a cheap preview or "
                    "full when the exact body is needed."
                ),
                action_type=OpenHandsArcRecallAction,
                observation_type=OpenHandsArcRecallObservation,
                executor=_OpenHandsArcRecallExecutor(runtime_id),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


class OpenHandsArcListAction(Action):
    query: str = Field(
        default="",
        description="Case-insensitive command/path substring; empty lists newest records.",
    )
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class OpenHandsArcListObservation(Observation):
    pass


class _OpenHandsArcListExecutor(ToolExecutor[OpenHandsArcListAction, OpenHandsArcListObservation]):
    def __init__(self, runtime_id: str):
        self.runtime_id = runtime_id

    def __call__(self, action: OpenHandsArcListAction, conversation=None):  # noqa: ARG002
        del conversation
        runtime = _get_runtime(self.runtime_id)
        if runtime is None:
            return OpenHandsArcListObservation.from_text(
                "ARC runtime is unavailable; resume the condenser with its WAL first.",
                is_error=True,
            )
        try:
            content = runtime.list_observations(action.query, action.cursor, action.limit)
        except ValueError as exc:
            return OpenHandsArcListObservation.from_text(str(exc), is_error=True)
        return OpenHandsArcListObservation.from_text(content)


class OpenHandsArcListTool(ToolDefinition[OpenHandsArcListAction, OpenHandsArcListObservation]):
    """Bounded command/id index sharing the matching condenser's store."""

    name: ClassVar[str] = "arc_list_observations"

    @classmethod
    def create(cls, conv_state=None, **params):  # noqa: ARG003
        del conv_state
        runtime_id = params.pop("runtime_id", None)
        if params:
            raise ValueError(f"Unsupported OpenHandsArcListTool params: {sorted(params)}")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("OpenHandsArcListTool requires runtime_id")
        return [
            cls(
                description=(
                    "Search the bounded ARC observation index by original command/path. "
                    "Use when a strict summary cap leaves bare or omitted handles, then "
                    "call arc_recall with the returned obs_id. When multiple matches are "
                    "returned, prefer the first result — observations are ordered newest first."
                ),
                action_type=OpenHandsArcListAction,
                observation_type=OpenHandsArcListObservation,
                executor=_OpenHandsArcListExecutor(runtime_id),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool("OpenHandsArcRecallTool", OpenHandsArcRecallTool)
register_tool("OpenHandsArcListTool", OpenHandsArcListTool)


__all__ = [
    "OpenHandsArcCondenser",
    "OpenHandsArcListAction",
    "OpenHandsArcListObservation",
    "OpenHandsArcListTool",
    "OpenHandsArcRecallAction",
    "OpenHandsArcRecallObservation",
    "OpenHandsArcRecallTool",
]
