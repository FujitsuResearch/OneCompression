"""MCP tools exposing the ARC compaction engine to agent platforms.

Exposes :class:`onecomp.compaction.ArcEngine` over stdio to MCP-capable
agent hosts.  MCP supplies storage/recall tools, but automatic context
reduction still requires a host integration that owns and replaces the
conversation buffer.  Two usage styles are supported:

*Framework-driven* (a condenser/memory hook owns the message buffer):
    ``compaction_start_session`` once per agent, then
    ``compaction_record_observation`` after every action execution and
    ``compaction_compact_messages`` whenever the history must shrink.
    The returned buffer contains bounded citations instead of dropped
    observations.

*LLM-driven recall* (the platform exposes MCP tools to the LLM itself):
    the LLM calls ``compaction_recall`` directly with an ``obs_id`` it
    sees in a citation — the MCP-native equivalent of emitting the
    ``_recall §<id>`` magic action.  ``compaction_handle_action`` is the
    text-protocol variant for frameworks that intercept raw actions.

Sessions are independent :class:`~onecomp.compaction.engine.ArcEngine`
instances keyed by an opaque id, so one server process can serve many
concurrent agents of a multi-agent system.  Everything here is pure
Python (no torch, no network); tools are safe on CPU-only hosts.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..compaction import (
    ArcEngineConfig,
    CompactionConfig,
    CompactionRunner,
    FormalArcError,
    RecallError,
    format_citation,
)
from ..compaction.adapters import total_text_chars
from ..compaction.formal import format_citation as format_formal_citation
from .archive_sessions import ManagedArcEngine, ManagedCompactionRunner, reserve_archive
from .protocol import ToolDefinition
from .session_registry import SessionRegistry

#: Guard against unbounded growth if a client keeps opening sessions.
_MAX_SESSIONS = 256

#: Thread-safe registry and compatibility view used by existing diagnostics.
_REGISTRY: SessionRegistry[Any] = SessionRegistry(max_sessions=_MAX_SESSIONS)
_SESSIONS: Dict[str, Any] = _REGISTRY.resources


def _lease_engine(
    session_id: str,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
):
    return _REGISTRY.lease(
        session_id,
        session_token=session_token,
        tenant_id=tenant_id,
    )


def _store(session: Any) -> Any:
    if isinstance(session, CompactionRunner):
        assert session.archive is not None
        return session.archive
    return session.store


def _resolve_wal_path(wal_path: Optional[str]) -> Optional[str]:
    """Resolve an MCP-requested WAL beneath the configured safe root.

    MCP tools are callable by the model, so accepting an unrestricted path
    would let a prompt append JSON to arbitrary files writable by the server.
    Relative paths are anchored at ``ONECOMP_MCP_WAL_ROOT`` (default: the
    server working directory), and absolute paths must remain under it.
    """
    if wal_path is None:
        return None
    if not isinstance(wal_path, str) or not wal_path.strip():
        raise ValueError("wal_path must be a non-empty string when provided")
    root = Path(os.environ.get("ONECOMP_MCP_WAL_ROOT", os.getcwd())).expanduser().resolve()
    requested = Path(wal_path).expanduser()
    resolved = (requested if requested.is_absolute() else root / requested).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"wal_path must stay within ONECOMP_MCP_WAL_ROOT ({root})") from exc
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"wal_path exists but is not a regular file: {resolved}")
    return str(resolved)


def close_all_sessions() -> None:
    """Close every live engine and clear the process-local session registry."""
    _REGISTRY.close_all()


def compaction_start_session(
    keep_recent: int = 1,
    result_max_chars: int = 500,
    summary_max_chars: int = 24000,
    recall_per_step: int = 2,
    recall_budget_chars: int = 200000,
    recall_max_chars: int = 0,
    wal_path: Optional[str] = None,
    resume_wal: bool = False,
    tenant_id: str = "default",
    ttl_seconds: int = 3600,
    require_session_token: bool = False,
    profile: str = "experimental-v1",
    max_recall_units: int = 4096,
    max_archive_bytes: int = 0,
) -> Dict[str, Any]:
    """Create an ARC compaction session and return its id.

    Args:
        keep_recent: Recent turn pairs kept uncompacted.
        result_max_chars: Observations at most this long stay verbatim
            in the summary; longer ones become citations.
        summary_max_chars: Character cap on the compaction summary
            (0 = unbounded).
        recall_per_step: Max recalls served per step.
        recall_budget_chars: Total verbatim-recall chars kept in context
            before LRU eviction (0 = no eviction).
        recall_max_chars: Window cap on a single full recall
            (0 = lossless).
        wal_path: Optional server-side archive. ``formal-v1`` uses durable
            SQLite transactions; ``experimental-v1`` provides process-crash
            recovery without per-record power-loss durability.
        resume_wal: Restore observations already present in ``wal_path``
            before accepting new records.
        max_archive_bytes: Exact UTF-8 body quota for ``formal-v1``; ``0``
            disables the core quota and delegates storage limits to deployment.

    Returns:
        ``session_id`` plus ``system_prompt_snippet`` to append to the
        agent's system prompt.
    """
    if not isinstance(resume_wal, bool):
        raise ValueError("resume_wal must be a boolean")
    if not isinstance(tenant_id, str) or not tenant_id or len(tenant_id) > 256:
        raise ValueError("tenant_id must be a non-empty string up to 256 characters")
    if (
        not isinstance(ttl_seconds, int)
        or isinstance(ttl_seconds, bool)
        or not 1 <= ttl_seconds <= 7 * 24 * 3600
    ):
        raise ValueError("ttl_seconds must be an integer between 1 and 604800")
    if not isinstance(require_session_token, bool):
        raise TypeError("require_session_token must be a boolean")
    if profile not in ("experimental-v1", "formal-v1"):
        raise ValueError("profile must be 'experimental-v1' or 'formal-v1'")
    if profile != "formal-v1" and max_archive_bytes != 0:
        raise ValueError("max_archive_bytes is supported only by profile='formal-v1'")
    resolved_wal_path = _resolve_wal_path(wal_path)
    if resume_wal and resolved_wal_path is None:
        raise ValueError("resume_wal=true requires wal_path")
    if resume_wal and resolved_wal_path is not None and not Path(resolved_wal_path).is_file():
        raise FileNotFoundError(f"compaction archive does not exist: {resolved_wal_path}")
    if (
        profile == "formal-v1"
        and resolved_wal_path is not None
        and Path(resolved_wal_path).exists()
        and not resume_wal
    ):
        raise ValueError("formal archive already exists; set resume_wal=true or choose a new path")
    if profile == "formal-v1":
        session_config = CompactionConfig(
            profile="formal-v1",
            archive_path=resolved_wal_path,
            keep_recent=keep_recent,
            result_max_chars=result_max_chars,
            summary_max_chars=summary_max_chars,
            max_recall_units=max_recall_units,
            max_archive_bytes=max_archive_bytes,
            recalls_per_step=recall_per_step,
        )
    else:
        session_config = ArcEngineConfig(
            keep_recent=keep_recent,
            result_max_chars=result_max_chars,
            summary_max_chars=summary_max_chars,
            recall_per_step=recall_per_step,
            recall_budget_chars=recall_budget_chars,
            recall_max_chars=recall_max_chars,
            wal_path=resolved_wal_path,
        )
    archive_lease = reserve_archive(resolved_wal_path)
    engine: Any = None
    try:
        if profile == "formal-v1":
            assert isinstance(session_config, CompactionConfig)
            engine = ManagedCompactionRunner(session_config)
            engine.attach_archive_lease(archive_lease)
            if resolved_wal_path is not None:
                assert engine.archive is not None
                engine.archive.bind_owner(tenant_id)
        else:
            assert isinstance(session_config, ArcEngineConfig)
            engine = (
                ManagedArcEngine.from_wal(resolved_wal_path, session_config)
                if resume_wal and resolved_wal_path is not None
                else ManagedArcEngine(session_config)
            )
            engine.attach_archive_lease(archive_lease)
        handle = _REGISTRY.create(
            engine,
            tenant_id=tenant_id,
            ttl_seconds=ttl_seconds,
            require_token=require_session_token,
        )
    except Exception:
        if engine is not None:
            engine.close()
        elif archive_lease is not None:
            archive_lease.release()
        raise
    return {
        "session_id": handle.session_id,
        "session_token": handle.session_token,
        "tenant_id": handle.tenant_id,
        "expires_in_seconds": handle.expires_in_seconds,
        "auth_required": handle.auth_required,
        "profile": profile,
        "system_prompt_snippet": engine.system_prompt_snippet(),
        "restored_observations": _store(engine).n_records if resume_wal else 0,
    }


def compaction_close_session(
    session_id: str,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Close a session, flush its WAL, and free its store."""
    _REGISTRY.close(
        session_id,
        session_token=session_token,
        tenant_id=tenant_id,
    )
    return {"closed": session_id}


def compaction_begin_step(
    session_id: str,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Mark a step boundary: advance the step counter and reset the
    per-step recall budget.

    Call once per LLM response when the integration wants precise
    ``recall_per_step`` semantics.  ``compaction_record_observation``
    also marks a boundary (an execution implies a new step), so
    integrations that record every execution can skip this tool.
    """
    with _lease_engine(session_id, session_token, tenant_id) as engine:
        return {"step": engine.begin_step()}


def compaction_record_observation(
    session_id: str,
    command: str,
    content: str,
    returncode: int = 0,
    advance_step: bool = True,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
    source_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Store an executed action's output verbatim in the session store.

    Call after every environment execution.  The returned ``obs_id`` is
    the stable handle citations and recalls use.  Recording also marks a
    step boundary (resetting the per-step recall budget), since an
    execution means the previous LLM response is complete.  Set
    ``advance_step=false`` for sibling results from one parallel tool-call
    response, or when ``compaction_begin_step`` was already called explicitly.

    Args:
        session_id: Session from ``compaction_start_session``.
        command: The action that produced the observation.
        content: Raw observation body (stored losslessly).
        returncode: Exit status, if applicable.
        advance_step: Advance/reset the ARC response-step counter before
            recording.  Defaults to true for backward compatibility.
    """
    if not isinstance(command, str) or not command or len(command) > 16_384:
        raise ValueError("command must be a non-empty string up to 16384 characters")
    if not isinstance(content, str) or len(content) > 67_108_864:
        raise ValueError("content must be a string up to 67108864 characters")
    if source_key is not None and (
        not isinstance(source_key, str) or not source_key or len(source_key) > 1024
    ):
        raise ValueError("source_key must be null or a non-empty string up to 1024 characters")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise ValueError("returncode must be an integer")
    if not isinstance(advance_step, bool):
        raise ValueError("advance_step must be a boolean")
    with _lease_engine(session_id, session_token, tenant_id) as engine:
        if advance_step:
            engine.begin_step()
        if isinstance(engine, CompactionRunner):
            rec = engine.record_observation(
                command=command,
                content=content,
                source_key=source_key,
                returncode=returncode,
            )
        else:
            rec = engine.record_observation(
                command=command, content=content, returncode=returncode
            )
        out = {
            "obs_id": rec.obs_id,
            "char_count": rec.char_count,
            "line_count": rec.line_count,
            "will_be_cited": rec.char_count > engine.config.result_max_chars,
        }
        if rec.char_count > engine.config.result_max_chars:
            if isinstance(engine, CompactionRunner):
                out["replacement_content"] = format_formal_citation(
                    rec,
                    head_chars=engine.config.citation_head_chars,
                    tail_chars=engine.config.citation_tail_chars,
                )
            else:
                out["replacement_content"] = format_citation(
                    obs_id=rec.obs_id,
                    cmd_signature=rec.cmd_signature,
                    returncode=rec.returncode,
                    line_count=rec.line_count,
                    char_count=rec.char_count,
                    head=rec.head,
                    tail=rec.tail,
                    head_chars=engine.config.citation_head_chars,
                    tail_chars=engine.config.citation_tail_chars,
                )
        return out


def compaction_compact_messages(
    session_id: str,
    messages: List[Dict[str, Any]],
    max_chars: int = 0,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Compact a message buffer with the session's citation-rich operator.

    Observation messages must carry the ``_obs_id`` returned by
    ``compaction_record_observation`` for their bodies to be preserved
    as recall-able citations; untagged long observations fall back to
    the legacy drop behaviour.

    Args:
        session_id: Session from ``compaction_start_session``.
        messages: Provider message/item list.  Leading system/developer
            instructions and the first user task are kept intact.
        max_chars: When > 0, shrink until the total content length fits
            this budget (proactive mode).  When 0, apply one reactive
            compaction pass.

    Returns:
        ``messages`` (the compacted buffer) and ``total_chars``.
    """
    # The JSON schema is not enforced by the server; validate the shape
    # here (system boundary) so a malformed client payload gets a clear
    # error instead of an AttributeError from deep inside the compactor.
    if (
        not isinstance(messages, list)
        or len(messages) > 100_000
        or not all(isinstance(m, dict) for m in messages)
    ):
        raise ValueError("messages must be a list of objects ({role, content, ...})")
    for index, message in enumerate(messages):
        role = message.get("role")
        item_type = message.get("type") or message.get("kind")
        if role is not None and not isinstance(role, str):
            raise ValueError(f"messages[{index}].role must be a string when provided")
        if role is None and not isinstance(item_type, str):
            raise ValueError(
                f"messages[{index}] must have a string role, type, or kind discriminator"
            )
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list, dict)):
            raise ValueError(
                f"messages[{index}].content must be text, content blocks, an object, or null"
            )
        if "_obs_id" in message and not isinstance(message["_obs_id"], str):
            raise ValueError(f"messages[{index}]._obs_id must be a string")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    with _lease_engine(session_id, session_token, tenant_id) as engine:
        if max_chars > 0:
            out = engine.enforce_char_budget(messages, max_chars)
        else:
            out = engine.compact(messages)
        return {
            "messages": out,
            "total_chars": total_text_chars(out),
            "n_messages": len(out),
        }


def compaction_handle_action(
    session_id: str,
    action: str,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Intercept a raw LLM action; serve it if it is a recall.

    Text-protocol variant for frameworks that pass the LLM's action
    string through before executing it.

    The ``recall_per_step`` budget is charged across consecutive recalls
    and resets at the next step boundary (``compaction_begin_step`` or
    ``compaction_record_observation``) — this tool deliberately does NOT
    reset it, otherwise the budget could never trigger over MCP.

    Returns:
        ``handled=False`` when the action is not a recall (execute it
        normally).  ``handled=True`` with ``content`` (inject as a
        user-role observation) on success, or with ``error`` (feed back
        to the LLM) when the recall could not be served.
    """
    if not isinstance(action, str) or len(action) > 512:
        raise ValueError("action must be a string up to 512 characters")
    with _lease_engine(session_id, session_token, tenant_id) as engine:
        scratch: List[Dict[str, Any]] = []
        try:
            outcome = engine.handle_action(action, scratch)
        except (RecallError, FormalArcError, ValueError) as exc:
            return {"handled": True, "error": str(exc)}
        if outcome is None:
            return {"handled": False}
        if not isinstance(engine, CompactionRunner):
            return {
                "handled": True,
                "kind": outcome.kind,
                "obs_id": outcome.obs_id,
                "content": outcome.message["content"],
                "evicted_obs_ids": list(outcome.evicted_obs_ids),
            }
        return {
            "handled": True,
            "kind": outcome.kind,
            "obs_id": outcome.obs_id,
            "content": outcome.body,
            "exact": outcome.exact,
            "metadata": dict(outcome.metadata),
            "evicted_obs_ids": [],
        }


def compaction_recall(
    session_id: str,
    obs_id: str,
    recall_kind: str = "full",
    chunk_index: Optional[int] = None,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Recall a stored observation by id (MCP-native recall).

    The tool-call equivalent of the ``_recall §<id>`` /
    ``_recall_meta §<id>`` text protocol, for platforms whose LLM calls
    MCP tools directly.  Inject the returned ``content`` into the
    conversation as a user-role observation.  Shares the per-step recall
    budget with ``compaction_handle_action`` (see there for the step
    boundary semantics).

    Args:
        session_id: Session from ``compaction_start_session``.
        obs_id: The ``§<obs_id>`` handle shown in a citation (without
            the section sign).
        recall_kind: ``"full"`` for the verbatim body, ``"meta"`` for
            bounded metadata, ``"chunk"`` for one deterministic chunk, or
            ``"range"`` for an exact half-open character range.  (Named
            ``recall_kind``, not
            ``kind``: several agent-framework MCP clients — e.g. the
            OpenHands SDK, which reserves ``kind`` as the discriminator
            field on every dynamically generated tool-call model —
            crash if a tool parameter is literally named ``kind``.)
    """
    if not isinstance(obs_id, str) or not 4 <= len(obs_id) <= 128:
        raise ValueError("obs_id must be a string between 4 and 128 characters")
    if recall_kind not in ("full", "meta", "chunk", "range"):
        raise ValueError(
            'recall_kind must be "full", "meta", "chunk", or "range", ' f"got {recall_kind!r}"
        )
    if recall_kind == "chunk":
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer for chunk recall")
        action = f"_recall_chunk \u00a7{obs_id} {chunk_index}"
    elif recall_kind == "range":
        if (
            not isinstance(range_start, int)
            or isinstance(range_start, bool)
            or not isinstance(range_end, int)
            or isinstance(range_end, bool)
            or range_start < 0
            or range_end < range_start
        ):
            raise ValueError("range recall requires 0 <= range_start <= range_end")
        action = f"_recall_range \u00a7{obs_id} {range_start}:{range_end}"
    else:
        verb = "_recall" if recall_kind == "full" else "_recall_meta"
        action = f"{verb} \u00a7{obs_id}"
    result = compaction_handle_action(
        session_id,
        action,
        session_token,
        tenant_id,
    )
    if "error" in result:
        return {"found": False, "error": result["error"]}
    return {
        "found": True,
        "obs_id": result["obs_id"],
        "recall_kind": result["kind"],
        "content": result["content"],
        "evicted_obs_ids": result.get("evicted_obs_ids", []),
    }


def compaction_list_observations(
    session_id: str,
    cursor: int = 0,
    limit: int = 20,
    query: Optional[str] = None,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Return a bounded index into the lossless store.

    A strict summary cap cannot keep an unbounded number of citation handles
    visible.  This operation lets MCP/host integrations rediscover handles
    without dumping the entire store back into model context.  ``query``
    case-insensitively filters command signatures, allowing the model to map a
    requested file/command to an old bare handle before recalling its body.
    The formal profile is stable occurrence-sequence order; the compatibility
    profile retains its historical newest-first offset order.
    """
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
        raise ValueError("cursor must be a non-negative integer")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    if query is not None and (not isinstance(query, str) or len(query) > 200):
        raise ValueError("query must be a string up to 200 characters or null")
    with _lease_engine(session_id, session_token, tenant_id) as engine:
        store = _store(engine)
        if isinstance(engine, CompactionRunner):
            assert engine.service is not None
            formal_page = engine.service.catalog(
                after_sequence=(None if cursor == 0 else cursor - 1),
                limit=limit,
                query=query or "",
            )
            return {
                "observations": [
                    {
                        "obs_id": record.obs_id,
                        "cmd_signature": record.cmd_signature[:200],
                        "returncode": record.returncode,
                        "line_count": record.line_count,
                        "char_count": record.char_count,
                        "created_step": record.created_step,
                    }
                    for record in formal_page.items
                ],
                "next_cursor": (
                    formal_page.items[-1].sequence + 1
                    if formal_page.has_more and formal_page.items
                    else None
                ),
                "total": store.catalog_count(query=query or ""),
            }
        ids = store.known_ids()
        if query:
            needle = query.casefold()
            ids = [
                obs_id
                for obs_id in ids
                if (record := store.get(obs_id)) is not None
                and needle in record.cmd_signature.casefold()
            ]
        selected_ids = list(reversed(ids))[cursor : cursor + limit]
        page = [store.get(obs_id) for obs_id in selected_ids]
        observations = [
            {
                "obs_id": record.obs_id,
                "cmd_signature": record.cmd_signature[:200],
                "returncode": record.returncode,
                "line_count": record.line_count,
                "char_count": record.char_count,
                "created_step": record.created_step,
            }
            for record in page
            if record is not None
        ]
        next_cursor = cursor + len(page)
        return {
            "observations": observations,
            "next_cursor": next_cursor if next_cursor < len(ids) else None,
            "total": len(ids),
        }


def compaction_verify_value(
    session_id: str,
    value: str,
    prefix: Optional[str] = None,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
    match_mode: str = "substring",
    obs_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Anti-confabulation guard: did ``value`` appear in any observation?

    Before committing an agent's final answer, check that the value
    actually occurred verbatim in the session's lossless store.  When it
    did not, the agent is likely confabulating (e.g. copying a
    similar-looking value from the wrong observation) and should
    ``_recall`` the relevant citation instead.

    Args:
        session_id: Session from ``compaction_start_session``.
        value: The exact value the agent is about to submit.
        prefix: Optional literal prefix that must immediately precede
            the value in the observation (e.g. ``"trace-key: "``),
            eliminating false-passes on coincidental substrings.
        match_mode: ``"substring"`` for backward compatibility or
            ``"exact_line"`` to reject partial-line coincidences.
        obs_id: Optionally restrict verification to one recalled occurrence.

    Returns:
        ``seen`` plus, when False, the most recent ``candidate_obs_ids``
        the agent could recall to find the correct value.
    """
    if not isinstance(value, str) or not value or len(value) > 1_048_576:
        raise ValueError("value must be a non-empty string up to 1048576 characters")
    if prefix is not None and (not isinstance(prefix, str) or len(prefix) > 1_048_576):
        raise ValueError("prefix must be a string up to 1048576 characters or null")
    if match_mode not in {"substring", "exact_line"}:
        raise ValueError("match_mode must be 'substring' or 'exact_line'")
    if obs_id is not None and (not isinstance(obs_id, str) or not 4 <= len(obs_id) <= 128):
        raise ValueError("obs_id must be a string between 4 and 128 characters or null")
    phrase = (prefix or "") + value
    if match_mode == "exact_line" and ("\n" in phrase or "\r" in phrase):
        raise ValueError("exact_line verification requires a single-line prefix and value")
    with _lease_engine(session_id, session_token, tenant_id) as engine:
        store = _store(engine)
        if obs_id is not None:
            record = store.get(obs_id)
            if record is None:
                raise ValueError(f"unknown observation id: §{obs_id}")
            if match_mode == "exact_line":
                seen = phrase in record.full_content.splitlines()
            else:
                seen = phrase in record.full_content
        elif match_mode == "exact_line":
            seen = store.contains_exact_line(phrase)
        elif prefix:
            seen = store.contains_value_with_prefix(prefix, value)
        else:
            seen = store.contains_value(value)
        out: Dict[str, Any] = {
            "seen": seen,
            "match_mode": match_mode,
            "obs_id": obs_id,
        }
        if not seen:
            recent = (
                store.recent_ids(5)
                if isinstance(engine, CompactionRunner)
                else store.known_ids()[-5:]
            )
            out["candidate_obs_ids"] = recent[::-1]
            out["hint"] = (
                f"The value {value!r} does not appear verbatim in any of the "
                f"{store.n_records} stored observations"
                + (f" (checked for the exact phrase {prefix!r} + value)" if prefix else "")
                + (f" within §{obs_id}" if obs_id else "")
                + ". Do not guess — recall the observation that contains the "
                "answer and copy it exactly."
            )
        return out


def compaction_session_stats(
    session_id: str,
    session_token: Optional[str] = None,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Telemetry for a session: store size, compactions, recalls, evictions."""
    with _lease_engine(session_id, session_token, tenant_id) as engine:
        store = _store(engine)
        if isinstance(engine, CompactionRunner):
            events = engine.events
            n_compactions = sum(event["kind"] == "compaction" for event in events)
            n_recalls = sum(event["kind"] == "action" for event in events)
            n_evictions = 0
            in_context = []
            profile = engine.profile
        else:
            n_compactions = len(engine.compaction_log)
            n_recalls = len(engine.recall_log)
            n_evictions = len(engine.eviction_log)
            in_context = list(engine.in_context_obs.keys())
            profile = "experimental-v1"
        return {
            "profile": profile,
            "step": engine.step,
            "n_observations": store.n_records,
            "n_compactions": n_compactions,
            "n_recalls": n_recalls,
            "n_evictions": n_evictions,
            "in_context_obs": in_context,
            "known_obs_ids": (
                store.recent_ids(100)
                if isinstance(engine, CompactionRunner)
                else store.known_ids()[-100:]
            ),
            "known_obs_ids_truncated": max(0, store.n_records - 100),
        }


_MESSAGES_SCHEMA = {
    "type": "array",
    "maxItems": 100000,
    "description": (
        "Provider message/item list. Supports role/content messages, OpenAI Responses "
        "items, Anthropic content blocks, and optional _obs_id tags."
    ),
    "items": {"type": "object"},
}

_SESSION_AUTH_PROPERTIES = {
    "session_token": {
        "type": "string",
        "minLength": 1,
        "maxLength": 128,
        "description": "Opaque secret returned by compaction_start_session.",
    },
    "tenant_id": {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
        "default": "default",
        "description": "Tenant binding selected when the session was created.",
    },
}

_COMPACTION_OUTPUT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "compaction_start_session": {
        "type": "object",
        "required": ["session_id", "session_token", "profile", "auth_required"],
        "properties": {
            "session_id": {"type": "string"},
            "session_token": {"type": "string"},
            "tenant_id": {"type": "string"},
            "profile": {"type": "string"},
            "auth_required": {"type": "boolean"},
            "expires_in_seconds": {"type": "integer"},
            "system_prompt_snippet": {"type": "string"},
            "restored_observations": {"type": "integer"},
        },
    },
    "compaction_close_session": {
        "type": "object",
        "required": ["closed"],
        "properties": {"closed": {"type": "string"}},
    },
    "compaction_begin_step": {
        "type": "object",
        "required": ["step"],
        "properties": {"step": {"type": "integer"}},
    },
    "compaction_record_observation": {
        "type": "object",
        "required": ["obs_id", "char_count", "line_count", "will_be_cited"],
        "properties": {
            "obs_id": {"type": "string"},
            "char_count": {"type": "integer"},
            "line_count": {"type": "integer"},
            "will_be_cited": {"type": "boolean"},
            "replacement_content": {"type": "string"},
        },
    },
    "compaction_compact_messages": {
        "type": "object",
        "required": ["messages", "total_chars", "n_messages"],
        "properties": {
            "messages": _MESSAGES_SCHEMA,
            "total_chars": {"type": "integer"},
            "n_messages": {"type": "integer"},
        },
    },
    "compaction_handle_action": {
        "type": "object",
        "required": ["handled"],
        "properties": {
            "handled": {"type": "boolean"},
            "kind": {"type": "string"},
            "obs_id": {"type": ["string", "null"]},
            "content": {"type": "string"},
            "exact": {"type": "boolean"},
            "metadata": {"type": "object"},
            "error": {"type": "string"},
            "evicted_obs_ids": {"type": "array", "items": {"type": "string"}},
        },
    },
    "compaction_recall": {
        "type": "object",
        "required": ["found"],
        "properties": {
            "found": {"type": "boolean"},
            "obs_id": {"type": "string"},
            "recall_kind": {"type": "string"},
            "content": {"type": "string"},
            "error": {"type": "string"},
            "evicted_obs_ids": {"type": "array", "items": {"type": "string"}},
        },
    },
    "compaction_list_observations": {
        "type": "object",
        "required": ["observations", "next_cursor", "total"],
        "properties": {
            "observations": {"type": "array", "items": {"type": "object"}},
            "next_cursor": {"type": ["integer", "null"]},
            "total": {"type": "integer"},
        },
    },
    "compaction_verify_value": {
        "type": "object",
        "required": ["seen"],
        "properties": {
            "seen": {"type": "boolean"},
            "match_mode": {"type": "string"},
            "obs_id": {"type": ["string", "null"]},
            "candidate_obs_ids": {"type": "array", "items": {"type": "string"}},
            "hint": {"type": "string"},
        },
    },
    "compaction_session_stats": {
        "type": "object",
        "required": ["profile", "step", "n_observations"],
        "properties": {
            "profile": {"type": "string"},
            "step": {"type": "integer"},
            "n_observations": {"type": "integer"},
            "n_compactions": {"type": "integer"},
            "n_recalls": {"type": "integer"},
            "n_evictions": {"type": "integer"},
            "in_context_obs": {"type": "array", "items": {"type": "string"}},
            "known_obs_ids": {"type": "array", "items": {"type": "string"}},
            "known_obs_ids_truncated": {"type": "integer"},
        },
    },
}


def _session_required(require_session_token: bool, *names: str) -> list[str]:
    required = ["session_id", *names]
    if require_session_token:
        required.append("session_token")
    return required


def build_compaction_tools(
    *,
    require_session_token: bool = True,
    profile: str = "formal-v1",
) -> List[ToolDefinition]:
    """Return the ARC compaction :class:`ToolDefinition` list."""
    if not isinstance(require_session_token, bool):
        raise TypeError("require_session_token must be a boolean")
    if profile not in ("experimental-v1", "formal-v1"):
        raise ValueError("profile must be 'experimental-v1' or 'formal-v1'")

    def start_session(**arguments: Any) -> Dict[str, Any]:
        arguments.pop("require_session_token", None)
        arguments.pop("profile", None)
        return compaction_start_session(
            **arguments,
            require_session_token=require_session_token,
            profile=profile,
        )

    tools = [
        ToolDefinition(
            name="compaction_start_session",
            description=(
                f"Start an ARC {profile} context-compaction session for one agent. "
                "Returns a session_id and a system-prompt snippet that "
                "teaches the LLM the _recall protocol. ARC preserves long "
                "observations losslessly and replaces them with bounded "
                "citations on compaction instead of dropping them."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keep_recent": {"type": "integer", "default": 1},
                    "result_max_chars": {"type": "integer", "default": 500},
                    "summary_max_chars": {"type": "integer", "default": 24000},
                    "recall_per_step": {"type": "integer", "default": 2},
                    "recall_budget_chars": {"type": "integer", "default": 200000},
                    "recall_max_chars": {"type": "integer", "default": 0},
                    "wal_path": {"type": "string", "maxLength": 4096},
                    "resume_wal": {"type": "boolean", "default": False},
                    "max_recall_units": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 4096,
                    },
                    "max_archive_bytes": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": (
                            "Exact UTF-8 body quota for the formal archive; "
                            "0 leaves quota policy to the deployment."
                        ),
                    },
                    "tenant_id": _SESSION_AUTH_PROPERTIES["tenant_id"],
                    "ttl_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 604800,
                        "default": 3600,
                    },
                },
            },
            handler=start_session,
        ),
        ToolDefinition(
            name="compaction_close_session",
            description="Close an ARC compaction session and free its store.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token),
            },
            handler=compaction_close_session,
        ),
        ToolDefinition(
            name="compaction_begin_step",
            description=(
                "Mark a step boundary (one LLM response): advances the step "
                "counter and resets the per-step recall budget. Optional — "
                "compaction_record_observation also marks a boundary."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token),
            },
            handler=compaction_begin_step,
        ),
        ToolDefinition(
            name="compaction_record_observation",
            description=(
                "Store an executed action's output verbatim in the session's "
                "lossless store. Call after every environment execution; the "
                "returned obs_id is the handle used by citations and recalls."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    "command": {"type": "string", "minLength": 1, "maxLength": 16384},
                    "content": {"type": "string", "maxLength": 67108864},
                    "returncode": {"type": "integer", "default": 0},
                    "advance_step": {"type": "boolean", "default": True},
                    "source_key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1024,
                        "description": "Unique host tool-call/event identifier.",
                    },
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token, "command", "content"),
            },
            handler=compaction_record_observation,
        ),
        ToolDefinition(
            name="compaction_compact_messages",
            description=(
                "Compact a message buffer with the ARC citation-rich "
                "operator. Long observations tagged with _obs_id become "
                "bounded recall-able citations instead of being dropped. "
                "Set max_chars > 0 to shrink until the buffer fits."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    "messages": _MESSAGES_SCHEMA,
                    "max_chars": {"type": "integer", "default": 0},
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token, "messages"),
            },
            handler=compaction_compact_messages,
        ),
        ToolDefinition(
            name="compaction_handle_action",
            description=(
                "Intercept a raw LLM action string before execution. If it "
                "is a `_recall §<id>` / `_recall_meta §<id>` command, serve "
                "it from the lossless store (handled=true, inject content); "
                "otherwise handled=false and the action should be executed "
                "normally."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    "action": {"type": "string", "maxLength": 512},
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token, "action"),
            },
            handler=compaction_handle_action,
        ),
        ToolDefinition(
            name="compaction_recall",
            description=(
                "Recall a previously observed output by its §<obs_id> "
                "citation handle. recall_kind='full' re-injects the verbatim "
                "body; recall_kind='meta' returns only the bounded metadata "
                "preview. Use when a citation in the conversation summary "
                "points to content you need to read again."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    "obs_id": {"type": "string", "minLength": 4, "maxLength": 128},
                    "recall_kind": {
                        "type": "string",
                        "enum": ["full", "meta", "chunk", "range"],
                        "default": "full",
                    },
                    "chunk_index": {"type": "integer", "minimum": 0},
                    "range_start": {"type": "integer", "minimum": 0},
                    "range_end": {"type": "integer", "minimum": 0},
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token, "obs_id"),
            },
            handler=compaction_recall,
        ),
        ToolDefinition(
            name="compaction_list_observations",
            description=(
                "List a bounded page of stored observation handles and metadata "
                "without bodies. Use this to rediscover handles omitted by a "
                "strict ARC summary cap without re-expanding observation bodies."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    "cursor": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    },
                    "query": {
                        "type": "string",
                        "maxLength": 200,
                        "description": (
                            "Optional case-insensitive substring of the original "
                            "command/path used to find a bare or omitted handle."
                        ),
                    },
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token),
            },
            handler=compaction_list_observations,
        ),
        ToolDefinition(
            name="compaction_verify_value",
            description=(
                "Anti-confabulation guard: check that a value the agent is "
                "about to submit appeared verbatim in some stored "
                "observation. Returns seen=false with candidate obs_ids to "
                "recall when it did not. Optionally require a literal "
                "prefix (e.g. 'trace-key: ') to precede the value, use an "
                "exact-line match, or bind the claim to one observation ID."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    "value": {"type": "string", "maxLength": 1048576},
                    "prefix": {"type": "string", "maxLength": 1048576},
                    "match_mode": {
                        "type": "string",
                        "enum": ["substring", "exact_line"],
                        "default": "substring",
                    },
                    "obs_id": {"type": "string", "minLength": 4, "maxLength": 128},
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token, "value"),
            },
            handler=compaction_verify_value,
        ),
        ToolDefinition(
            name="compaction_session_stats",
            description=(
                "Telemetry for an ARC compaction session: observations "
                "stored, compactions, recalls, evictions, live obs_ids."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "minLength": 12, "maxLength": 12},
                    **_SESSION_AUTH_PROPERTIES,
                },
                "required": _session_required(require_session_token),
            },
            handler=compaction_session_stats,
        ),
    ]
    for tool in tools:
        tool.output_schema = _COMPACTION_OUTPUT_SCHEMAS[tool.name]
    return tools
