"""Pure message-list compaction operators.

All functions here are pure: they take a message buffer (a
``list[dict]`` of OpenAI-style ``{"role": ..., "content": ...}``
messages) and return a new, shorter buffer.  No I/O, no store mutation,
no framework dependency — the caller decides *when* to compact (e.g. on
a context-window error or proactively before a request).

Three operators are provided, from lossiest to most recoverable:

:func:`compact_messages`
    Baseline: truncate assistant thoughts, keep short observations,
    drop long ones, inherit prior summaries.
:func:`compact_messages_max`
    Same, but prior summaries are *discarded* on re-compaction (the
    "AutoComp" baseline in the ARC paper; loss points L3 + L4).
:func:`compact_messages_arc`
    Citation-rich: long observations become bounded-length *citations*
    pointing into a lossless :class:`~onecomp.compaction.store.ObsStore`
    and verified entries from prior summaries are carried forward, so any
    observation cited once stays recall-able for the remainder of the
    trajectory.

Message-buffer convention
-------------------------

Leading system/developer messages and the first user task are kept intact.
This covers OpenAI Chat and Responses items, Anthropic content blocks, and
OpenHands-style serialised messages rather than assuming fixed list indices.
Observation messages produced through the ARC flow carry an ``"_obs_id"``
key (see
:meth:`onecomp.compaction.engine.ArcEngine.make_observation_message`);
messages without it fall back to the string-only path.  Use
:func:`onecomp.compaction.engine.sanitize_messages` to strip private
keys before handing the buffer to an LLM API that rejects extra fields.

Ported from ARC (Addressable Recall for Compaction), Apache-2.0.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
from typing import Any, Mapping, Optional

from .adapters.messages import (
    message_fragments,
    message_role,
    message_text,
    protected_prefix_count,
    recent_turn_start,
)
from .formal.protocol import format_citation as format_formal_citation
from .protocol import format_citation
from .store import ObsStore

THOUGHT_MAX_CHARS = 300
CMD_MAX_CHARS = 150
RESULT_MAX_CHARS = 500
USER_MAX_CHARS = 1000

#: Default regex extracting the action body from an assistant message.
DEFAULT_ACTION_REGEX = r"```bash\s*\n(.*?)\n```"
#: Default regex extracting the observation body from a user message.
DEFAULT_OBSERVATION_REGEX = r"<output>\s*(.*?)\s*</output>"

SUMMARY_HEADER = "The following is a summary of your previous actions (long observations omitted):"

# Separator placed between summary parts; counted against the budget so
# the cap reflects the real assembled size.
_SUMMARY_SEP = "\n\n---\n\n"

# Hard ceiling on the number of obs_ids listed in a single collapsed
# id-only line.  Keeps the collapsed fallback bounded even on very long
# trajectories that accumulate thousands of observations.
_MAX_COLLAPSED_IDS = 256

# Appended to the summary header when one or more entries were collapsed
# to id-only references, so the model knows a bare ``§<id>`` is
# recall-able.
_ELISION_NOTE = (
    "Older observations are elided to keep this summary within the context "
    "budget and reduced to bare `\u00a7<id>` handles. Listed handles remain "
    "recall-able; bodies remain lossless in the ARC store."
)

_ARC_SUMMARY_KEY = "_arc_summary"
_ARC_SUMMARY_VERSION = 2
_OBSERVATION_HANDLE_RE = re.compile(r"\u00a7([0-9a-fA-F]{4,128})")


class _SummaryProvenance:
    def __deepcopy__(self, _memo: dict[int, Any]) -> "_SummaryProvenance":
        return self


_SUMMARY_PROVENANCE = _SummaryProvenance()


def _compact_assistant(
    content: str,
    action_regex: str | re.Pattern[str],
    thought_max_chars: int = THOUGHT_MAX_CHARS,
) -> str:
    """Compact an assistant message: truncate thought, keep the command signature."""
    action_pattern = (
        action_regex
        if isinstance(action_regex, re.Pattern)
        else _compile_extractor("action_regex", action_regex)
    )
    actions = action_pattern.findall(content)
    thought_match = re.match(r"(?s)(.*?)```bash", content)
    thought = thought_match.group(1).strip() if thought_match else content.split("```")[0].strip()
    if thought_max_chars == 0:
        thought = ""
    elif len(thought) > thought_max_chars:
        if thought_max_chars <= 3:
            thought = thought[:thought_max_chars]
        else:
            thought = thought[: thought_max_chars - 3] + "..."
    if actions:
        # Split on && / ||, truncate each sub-command individually.
        parts = re.split(r"(\s*(?:&&|\|\|)\s*)", actions[0].strip())
        cmd = "".join(p[:CMD_MAX_CHARS] if i % 2 == 0 else p for i, p in enumerate(parts))
        if len(cmd) > CMD_MAX_CHARS:
            cmd = cmd[: CMD_MAX_CHARS - 1] + "…"
        if thought:
            return f"{thought}\n\n```bash\n{cmd}\n```"
        return f"```bash\n{cmd}\n```"
    return thought


def _compact_user(content: str, max_chars: int = USER_MAX_CHARS) -> str:
    """Bound an old human follow-up while preserving both ends."""
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content
    marker = "\n... [older user message truncated by ARC] ...\n"
    if max_chars <= len(marker):
        return content[:max_chars]
    payload = max(0, max_chars - len(marker))
    head_n = (payload * 2) // 3
    tail_n = payload - head_n
    tail = content[-tail_n:] if tail_n else ""
    return content[:head_n] + marker + tail


def _compile_extractor(name: str, pattern: str) -> re.Pattern[str]:
    if not isinstance(pattern, str):
        raise TypeError(f"{name} must be a string")
    try:
        compiled = re.compile(pattern, re.DOTALL)
    except re.error as exc:
        raise ValueError(f"{name} is not a valid regular expression") from exc
    if compiled.groups != 1:
        raise ValueError(f"{name} must contain exactly one capture group")
    return compiled


def _summary_manifest(content: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    state_entries = [
        {
            "full": str(entry["full"]),
            "obs_ids": [str(obs_id).lower() for obs_id in entry.get("obs_ids", [])],
        }
        for entry in entries
        if entry.get("full")
    ]
    observation_ids = list(
        dict.fromkeys(obs_id for entry in state_entries for obs_id in entry["obs_ids"])
    )
    state_json = json.dumps(
        state_entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "version": _ARC_SUMMARY_VERSION,
        "provenance": _SUMMARY_PROVENANCE,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "state_sha256": hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
        "observation_ids": observation_ids,
        "entries": state_entries,
    }


def _trusted_summary(message: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Return verified structured entries and represented handles."""
    content = message.get("content")
    manifest = message.get(_ARC_SUMMARY_KEY)
    if not isinstance(content, str) or not content.startswith(SUMMARY_HEADER):
        return None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("version") != _ARC_SUMMARY_VERSION
        or manifest.get("provenance") is not _SUMMARY_PROVENANCE
    ):
        return None
    digest = manifest.get("content_sha256")
    if not isinstance(digest, str) or not hmac.compare_digest(
        digest,
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    ):
        return None
    ids = manifest.get("observation_ids")
    if not isinstance(ids, list):
        return None
    visible = {match.group(1).lower() for match in _OBSERVATION_HANDLE_RE.finditer(content)}
    normalized: list[str] = []
    seen: set[str] = set()
    for obs_id in ids:
        if (
            not isinstance(obs_id, str)
            or not 4 <= len(obs_id) <= 128
            or any(character not in "0123456789abcdefABCDEF" for character in obs_id)
        ):
            return None
        canonical = obs_id.lower()
        if canonical not in visible:
            return None
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        return None
    entries: list[dict[str, Any]] = []
    entry_ids: list[str] = []
    cursor = len(SUMMARY_HEADER)
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            return None
        full = entry.get("full")
        obs_ids = entry.get("obs_ids")
        if not isinstance(full, str) or not full or not isinstance(obs_ids, list):
            return None
        position = content.find(full, cursor)
        if position < 0:
            return None
        cursor = position + len(full)
        normalized_entry_ids: list[str] = []
        for obs_id in obs_ids:
            if not isinstance(obs_id, str):
                return None
            canonical = obs_id.lower()
            if canonical not in seen or f"§{canonical}" not in full.lower():
                return None
            normalized_entry_ids.append(canonical)
            entry_ids.append(canonical)
        entries.append({"full": full, "obs_ids": normalized_entry_ids})
    if list(dict.fromkeys(entry_ids)) != normalized:
        return None
    state_json = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    state_digest = manifest.get("state_sha256")
    if not isinstance(state_digest, str) or not hmac.compare_digest(
        state_digest,
        hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
    ):
        return None
    return entries, normalized


def compact_messages(
    messages: list[dict],
    *,
    keep_recent: int = 10,
    action_regex: str = DEFAULT_ACTION_REGEX,
    observation_regex: str = DEFAULT_OBSERVATION_REGEX,
    thought_max_chars: int = THOUGHT_MAX_CHARS,
    result_max_chars: int = RESULT_MAX_CHARS,
    inherit_prior_summary: bool = True,
) -> list[dict]:
    """Compact older messages to reduce context size (string-only baseline).

    Returns the (possibly compacted) message list.  Does not mutate the
    input.

    Args:
        messages: Full message history.
        keep_recent: Number of recent turn pairs (assistant + user) to
            keep uncompacted.
        action_regex: Regex extracting the action from assistant messages.
        observation_regex: Regex extracting the observation body from
            user messages.
        thought_max_chars: Max chars for thought truncation.
        result_max_chars: Observations longer than this are dropped.
        inherit_prior_summary: When True (default), the body of a prior
            summary is folded into the new summary.  When False, prior
            summaries are dropped (the "max" baseline's L4
            information-loss point).
    """
    action_pattern = _compile_extractor("action_regex", action_regex)
    observation_pattern = _compile_extractor("observation_regex", observation_regex)
    prefix_count = protected_prefix_count(messages)
    body = messages[prefix_count:]
    tail_start = recent_turn_start(body, keep_recent)
    if tail_start <= 0:
        return messages

    to_compact = body[:tail_start]
    to_keep = body[tail_start:]

    entries: list[dict[str, Any]] = []
    retained_instructions: list[dict[str, Any]] = []
    for msg in to_compact:
        for fragment in message_fragments(msg):
            role = message_role(fragment)
            if role == "assistant":
                full = _compact_assistant(
                    message_text(fragment), action_pattern, thought_max_chars
                )
                if full:
                    entries.append({"full": full, "obs_ids": []})
            elif role in {"system", "developer"}:
                content = _compact_user(message_text(fragment).strip())
                if content:
                    retained_instructions.append({"role": role, "content": content})
            elif role in {"user", "tool", "function"}:
                content = message_text(fragment).strip()
                trusted = _trusted_summary(fragment)
                if trusted is not None:
                    if inherit_prior_summary:
                        entries.extend(trusted[0])
                    continue
                match = observation_pattern.search(content)
                if match and len(match.group(1)) <= result_max_chars:
                    entries.append({"full": f"result: {match.group(1)}", "obs_ids": []})
                elif role == "user" and content:
                    entries.append({"full": f"user: {_compact_user(content)}", "obs_ids": []})

    parts = [entry["full"] for entry in entries]
    summary = SUMMARY_HEADER + "\n\n" + "\n\n---\n\n".join(parts) if parts else SUMMARY_HEADER

    summary_message = {
        "role": "user",
        "content": summary,
        _ARC_SUMMARY_KEY: _summary_manifest(summary, entries),
    }
    return [
        *messages[:prefix_count],
        *retained_instructions,
        summary_message,
        *to_keep,
    ]


def compact_messages_max(
    messages: list[dict],
    *,
    keep_recent: int = 1,
    action_regex: str = DEFAULT_ACTION_REGEX,
    observation_regex: str = DEFAULT_OBSERVATION_REGEX,
    thought_max_chars: int = THOUGHT_MAX_CHARS,
    result_max_chars: int = RESULT_MAX_CHARS,
) -> list[dict]:
    """Compact older messages, dropping any previous summary.

    Unlike :func:`compact_messages`, which inherits the body of prior
    summaries into the new one, this variant discards them entirely
    (the "AutoComp" baseline in the ARC paper).
    """
    return compact_messages(
        messages,
        keep_recent=keep_recent,
        action_regex=action_regex,
        observation_regex=observation_regex,
        thought_max_chars=thought_max_chars,
        result_max_chars=result_max_chars,
        inherit_prior_summary=False,
    )


def _collapsed_id_line(ids: list[str]) -> str:
    """Render the id-only fallback for a group of evicted citations.

    ``ids`` is an ordered, de-duplicated list (oldest first).  Only the
    most recent ``_MAX_COLLAPSED_IDS`` are listed verbatim; any overflow
    is noted as a count so the line stays bounded.
    """
    if not ids:
        return ""
    shown = ids[-_MAX_COLLAPSED_IDS:]
    dropped = len(ids) - len(shown)
    prefix = f"(+{dropped} older elided) " if dropped else ""
    return prefix + " ".join(f"\u00a7{i}" for i in shown)


def _elision_notice(omitted_ids: int) -> str:
    notice = _ELISION_NOTE
    if omitted_ids:
        notice += (
            f" {omitted_ids} older handle(s) could not fit; use the host's "
            "bounded observation-list operation to rediscover them."
        )
    return notice


def _assemble_summary(parts: list[str], *, collapsed: bool, omitted_ids: int = 0) -> str:
    """Render a summary, including truthful bounded-elision metadata."""
    header = SUMMARY_HEADER
    if collapsed:
        header += "\n" + _elision_notice(omitted_ids)
    body = _SUMMARY_SEP.join(parts)
    return header + ("\n\n" + body if body else "")


def _render_summary_budget(
    entries: list[dict], summary_max_chars: int
) -> tuple[str, bool, list[dict[str, Any]]]:
    """Render entries and return the exact structured state that remains visible.

    Recent entries retain their rich form.  Demoted observations share one
    bounded id-only line, newest first for admission and chronological in the
    final rendering.  When even all handles cannot fit, the oldest handles are
    omitted explicitly; their records remain in the lossless store and are
    discoverable through a host/MCP observation-list operation.
    """
    entries = [entry for entry in entries if entry.get("full")]
    full_parts = [entry["full"] for entry in entries]
    if summary_max_chars <= 0:
        return _assemble_summary(full_parts, collapsed=False), False, list(entries)

    def rendered_length(
        parts_chars: int,
        parts_count: int,
        *,
        omitted_ids: int = 0,
    ) -> int:
        header_length = len(SUMMARY_HEADER) + 1 + len(_elision_notice(omitted_ids))
        if not parts_count:
            return header_length
        return header_length + 2 + parts_chars + len(_SUMMARY_SEP) * (parts_count - 1)

    unbounded_length = len(SUMMARY_HEADER)
    if full_parts:
        unbounded_length += (
            2 + sum(map(len, full_parts)) + len(_SUMMARY_SEP) * (len(full_parts) - 1)
        )
    if unbounded_length <= summary_max_chars:
        return _assemble_summary(full_parts, collapsed=False), False, list(entries)

    # Pick a contiguous rich suffix.  Accounting uses the final collapsed
    # header so adding the elision note can never push the result over cap.
    rich_start = len(entries)
    rich_chars = 0
    rich_count = 0
    for index in range(len(entries) - 1, -1, -1):
        candidate_chars = rich_chars + len(entries[index]["full"])
        if rendered_length(candidate_chars, rich_count + 1) <= summary_max_chars:
            rich_chars = candidate_chars
            rich_count += 1
            rich_start = index
        else:
            break

    demoted_ids: list[str] = []
    seen: set[str] = set()

    def demote(entry: dict) -> None:
        for obs_id in entry.get("obs_ids", []):
            canonical = obs_id.lower()
            if canonical not in seen:
                seen.add(canonical)
                demoted_ids.append(canonical)

    for entry in entries[:rich_start]:
        demote(entry)

    while True:
        # The hard per-line ceiling is applied before the exact budget pass.
        omitted = max(0, len(demoted_ids) - _MAX_COLLAPSED_IDS)
        shown_ids = demoted_ids[-_MAX_COLLAPSED_IDS:]

        # First admit as many recent handles as the exact cap permits.
        while shown_ids:
            collapsed_line = _collapsed_id_line(shown_ids)
            if (
                rendered_length(
                    len(collapsed_line) + rich_chars,
                    rich_count + 1,
                    omitted_ids=omitted,
                )
                <= summary_max_chars
            ):
                rich_suffix = [entry["full"] for entry in entries[rich_start:]]
                rendered_entries: list[dict[str, Any]] = [
                    {"full": _elision_notice(omitted), "obs_ids": []}
                ]
                rendered_entries.extend(
                    {"full": f"§{obs_id}", "obs_ids": [obs_id]} for obs_id in shown_ids
                )
                rendered_entries.extend(entries[rich_start:])
                return (
                    _assemble_summary(
                        [collapsed_line, *rich_suffix],
                        collapsed=True,
                        omitted_ids=omitted,
                    ),
                    True,
                    rendered_entries,
                )
            shown_ids.pop(0)
            omitted += 1

        if rendered_length(rich_chars, rich_count, omitted_ids=omitted) <= summary_max_chars:
            rich_suffix = [entry["full"] for entry in entries[rich_start:]]
            rendered_entries = [
                {"full": _elision_notice(omitted), "obs_ids": []},
                *entries[rich_start:],
            ]
            return (
                _assemble_summary(rich_suffix, collapsed=True, omitted_ids=omitted),
                True,
                rendered_entries,
            )
        if rich_start < len(entries):
            # Extremely small caps or an oversized most-recent short result:
            # demote another rich entry and recompute its exact observation-id
            # contribution.  Assistant text has no handle and must not inflate
            # the omitted-handle count.
            demote(entries[rich_start])
            rich_chars -= len(entries[rich_start]["full"])
            rich_count -= 1
            rich_start += 1
            continue
        # A pathological cap smaller than the summary header cannot preserve
        # structure.  Honour the caller's hard bound rather than overflowing.
        return "", True, []


def _apply_summary_budget(entries: list[dict], summary_max_chars: int) -> tuple[str, bool]:
    """Compatibility wrapper returning the rendered summary and collapse flag."""
    summary, collapsed, _rendered_entries = _render_summary_budget(entries, summary_max_chars)
    return summary, collapsed


def compact_messages_arc(
    messages: list[dict],
    *,
    obs_store: Optional[ObsStore] = None,
    keep_recent: int = 1,
    action_regex: str = DEFAULT_ACTION_REGEX,
    observation_regex: str = DEFAULT_OBSERVATION_REGEX,
    thought_max_chars: int = THOUGHT_MAX_CHARS,
    result_max_chars: int = RESULT_MAX_CHARS,
    citation_head_chars: int = 200,
    citation_tail_chars: int = 200,
    summary_max_chars: int = 0,
) -> list[dict]:
    """Compact older messages into a citation-rich summary (ARC operator).

    Instead of *dropping* observations longer than ``result_max_chars``,
    observations tagged with an ``_obs_id`` present in ``obs_store`` are
    rendered as structured citations (bounded-length pointers into the
    lossless store).  Verified structured entries from prior summaries are
    carried forward, so an obs_id cited once stays accessible for the rest
    of the trajectory without treating arbitrary summary text as provenance.

    Degeneration property: with ``obs_store=None`` and no ``_obs_id``
    tags, the output is identical to :func:`compact_messages_max` except
    that verified entries from prior summaries are preserved.

    Args:
        messages: Full provider message/item history.  Leading instructions
            and the first user task are kept intact.
        obs_store: The lossless observation store.  ``None`` falls back
            to the legacy drop behaviour.
        keep_recent: Number of recent turn pairs to keep uncompacted.
        action_regex: Regex extracting the action from assistant messages.
        observation_regex: Regex extracting the observation body from
            user messages (store-less fallback path).
        thought_max_chars: Max chars of assistant thought retained.
        result_max_chars: Observations at most this long are kept
            verbatim; longer ones become citations (or are dropped when
            not in the store).
        citation_head_chars: Characters of head preview per citation.
        citation_tail_chars: Characters of tail preview per citation.
        summary_max_chars: Character cap on the assembled summary body.
            ``0`` (default) disables the cap.  When > 0, the most recent
            entries stay verbatim and older citations collapse to
            id-only ``§<id>`` references, bounding the compacted context
            regardless of trajectory length.

    Returns:
        A new ``list[dict]`` with the protected prompt prefix, one user-role
        summary, and the recent ``keep_recent`` turns appended verbatim.
    """
    action_pattern = _compile_extractor("action_regex", action_regex)
    observation_pattern = _compile_extractor("observation_regex", observation_regex)
    prefix_count = protected_prefix_count(messages)
    body = messages[prefix_count:]
    tail_start = recent_turn_start(body, keep_recent)
    if tail_start <= 0:
        return messages

    to_compact = body[:tail_start]
    to_keep = body[tail_start:]

    # Each entry is ``{"full": <verbatim rendering>, "collapsed":
    # <id-only fallback or "">}``; the budget pass chooses which form to
    # emit.  ``emitted_obs_ids`` de-duplicates citations across prior
    # summaries and current messages (ids normalised to lowercase).
    entries: list[dict] = []
    emitted_obs_ids: set[str] = set()
    retained_instructions: list[dict[str, Any]] = []

    def _add(full: str, obs_ids: Optional[list[str]] = None) -> None:
        if full:
            entries.append({"full": full, "obs_ids": obs_ids or []})

    for msg in to_compact:
        for fragment in message_fragments(msg):
            role = message_role(fragment)
            if role == "assistant":
                _add(_compact_assistant(message_text(fragment), action_pattern, thought_max_chars))
                continue

            if role in {"system", "developer"}:
                content = _compact_user(message_text(fragment).strip())
                if content:
                    retained_instructions.append({"role": role, "content": content})
                continue

            if role not in {"user", "tool", "function"}:
                continue

            content = message_text(fragment).strip()
            trusted = _trusted_summary(fragment)
            if trusted is not None:
                inherited_entries, inherited_ids = trusted
                emitted_obs_ids.update(inherited_ids)
                for inherited_entry in inherited_entries:
                    _add(inherited_entry["full"], inherited_entry["obs_ids"])
                continue

            obs_id = fragment.get("_obs_id")
            if not isinstance(obs_id, str):
                obs_id = None
            if obs_id is not None and obs_id.lower() in emitted_obs_ids:
                continue
            if obs_id is not None and obs_store is not None and obs_store.has(obs_id):
                rec = obs_store.get(obs_id)
                assert rec is not None
                if rec.char_count <= result_max_chars:
                    escaped = html.escape(rec.full_content, quote=False)
                    full = (
                        f'<observation_inline id="\u00a7{rec.obs_id}" '
                        'trust="untrusted_tool_output">'
                        f"<body>{escaped}</body></observation_inline>"
                    )
                elif hasattr(rec, "content_sha256") and hasattr(rec, "sequence"):
                    full = format_formal_citation(
                        rec,
                        head_chars=citation_head_chars,
                        tail_chars=citation_tail_chars,
                    )
                else:
                    full = format_citation(
                        obs_id=rec.obs_id,
                        cmd_signature=rec.cmd_signature,
                        returncode=rec.returncode,
                        line_count=rec.line_count,
                        char_count=rec.char_count,
                        head=rec.head,
                        tail=rec.tail,
                        head_chars=citation_head_chars,
                        tail_chars=citation_tail_chars,
                    )
                emitted_obs_ids.add(rec.obs_id.lower())
                _add(full, [rec.obs_id])
                continue

            match = observation_pattern.search(content)
            if match and len(match.group(1)) <= result_max_chars:
                escaped = html.escape(match.group(1), quote=False)
                _add(
                    '<observation_inline trust="untrusted_tool_output">'
                    f"<body>{escaped}</body></observation_inline>"
                )
            elif role == "user" and content:
                _add(f"user: {_compact_user(content)}")

    summary_text, _collapsed_any, rendered_entries = _render_summary_budget(
        entries, summary_max_chars
    )
    summary_message: dict[str, Any] = {"role": "user", "content": summary_text}
    if summary_text.startswith(SUMMARY_HEADER):
        summary_message[_ARC_SUMMARY_KEY] = _summary_manifest(summary_text, rendered_entries)

    return [
        *messages[:prefix_count],
        *retained_instructions,
        summary_message,
        *to_keep,
    ]
