"""Provider-neutral helpers for agent message payloads.

The ARC core keeps provider objects opaque whenever they remain in the live
tail, but it still needs a textual view of older objects for compaction and
budget accounting.  These helpers cover the shapes used by OpenAI Chat and
Responses, Anthropic Messages / Claude Code, and OpenHands serialised events
without importing any of those SDKs.

Unknown non-text blocks are represented by a short marker instead of their
full JSON encoding.  This avoids accidentally copying base64 image payloads
into an ARC summary.  Recent messages are returned unchanged by the compactor,
so the marker is used only after a block has moved into compacted history.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import math
from typing import Any

_MAX_PROJECTION_DEPTH = 32
_MAX_PROJECTION_NODES = 100_000
_MAX_PROJECTION_CHARS = 1_000_000

_ASSISTANT_ITEM_TYPES = {
    "assistant",
    "computer_call",
    "custom_tool_call",
    "function_call",
    "local_shell_call",
    "message",
    "reasoning",
    "shell_call",
}
_TOOL_ITEM_TYPES = {
    "computer_call_output",
    "custom_tool_call_output",
    "function_call_output",
    "local_shell_call_output",
    "shell_call_output",
    "tool_result",
}
_TEXT_BLOCK_TYPES = {"input_text", "output_text", "text"}


def message_role(message: dict[str, Any]) -> str:
    """Return a canonical role for common provider message/item shapes."""
    role = message.get("role")
    if isinstance(role, str):
        canonical = role.lower()
        # Anthropic represents tool results as content blocks inside a
        # user-role message.  Treat those as observations during compaction;
        # otherwise an untagged multi-kilobyte tool result is mistaken for a
        # human follow-up and copied into the summary.
        if canonical == "user" and _is_pure_tool_result(message.get("content")):
            return "tool"
        return canonical

    item_type = message.get("type") or message.get("kind")
    if not isinstance(item_type, str):
        return ""
    item_type = item_type.lower()
    if item_type in _TOOL_ITEM_TYPES or item_type.endswith("_call_output"):
        return "tool"
    if item_type in _ASSISTANT_ITEM_TYPES or item_type.endswith("_call"):
        return "assistant"
    if item_type in {"user", "message_event", "observation", "observation_event"}:
        return "user"
    if item_type in {"developer", "system", "system_prompt_event"}:
        return "system" if item_type.startswith("system") else "developer"
    return ""


def content_to_text(content: Any, *, max_chars: int | None = _MAX_PROJECTION_CHARS) -> str:
    """Extract text from string or provider content-block payloads.

    This function is deliberately total for JSON-like values: malformed or
    unfamiliar blocks cannot crash context-budget enforcement.
    """
    if max_chars is not None and (
        not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0
    ):
        raise ValueError("max_chars must be a non-negative integer or None")
    active: set[int] = set()
    nodes = 0

    def project(value: Any, depth: int) -> str:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_PROJECTION_NODES:
            if max_chars is None:
                raise ValueError("content exceeds the exact projection node limit")
            return "[content projection node limit reached]"
        if depth > _MAX_PROJECTION_DEPTH:
            if max_chars is None:
                raise ValueError("content exceeds the exact projection depth limit")
            return "[content projection depth limit reached]"
        if value is None:
            return ""
        if isinstance(value, str):
            return _bounded_projection(value, max_chars)
        if isinstance(value, (int, float, bool)):
            if max_chars is None and isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite content has no exact JSON projection")
            return str(value)
        if not isinstance(value, (list, tuple, dict)):
            if max_chars is None:
                raise ValueError("non-JSON content has no exact textual projection")
            return _bounded_projection(str(value), max_chars)

        identity = id(value)
        if identity in active:
            if max_chars is None:
                raise ValueError("cyclic content has no exact textual projection")
            return "[cyclic content omitted]"
        active.add(identity)
        try:
            if isinstance(value, (list, tuple)):
                parts: list[str] = []
                used = 0
                for item in value:
                    part = project(item, depth + 1)
                    if not part:
                        continue
                    separator = 1 if parts else 0
                    if max_chars is None:
                        parts.append(part)
                        used += separator + len(part)
                    else:
                        remaining = max_chars - used - separator
                        if remaining <= 0:
                            break
                        parts.append(part[:remaining])
                        used += separator + min(len(part), remaining)
                        if len(part) > remaining:
                            break
                    if nodes > _MAX_PROJECTION_NODES:
                        break
                return "\n".join(parts)

            block_type = value.get("type") or value.get("kind")
            normalized_type = block_type.lower() if isinstance(block_type, str) else block_type
            if normalized_type in _TEXT_BLOCK_TYPES and isinstance(value.get("text"), str):
                return _bounded_projection(value["text"], max_chars)
            if normalized_type == "tool_use":
                name = value.get("name", "tool")
                payload = value.get("input", {})
                json_limit = None if max_chars is None else min(max_chars, 240)
                return _bounded_projection(
                    f"{name}({_compact_json(payload, max_chars=json_limit)})",
                    max_chars,
                )
            if _is_non_text_block_type(block_type):
                return f"[{block_type} block omitted from compacted text]"
            if _is_tool_result_block(value):
                for key in ("content", "output", "text"):
                    if key in value:
                        return project(value[key], depth + 1)

            for key in ("text", "content", "output", "result", "summary"):
                if key in value:
                    text = project(value[key], depth + 1)
                    if text:
                        return text

            label = block_type if isinstance(block_type, str) else "non-text"
            return f"[{label} block omitted from compacted text]"
        finally:
            active.remove(identity)

    text = project(content, 0)
    return _bounded_projection(text, max_chars)


def _bounded_projection(value: str, max_chars: int | None) -> str:
    if max_chars is None or len(value) <= max_chars:
        return value
    if max_chars == 0:
        return ""
    return value[: max_chars - 1] + "…"


def message_fragments(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a mixed user payload into ordered human and tool fragments.

    Anthropic may place text and one or more ``tool_result`` blocks in a
    single user-role message. Treating the complete message as a tool result
    drops the human text during compaction. The live tail remains opaque;
    this helper is intended only for history that is already being compacted.
    """
    role = message.get("role")
    if (
        not isinstance(role, str)
        or role.lower() != "user"
        or not isinstance(message.get("content"), list)
    ):
        return [message]
    blocks = message["content"]
    kinds = [_is_tool_result_block(block) for block in blocks]
    if not any(kinds) or all(kinds):
        return [message]

    tool_blocks = sum(kinds)
    fragments: list[dict[str, Any]] = []
    start = 0
    while start < len(blocks):
        is_tool = kinds[start]
        end = start + 1
        while end < len(blocks) and kinds[end] == is_tool:
            end += 1
        fragment = dict(message)
        fragment["content"] = blocks[start:end]
        fragment["role"] = "tool" if is_tool else "user"
        if not is_tool or tool_blocks != 1:
            fragment.pop("_obs_id", None)
        fragments.append(fragment)
        start = end
    return fragments


def message_text(
    message: dict[str, Any],
    *,
    max_chars: int | None = _MAX_PROJECTION_CHARS,
) -> str:
    """Return the model-visible textual projection of one message/item."""
    item_type = message.get("type") or message.get("kind")
    normalized_type = item_type.lower() if isinstance(item_type, str) else item_type
    if isinstance(normalized_type, str) and (
        normalized_type in _TOOL_ITEM_TYPES or normalized_type.endswith("_call_output")
    ):
        for key in ("output", "content", "result"):
            if key in message:
                return content_to_text(message[key], max_chars=max_chars)

    if isinstance(normalized_type, str) and (
        normalized_type in _ASSISTANT_ITEM_TYPES or normalized_type.endswith("_call")
    ):
        if normalized_type.endswith("_call") or normalized_type in {
            "function_call",
            "custom_tool_call",
            "local_shell_call",
            "shell_call",
        }:
            name = message.get("name") or normalized_type
            arguments = message.get("arguments", message.get("input", ""))
            json_limit = None if max_chars is None else min(max_chars, 240)
            return _bounded_projection(
                f"{name}({_compact_json(arguments, max_chars=json_limit)})",
                max_chars,
            )

    parts: list[str] = []
    projected_chars = 0

    def append_part(part: str) -> bool:
        """Append one part and return whether another part can still fit."""
        nonlocal projected_chars
        if not part:
            return True
        separator_chars = 1 if parts else 0
        if max_chars is None:
            parts.append(part)
            projected_chars += separator_chars + len(part)
            return True
        remaining = max_chars - projected_chars - separator_chars
        if remaining <= 0:
            return False
        parts.append(part[:remaining])
        projected_chars += separator_chars + min(len(part), remaining)
        return len(part) <= remaining and projected_chars < max_chars

    content = content_to_text(message.get("content"), max_chars=max_chars)
    if not append_part(content):
        return _bounded_projection("\n".join(parts), max_chars)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, call in enumerate(tool_calls):
            if index >= _MAX_PROJECTION_NODES:
                if max_chars is None:
                    raise ValueError("tool_calls exceeds the exact projection node limit")
                append_part("[tool call projection node limit reached]")
                break
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            name = function.get("name", "tool")
            arguments = function.get("arguments", function.get("input", ""))
            json_limit = None if max_chars is None else min(max_chars, 240)
            if not append_part(f"{name}({_compact_json(arguments, max_chars=json_limit)})"):
                break
    return _bounded_projection("\n".join(parts), max_chars)


def protected_prefix_count(messages: list[dict[str, Any]]) -> int:
    """Count the immutable prompt prefix for common agent histories.

    Leading system/developer messages and the first user task are protected.
    This yields two messages for the original ARC ReAct layout, one for an
    Anthropic message list whose system prompt is supplied separately, and any
    number of leading developer/system instructions for OpenAI APIs.
    """
    index = 0
    while index < len(messages) and message_role(messages[index]) in {
        "system",
        "developer",
    }:
        index += 1
    if index < len(messages) and message_role(messages[index]) == "user":
        index += 1
    return index


def recent_turn_start(messages: list[dict[str, Any]], keep_recent: int) -> int:
    """Index where the newest ``keep_recent`` assistant turns begin."""
    if keep_recent <= 0:
        return len(messages)
    assistant_indices = [
        index for index, message in enumerate(messages) if message_role(message) == "assistant"
    ]
    if assistant_indices:
        # If a prior ARC summary precedes the only live assistant turn, its
        # index is > 0.  Returning that index permits progressive re-compaction
        # with a tighter cap while still preserving the complete live turn.
        index = assistant_indices[max(0, len(assistant_indices) - keep_recent)]
        # A provider response can contain consecutive reasoning, message, and
        # tool-call items. Walk back over the complete machine-authored group
        # so compaction never cuts a logical response at an item boundary.
        while index > 0 and message_role(messages[index - 1]) == "assistant":
            index -= 1
        return index
    # Observation-only/event-only histories still get a bounded live tail.
    return max(0, len(messages) - keep_recent)


def total_text_chars(messages: list[dict[str, Any]]) -> int:
    """Character count of the textual projection sent to a model."""
    return sum(len(message_text(message, max_chars=None)) for message in messages)


def _compact_json(value: Any, max_chars: int | None = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            if max_chars is None:
                raise ValueError("tool arguments have no exact JSON projection") from exc
            text = str(value)
    text = " ".join(text.split())
    if max_chars == 0:
        return ""
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def _is_tool_result_block(content: Any) -> bool:
    if not isinstance(content, dict):
        return False
    block_type = content.get("type") or content.get("kind")
    normalized = block_type.lower() if isinstance(block_type, str) else block_type
    return isinstance(normalized, str) and (
        normalized in _TOOL_ITEM_TYPES or normalized.endswith("_call_output")
    )


def _is_pure_tool_result(content: Any) -> bool:
    if isinstance(content, dict):
        return _is_tool_result_block(content)
    if isinstance(content, (list, tuple)):
        meaningful = [item for item in content if item is not None]
        return bool(meaningful) and all(_is_tool_result_block(item) for item in meaningful)
    return False


def _is_non_text_block_type(block_type: Any) -> bool:
    if not isinstance(block_type, str):
        return False
    normalized = block_type.lower()
    return normalized in {
        "audio",
        "binary",
        "document",
        "file",
        "image",
        "image_url",
        "input_audio",
        "input_file",
        "input_image",
        "input_video",
        "video",
    } or normalized.endswith(("_audio", "_image", "_video"))


__all__ = [
    "content_to_text",
    "message_role",
    "message_fragments",
    "message_text",
    "protected_prefix_count",
    "recent_turn_start",
    "total_text_chars",
]
