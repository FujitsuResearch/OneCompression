"""Wire helpers for current Codex ``PostToolUse`` hook contracts.

Current Codex replaces a completed tool result with the post-tool ``reason``
when ``decision`` is ``"block"``. OneComp names that behavior ``replace``;
``block`` remains as a wire-compatible alias and ``context`` keeps the original
result while adding archival context.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Any, Literal

CodexResultMode = Literal["replace", "context", "stop", "block"]


def post_tool_use_output(
    citation: str,
    *,
    mode: CodexResultMode = "replace",
) -> dict[str, Any]:
    """Encode replacement, archival context, or a stop decision."""
    if not isinstance(citation, str) or not citation:
        raise ValueError("citation must be a non-empty string")
    if mode == "context":
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": citation,
            }
        }
    if mode == "stop":
        return {"continue": False, "stopReason": citation}
    if mode in {"replace", "block"}:
        return {"decision": "block", "reason": citation}
    raise ValueError("mode must be 'replace', 'context', 'stop', or 'block'")


__all__ = ["CodexResultMode", "post_tool_use_output"]
