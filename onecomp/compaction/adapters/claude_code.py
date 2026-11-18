"""
Wire response helpers for a Claude Code ``PostToolUse`` hook.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Any, Optional


def replace_tool_output(response: Any, citation: str) -> Optional[Any]:
    """Preserve a built-in tool's output shape while replacing its body.

    Unknown object shapes fail open. Returning an invalid replacement can make
    a host ignore compaction while giving the integration a false success
    signal, so unsupported shapes are never guessed.
    """
    if not isinstance(citation, str) or not citation:
        raise ValueError("citation must be a non-empty string")
    if isinstance(response, str):
        return citation
    if (
        response
        and isinstance(response, list)
        and all(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            for block in response
        )
    ):
        return [{"type": "text", "text": citation}]
    if isinstance(response, dict) and isinstance(response.get("stdout"), str):
        replacement = dict(response)
        replacement["stdout"] = citation
        if isinstance(replacement.get("stderr"), str):
            replacement["stderr"] = ""
        return replacement
    return None


def post_tool_use_output(response: Any, citation: str) -> dict[str, Any]:
    """Encode a schema-compatible ``updatedToolOutput`` response."""
    replacement = replace_tool_output(response, citation)
    if replacement is None:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": replacement,
        }
    }


__all__ = ["post_tool_use_output", "replace_tool_output"]
