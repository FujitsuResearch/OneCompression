"""ARC PostToolUse hook adapters for supported agent host contracts.

The command entry point reads one hook event from stdin, stores a long tool
response in a per-session WAL. Claude Code, Codex, and OpenCode can replace the
model-visible result with a bounded citation. The standalone command wrapper
remains available for older hosts and command classes without replacement.
A paired recall entry point prints the exact stored response on demand.

These adapters operate at tool-result ingestion time.  They do not pretend
that merely exposing the OneComp MCP server can rewrite a host application's
already-owned conversation history.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from ..config import CompactionConfig
from ..formal import (
    ArcCoreService,
    FormalArcError,
    RecallRequiresChunking,
    SQLiteObservationArchive,
    format_citation,
)
from ..runner import CompactionRunner
from .claude_code import post_tool_use_output as claude_post_tool_use_output
from .claude_code import replace_tool_output as replace_claude_tool_output
from .codex import CodexResultMode
from .codex import post_tool_use_output as codex_post_tool_use_output

_OBS_ID_RE = re.compile(r"^[0-9a-fA-F]{4,128}$")
_RECALL_COMMANDS = ("onecomp-arc-recall", "onecomp.compaction.adapters.hooks recall")


def process_post_tool_use(
    payload: dict[str, Any],
    *,
    platform: Literal["claude", "codex", "opencode"],
    result_max_chars: int = 500,
    hook_root: Optional[Path | str] = None,
    codex_result_mode: CodexResultMode = "replace",
    max_recall_units: int = 4096,
    max_archive_bytes: int = 0,
) -> dict[str, Any]:
    """Store one PostToolUse response and apply the host's supported contract.

    Returns an empty object when the response is short, unsupported, or is the
    paired recall command itself.  Hook callers can therefore configure a broad
    matcher while small ordinary results pass through unchanged.
    """
    if platform not in {"claude", "codex", "opencode"}:
        raise ValueError("platform must be 'claude', 'codex', or 'opencode'")
    if codex_result_mode not in {"replace", "context", "stop", "block"}:
        raise ValueError("codex_result_mode must be 'replace', 'context', 'stop', or 'block'")
    if not isinstance(payload, dict):
        raise ValueError("hook payload must be an object")
    if not isinstance(result_max_chars, int) or isinstance(result_max_chars, bool):
        raise ValueError("result_max_chars must be a non-negative integer")
    if result_max_chars < 0:
        raise ValueError("result_max_chars must be a non-negative integer")
    if (
        not isinstance(max_recall_units, int)
        or isinstance(max_recall_units, bool)
        or max_recall_units <= 0
    ):
        raise ValueError("max_recall_units must be a positive integer")
    if (
        not isinstance(max_archive_bytes, int)
        or isinstance(max_archive_bytes, bool)
        or max_archive_bytes < 0
    ):
        raise ValueError("max_archive_bytes must be a non-negative integer")

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 4096:
        raise ValueError("hook payload requires a session_id up to 4096 characters")
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name or len(tool_name) > 1024:
        raise ValueError("hook payload requires a tool_name up to 1024 characters")
    tool_input = payload.get("tool_input", {})
    command = _command_signature(tool_name, tool_input)
    if any(marker in command for marker in _RECALL_COMMANDS):
        return {}

    response = payload.get("tool_response")
    content = _lossless_json_text(response)
    if len(content) <= result_max_chars:
        return {}
    # Claude accepts only replacement values whose original wire shape is
    # known.  Decide that before committing so an unsupported third-party
    # result cannot leave an unreachable archive occurrence behind.
    if platform == "claude" and replace_claude_tool_output(response, "probe") is None:
        return {}

    archive_path = _archive_path(session_id, hook_root)
    runner = CompactionRunner(
        CompactionConfig(
            profile="formal-v1",
            archive_path=archive_path,
            result_max_chars=result_max_chars,
            max_recall_units=max_recall_units,
            max_archive_bytes=max_archive_bytes,
        )
    )
    try:
        record = runner.record_observation(
            command=command,
            content=content,
            source_key=_source_key(payload, platform),
            returncode=_returncode(response),
            extra={
                "platform": platform,
                "tool_name": tool_name,
                "tool_use_id": str(payload.get("tool_use_id", ""))[:1024],
            },
        )
        assert runner.archive is not None
        # The citation can be consumed by a sandbox that is unable to lock or
        # mutate the hook-owned WAL. Publish the committed row to the main
        # database before returning that citation.
        runner.archive.checkpoint(truncate=True)
    finally:
        runner.close()

    citation = format_citation(record)
    recall_command = (
        f"{_recall_launcher()} --session-id " f"{shlex.quote(session_id)} --obs-id {record.obs_id}"
    )
    citation += (
        "\n<host_recovery>Tool completed. For exact bounded recovery run "
        f"`{recall_command}`. Add --chunk-index or --range for oversized "
        "observations.</host_recovery>"
    )

    if platform == "codex":
        return codex_post_tool_use_output(citation, mode=codex_result_mode)
    if platform == "claude":
        return claude_post_tool_use_output(response, citation)
    return {"replacement": citation}


def _recall_launcher() -> str:
    """Address recall through the hook's installation, independent of PATH."""
    entry_point = Path(sys.executable).with_name("onecomp-arc-recall")
    if entry_point.is_file():
        return shlex.quote(str(entry_point))
    return f"{shlex.quote(sys.executable)} -m onecomp.compaction.adapters.hooks recall"


def _command_signature(tool_name: str, tool_input: Any) -> str:
    if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str):
        command = tool_input["command"].strip()
        if command:
            return _bounded_signature(command)
    return _bounded_signature(f"{tool_name}({_lossless_json_text(tool_input)})")


def _bounded_signature(value: str) -> str:
    if len(value) <= 16_384:
        _utf8_bytes("command signature", value)
        return value
    digest = hashlib.sha256(_utf8_bytes("command signature", value)).hexdigest()
    suffix = f"… [sha256={digest}]"
    return value[: 16_384 - len(suffix)] + suffix


def _lossless_json_text(value: Any) -> str:
    if isinstance(value, str):
        _utf8_bytes("hook string value", value)
        return value
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        _utf8_bytes("hook JSON value", text)
        return text
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("hook values must be JSON-serializable without NaN") from exc


def _returncode(response: Any) -> int:
    if isinstance(response, dict):
        for key in ("returncode", "return_code", "exit_code", "exitCode"):
            value = response.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return 0


def _source_key(payload: dict[str, Any], platform: str) -> Optional[str]:
    """Use a host occurrence identity only when the host supplied one.

    Content-derived keys incorrectly merge two distinct executions that happen
    to have equal inputs and outputs. Without a stable host event identifier,
    preserving each occurrence is safer than pretending retries are detectable.
    """
    for key in ("tool_use_id", "call_id", "tool_call_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            identity = f"{platform}\x1f{payload['session_id']}\x1f{value}"
            digest = hashlib.sha256(_utf8_bytes("hook occurrence identity", identity)).hexdigest()
            return f"hook:{platform}:{digest}"
    return None


def _hook_root(root: Optional[Path | str] = None) -> Path:
    resolved = Path(
        root
        if root is not None
        else os.environ.get(
            "ONECOMP_ARC_HOOK_ROOT",
            str(Path.home() / ".cache" / "onecomp" / "arc-hooks"),
        )
    ).expanduser()
    root_existed = resolved.exists()
    resolved.mkdir(parents=True, exist_ok=True)
    if not root_existed:
        try:
            resolved.chmod(0o700)
        except OSError:
            pass
    return resolved.resolve()


def _archive_path(session_id: str, root: Optional[Path | str] = None) -> Path:
    digest = hashlib.sha256(_utf8_bytes("session_id", session_id)).hexdigest()[:32]
    return _hook_root(root) / f"{digest}.sqlite3"


def _utf8_bytes(name: str, value: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc


def hook_main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="ARC PostToolUse hook")
    parser.add_argument("--platform", choices=("claude", "codex", "opencode"), required=True)
    parser.add_argument("--result-max-chars", type=int, default=500)
    parser.add_argument("--max-recall-units", type=int, default=4096)
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=0,
        help="exact UTF-8 body quota for each session archive (0 = deployment policy)",
    )
    parser.add_argument(
        "--codex-result-mode",
        choices=("replace", "context", "stop", "block"),
        default="replace",
        help="Codex result handling mode (default: replace verbose output)",
    )
    args = parser.parse_args(argv)
    try:
        payload = json.load(sys.stdin)
        result = process_post_tool_use(
            payload,
            platform=args.platform,
            result_max_chars=args.result_max_chars,
            max_recall_units=args.max_recall_units,
            max_archive_bytes=args.max_archive_bytes,
            codex_result_mode=args.codex_result_mode,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Hook failure is intentionally fail-open: never hide a tool result if
        # persistence, serialization, or payload parsing failed.  This broad
        # catch belongs only at the host process boundary; the library APIs
        # retain their precise exceptions.
        print(f"onecomp ARC hook skipped: {exc}", file=sys.stderr)
        result = {}
    print(json.dumps(result, ensure_ascii=False))


def recall_main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Recall one ARC hook observation")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--obs-id", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chunk-index", type=int)
    mode.add_argument("--range", dest="char_range")
    parser.add_argument("--max-recall-units", type=int, default=4096)
    args = parser.parse_args(argv)
    if not _OBS_ID_RE.fullmatch(args.obs_id):
        parser.error("--obs-id must be 4-128 hexadecimal characters")
    archive_path = _archive_path(args.session_id)
    if not archive_path.is_file():
        parser.error(f"session archive does not exist: {archive_path}")
    try:
        # Agent sandboxes commonly allow the hook process to write its cache
        # while tool commands can only read it. Exact recovery needs no schema
        # mutation or telemetry fsync, so open the committed archive read-only.
        with SQLiteObservationArchive(archive_path, read_only=True) as archive:
            service = ArcCoreService(archive, max_recall_units=args.max_recall_units)
            if args.chunk_index is not None:
                body = service.recall_text_chunk(args.obs_id, args.chunk_index).text
            elif args.char_range is not None:
                try:
                    start_text, end_text = args.char_range.split(":", 1)
                    start, end = int(start_text), int(end_text)
                except (ValueError, TypeError):
                    parser.error("--range must be START:END")
                body = service.recall_range(args.obs_id, start, end)
            else:
                body = service.recall_full(args.obs_id)
    except RecallRequiresChunking as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3) from exc
    except (FormalArcError, OSError, ValueError) as exc:
        parser.exit(2, f"onecomp-arc-recall: {type(exc).__name__}: {exc}\n")
    sys.stdout.write(body)


def main(argv: Optional[list[str]] = None) -> None:
    """Module entry point used by generated host plugins."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"hook", "recall"}:
        raise SystemExit("usage: python -m onecomp.compaction.adapters.hooks {hook|recall} ...")
    command, remainder = arguments[0], arguments[1:]
    if command == "hook":
        hook_main(remainder)
    else:
        recall_main(remainder)


if __name__ == "__main__":
    main()


__all__ = ["hook_main", "main", "process_post_tool_use", "recall_main"]
