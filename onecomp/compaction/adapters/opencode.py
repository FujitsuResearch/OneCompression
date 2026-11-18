"""
Generate a dependency-free OpenCode plugin for per-tool ARC ingestion.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def render_plugin(
    *,
    python_executable: str = "python3",
    result_max_chars: int = 500,
    max_archive_bytes: int = 0,
) -> str:
    """Render a TypeScript plugin using the host's documented mutable output.

    The generated plugin calls OneComp after each tool execution. Long output
    is committed before ``output.output`` is replaced, while short output is
    left untouched. A compaction hook adds the exact-recall protocol to the
    host's compaction prompt.
    """
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("python_executable must be a non-empty string")
    if (
        not isinstance(result_max_chars, int)
        or isinstance(result_max_chars, bool)
        or result_max_chars < 0
    ):
        raise ValueError("result_max_chars must be a non-negative integer")
    if (
        not isinstance(max_archive_bytes, int)
        or isinstance(max_archive_bytes, bool)
        or max_archive_bytes < 0
    ):
        raise ValueError("max_archive_bytes must be a non-negative integer")
    python_literal = json.dumps(python_executable)
    return f"""import type {{ Plugin }} from "@opencode-ai/plugin"

const pythonExecutable = {python_literal}
const resultMaxChars = {result_max_chars}
const maxArchiveBytes = {max_archive_bytes}
const hookTimeoutMs = 30_000

async function compact(payload: unknown): Promise<Record<string, unknown>> {{
  const process = Bun.spawn(
    [pythonExecutable, "-m", "onecomp.compaction.adapters.hooks", "hook",
     "--platform", "opencode", "--result-max-chars", String(resultMaxChars),
     "--max-archive-bytes", String(maxArchiveBytes)],
    {{ stdin: "pipe", stdout: "pipe", stderr: "inherit" }},
  )
  process.stdin.write(JSON.stringify(payload))
  process.stdin.end()
  let timedOut = false
  const timeout = setTimeout(() => {{
    timedOut = true
    process.kill()
  }}, hookTimeoutMs)
  try {{
    const stdout = await new Response(process.stdout).text()
    if (timedOut || (await process.exited) !== 0) return {{}}
    const decoded: unknown = JSON.parse(stdout)
    if (decoded === null || Array.isArray(decoded) || typeof decoded !== "object") return {{}}
    return decoded as Record<string, unknown>
  }} finally {{
    clearTimeout(timeout)
  }}
}}

export const OneCompCompaction: Plugin = async () => ({{
  "tool.execute.after": async (input, output) => {{
    if (typeof output.output !== "string" || output.output.length <= resultMaxChars) return
    try {{
      const response = await compact({{
        session_id: input.sessionID,
        tool_use_id: input.callID,
        tool_name: input.tool,
        tool_input: input.args,
        tool_response: output.output,
      }})
      const replacement = response.replacement
      if (typeof replacement === "string") output.output = replacement
    }} catch (error) {{
      // Fail open: preserve the original output unless durable persistence is proven.
      console.warn("OneComp compaction skipped", error)
    }}
  }},
  "experimental.session.compacting": async (_input, output) => {{
    if (!Array.isArray(output.context)) return
    output.context.push(
      "OneComp preserves large tool results under §<id>. Use the configured " +
      "onecomp-arc-recall command for exact bounded recovery; never guess from a preview.",
    )
  }},
}})
"""


def write_plugin(
    path: str | Path,
    *,
    python_executable: str = "python3",
    result_max_chars: int = 500,
    max_archive_bytes: int = 0,
    overwrite: bool = False,
) -> Path:
    """Atomically write a generated plugin without overwriting by default."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"integration file already exists: {destination}")
    payload = render_plugin(
        python_executable=python_executable,
        result_max_chars=result_max_chars,
        max_archive_bytes=max_archive_bytes,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(f"integration file already exists: {destination}") from exc
            temporary.unlink()
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["render_plugin", "write_plugin"]
