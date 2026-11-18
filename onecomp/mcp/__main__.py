"""Entry point for the OneComp MCP server: ``python -m onecomp.mcp``.

Starts an MCP server on stdio (newline-delimited JSON-RPC 2.0) exposing
the OneComp tools: estimate_memory, recommend_recipe, quantize, and
evaluate.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import argparse
import os
from pathlib import Path


def main(argv=None):
    """Run the MCP server on stdio until stdin is closed."""
    parser = argparse.ArgumentParser(description="OneComp MCP stdio server")
    parser.add_argument(
        "--protocol",
        choices=("auto", "modern", "legacy"),
        default="auto",
        help="auto accepts initialized legacy and stateless modern requests",
    )
    parser.add_argument(
        "--allow-unauthenticated-sessions",
        action="store_true",
        help="compatibility mode: permit compaction session IDs without a token",
    )
    parser.add_argument(
        "--compaction-only",
        action="store_true",
        help="serve dependency-free formal compaction tools without loading model tooling",
    )
    parser.add_argument(
        "--wal-root",
        type=Path,
        help="confine persistent compaction archives below this server-owned directory",
    )
    arguments = parser.parse_args(argv)

    if arguments.wal_root is not None:
        wal_root = arguments.wal_root.expanduser().resolve()
        existed = wal_root.exists()
        if existed and not wal_root.is_dir():
            parser.error(f"--wal-root is not a directory: {wal_root}")
        try:
            wal_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.error(f"cannot create --wal-root {wal_root}: {exc}")
        if not existed:
            try:
                os.chmod(wal_root, 0o700)
            except OSError:
                pass
        os.environ["ONECOMP_MCP_WAL_ROOT"] = str(wal_root)

    # Parse first so ``--help`` remains dependency-free. Model tooling is loaded
    # only for the full server; the compaction-only path imports no torch,
    # transformers, model, or accelerator package.
    from .compaction_tools import (  # pylint: disable=import-outside-toplevel
        build_compaction_tools,
        close_all_sessions,
    )
    from .modern_protocol import (  # pylint: disable=import-outside-toplevel
        CompatibleMCPServer,
        ModernMCPServer,
    )
    from .protocol import MCPServer  # pylint: disable=import-outside-toplevel

    server_class = {
        "auto": CompatibleMCPServer,
        "modern": ModernMCPServer,
        "legacy": MCPServer,
    }[arguments.protocol]
    if arguments.compaction_only:
        tools = build_compaction_tools(
            require_session_token=not arguments.allow_unauthenticated_sessions,
            profile="formal-v1",
        )
    else:
        from .tools import build_tools  # pylint: disable=import-outside-toplevel

        tools = build_tools(
            require_session_token=not arguments.allow_unauthenticated_sessions,
            compaction_profile="formal-v1",
        )
    server = server_class(tools=tools)
    try:
        server.serve_forever()
    finally:
        close_all_sessions()


if __name__ == "__main__":
    main()
