"""Dependency-isolation checks for the formal-compaction MCP entry point."""

import json
import subprocess
import sys
from pathlib import Path


def test_compaction_only_mcp_runs_without_site_packages():
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "server/discover",
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        },
    }
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "onecomp.mcp", "--compaction-only"],
        cwd=repository,
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["id"] == 1
    assert response["result"]["supportedVersions"] == ["2026-07-28"]
    assert "torch" not in result.stderr


def test_mcp_help_runs_without_site_packages():
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "onecomp.mcp", "--help"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--compaction-only" in result.stdout
    assert "--wal-root" in result.stdout


def test_ci_compaction_imports_do_not_require_torch():
    repository = Path(__file__).resolve().parents[3]
    script = """
import builtins

real_import = builtins.__import__

def without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch intentionally unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = without_torch
from onecomp.analyzer.compaction import build_run_report
from onecomp.mcp.tools import build_tools
assert callable(build_run_report)
assert callable(build_tools)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_compaction_only_mcp_creates_private_explicit_wal_root(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    wal_root = tmp_path / "private-wal"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "onecomp.mcp",
            "--compaction-only",
            "--wal-root",
            str(wal_root),
        ],
        cwd=repository,
        input="",
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert wal_root.is_dir()
    assert wal_root.stat().st_mode & 0o077 == 0
