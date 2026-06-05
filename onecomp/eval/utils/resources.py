"""Resolution of the bundled MT-Bench data snapshot.

Resolution order:

1. Explicit data_dir (e.g. evals.mt_bench.data_dir Hydra override).
2. MT_BENCH_DATA_DIR environment variable.
3. onecomp/eval/data/mt_bench_en/ (default English snapshot).
4. onecomp/eval/data/mt_bench_jp/ and legacy paths: onecomp/eval/data/, onecomp/eval/data/mt_bench/, data/mt_bench/.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import os
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)

_MT_BENCH_MARKER = "question.jsonl"
_EVAL_DATA = Path(__file__).resolve().parent.parent / "data"
_PKG_MT_BENCH_DATA_CANDIDATES: tuple[Path, ...] = (
    _EVAL_DATA / "mt_bench_en",
    _EVAL_DATA / "mt_bench_jp",
    _EVAL_DATA,
)
_MT_BENCH_REL_PATHS: tuple[Path, ...] = (
    Path("onecomp") / "eval" / "data" / "mt_bench_en",
    Path("onecomp") / "eval" / "data" / "mt_bench_jp",
    Path("onecomp") / "eval" / "data",
    Path("onecomp") / "eval" / "data" / "mt_bench",
    Path("data") / "mt_bench",
)


def find_bundled_mt_bench_data_dir() -> Path | None:
    """Return the bundled MT-Bench data dir, or None if absent."""
    env = os.environ.get("MT_BENCH_DATA_DIR", "").strip()
    if env:
        candidate = Path(env)
        if (candidate / _MT_BENCH_MARKER).is_file():
            return candidate
        logger.warning(
            "MT_BENCH_DATA_DIR=%s does not contain %s; ignoring.",
            env,
            _MT_BENCH_MARKER,
        )

    for candidate in _PKG_MT_BENCH_DATA_CANDIDATES:
        if (candidate / _MT_BENCH_MARKER).is_file():
            return candidate

    here = Path(__file__).resolve()
    for parent in here.parents:
        for rel in _MT_BENCH_REL_PATHS:
            candidate = parent / rel
            if (candidate / _MT_BENCH_MARKER).is_file():
                return candidate

    return None


def resolve_mt_bench_data_dir(explicit: str | os.PathLike[str] | None) -> Path:
    """Resolve the MT-Bench data dir or raise FileNotFoundError."""
    if explicit:
        path = Path(explicit)
        if not (path / _MT_BENCH_MARKER).is_file():
            raise FileNotFoundError(
                f"MT-Bench question file not found: {path / _MT_BENCH_MARKER}. "
                "Set evals.mt_bench.data_dir to a directory that contains "
                "question.jsonl, judge_prompts.jsonl, and reference_answer/."
            )
        return path

    bundled = find_bundled_mt_bench_data_dir()
    if bundled is not None:
        logger.info("Using bundled MT-Bench data at %s", bundled)
        return bundled

    raise FileNotFoundError(
        "Could not locate MT-Bench data. Pass evals.mt_bench.data_dir=... "
        "or set MT_BENCH_DATA_DIR."
    )
