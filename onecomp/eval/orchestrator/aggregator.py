"""Aggregate per-evaluator result.json files into a unified summary.

The aggregator's contract is intentionally minimal:

- It reads every <output_dir>/<eval_name>/result.json whose schema
  matches TaskResult.
- It flattens scores into one row per (model, evaluator, score key)
  for the CSV output.
- It writes <output_dir>/summary.json and <output_dir>/summary.csv
  according to cfg.summary.formats.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Iterable

from ..schema import JST, TaskResult

logger = getLogger(__name__)


def aggregate_results(
    *,
    output_dir: Path,
    results: Iterable[TaskResult],
    include: list[str] | None = None,
    formats: list[str] | None = None,
) -> dict:
    """Collate per-evaluator results into a summary dict (and write files).

    Args:
        output_dir: Run-level output directory.
        results: Iterable of TaskResult.
        include: Optional whitelist of eval_name to fold into the
            summary; None keeps every successful result.
        formats: Output formats ("json", "csv"). Defaults to
            both.

    Returns:
        The summary dict that is also written to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["json", "csv"]

    results_list = list(results)
    selected = _select(results_list, include)

    summary = _build_summary(results_list, selected)

    if "json" in formats:
        path = output_dir / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        logger.info("Wrote %s", path)

    if "csv" in formats:
        rows = _to_csv_rows(selected)
        path = output_dir / "summary.csv"
        _write_csv(path, rows)
        logger.info("Wrote %s", path)

    return summary


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _select(
    results: list[TaskResult],
    include: list[str] | None,
) -> list[TaskResult]:
    """Filter results that contribute to the summary."""
    selected: list[TaskResult] = []
    for r in results:
        if include is not None and r.eval_name not in include:
            continue
        if r.status != "success":
            logger.warning(
                "Excluding %s from summary (status=%s)",
                r.eval_name,
                r.status,
            )
            continue
        selected.append(r)
    return selected


def _build_summary(
    all_results: list[TaskResult],
    selected: list[TaskResult],
) -> dict:
    """Construct the summary dict written to summary.json."""
    return {
        "timestamp": datetime.now(JST).isoformat(timespec="seconds"),
        "selected_evals": [r.eval_name for r in selected],
        "all_evals": [
            {
                "eval_name": r.eval_name,
                "status": r.status,
                "model": r.model,
                "error": r.error,
            }
            for r in all_results
        ],
        "scores": {
            r.eval_name: {
                "model": r.model,
                "scores": r.scores,
                "artifacts": r.artifacts,
                "metadata": r.metadata,
            }
            for r in selected
        },
        "raw_results": [asdict(r) for r in all_results],
    }


def _to_csv_rows(selected: list[TaskResult]) -> list[dict]:
    """Flatten scores into a long-format table for the CSV output."""
    rows: list[dict] = []
    for r in selected:
        for key, value in _walk_scores(r.scores):
            rows.append(
                {
                    "eval_name": r.eval_name,
                    "model": r.model,
                    "metric": key,
                    "value": value,
                }
            )
        if not r.scores:
            rows.append(
                {
                    "eval_name": r.eval_name,
                    "model": r.model,
                    "metric": "(no scores)",
                    "value": "",
                }
            )
    return rows


def _walk_scores(scores: dict, prefix: str = "") -> Iterable[tuple[str, object]]:
    """Yield ("category.subkey", value) pairs for nested score dicts."""
    for k, v in scores.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _walk_scores(v, prefix=f"{key}.")
        else:
            yield key, v


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["eval_name", "model", "metric", "value"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_results_from_dir(output_dir: Path) -> list[TaskResult]:
    """Glob every <output_dir>/*/result.json and load them."""
    output_dir = Path(output_dir)
    results: list[TaskResult] = []
    for path in sorted(output_dir.glob("*/result.json")):
        try:
            results.append(TaskResult.load(path))
        except (OSError, ValueError, TypeError) as e:
            logger.warning("Skipping unreadable result %s: %s", path, e)
    return results
