"""Collect lm-eval outputs into a compact paper JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC_PRIORITY = (
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "f1,none",
    "pass@1,create_test",
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _iter_result_jsons(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("samples_"):
            continue
        try:
            data = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            yield path, data


def _select_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    for key in METRIC_PRIORITY:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in sorted(metrics.items()):
        if key.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def _relative_parts(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    checkpoint = parts[0] if parts else "unknown"
    task_from_path = parts[1] if len(parts) > 2 else ""
    return checkpoint, task_from_path


def collect(root: Path) -> dict[str, dict]:
    collected: dict[str, dict] = {}
    files = []
    for path, data in _iter_result_jsons(root):
        checkpoint, task_from_path = _relative_parts(path, root)
        files.append(str(path))
        for task, metrics in data["results"].items():
            if not isinstance(metrics, dict):
                continue
            chosen = _select_metric(metrics)
            if chosen is None:
                continue
            metric, value = chosen
            entry = {
                "metric": metric,
                "value": value,
                "source": str(path),
            }
            if task_from_path:
                entry["task_dir"] = task_from_path
            collected.setdefault(checkpoint, {})[task] = entry
    return {
        "results": collected,
        "num_checkpoints": len(collected),
        "num_files": len(files),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = collect(args.root)
    _write_json(args.output_json, data)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "num_checkpoints": data["num_checkpoints"],
                "num_files": data["num_files"],
                "tasks_by_checkpoint": {
                    ckpt: sorted(tasks) for ckpt, tasks in data["results"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
