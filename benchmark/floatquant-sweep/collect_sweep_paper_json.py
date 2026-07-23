"""Merge split FloatQuant sweep runs into paper-ready PPL grid JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_paper_json import convert

DEFAULT_MODEL_OUTPUTS = {
    "Qwen2.5-0.5B": "ppl_grid_qwen05b.json",
    "Qwen2.5-1.5B": "ppl_grid_qwen15b.json",
    "Qwen2.5-7B": "ppl_grid_qwen7b.json",
}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _parse_model_outputs(values: list[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_MODEL_OUTPUTS)
    for value in values:
        fragment, filename = value.split("=", 1)
        mapping[fragment] = filename
    return mapping


def _infer_output_name(run_dir: Path, mapping: dict[str, str]) -> str | None:
    text = str(run_dir)
    matches = [filename for fragment, filename in mapping.items() if fragment in text]
    if len(matches) == 1:
        return matches[0]
    return None


def _better_entry(old: dict, new: dict) -> dict:
    old_ppl = old.get("wikitext2_ppl")
    new_ppl = new.get("wikitext2_ppl")
    if old_ppl is None:
        return new
    if new_ppl is None:
        return old
    return new if float(new_ppl) < float(old_ppl) else old


def _merge_entry(
    merged: dict[str, dict],
    key: str,
    entry: dict,
    *,
    source: str,
    sources: dict[str, str],
    duplicate_policy: str,
) -> None:
    if key not in merged:
        merged[key] = entry
        sources[key] = source
        return
    if merged[key] == entry:
        return
    if duplicate_policy == "error":
        raise SystemExit(
            f"Duplicate key {key!r} from {source}; existing source is {sources[key]}. "
            "Use --duplicate-policy best-ppl or latest if this is intentional."
        )
    if duplicate_policy == "latest":
        merged[key] = entry
        sources[key] = source
        return
    if duplicate_policy == "best-ppl":
        chosen = _better_entry(merged[key], entry)
        if chosen is entry:
            sources[key] = source
        merged[key] = chosen
        return
    raise AssertionError(f"unknown duplicate policy: {duplicate_policy}")


def collect(args: argparse.Namespace) -> dict[str, dict]:
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    model_outputs = _parse_model_outputs(args.model_output)
    merged_by_file: dict[str, dict] = {}
    sources_by_file: dict[str, dict[str, str]] = {}
    scanned = 0
    skipped = []

    for results_path in sorted(root.rglob("floatquant_sweep_results.json")):
        run_dir = results_path.parent
        output_name = _infer_output_name(run_dir, model_outputs)
        if output_name is None:
            skipped.append(str(run_dir))
            continue
        scanned += 1
        converted = convert(run_dir, args.dataset_name)
        merged = merged_by_file.setdefault(output_name, {})
        sources = sources_by_file.setdefault(output_name, {})
        for key, entry in converted.items():
            _merge_entry(
                merged,
                key,
                entry,
                source=str(run_dir),
                sources=sources,
                duplicate_policy=args.duplicate_policy,
            )

    summary = {
        "root": str(root),
        "scanned_runs": scanned,
        "skipped_runs": skipped,
        "outputs": {},
    }
    for filename, data in sorted(merged_by_file.items()):
        path = out_dir / filename
        _write_json(path, data)
        summary["outputs"][filename] = {
            "path": str(path),
            "num_rows": len(data),
            "keys": sorted(data),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="Root directory containing split sweep runs."
    )
    parser.add_argument("--out-dir", required=True, help="Directory for merged paper JSONs.")
    parser.add_argument("--dataset-name", default="wikitext2")
    parser.add_argument(
        "--model-output",
        action="append",
        default=[],
        help="Path-fragment=output-filename mapping. Defaults cover Qwen2.5 0.5B/1.5B/7B.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "latest", "best-ppl"),
        default="error",
        help="How to handle conflicting duplicate paper keys.",
    )
    return parser.parse_args()


def main() -> int:
    summary = collect(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
