"""Convert FloatQuant sweep outputs to the paper asset JSON schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiment_specs import SWEEP_MODE_TO_PAPER_SUFFIX


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _iter_relative_weight_errors(obj: Any):
    if isinstance(obj, dict):
        if obj.get("relative_weight_squared_error") is not None:
            value = obj["relative_weight_squared_error"]
            if value is not None and value >= 0:
                yield math.sqrt(float(value))
        for value in obj.values():
            yield from _iter_relative_weight_errors(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_relative_weight_errors(value)


def _mean_rel_error(stats_path: Path | None) -> float | None:
    if stats_path is None or not stats_path.exists():
        return None
    values = list(_iter_relative_weight_errors(_load_json(stats_path)))
    if not values:
        return None
    return sum(values) / len(values)


def _paper_tag(record: dict) -> str | None:
    fmt = record["fmt"]
    mode = record["name"].removeprefix(f"FloatQuant_{fmt}_")
    for qep_suffix in ("_noqep", "_qep"):
        mode = mode.removesuffix(qep_suffix)
    for strategy_suffix in ("_local", "_full", "_adaptive"):
        mode = mode.removesuffix(strategy_suffix)
    mapped = SWEEP_MODE_TO_PAPER_SUFFIX.get(mode)
    if mapped is None:
        return None
    if record.get("qep") and mapped.startswith("hessian_"):
        mapped = "qep_" + mapped
    return f"{fmt}_{mapped}"


def _record_ppl(record: dict, dataset_name: str) -> tuple[float | None, float | None]:
    dataset = record.get("ppl", {}).get(dataset_name, {})
    quantized = dataset.get(record["name"])
    if quantized is None:
        quantized = dataset.get("dequantized") or dataset.get("quantized")
    original = dataset.get("original")
    return original, quantized


def convert(input_dir: Path, dataset_name: str) -> dict:
    results_path = input_dir / "floatquant_sweep_results.json"
    records = _load_json(results_path)
    output: dict[str, dict[str, float]] = {}
    baseline = None
    for record in records:
        original_ppl, quant_ppl = _record_ppl(record, dataset_name)
        if original_ppl is not None:
            baseline = original_ppl
        tag = _paper_tag(record)
        if tag is None or quant_ppl is None:
            continue
        stats_path = input_dir / f"quantization_statistics_{record['name']}.json"
        entry = {
            "wikitext2_ppl": float(quant_ppl),
            "quant_seconds": float(record.get("elapsed_seconds", 0.0)),
        }
        rel = _mean_rel_error(stats_path)
        if rel is not None:
            entry["mean_rel_error"] = rel
        output[tag] = entry
    if baseline is not None:
        output["fp16_baseline"] = {"wikitext2_ppl": float(baseline)}
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dataset-name", default="wikitext2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted = convert(args.input_dir, args.dataset_name)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(converted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(converted)} paper rows to {args.output_json}")


if __name__ == "__main__":
    main()
