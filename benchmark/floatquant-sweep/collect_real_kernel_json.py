"""Collect real-kernel benchmark records into paper JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_specs import (
    REAL_KERNEL_FULL_RECORD_ORDER,
    REAL_KERNEL_NATIVE_05_PAPER_KEYS,
    REAL_KERNEL_NATIVE_7B_PAPER_KEYS,
    real_kernel_record_key,
)


def _load_records(records_dir: Path, model_tag: str) -> dict[str, dict]:
    records = {}
    for path in sorted(records_dir.glob(f"{model_tag}_*.json")):
        with path.open(encoding="utf-8") as f:
            record = json.load(f)
        records[
            real_kernel_record_key(
                record["mode"],
                qep=bool(record.get("qep")),
                w4a4=bool(record.get("w4a4")),
            )
        ] = record
    return records


def _paper_entry(record: dict) -> dict:
    entry = {"status": record.get("status", "FAILED")}
    if record.get("status") != "OK":
        if "error" in record:
            entry["error"] = record["error"]
        return entry

    ppl = record.get("ppl", {})
    if "wikitext2_ppl" in ppl:
        entry["wikitext2_ppl"] = round(float(ppl["wikitext2_ppl"]), 4)
    if "num_scored_tokens" in ppl:
        entry["num_scored_tokens"] = ppl["num_scored_tokens"]
    if "weight_mb" in record:
        entry["weight_mb"] = round(float(record["weight_mb"]), 2)
    if "weight_gb" in record:
        entry["weight_gb"] = round(float(record["weight_gb"]), 2)
    if "checkpoint" in record:
        entry["checkpoint"] = record["checkpoint"]
    return entry


def _speed_entry(record: dict) -> dict:
    entry = _paper_entry(record)
    speed = record.get("speed", {})
    for key in ("bs1", "bs8", "bs32"):
        if key in speed:
            entry[key] = round(float(speed[key]))
    return entry


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            existing.update(data)
            data = existing
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qwen05-tag", default="qwen25_05b")
    parser.add_argument("--qwen7-tag", default="qwen25_7b")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records_dir = Path(args.records_dir)
    out_dir = Path(args.out_dir)

    q05 = _load_records(records_dir, args.qwen05_tag)
    native05 = {
        paper_key: _paper_entry(q05[record_key])
        for record_key, paper_key in REAL_KERNEL_NATIVE_05_PAPER_KEYS.items()
        if record_key in q05
    }
    if native05:
        _write(out_dir / "native_wikitext_results.json", native05)
    full05 = {
        record_key: _paper_entry(q05[record_key])
        for record_key in REAL_KERNEL_FULL_RECORD_ORDER
        if record_key in q05
    }
    if full05:
        _write(out_dir / "real_kernel_full_results.json", full05)

    q7 = _load_records(records_dir, args.qwen7_tag)
    native7 = {
        paper_key: _paper_entry(q7[record_key])
        for record_key, paper_key in REAL_KERNEL_NATIVE_7B_PAPER_KEYS.items()
        if record_key in q7
    }
    if native7:
        _write(out_dir / "native7b_results.json", native7)

    speed7 = {
        paper_key: _speed_entry(q7[record_key])
        for record_key, paper_key in REAL_KERNEL_NATIVE_7B_PAPER_KEYS.items()
        if record_key in q7 and "speed" in q7[record_key]
    }
    if speed7:
        _write(out_dir / "speed7b_full_results.json", speed7)

    full7 = {
        record_key: _paper_entry(q7[record_key])
        for record_key in REAL_KERNEL_FULL_RECORD_ORDER
        if record_key in q7
    }
    if full7:
        _write(out_dir / "real_kernel_full_qwen7b_results.json", full7)

    speed7_full = {
        record_key: _speed_entry(q7[record_key])
        for record_key in REAL_KERNEL_FULL_RECORD_ORDER
        if record_key in q7 and "speed" in q7[record_key]
    }
    if speed7_full:
        _write(out_dir / "speed7b_full_method_results.json", speed7_full)

    print(
        json.dumps(
            {
                "native05": sorted(native05),
                "full05": sorted(full05),
                "native7": sorted(native7),
                "speed7": sorted(speed7),
                "full7": sorted(full7),
                "speed7_full": sorted(speed7_full),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
