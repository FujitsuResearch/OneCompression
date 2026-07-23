"""Compare common tensors between two exported safetensors checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def _tensor_hash(handle, key: str) -> str:
    tensor = handle.get_tensor(key).detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _safetensors_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "model.safetensors"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find model.safetensors under {path}")


def compare(left: Path, right: Path) -> dict:
    left_path = _safetensors_path(left)
    right_path = _safetensors_path(right)
    with (
        safe_open(left_path, framework="pt", device="cpu") as lhs,
        safe_open(right_path, framework="pt", device="cpu") as rhs,
    ):
        left_keys = set(lhs.keys())
        right_keys = set(rhs.keys())
        common = sorted(left_keys & right_keys)
        mismatched = []
        for key in common:
            lt = lhs.get_tensor(key)
            rt = rhs.get_tensor(key)
            if lt.shape != rt.shape or lt.dtype != rt.dtype:
                mismatched.append(
                    {
                        "key": key,
                        "left": str((lt.shape, lt.dtype)),
                        "right": str((rt.shape, rt.dtype)),
                    }
                )
                continue
            if _tensor_hash(lhs, key) != _tensor_hash(rhs, key):
                mismatched.append({"key": key, "reason": "sha256 mismatch"})
        return {
            "left": str(left_path),
            "right": str(right_path),
            "num_left_tensors": len(left_keys),
            "num_right_tensors": len(right_keys),
            "num_common_tensors": len(common),
            "left_only": sorted(left_keys - right_keys),
            "right_only": sorted(right_keys - left_keys),
            "num_mismatched_common_tensors": len(mismatched),
            "mismatched_common_tensors": mismatched[:50],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare(args.left, args.right)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["num_mismatched_common_tensors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
