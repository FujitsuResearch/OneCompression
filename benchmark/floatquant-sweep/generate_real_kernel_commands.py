"""Generate Slurm-array commands for real-kernel FloatQuant evaluation."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from experiment_specs import (
    REAL_KERNEL_QEP_COMPATIBLE_MODES,
    REAL_KERNEL_W4A4_CAPABLE_MODES,
    real_kernel_modes_for_profile,
)


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _w4a4_modes(args: argparse.Namespace, modes: tuple[str, ...]) -> set[str]:
    if not args.include_w4a4:
        return set()
    if args.w4a4_modes == "all":
        return {mode for mode in modes if mode in REAL_KERNEL_W4A4_CAPABLE_MODES}
    selected = {item.strip() for item in args.w4a4_modes.split(",") if item.strip()}
    unknown = sorted(selected - set(modes))
    if unknown:
        raise SystemExit(f"--w4a4-modes contains mode(s) not in profile: {', '.join(unknown)}")
    invalid = sorted(mode for mode in selected if mode not in REAL_KERNEL_W4A4_CAPABLE_MODES)
    if invalid:
        raise SystemExit(f"--w4a4-modes must be NVFP4 modes only: {', '.join(invalid)}")
    return selected


def build_commands(args: argparse.Namespace) -> list[str]:
    commands = []
    modes = real_kernel_modes_for_profile(args.profile)
    w4a4_modes = _w4a4_modes(args, modes)
    for model_spec in args.model:
        model_tag, model_path = model_spec.split("=", 1)
        for mode in modes:
            qep_values = (
                (False, True)
                if args.include_qep and mode in REAL_KERNEL_QEP_COMPATIBLE_MODES
                else (False,)
            )
            for qep in qep_values:
                base = [
                    _quote(args.python),
                    "real_kernel_benchmark.py",
                    "--model-path",
                    _quote(model_path),
                    "--model-tag",
                    _quote(model_tag),
                    "--mode",
                    mode,
                    "--output-dir",
                    _quote(args.output_dir),
                    "--calibration-samples",
                    str(args.calibration_samples),
                    "--calibration-max-length",
                    str(args.calibration_max_length),
                    "--ppl-max-length",
                    str(args.ppl_max_length),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--skip-existing",
                ]
                if args.ppl_max_samples is not None:
                    base.extend(["--ppl-max-samples", str(args.ppl_max_samples)])
                if args.num_layers is not None:
                    base.extend(["--num-layers", str(args.num_layers)])
                if args.enforce_eager:
                    base.append("--enforce-eager")
                if qep:
                    base.append("--qep")
                if model_tag.endswith("7b") and args.speed_batches:
                    base.extend(["--speed-batches", args.speed_batches])

                commands.append(" ".join(base))

                if mode in w4a4_modes:
                    w4a4 = list(base)
                    w4a4.append("--w4a4")
                    commands.append(" ".join(w4a4))
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="MODEL_TAG=MODEL_PATH, e.g. qwen25_05b=/path/to/model.",
    )
    parser.add_argument("--profile", choices=("smoke", "native", "full"), default="smoke")
    parser.add_argument("--include-qep", action="store_true")
    parser.add_argument("--include-w4a4", action="store_true")
    parser.add_argument(
        "--w4a4-modes",
        default="nvfp4_rtn_sweep",
        help=(
            "Comma-separated NVFP4 modes for W4A4 companion runs, or 'all'. "
            "The default preserves the native same-weight RTN+sweep comparison; "
            "use 'all' for strongest full-method W4A4 coverage."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/floatquant-real-kernel")
    parser.add_argument("--commands-file", default="outputs/floatquant-real-kernel/commands.txt")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--calibration-max-length", type=int, default=512)
    parser.add_argument("--ppl-max-samples", type=int, default=None)
    parser.add_argument("--ppl-max-length", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--speed-batches", default="")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    path = Path(args.commands_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {path}")


if __name__ == "__main__":
    main()
