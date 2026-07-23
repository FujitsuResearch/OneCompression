"""Generate Slurm-array command files for FloatQuant sweeps.

The benchmark script itself loops over modes/formats/QEP values.  For
large experiments, however, isolating one model/seed/mode/format/QEP
tuple per Slurm task is easier to retry and produces cleaner logs.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from experiment_specs import (
    SWEEP_HESSIAN_MODES,
    sweep_formats_for_profile,
    sweep_modes_for_profile,
)


def _list_arg(values: list[str] | tuple[str, ...]) -> str:
    return "[" + ",".join(values) + "]"


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _qep_values(mode: str, include_qep: bool) -> tuple[str, ...]:
    if include_qep and mode in SWEEP_HESSIAN_MODES:
        return ("false", "true")
    return ("false",)


def build_commands(args: argparse.Namespace) -> list[str]:
    """Build one Hydra command per Slurm array task."""
    modes = sweep_modes_for_profile(args.profile)
    formats = sweep_formats_for_profile(args.profile)
    commands: list[str] = []
    for model in args.model:
        model_name = Path(model).name.replace("/", "_")
        for seed in args.seed:
            for fmt in formats:
                for mode in modes:
                    for qep in _qep_values(mode, args.include_qep):
                        output_dir = (
                            Path(args.output_dir)
                            / args.profile
                            / model_name
                            / f"seed{seed}"
                            / fmt
                            / mode
                            / f"qep_{qep}"
                        )
                        cmd = [
                            _quote(args.python),
                            "quant_benchmark.py",
                            f"model_path={_quote(model)}",
                            f"output_dir={_quote(output_dir)}",
                            f"filters.formats={_list_arg([fmt])}",
                            f"filters.modes={_list_arg([mode])}",
                            f"qep.enabled={_list_arg([qep])}",
                            f"calibration.seed={seed}",
                            f"timing.warmup_runs={args.warmup_runs}",
                            f"timing.repeats={args.repeats}",
                            f"timing.randomize_order=false",
                        ]
                        if args.num_layers is not None:
                            cmd.append(f"floatquant.num_layers={args.num_layers}")
                        commands.append(" ".join(cmd))
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="HF model path/id.")
    parser.add_argument("--seed", action="append", type=int, default=[0])
    parser.add_argument("--profile", choices=("smoke", "default"), default="smoke")
    parser.add_argument("--include-qep", action="store_true")
    parser.add_argument("--output-dir", default="outputs/floatquant-sweep")
    parser.add_argument("--commands-file", default="outputs/floatquant-sweep/commands.txt")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-layers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    output_path = Path(args.commands_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {output_path}")


if __name__ == "__main__":
    main()
