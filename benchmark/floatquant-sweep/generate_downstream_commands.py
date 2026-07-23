"""Generate lm-eval Slurm commands for exported checkpoints."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

DEFAULT_TASKS = (
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "ifeval",
    "gsm8k",
    "mmlu_pro",
)


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_commands(args: argparse.Namespace) -> list[str]:
    task_groups = [args.tasks or ",".join(DEFAULT_TASKS)]
    if args.split_tasks:
        task_groups = [task.strip() for task in task_groups[0].split(",") if task.strip()]
    commands = []
    for spec in args.checkpoint:
        tag, path = spec.split("=", 1)
        for tasks in task_groups:
            safe_tasks = tasks.replace(",", "__").replace("/", "_")
            output_path = Path(args.output_dir) / tag
            if args.split_tasks:
                output_path = output_path / safe_tasks
            if args.backend == "vllm":
                model_args = ",".join(
                    [
                        f"pretrained={path}",
                        "trust_remote_code=True",
                        f"gpu_memory_utilization={args.gpu_memory_utilization}",
                        f"max_model_len={args.max_model_len}",
                        f"enforce_eager={str(args.enforce_eager)}",
                    ]
                )
                model = "vllm"
            else:
                model_args = ",".join(
                    [
                        f"pretrained={path}",
                        "trust_remote_code=True",
                        "dtype=auto",
                    ]
                )
                model = "hf"
            cmd = [
                _quote(args.python),
                "-m",
                "lm_eval",
                "--model",
                model,
                "--model_args",
                _quote(model_args),
                "--tasks",
                _quote(tasks),
                "--batch_size",
                _quote(args.batch_size),
                "--output_path",
                _quote(output_path),
            ]
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            commands.append(" ".join(cmd))
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="TAG=CHECKPOINT_PATH.",
    )
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--tasks", default="")
    parser.add_argument(
        "--split-tasks",
        action="store_true",
        help="Emit one Slurm command per checkpoint/task instead of grouping all tasks.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--output-dir", default="outputs/floatquant-downstream")
    parser.add_argument("--commands-file", default="outputs/floatquant-downstream/commands.txt")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=4096)
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
