"""Child entrypoint for the throughput evaluator.

Invoked by the orchestrator as:

    python -m onecomp.eval.evals.throughput.run \
        --config <task_config.yaml> \
        --output-dir <run_output_dir> \
        --model-name <model_name>

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import sys
from logging import getLogger
from pathlib import Path

from omegaconf import DictConfig

from ...schema import TaskResult
from ..base import child_main
from .bench import aggregate_trial_metrics, run_throughput_benchmark

logger = getLogger(__name__)

EVAL_NAME = "throughput"


def run_throughput(
    cfg: DictConfig,
    output_dir: Path,
    model_name: str,
) -> TaskResult:
    """Run Chat Completions throughput benchmark and return TaskResult."""
    trials, trials_path, prompt_path = run_throughput_benchmark(
        output_dir=output_dir,
        model_name=model_name,
        prompt_tokens=int(cfg.get("prompt_tokens", 512)),
        max_tokens=int(cfg.get("max_tokens", 512)),
        num_warmup=int(cfg.get("num_warmup", 2)),
        num_trials=int(cfg.get("num_trials", 5)),
        temperature=float(cfg.get("temperature", 0.0)),
        prompt_seed_text=str(cfg.get("prompt_seed_text", "")),
        request_timeout_sec=int(cfg.get("request_timeout_sec", 600)),
        save_responses=bool(cfg.get("save_responses", True)),
        save_warmup_responses=bool(cfg.get("save_warmup_responses", False)),
        min_completion_tokens=int(cfg.get("min_completion_tokens", 32)),
    )

    measured = [t for t in trials if not t.is_warmup]
    failures = [t for t in measured if t.error]
    if failures and len(failures) == len(measured):
        return TaskResult.create(
            eval_name=EVAL_NAME,
            model=model_name,
            status="failed",
            error=f"all {len(measured)} trials failed",
            artifacts={
                "trials": str(trials_path),
                "prompt": str(prompt_path),
            },
            metadata=_run_metadata(cfg),
        )

    scores = aggregate_trial_metrics(trials)
    unhealthy = [t.trial_index for t in measured if not t.response_ok]
    metadata = _run_metadata(cfg)
    metadata["unhealthy_trial_indices"] = unhealthy

    status = "success" if scores.get("n_success", 0) > 0 else "failed"
    error = ""
    if status == "failed":
        error = "no successful measurement trials"
    elif scores.get("response_health_ok", 1.0) < 1.0:
        error = (
            f"unhealthy responses in {len(unhealthy)} trial(s): "
            f"{scores.get('response_issue_counts', {})}"
        )
        logger.warning("[throughput] %s", error)

    return TaskResult.create(
        eval_name=EVAL_NAME,
        model=model_name,
        status=status,
        scores=scores,
        artifacts={
            "trials": str(trials_path),
            "prompt": str(prompt_path),
        },
        metadata=metadata,
        error=error,
    )


def _run_metadata(cfg: DictConfig) -> dict[str, object]:
    return {
        "prompt_tokens": int(cfg.get("prompt_tokens", 512)),
        "max_tokens": int(cfg.get("max_tokens", 512)),
        "num_warmup": int(cfg.get("num_warmup", 2)),
        "num_trials": int(cfg.get("num_trials", 5)),
        "temperature": float(cfg.get("temperature", 0.0)),
        "min_completion_tokens": int(cfg.get("min_completion_tokens", 32)),
        "save_responses": bool(cfg.get("save_responses", True)),
    }


def main(argv: list[str] | None = None) -> int:
    return child_main(EVAL_NAME, run_throughput, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
