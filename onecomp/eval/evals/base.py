"""Base class / helper for evaluator adapters and their child entrypoints.

Two things live here:

1. EvalAdapter -- describes how the parent orchestrator dispatches
   one evaluator: which module to python -m, how to extract its
   config, and which env vars to inject.
2. child_main -- common argv parsing / config-loading boilerplate
   used by every child run.py so they all behave the same way.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from ..schema import TaskResult

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Parent-side: adapter description
# ---------------------------------------------------------------------------


def _no_env(_cfg: DictConfig) -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class EvalAdapter:
    """Static metadata describing how to dispatch one evaluator subprocess.

    Args:
        name: Short identifier matching the field on cfg.evals
            (e.g. "mt_bench").
        module: Dotted path of the child entrypoint
            (e.g. "onecomp.eval.evals.mt_bench.run").
        extract_config: Callable mapping the full Hydra config onto the
            per-task slice. Use default_extract for the common
            cfg.evals.<name> case.
        extra_env: Optional callable producing additional env vars for
            the child subprocess.
    """

    name: str
    module: str
    extract_config: Callable[[DictConfig], Any]
    extra_env: Callable[[DictConfig], dict[str, str]] = _no_env


def default_extract(name: str) -> Callable[[DictConfig], Any]:
    """Default extractor: return cfg.evals.<name>."""

    def _extract(cfg: DictConfig) -> Any:
        return getattr(cfg.evals, name)

    return _extract


# ---------------------------------------------------------------------------
# Child-side: standard argv + config-loading wrapper
# ---------------------------------------------------------------------------


def _parse_child_argv(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluator subprocess entrypoint",
    )
    parser.add_argument("--config", required=True, help="Path to per-task YAML config")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Run output dir; result.json goes under <output-dir>/<eval_name>/",
    )
    parser.add_argument(
        "--model-name", required=True, help="Model identifier (used for output file naming)"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def child_main(
    eval_name: str,
    run_fn: Callable[[DictConfig, Path, str], TaskResult],
    argv: list[str] | None = None,
) -> int:
    """Common harness for every evaluator's run.py.

    Boilerplate flow:

    1. Parse standard argv (--config, --output-dir, --model-name).
    2. Load the per-task config via OmegaConf.
    3. Call run_fn(task_cfg, output_dir, model_name) to get a TaskResult.
    4. Save the result to <output-dir>/<eval_name>/result.json.

    Errors are caught and turned into a status="failed" TaskResult so
    the parent orchestrator can keep going.

    Args:
        eval_name: Short evaluator name (matches the output subdir).
        run_fn: Callable that actually runs the evaluator.
        argv: Optional argv override (for tests).

    Returns:
        Process exit code (0 on success / skipped, 1 on failure).
    """
    args = _parse_child_argv(argv)
    _configure_logging(args.log_level)

    output_dir = Path(args.output_dir).resolve()
    eval_dir = output_dir / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    result_path = eval_dir / "result.json"

    try:
        task_cfg = OmegaConf.load(args.config)
        result = run_fn(task_cfg, output_dir, args.model_name)
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] evaluator crashed: %s", eval_name, e)
        traceback.print_exc()
        result = TaskResult.create(
            eval_name=eval_name,
            model=args.model_name,
            status="failed",
            error=f"{type(e).__name__}: {e}",
        )
        result.save(result_path)
        return 1

    if result.eval_name != eval_name:
        result.eval_name = eval_name
    result.save(result_path)
    return 0 if result.status in ("success", "skipped") else 1


__all__ = ["EvalAdapter", "default_extract", "child_main"]
