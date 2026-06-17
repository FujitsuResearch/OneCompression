"""Launch a single evaluator as a subprocess and recover its result file.

The parent (orchestrator) writes the evaluator's slice of the config to
disk, then runs python -m <module> --config <path> --output-dir <dir>.
The child is responsible for writing its result.json per the
TaskResult schema.

If the child crashes or exits non-zero without producing a result file,
the parent synthesises a status="failed" TaskResult so the
aggregator always sees a uniform input.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from logging import getLogger
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..schema import TaskResult
from ..utils.secrets import strip_sensitive_fields

logger = getLogger(__name__)

RESULT_FILE = "result.json"
CONFIG_FILE = "task_config.yaml"
LOG_FILE = "subprocess.log"


def run_subprocess_eval(
    *,
    eval_name: str,
    module: str,
    task_cfg: Any,
    model_name: str,
    output_root: Path,
    env_overrides: dict[str, str] | None = None,
    timeout_sec: int = 7200,
) -> TaskResult:
    """Run one evaluator as a subprocess and return its TaskResult.

    Args:
        eval_name: Short identifier (e.g. "mt_bench"). Used for the
            per-eval output directory.
        module: Dotted module path of the child entrypoint
            (e.g. "onecomp.eval.evals.mt_bench.run").
        task_cfg: Subtree of the Hydra config for this evaluator. Any
            omegaconf.DictConfig or plain dataclass works.
        model_name: Model identifier (forwarded to the child for output
            file naming).
        output_root: Parent output directory; the child writes under
            <output_root>/<eval_name>/.
        env_overrides: Additional environment variables passed to the
            child (e.g. OPENAI_BASE_URL for the vLLM HTTP endpoint).
        timeout_sec: Wall-clock limit. -1 disables.

    Returns:
        TaskResult. status is either the child's reported
        value or "failed" if the child crashed.
    """
    task_dir = output_root / eval_name
    task_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = task_dir / CONFIG_FILE
    result_path = task_dir / RESULT_FILE
    log_path = task_dir / LOG_FILE

    # Dump the per-task config as plain YAML so the child does not need
    # to depend on Hydra. Sensitive values (e.g. API keys) are stripped.
    safe_cfg = strip_sensitive_fields(task_cfg)
    OmegaConf.save(safe_cfg, cfg_path)

    if result_path.exists():
        result_path.unlink()

    argv = [
        sys.executable,
        "-m",
        module,
        "--config",
        str(cfg_path),
        "--output-dir",
        str(output_root),
        "--model-name",
        model_name,
    ]

    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)

    logger.info("[%s] launching subprocess: %s", eval_name, " ".join(argv))
    logger.info("[%s] log: %s", eval_name, log_path)

    start = time.monotonic()
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_fh:
        try:
            completed = subprocess.run(
                argv,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=None if timeout_sec < 0 else timeout_sec,
                check=False,
            )
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = -signal_or_default()
            timed_out = True

    duration = time.monotonic() - start
    logger.info(
        "[%s] subprocess exit=%d duration=%.1fs (timed_out=%s)",
        eval_name,
        returncode,
        duration,
        timed_out,
    )

    if result_path.exists():
        try:
            result = TaskResult.load(result_path)
            # Augment with parent-side metadata
            result.metadata.setdefault("duration_sec", round(duration, 2))
            result.metadata.setdefault("subprocess_exit_code", returncode)
            if timed_out:
                result.status = "failed"
                result.error = (result.error or "") + " timed_out"
            result.save(result_path)
            return result
        except (OSError, ValueError) as e:
            logger.warning(
                "[%s] result.json was unreadable (%s); synthesising failure",
                eval_name,
                e,
            )

    # Either the child crashed before writing the result, or the file is
    # malformed. Synthesise a failure record so the aggregator sees a
    # uniform schema.
    failure = TaskResult.create(
        eval_name=eval_name,
        model=model_name,
        status="failed",
        error=(
            f"subprocess exited with code {returncode} and produced no "
            f"valid result.json (see {log_path})" + (" [timed out]" if timed_out else "")
        ),
        metadata={
            "duration_sec": round(duration, 2),
            "subprocess_exit_code": returncode,
            "log": str(log_path),
        },
    )
    failure.save(result_path)
    return failure


def signal_or_default() -> int:
    """SIGTERM number where available; fallback constant otherwise."""
    try:
        import signal

        return int(signal.SIGTERM)
    except Exception:
        return 15
