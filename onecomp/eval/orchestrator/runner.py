"""End-to-end orchestration: server lifecycle, evaluator dispatch, aggregation.

This is the single function called by onecomp.eval.run_evaluate.
It owns the vLLM server context, dispatches every enabled evaluator
through run_subprocess_eval, then runs the aggregator.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import os
from logging import getLogger
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..evals import iter_evaluators
from ..schema import TaskResult
from .aggregator import aggregate_results
from .server import VllmServerManager
from .subprocess_runner import run_subprocess_eval

logger = getLogger(__name__)


def run_pipeline(cfg: DictConfig) -> dict:
    """Execute the configured evaluation pipeline.

    Args:
        cfg: Resolved Hydra config (root EvalConfig).

    Returns:
        The aggregator summary dict (also written to summary.json).
    """
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.model.path
    model_name = cfg.model.name or _default_model_name(model_path)
    logger.info("Model path: %s", model_path)
    logger.info("Model name: %s", model_name)
    logger.info("Output dir: %s", output_dir)

    enabled = _select_enabled_evaluators(cfg)
    if not enabled:
        logger.warning("No evaluators enabled; nothing to do.")
        return aggregate_results(
            output_dir=output_dir,
            results=[],
            include=_summary_include(cfg),
            formats=list(cfg.summary.formats),
        )

    logger.info("Enabled evaluators: %s", [a.name for a in enabled])

    server_cm, env_overrides = _build_server_context(cfg, model_path, output_dir)

    results: list[TaskResult] = []
    with server_cm as server:
        env_overrides.update(
            {
                "OPENAI_BASE_URL": server.base_url,
                "OPENAI_API_KEY": server.api_key,
            }
        )
        judge_key = _judge_api_key_from_parent()
        if judge_key:
            env_overrides["ONECOMP_JUDGE_OPENAI_API_KEY"] = judge_key
        logger.info("Evaluators will hit %s", server.base_url)

        for adapter in enabled:
            task_cfg = adapter.extract_config(cfg)
            child_env = dict(env_overrides)
            child_env.update(adapter.extra_env(cfg))

            logger.info("=== Running %s ===", adapter.name)
            result = run_subprocess_eval(
                eval_name=adapter.name,
                module=adapter.module,
                task_cfg=task_cfg,
                model_name=model_name,
                output_root=output_dir,
                env_overrides=child_env,
                timeout_sec=int(getattr(task_cfg, "subprocess_timeout_sec", 7200)),
            )
            logger.info(
                "=== %s done: status=%s ===",
                adapter.name,
                result.status,
            )
            results.append(result)

    summary = aggregate_results(
        output_dir=output_dir,
        results=results,
        include=_summary_include(cfg),
        formats=list(cfg.summary.formats),
    )
    return summary


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_server_context(
    cfg: DictConfig,
    model_path: str,
    output_dir: Path,
) -> tuple[VllmServerManager, dict[str, str]]:
    """Return (vLLM server manager, env overrides for evaluator subprocesses)."""
    mode = str(cfg.inference.mode)
    if mode != "vllm_server":
        raise ValueError(f"Unsupported inference.mode={mode!r}; only 'vllm_server' is supported.")

    from ..schema import InferenceConfig

    server_cfg = OmegaConf.to_object(cfg.inference)
    if not isinstance(server_cfg, InferenceConfig):
        server_cfg = InferenceConfig(**OmegaConf.to_container(cfg.inference, resolve=True))  # type: ignore[arg-type]
    manager = VllmServerManager(
        cfg=server_cfg,
        model_path=model_path,
        log_dir=output_dir / "_logs",
    )
    env: dict[str, str] = {"ONECOMP_INFERENCE_MODE": "vllm_server"}
    return manager, env


def _select_enabled_evaluators(cfg: DictConfig) -> list:
    """Return adapter objects for every evaluator whose enabled is true."""
    enabled = []
    for adapter in iter_evaluators():
        sub = getattr(cfg.evals, adapter.name, None)
        if sub is not None and bool(getattr(sub, "enabled", False)):
            enabled.append(adapter)
    return enabled


def _summary_include(cfg: DictConfig) -> list[str] | None:
    inc = cfg.summary.include
    if inc is None:
        return None
    return list(inc)


def _default_model_name(model_path: str) -> str:
    p = Path(model_path)
    parent = p.parent.name
    return f"{parent}_{p.name}" if parent else p.name


def _judge_api_key_from_parent() -> str:
    """Capture the judge key before OPENAI_API_KEY is overwritten for vLLM."""
    for name in ("ONECOMP_JUDGE_OPENAI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value and value != "EMPTY":
            return value
    return ""
