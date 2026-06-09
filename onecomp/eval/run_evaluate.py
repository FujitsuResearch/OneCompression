"""Hydra-driven evaluation pipeline (single entry point).

This is the only CLI in the package. It is registered as
onecomp-eval in pyproject.toml:

    onecomp-eval model.path=/path/to/model
    onecomp-eval model.path=... evals.mt_bench.enabled=false

Internally it just sets up logging, validates the model path, and hands
off to onecomp.eval.orchestrator.runner.run_pipeline.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import logging
import sys
from logging import getLogger
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .orchestrator import run_pipeline
from .schema import EvalConfig
from .utils.secrets import (
    drop_sensitive_fields,
    redact_config_for_log,
    sanitize_hydra_run_dir,
    warn_on_sensitive_overrides,
)

logger = getLogger(__name__)

CONFIG_PATH = "conf"
CONFIG_NAME = "eval_config"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.3")
def main(cfg: DictConfig) -> None:
    warn_on_sensitive_overrides()
    sanitize_hydra_run_dir()

    cfg = OmegaConf.merge(OmegaConf.structured(EvalConfig), cfg)
    drop_sensitive_fields(cfg)
    _setup_logging(cfg.log_level)

    logger.info("============================================================")
    logger.info(" Resolved configuration")
    logger.info(
        "============================================================\n%s",
        OmegaConf.to_yaml(redact_config_for_log(cfg), resolve=True),
    )

    _validate(cfg)

    summary = run_pipeline(cfg)

    logger.info("============================================================")
    logger.info(" Pipeline complete")
    logger.info("============================================================")
    for entry in summary.get("all_evals", []):
        logger.info(
            "  %-15s %-8s  %s",
            entry["eval_name"],
            entry["status"],
            entry.get("error", "") or "",
        )
    logger.info("Summary written to %s/summary.{json,csv}", cfg.output_dir)


def _validate(cfg: DictConfig) -> None:
    model_path = Path(cfg.model.path)
    if not model_path.exists():
        logger.error("Model path does not exist: %s", model_path)
        sys.exit(2)


def _setup_logging(level: str) -> None:
    try:
        from onecomp import setup_logger

        setup_logger()
    except ImportError:
        pass
    logging.getLogger().setLevel(level.upper())


if __name__ == "__main__":
    main()
