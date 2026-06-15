"""OneComp Evaluation Harness.

End-to-end evaluation pipeline:

- run_evaluate.main is the Hydra-driven entry point (CLI:
  onecomp-eval). It manages vLLM server lifecycle, dispatches each
  enabled evaluator as a subprocess, and aggregates the per-evaluator
  result files into a unified summary.
- Evaluators live under onecomp.eval.evals and each
  exposes a child run.py that the orchestrator launches via
  python -m. Result files conform to TaskResult.

Currently supported evaluators:

- mt_bench -- MT-Bench (default: English), full pipeline

Copyright 2025-2026 Fujitsu Ltd.
"""

from .orchestrator import (
    VllmServerManager,
    aggregate_results,
    run_pipeline,
    run_subprocess_eval,
)
from .schema import (
    EvalConfig,
    EvalsConfig,
    InferenceConfig,
    ModelConfig,
    MtBenchConfig,
    SummaryConfig,
    TaskResult,
    ThroughputConfig,
    VllmServerConfig,
)

__all__ = [
    "EvalConfig",
    "EvalsConfig",
    "ModelConfig",
    "MtBenchConfig",
    "ThroughputConfig",
    "SummaryConfig",
    "InferenceConfig",
    "TaskResult",
    "VllmServerConfig",
    "VllmServerManager",
    "aggregate_results",
    "run_pipeline",
    "run_subprocess_eval",
]
