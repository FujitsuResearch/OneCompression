"""Orchestration layer.

- VllmServerManager -- start/stop a vLLM OpenAI HTTP server.
- run_subprocess_eval -- launch one evaluator as a subprocess.
- aggregate_results -- merge per-evaluator result.json files.
- run_pipeline -- end-to-end orchestrator used by run_evaluate.
"""

from .aggregator import aggregate_results
from .runner import run_pipeline
from .server import VllmServerManager
from .subprocess_runner import run_subprocess_eval

__all__ = [
    "VllmServerManager",
    "run_subprocess_eval",
    "aggregate_results",
    "run_pipeline",
]
