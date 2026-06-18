"""Per-evaluator adapters.

Each evaluator lives in its own subpackage and registers an
EvalAdapter describing:

- The dotted module path of the child entrypoint.
- How to slice its config out of the root EvalConfig.
- Optional extra environment variables to inject into the subprocess.

The orchestrator dispatches every adapter listed by
iter_evaluators whose cfg.evals.<name>.enabled is true.
"""

from __future__ import annotations

from .base import EvalAdapter, child_main
from .mt_bench.adapter import ADAPTER as MT_BENCH
from .throughput.adapter import ADAPTER as THROUGHPUT


def iter_evaluators() -> list[EvalAdapter]:
    """Return every registered evaluator adapter."""
    return [MT_BENCH, THROUGHPUT]


__all__ = ["EvalAdapter", "child_main", "iter_evaluators"]
