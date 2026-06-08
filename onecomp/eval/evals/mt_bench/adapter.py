"""Parent-side adapter for the MT-Bench evaluator.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from ..base import EvalAdapter, default_extract

ADAPTER = EvalAdapter(
    name="mt_bench",
    module="onecomp.eval.evals.mt_bench.run",
    extract_config=default_extract("mt_bench"),
)
