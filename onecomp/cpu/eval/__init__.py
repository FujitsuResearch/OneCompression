"""CPU-side evaluation tools for GGUF models.

Modules:
    inspect_gguf -- per-tensor quant types, size and effective bit-width
    perplexity   -- CPU perplexity of a GGUF model on text
    parity       -- HF (PyTorch) vs GGUF (llama.cpp) output agreement
    benchmark    -- CPU prefill / decode throughput

Copyright 2025-2026 Fujitsu Ltd.
"""

from onecomp.cpu.eval.benchmark import BenchmarkResult, benchmark, benchmark_model
from onecomp.cpu.eval.inspect_gguf import GGUFReport, format_report, inspect_gguf
from onecomp.cpu.eval.parity import (
    GreedyParity,
    TeacherForcedParity,
    compare_greedy,
    compare_logits,
    teacher_forced_parity,
)
from onecomp.cpu.eval.perplexity import (
    PerplexityResult,
    perplexity,
    perplexity_from_tokens,
)

__all__ = [
    "inspect_gguf",
    "format_report",
    "GGUFReport",
    "perplexity",
    "perplexity_from_tokens",
    "PerplexityResult",
    "teacher_forced_parity",
    "compare_logits",
    "compare_greedy",
    "TeacherForcedParity",
    "GreedyParity",
    "benchmark",
    "benchmark_model",
    "BenchmarkResult",
]
