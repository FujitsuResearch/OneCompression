"""CPU throughput / latency benchmark for GGUF models via llama-cpp-python.

Measures prefill (prompt ingestion) and decode (token generation) speed, which
are the two numbers that matter for CPU serving. Reports tokens/sec for each
phase plus end-to-end wall time, averaged over a few runs.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

logger = getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Throughput / latency of one CPU benchmark run."""

    prompt_tokens: int
    gen_tokens: int
    prefill_s: float
    decode_s: float
    n_threads: int

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.prefill_s if self.prefill_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        return self.gen_tokens / self.decode_s if self.decode_s > 0 else 0.0

    def summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "n_threads": self.n_threads,
            "prefill_tok_per_s": round(self.prefill_tps, 2),
            "decode_tok_per_s": round(self.decode_tps, 2),
            "prefill_s": round(self.prefill_s, 4),
            "decode_s": round(self.decode_s, 4),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"prefill={self.prefill_tps:.1f} tok/s ({self.prompt_tokens} tok), "
            f"decode={self.decode_tps:.1f} tok/s ({self.gen_tokens} tok), "
            f"threads={self.n_threads}"
        )


def benchmark_model(
    model,
    prompt: str = "Fujitsu is a Japanese multinational company that",
    gen_tokens: int = 64,
    warmup: int = 1,
) -> BenchmarkResult:
    """Benchmark an existing :class:`onecomp.cpu.inference.LlamaCppModel`.

    Splits timing into the prefill (single eval of the prompt tokens) and the
    decode loop (one token at a time, greedy/argmax).
    """
    import numpy as np

    llm = model._llm
    prompt_ids = model.tokenize(prompt, add_bos=True)

    for _ in range(max(warmup, 0)):
        llm.reset()
        llm.eval(prompt_ids)

    # Prefill: time a single forward over the whole prompt.
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(prompt_ids)
    prefill_s = time.perf_counter() - t0

    # Decode: greedy, one token per step.
    cur_len = len(prompt_ids)
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        logits = np.asarray(llm.scores[cur_len - 1], dtype=np.float32)
        nt = int(logits.argmax())
        llm.eval([nt])
        cur_len += 1
    decode_s = time.perf_counter() - t0

    n_threads = getattr(model, "n_threads", None) or 0
    return BenchmarkResult(
        prompt_tokens=len(prompt_ids),
        gen_tokens=gen_tokens,
        prefill_s=prefill_s,
        decode_s=decode_s,
        n_threads=int(n_threads),
    )


def benchmark(
    gguf_path: str,
    prompt: str = "Fujitsu is a Japanese multinational company that",
    gen_tokens: int = 64,
    n_ctx: int = 2048,
    n_threads: Optional[int] = None,
    runs: int = 1,
) -> List[BenchmarkResult]:
    """Convenience wrapper: load ``gguf_path`` and benchmark it ``runs`` times."""
    from onecomp.cpu.inference import LlamaCppModel

    model = LlamaCppModel(gguf_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=True)
    return [benchmark_model(model, prompt=prompt, gen_tokens=gen_tokens) for _ in range(runs)]
