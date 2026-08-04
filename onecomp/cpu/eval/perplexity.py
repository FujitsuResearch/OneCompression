"""CPU perplexity evaluation for GGUF models via llama-cpp-python.

A light-weight, dependency-free (numpy-only) perplexity estimator. It tokenizes
the input text, slices it into windows of ``n_ctx`` tokens, runs a single
teacher-forced forward per window with ``logits_all=True`` and accumulates the
negative log-likelihood of each next token. This is the standard metric for
checking how much a quantization recipe degrades a model, runnable entirely on
CPU.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

import numpy as np

logger = getLogger(__name__)


@dataclass
class PerplexityResult:
    """Outcome of a perplexity run."""

    perplexity: float
    nll: float  # mean negative log-likelihood (nats)
    n_tokens: int  # number of scored (predicted) tokens
    n_windows: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"perplexity={self.perplexity:.4f} nll={self.nll:.4f} "
            f"tokens={self.n_tokens} windows={self.n_windows}"
        )


def _log_softmax_gather(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return log p(target) for each row of ``logits`` (numerically stable)."""
    logits = logits.astype(np.float64, copy=False)
    m = logits.max(axis=-1, keepdims=True)
    shifted = logits - m
    logZ = np.log(np.exp(shifted).sum(axis=-1)) + m[:, 0]
    chosen = logits[np.arange(logits.shape[0]), targets]
    return chosen - logZ


def perplexity_from_tokens(
    model,
    token_ids: List[int],
    n_ctx: Optional[int] = None,
    stride: Optional[int] = None,
) -> PerplexityResult:
    """Compute perplexity over a token id list using an existing ``LlamaCppModel``.

    Args:
        model: an :class:`onecomp.cpu.inference.LlamaCppModel` built with
            ``logits_all=True`` (required to read per-position logits).
        token_ids: the full token sequence to score.
        n_ctx: window size; defaults to the model's ``n_ctx``.
        stride: window advance; defaults to ``n_ctx`` (non-overlapping windows).

    Returns:
        :class:`PerplexityResult`.
    """
    if not getattr(model, "logits_all", False):
        raise ValueError(
            "perplexity_from_tokens requires logits_all=True when constructing "
            "LlamaCppModel(..., logits_all=True); otherwise only the final "
            "position's logits are valid and perplexity is meaningless."
        )
    llm = model._llm
    if n_ctx is None:
        n_ctx = int(llm.n_ctx())
    if stride is None:
        stride = n_ctx

    total_nll = 0.0
    total_tokens = 0
    n_windows = 0
    n = len(token_ids)
    if n < 2:
        raise ValueError("Need at least 2 tokens to compute perplexity.")

    start = 0
    while start < n - 1:
        window = token_ids[start : start + n_ctx]
        if len(window) < 2:
            break
        llm.reset()
        llm.eval(window)
        scores = np.array(llm.scores[: len(window)], dtype=np.float32)
        if scores.ndim == 1:
            scores = scores.reshape(len(window), -1)
        # position i predicts token i+1
        logits = scores[:-1]
        targets = np.asarray(window[1:], dtype=np.int64)
        logp = _log_softmax_gather(logits, targets)
        total_nll += float(-logp.sum())
        total_tokens += len(targets)
        n_windows += 1
        if stride <= 0:
            break
        start += stride

    mean_nll = total_nll / max(total_tokens, 1)
    return PerplexityResult(
        perplexity=float(math.exp(mean_nll)),
        nll=mean_nll,
        n_tokens=total_tokens,
        n_windows=n_windows,
    )


def perplexity(
    gguf_path: str,
    text: str,
    n_ctx: int = 512,
    stride: Optional[int] = None,
    n_threads: Optional[int] = None,
) -> PerplexityResult:
    """Convenience wrapper: load ``gguf_path`` and score ``text``.

    Args:
        gguf_path: path to a ``.gguf`` model.
        text: raw text to evaluate.
        n_ctx: context window for both the model and the perplexity stride.
        stride: window advance (defaults to ``n_ctx``).
        n_threads: CPU threads (defaults to all cores).
    """
    from onecomp.cpu.inference import LlamaCppModel

    model = LlamaCppModel(gguf_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=True)
    token_ids = model.tokenize(text, add_bos=True)
    return perplexity_from_tokens(model, token_ids, n_ctx=n_ctx, stride=stride)
