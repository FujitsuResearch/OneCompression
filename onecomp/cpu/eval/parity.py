"""Quantify how closely a GGUF model reproduces its HF (PyTorch) counterpart.

Because the direct GPTQ -> GGUF export repacks the *same* integer codes, both
engines run mathematically identical weights; the only differences come from the
kernels (llama.cpp quantizes activations to 8-bit and accumulates in fp32, HF
uses float matmuls). This module measures the residual gap with:

  * teacher-forced parity: feed the *same* token ids to both engines and compare
    per-position top-1 agreement, the logit Pearson correlation and MSE;
  * greedy parity: greedily decode from both and report the first divergence.

These metrics are the honest way to validate the conversion: exact bit-for-bit
logit equality between a CPU integer kernel and a float kernel is not physically
attainable, but token-level agreement should be ~100% for a lossless repack.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional, Sequence

import numpy as np

logger = getLogger(__name__)


@dataclass
class TeacherForcedParity:
    """Per-position agreement of two engines over a fixed token sequence."""

    n_positions: int
    top1_agreement: float  # fraction of positions whose argmax matches
    pearson: float
    mse: float
    last_argmax_match: bool

    def summary(self) -> dict:
        return {
            "n_positions": self.n_positions,
            "top1_agreement_pct": round(self.top1_agreement * 100, 4),
            "pearson": round(self.pearson, 6),
            "mse": round(self.mse, 6),
            "last_argmax_match": self.last_argmax_match,
        }


@dataclass
class GreedyParity:
    """Greedy-decode agreement of two engines from the same prompt."""

    hf_ids: List[int]
    gguf_ids: List[int]
    first_divergence: int  # index of first mismatch (== len if identical)
    n_new: int

    @property
    def identical(self) -> bool:
        return self.first_divergence >= min(len(self.hf_ids), len(self.gguf_ids))

    def summary(self) -> dict:
        return {
            "n_new": self.n_new,
            "first_divergence": self.first_divergence,
            "identical": self.identical,
            "hf_ids": self.hf_ids,
            "gguf_ids": self.gguf_ids,
        }


def teacher_forced_parity(hf_logits: np.ndarray, gguf_logits: np.ndarray) -> TeacherForcedParity:
    """Compare two ``(n_positions, vocab)`` logit matrices position by position."""
    v = min(hf_logits.shape[-1], gguf_logits.shape[-1])
    hf = np.asarray(hf_logits)[:, :v].astype(np.float64)
    lc = np.asarray(gguf_logits)[:, :v].astype(np.float64)
    n = min(hf.shape[0], lc.shape[0])
    hf, lc = hf[:n], lc[:n]

    hf_arg = hf.argmax(-1)
    lc_arg = lc.argmax(-1)
    agree = float((hf_arg == lc_arg).mean()) if n else 0.0
    pearson = float(np.corrcoef(hf.reshape(-1), lc.reshape(-1))[0, 1]) if n else float("nan")
    mse = float(np.mean((hf - lc) ** 2)) if n else float("nan")
    return TeacherForcedParity(
        n_positions=n,
        top1_agreement=agree,
        pearson=pearson,
        mse=mse,
        last_argmax_match=bool(hf_arg[-1] == lc_arg[-1]) if n else False,
    )


def gguf_logits_for_tokens(model, token_ids: Sequence[int]) -> np.ndarray:
    """Per-position logits ``(len, vocab)`` from a ``LlamaCppModel`` (logits_all=True)."""
    if not getattr(model, "logits_all", False):
        raise ValueError(
            "Teacher-forced parity needs per-position logits; build the GGUF "
            "model with logits_all=True (LlamaCppModel(..., logits_all=True)). "
            "Without it only the last position is valid and parity is meaningless."
        )
    llm = model._llm
    llm.reset()
    llm.eval(list(token_ids))
    scores = np.array(llm.scores[: len(token_ids)], dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(len(token_ids), -1)
    return scores


def gguf_greedy(model, token_ids: Sequence[int], max_new: int) -> List[int]:
    """Greedy decode ``max_new`` tokens from a ``LlamaCppModel`` via manual argmax."""
    llm = model._llm
    cur = list(token_ids)
    llm.reset()
    llm.eval(cur)
    new: List[int] = []
    for _ in range(max_new):
        logits = np.asarray(llm.scores[len(cur) - 1], dtype=np.float64)
        nt = int(logits.argmax())
        new.append(nt)
        cur.append(nt)
        llm.eval([nt])
    return new


def compare_logits(
    hf_logits: np.ndarray,
    gguf_model,
    token_ids: Sequence[int],
) -> TeacherForcedParity:
    """Teacher-forced parity given precomputed HF logits and a GGUF model.

    ``hf_logits`` must be ``(len(token_ids), vocab)`` for the *same* ids.
    """
    lc = gguf_logits_for_tokens(gguf_model, token_ids)
    return teacher_forced_parity(np.asarray(hf_logits), lc)


def compare_greedy(
    hf_new_ids: Sequence[int],
    gguf_model,
    prompt_ids: Sequence[int],
    max_new: int,
) -> GreedyParity:
    """Greedy parity given the HF continuation ids and a GGUF model."""
    lc_new = gguf_greedy(gguf_model, prompt_ids, max_new)
    hf_new = list(hf_new_ids)
    div = next(
        (i for i, (a, b) in enumerate(zip(hf_new, lc_new)) if a != b),
        min(len(hf_new), len(lc_new)),
    )
    return GreedyParity(
        hf_ids=hf_new,
        gguf_ids=lc_new,
        first_divergence=div,
        n_new=max_new,
    )
