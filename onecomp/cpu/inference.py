"""Thin CPU inference wrapper around llama-cpp-python for OneComp GGUF models.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import os
from logging import getLogger
from typing import Iterator, List, Optional

logger = getLogger(__name__)


class LlamaCppModel:
    """Load a GGUF model and run CPU text generation via llama-cpp-python."""

    def __init__(
        self,
        gguf_path: str,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **llama_kwargs,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "llama-cpp-python is required for CPU inference. "
                "Install with: pip install 'onecomp[llamacpp]'."
            ) from exc

        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(gguf_path)

        if n_threads is None:
            n_threads = os.cpu_count() or 1

        self.gguf_path = gguf_path
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        # Per-position logits (needed for teacher-forced parity / perplexity) are
        # only retained when llama.cpp is built with logits_all=True.
        self.logits_all = bool(llama_kwargs.get("logits_all", False))
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            **llama_kwargs,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Greedy/sampled completion for a single prompt; returns the new text."""
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            **kwargs,
        )
        return out["choices"][0]["text"]

    def stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Yield generated text incrementally (token chunks) for a single prompt."""
        for chunk in self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=True,
            **kwargs,
        ):
            piece = chunk["choices"][0].get("text", "")
            if piece:
                yield piece

    def chat(self, messages: List[dict], max_tokens: int = 256, temperature: float = 0.0, **kw):
        """Chat completion using the GGUF model's embedded chat template."""
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature, **kw
        )
        return out["choices"][0]["message"]["content"]

    def logits_for_tokens(self, token_ids: List[int]):
        """Return the per-position logits ``(len(token_ids), vocab)`` for a sequence.

        Requires the model to have been constructed with ``logits_all=True``
        (pass it through ``**llama_kwargs``); otherwise llama.cpp only retains
        the logits of the final position and the earlier rows are unusable.
        """
        import numpy as np

        if not self.logits_all:
            raise ValueError(
                "logits_for_tokens requires the model to be built with "
                "logits_all=True (e.g. LlamaCppModel(..., logits_all=True)); "
                "otherwise only the final position's logits are valid."
            )
        self._llm.reset()
        self._llm.eval(token_ids)
        scores = np.array(self._llm.scores[: len(token_ids)], dtype=np.float64)
        return scores

    # Backwards-compatible alias.
    logprobs_for_tokens = logits_for_tokens

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)
