# Copyright 2025-2026 Fujitsu Ltd.

"""ASAG attention statistics from already-materialized eager attention weights."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def attention_statistics(
    attentions: Sequence[Any],
    query_start: int = 0,
    query_end: int | None = None,
    previous_span: tuple[int, int] | None = None,
    current_span: tuple[int, int] | None = None,
    last_layers: int = 4,
) -> dict[str, float | int | None]:
    """ASAG equations 2, 6, 8 from already-materialized causal attention.

    Each layer has shape [1, heads, queries, keys] or [heads, queries, keys].
    Supply only this request's unpadded keys. Query bounds index the supplied
    tensor, whereas thought spans index keys. The denominator is log(keys), not
    log(each row's causal support). Entropy is summed, not averaged, across
    selected query rows, heads and layers. Last-four selection uses available
    layers on tiny test models and reports their actual count.

    This reference reduction does not imply fused-kernel instrumentation. The
    caller must account for forwards used to produce these attention tensors.
    """
    import torch  # pylint: disable=import-outside-toplevel

    if last_layers < 1 or not attentions:
        raise ValueError("At least one attention layer is required")
    selected = attentions[-last_layers:]
    entropy = None
    key_mass = None
    shape = None
    for tensor in selected:
        values = tensor.detach()
        if values.ndim == 4:
            if values.shape[0] != 1:
                raise ValueError("Attention statistics require a single unpadded request")
            values = values[0]
        if values.ndim != 3:
            raise ValueError("Expected attention [heads, queries, keys]")
        heads, queries, keys = values.shape
        stop = queries if query_end is None else query_end
        if heads < 1 or keys < 1 or not 0 <= query_start < stop <= queries:
            raise ValueError("Invalid attention dimensions or query window")
        if shape is not None and shape != (heads, queries, keys):
            raise ValueError("Attention layers must have matching dimensions")
        shape = (heads, queries, keys)
        window = values[:, query_start:stop, :].float()
        if not bool(torch.isfinite(window).all()) or bool((window < 0).any()):
            raise ValueError("Attention weights must be finite and nonnegative")
        if not torch.allclose(window.sum(-1), torch.ones_like(window.sum(-1)), atol=2e-2, rtol=0):
            raise ValueError("Attention rows must be normalized probabilities")
        layer_entropy = (
            -torch.special.xlogy(window, window).sum() / math.log(keys)
            if keys > 1
            else window.new_zeros(())
        )
        entropy = layer_entropy if entropy is None else entropy + layer_entropy
        # Average heads and layers; summing query rows cancels in the trap ratio.
        layer_mass = window.mean(dim=0).sum(dim=0) / len(selected)
        key_mass = layer_mass if key_mass is None else key_mass + layer_mass

    def span_mean(span: tuple[int, int] | None) -> float | None:
        if span is None:
            return None
        start, end = span
        if not 0 <= start < end <= shape[-1]:
            raise ValueError("Thought spans must be nonempty and within the key prefix")
        return float(key_mass[start:end].mean())

    previous = span_mean(previous_span)
    current = span_mean(current_span)
    ratio = previous / current if previous is not None and current not in (None, 0.0) else None
    return {
        "attention_entropy": float(entropy),
        "previous_attention": previous,
        "current_attention": current,
        "backward_ratio": ratio,
        "attention_layers": len(selected),
        "attention_query_rows": stop - query_start,
        "attention_key_tokens": shape[-1],
    }
