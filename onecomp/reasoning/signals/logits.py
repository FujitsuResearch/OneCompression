# Copyright 2025-2026 Fujitsu Ltd.

"""Output-logit observations that need no extra model forward."""

from __future__ import annotations

import math
from typing import Any


def logit_observation(
    logits: Any, end_token_id: int, temperature: float = 1.0
) -> dict[str, float | None]:
    """Observe ThinkBrake's maximum non-end minus end log-probability margin.

    At positive temperature the logit difference is divided by temperature.
    Greedy experiments use an explicit reference temperature of one. Negative
    infinity masks are supported; nonfinite scalar observations become None so
    result records remain valid JSON. No extra model forward is required.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Signal temperature must be finite and positive")
    if hasattr(logits, "detach"):
        import torch  # pylint: disable=import-outside-toplevel

        values = logits.detach().float()
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError("Expected one vocabulary vector with at least two tokens")
        if not 0 <= end_token_id < values.numel():
            raise ValueError("End-token index is outside the vocabulary")
        if torch.isnan(values).any() or torch.isposinf(values).any():
            raise ValueError("Logits cannot contain NaN or positive infinity")
        # topk avoids copying/mutating the caller's logits to exclude the end ID.
        top = torch.topk(values, 2)
        other = top.values[1] if int(top.indices[0]) == end_token_id else top.values[0]
        end = float(values[end_token_id])
        normalizer = float(torch.logsumexp(values / temperature, dim=-1))
        best_other = float(other)
    else:
        values = [float(value) for value in logits]
        if len(values) < 2 or not 0 <= end_token_id < len(values):
            raise ValueError("Expected a vocabulary vector and a valid end-token index")
        if any(math.isnan(value) or value == math.inf for value in values):
            raise ValueError("Logits cannot contain NaN or positive infinity")
        end = values[end_token_id]
        best_other = max(value for index, value in enumerate(values) if index != end_token_id)
        maximum = max(values) / temperature
        normalizer = (
            maximum
            + math.log(math.fsum(math.exp(value / temperature - maximum) for value in values))
            if math.isfinite(maximum)
            else -math.inf
        )
    if not math.isfinite(normalizer):
        raise ValueError("At least one token must have a finite logit")
    margin = (best_other - end) / temperature
    return {
        "end_logit": end if math.isfinite(end) else None,
        "logit_margin": margin if math.isfinite(margin) else None,
        "end_probability": math.exp(end / temperature - normalizer),
    }
