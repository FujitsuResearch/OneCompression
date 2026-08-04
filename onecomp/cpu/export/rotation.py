"""Fold the online Hadamard of rotation-preprocessed models back into weights.

OneComp's rotation pre-processing (QuaRot/SpinQuant style) fuses R1/R2/scaling
into the linear weights *offline*, with one exception: the ``down_proj`` input
is Hadamard-transformed *online* by a forward pre-hook (``rotate_down_proj`` +
``register_online_hadamard_hooks``). ``llama.cpp`` has no way to apply that
online transform, so a rotated model would otherwise produce wrong outputs.

Because the Hadamard ``H`` used there is orthonormal, the online transform on the
activation can be cancelled by applying the *inverse* transform ``H^T`` to the
stored ``down_proj`` weight. The resulting weight is mathematically equivalent
*without* any online operation, so the exported GGUF runs correctly on stock
``llama.cpp``.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

from logging import getLogger

import torch

logger = getLogger(__name__)


def defold_down_proj_hadamard(weight: torch.Tensor, fp32_had: bool = False) -> torch.Tensor:
    """Invert the online input-Hadamard baked into a ``down_proj`` weight.

    Args:
        weight: Dense ``down_proj`` weight ``(out_features, in_features)`` whose
            input dimension carries the online Hadamard (i.e. it was produced by
            ``matmul_hadU_cuda(W, *get_hadK(in_features))``).
        fp32_had: Match OneComp's ``fp32_had`` setting (compute in fp32).

    Returns:
        The equivalent weight with the online Hadamard removed, so no forward
        pre-hook is needed at inference time.
    """
    from onecomp.pre_process.hadamard_utils import get_hadK, matmul_hadU_cuda

    in_features = weight.shape[-1]
    had_K, K = get_hadK(in_features, transpose=True)
    if fp32_had:
        out = matmul_hadU_cuda(weight.float(), had_K, K).to(weight.dtype)
    else:
        out = matmul_hadU_cuda(weight, had_K, K)
    return out.contiguous()


def defold_rotated_dense_state(dense_state: dict, fp32_had: bool = False) -> int:
    """In-place: remove the online down_proj Hadamard from every down_proj weight.

    Returns the number of ``down_proj`` weights that were de-folded.
    """
    n = 0
    for key in list(dense_state.keys()):
        if key.endswith("down_proj.weight"):
            dense_state[key] = defold_down_proj_hadamard(dense_state[key], fp32_had=fp32_had)
            n += 1
    logger.info("De-folded online Hadamard from %d down_proj weights (rotated model)", n)
    return n
