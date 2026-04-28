"""RTN (Round-To-Nearest) quantization module

This module provides RTN quantization functionality for neural network weights.
RTN is the simplest quantization method that rounds weights to the nearest quantization level.
It does not require calibration data or Hessian matrices.

Classes:
    RTNResult: Result class for RTN quantization containing quantized weights and parameters.
    RTN: RTN quantizer class that performs round-to-nearest quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from typing import Optional

import torch
import torch.nn as nn
import transformers


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: int,
    q_max: int,
) -> torch.Tensor:
    """Quantize floating-point values to integers.

    Computes ``clamp(round(x / scale) + zero_point, q_min, q_max)``.

    Args:
        x: Input tensor (floating-point).
        scale: Scale coefficient.
        zero_point: Zero point.
        q_min: Minimum quantization level.
        q_max: Maximum quantization level.

    Returns:
        Quantized integer tensor (clamped to the range [q_min, q_max]).
    """
    w_int = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max).int()
    return w_int


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize integer values back to floating-point.

    Args:
        quantized: Quantized integer tensor.
        scale: Scale coefficient.
        zero_point: Zero point.

    Returns:
        Dequantized floating-point tensor.
    """
    return (quantized.float() - zero_point) * scale


def pseudo_quantize_tensor(
    w: torch.Tensor,
    n_bit: int = 8,
    q_group_size: int = -1,
    zero_point: bool = True,
    inplace: bool = False,
    perchannel: bool = True,
    mse: bool = False,
    norm: float = 2.4,
    grid: int = 100,
    maxshrink: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pseudo-quantize a tensor using the Round-To-Nearest method.

    Args:
        w: Weight tensor to quantize.
        n_bit: Number of quantization bits.
        q_group_size: Group size (-1 means the entire row).
        zero_point: If True, asymmetric quantisation. If False,
            symmetric quantisation (min/max symmetrised around zero).
        inplace: Whether to perform in-place operations.
        perchannel: If True, compute one scale/zero per output channel
            (row). If False, use a single scale/zero for the entire
            tensor (per-tensor). Ignored when ``q_group_size > 0``.
        mse: If True, search over shrunk min/max ranges to minimise
            the quantisation error measured by ``norm``.
        norm: Exponent of the Lp norm used in the MSE grid search.
        grid: Number of candidate shrink levels evaluated.
        maxshrink: Maximum fraction by which the range is shrunk.

    Returns:
        w_quant: Dequantized weights (floating-point).
        scale: Scale coefficient.
        zero_point_val: Zero point.
        w_int: Quantized weights (integer values).
    """
    if not inplace:
        w = w.clone()

    # Save the original shape
    org_w_shape = w.shape

    # Configure group size
    if q_group_size > 0:
        # (out_features, in_features) -> (out_features, num_groups, group_size)
        if w.shape[-1] % q_group_size != 0:
            raise ValueError(
                f"Tensor shape {w.shape[-1]} must be divisible by group size {q_group_size}"
            )
        w = w.reshape(-1, w.shape[-1] // q_group_size, q_group_size)
    elif perchannel:
        # Treat the entire row as a single group
        w = w.reshape(-1, 1, w.shape[-1])
    else:
        # Per-tensor: single scale/zero for the entire weight
        w = w.flatten().reshape(1, 1, -1)

    # Quantization levels: always unsigned [0, 2^n_bit - 1]
    sym = not zero_point
    q_max = 2**n_bit - 1
    q_min = 0

    # Compute min and max values per group (with zero included in the range)
    tmp = torch.zeros(1, device=w.device)
    w_max = torch.maximum(w.amax(dim=-1, keepdim=True), tmp)
    w_min = torch.minimum(w.amin(dim=-1, keepdim=True), tmp)

    # Symmetric: symmetrise range around zero
    if sym:
        w_max = torch.maximum(torch.abs(w_min), w_max)
        w_min = -w_max

    # Handle all-zero groups
    dead = (w_min == 0) & (w_max == 0)
    w_min[dead] = -1
    w_max[dead] = +1

    # Compute scale and zero point
    scale = ((w_max - w_min) / q_max).clamp(min=1e-5)

    if sym:
        zero_point_val = torch.full_like(scale, (q_max + 1) / 2)
    else:
        zero_point_val = torch.round(-w_min / scale)

    # MSE grid search: try progressively shrunk ranges and keep the best
    if mse:
        best = torch.full(w.shape[:-1], float("inf"), device=w.device)
        for i in range(int(maxshrink * grid)):
            p = 1 - i / grid
            wmin1 = p * w_min
            wmax1 = p * w_max
            scale1 = ((wmax1 - wmin1) / q_max).clamp(min=1e-5)
            if sym:
                zp1 = zero_point_val
            else:
                zp1 = torch.round(-wmin1 / scale1)
            q = torch.clamp(torch.round(w / scale1) + zp1, q_min, q_max)
            dq = (q - zp1) * scale1
            err = (dq - w).abs_().pow_(norm).sum(dim=-1)
            improved = (err < best).unsqueeze(-1)
            best = torch.where(improved.squeeze(-1), err, best)
            scale = torch.where(improved, scale1, scale)
            zero_point_val = torch.where(improved, zp1, zero_point_val)

    # Quantize and dequantize
    w_int = quantize(w, scale, zero_point_val, q_min, q_max)
    w_quant = dequantize(w_int, scale, zero_point_val)

    # Restore original shape
    w_quant = w_quant.reshape(org_w_shape)
    w_int = w_int.reshape(org_w_shape)
    scale = scale.squeeze(-1)
    zero_point_val = zero_point_val.squeeze(-1)

    return w_quant, scale, zero_point_val, w_int
