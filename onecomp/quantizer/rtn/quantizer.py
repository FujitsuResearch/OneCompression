"""RTN (Round-To-Nearest) quantization module

This module provides RTN quantization functionality for neural network weights.
RTN is the simplest quantization method that rounds weights to the nearest quantization level.
It does not require calibration data or Hessian matrices.

Classes:
    RTNResult: Result class for RTN quantization containing quantized weights and parameters.
    RTN: RTN quantizer class that performs round-to-nearest quantization.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)
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
    
    Args:
        x: Input tensor (floating-point).
        scale: Scale coefficient.
        zero_point: Zero point.
        q_min: Minimum quantization level.
        q_max: Maximum quantization level.
    
    Returns:
        Quantized integer tensor (clamped to the range [q_min, q_max]).
    """
    w_int = torch.round(x / scale + zero_point)
    w_int = w_int.clamp(q_min, q_max).int()
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pseudo-quantize a tensor using the Round-To-Nearest method.

    Args:
        w: Weight tensor to quantize.
        n_bit: Number of quantization bits.
        q_group_size: Group size (-1 means the entire row).
        zero_point: Whether to use a zero point (asymmetric quantization).
        inplace: Whether to perform in-place operations.

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
        assert w.shape[-1] % q_group_size == 0, (
            f"Tensor shape {w.shape[-1]} must be divisible by group size {q_group_size}"
        )
        w = w.reshape(-1, w.shape[-1] // q_group_size, q_group_size)
    else:
        # Treat the entire row as a single group
        w = w.reshape(-1, 1, w.shape[-1])
    
    # Quantization levels
    if zero_point:
        # Asymmetric quantization: [0, 2^n_bit - 1]
        q_max = 2 ** n_bit - 1
        q_min = 0
    else:
        # Symmetric quantization: [-2^(n_bit-1), 2^(n_bit-1) - 1]
        q_max = 2 ** (n_bit - 1) - 1
        q_min = -(2 ** (n_bit - 1))

    # Compute min and max values per group
    w_max = w.amax(dim=-1, keepdim=True)
    w_min = w.amin(dim=-1, keepdim=True)
    
    # Compute scale and zero point
    scale = (w_max - w_min) / (q_max - q_min)
    # Prevent division by zero
    scale = scale.clamp(min=1e-5)

    if zero_point:
        zero_point_val = torch.round(q_min - w_min / scale)
        zero_point_val = zero_point_val.clamp(q_min, q_max)
    else:
        zero_point_val = torch.zeros_like(scale)

    # Quantize and dequantize
    w_int = quantize(w, scale, zero_point_val, q_min, q_max)
    w_quant = dequantize(w_int, scale, zero_point_val)

    # Restore original shape
    w_quant = w_quant.reshape(org_w_shape)
    w_int = w_int.reshape(org_w_shape)
    scale = scale.squeeze(-1)
    zero_point_val = zero_point_val.squeeze(-1)

    return w_quant, scale, zero_point_val, w_int

