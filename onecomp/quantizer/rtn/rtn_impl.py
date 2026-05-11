"""RTN (Round-To-Nearest) quantization implementation

This module provides the run_rtn function for RTN quantization.

Functions:
    run_rtn(layer, wbits, groupsize, sym, mse, norm, grid): Execute RTN quantization on a layer.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import torch
import torch.nn as nn
import transformers
import gc

from .quantizer import pseudo_quantize_tensor


def run_rtn(
    layer,
    wbits=16,
    groupsize=-1,
    sym=False,
    mse=False,
    norm=2.4,
    grid=100,
):
    """Execute quantization using RTN.

    Performs quantization using RTN (Round-To-Nearest).
    The simplest quantization method that rounds weights to the nearest quantization level.
    Does not require calibration data or Hessian matrices, performing quantization
    using only weight statistics.

    For Conv2d layers, weights are flattened to 2D for processing.
    For transformers.Conv1D layers, weights are transposed for processing.
    All tensors are moved to CPU after processing.
    Weight reconstruction: W = scale * (quantized_weight - zero)

    Args:
        layer (torch.nn.Module): Layer module to quantize (Linear, Conv2d, Conv1D, etc.).
        wbits (int, optional): Number of quantization bits. Default is 16.
        groupsize (int, optional): Group size. Computes independent scale
            and zero point for each group.
            -1 means no grouping (single scale and zero point for entire row). Default is -1.
        sym (bool, optional): Whether to use symmetric quantization. If True,
            zero point is placed at center.
            Default is False.
        mse (bool, optional): Enable MSE grid search for optimal clipping.
            Default is False.
        norm (float, optional): Lp norm exponent for MSE search. Default is 2.4.
        grid (int, optional): Number of candidate shrink levels. Default is 100.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing quantization results with the following keys:
            - "dequantized_weight": Dequantized weights (original shape, original dtype, CPU).
            - "quantized_weight": Quantized weights (INT type, CPU).
            - "scale": Scale coefficients (FP16, CPU).
            - "zero": Zero point (FP16, CPU).
    """
    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    Q, scale, zero_point, Q_int = pseudo_quantize_tensor(
        W,
        n_bit=wbits,
        q_group_size=groupsize,
        zero_point=not sym,
        inplace=False,
        perchannel=True,
        mse=mse,
        norm=norm,
        grid=grid,
    )

    if isinstance(layer, transformers.Conv1D):
        Q = Q.t()
        Q_int = Q_int.t()

    dequantized_weight = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype).cpu()
    quantized_weight = Q_int.reshape(layer.weight.shape).cpu()

    scale = scale.reshape(-1, scale.shape[-1]).to(dtype=torch.float16, device="cpu")
    zero = zero_point.reshape(-1, zero_point.shape[-1]).to(dtype=torch.float16, device="cpu")

    del W, Q, Q_int
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "dequantized_weight": dequantized_weight,
        "quantized_weight": quantized_weight,
        "scale": scale,
        "zero": zero,
    }
