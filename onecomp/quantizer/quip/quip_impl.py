"""QUIP (Quantization with Incoherence Processing) quantization module

This module provides QUIP quantization functionality for neural network weights.
QUIP is a quantization method using incoherence processing that quantizes weights
using Hessian matrices, similar to GPTQ but with improved accuracy through incoherence processing.

Functions:
    run_quip(H, layer, percdamp, wbits, incoh_mode): Execute QUIP quantization on a layer.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import gc
import torch
import torch.nn as nn
import transformers

from .quant_quip import QuantizerQFN
from .vector_balance import quantize_weight_vecbal
from .utils import rand_ortho_butterfly_noblock
from .utils_had import RHT_H, RHT_W, REVERSE_RHT_W


def run_quip(
    H: torch.Tensor,
    layer: torch.nn.Module,
    percdamp: float = 0.01,
    wbits: int = 4,
    incoh_mode: str = "kron",
) -> dict[str, torch.Tensor]:
    """Execute quantization using QUIP.

    Performs quantization using QUIP (Quantization with Incoherence Processing).
    Transforms weights and Hessian using incoherence processing and quantizes using
    LDLQ (LDL decomposition-based quantization). After quantization, applies inverse
    transformation to return to original space.

    For Conv2d layers, weights are flattened to 2D for processing.
    For transformers.Conv1D layers, weights are transposed for processing.
    All tensors are moved to CPU after processing.
    Incoherence processing transforms weights and Hessian before and after quantization.
    When qfn="b", zero is None.

    Args:
        H (torch.Tensor): Hessian matrix. Hessian matrix of activations computed from
            calibration data.
        layer (torch.nn.Module): Layer module to quantize (Linear, Conv2d, Conv1D, etc.).
        percdamp (float, optional): Damping coefficient. Ratio of damping added to diagonal elements
            of Hessian matrix. Default is 0.01.
        wbits (int, optional): Number of quantization bits. Default is 4.
        incoh_mode (str, optional): Incoherence mode. Choose from "kron" or "had".
            - "kron": Orthogonal transformation based on Kronecker product (uses Butterfly matrices)
            - "had": Hadamard transform-based processing
            Default is "kron".

    Returns:
        dict[str, torch.Tensor]: Dictionary containing quantization results with the following keys:
            - "dequantized_weight": Dequantized weights (original shape, original dtype, CPU).
            - "quantized_weight": Quantized weights (INT type, CPU).
            - "scale": Scale coefficients (FP16, CPU).
            - "zero": Zero point (FP16, CPU, None when qfn="b").
            - "maxq": Maximum quantization level (CPU).
    """
    quantizer = QuantizerQFN()
    quantizer.configure(wbits, perchannel=True, sym=False, qfn="b", mse=False)

    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()
    H = H.clone()

    quantizer.find_params(W, weight=True)

    # preproc_rescale
    H /= H.abs().max()
    diagH = torch.diag(H)
    diagW2 = torch.diag(W.T @ W)
    diagH = torch.clamp(diagH, min=1e-8)
    diagW2 = torch.clamp(diagW2, min=1e-8)
    scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
    scaleWH = scaleWH.clamp(min=1e-8)
    W *= scaleWH[None, :]
    H /= scaleWH[None, :]
    H /= scaleWH[:, None]
    scaleWH = scaleWH.cpu()

    # preproc_proj
    if incoh_mode == "kron":
        U = rand_ortho_butterfly_noblock(W.shape[0]).to(torch.float32).to(W.device)
        V = rand_ortho_butterfly_noblock(W.shape[1]).to(torch.float32).to(W.device)
        H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(
            H.shape[0], device=W.device
        )
        W = U @ W @ V.T
        H = V @ H @ V.T
        U = U.cpu()
        V = V.cpu()
    elif incoh_mode == "had":
        U = (torch.randn(W.shape[0], device=W.device).sign() + 1e-5).sign().to(torch.float32)
        V = (torch.randn(W.shape[1], device=W.device).sign() + 1e-5).sign().to(torch.float32)
        H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(
            H.shape[0], device=W.device
        )
        W = RHT_W(W, U, V)
        H = RHT_H(H, V)
        U = U.cpu()
        V = V.cpu()

    # H modification from gptq
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp

    # quantization
    Q, Q_int, scale_actual = quantize_weight_vecbal(
        w=W,
        H=H,
        nbits=wbits,
        npasses=0,
        scale=quantizer.scale,
        zero=quantizer.zero,
        maxq=quantizer.maxq,
        unbiased=False,
        qfn=quantizer.qfn,
        qmethod="ldlq",
    )
    Q = Q.float()

    # postproc
    if incoh_mode == "kron":
        U = U.to(Q.device)
        V = V.to(Q.device)
        Q = U.T @ Q @ V
    elif incoh_mode == "had":
        U = U.to(Q.device)
        V = V.to(Q.device)
        Q = REVERSE_RHT_W(Q, U, V)

    scaleWH = scaleWH.to(Q.device)
    Q = Q / scaleWH[None, :]

    if isinstance(layer, transformers.Conv1D):
        Q = Q.t()
        Q_int = Q_int.t()

    # Store quantized weights on CPU
    dequantized_weight = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype).cpu()
    quantized_weight = Q_int.reshape(layer.weight.shape).cpu()

    if quantizer.qfn == "b":
        scale_final = scale_actual.cpu()
        zero = None
    else:
        scale_final = quantizer.scale.cpu()
        if quantizer.zero is not None:
            zero = quantizer.zero.cpu()
        else:
            zero = None

    maxq = quantizer.maxq.cpu()

    del H, W, Q, Q_int, U, V, scaleWH
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "dequantized_weight": dequantized_weight,
        "quantized_weight": quantized_weight,
        "scale": scale_final,
        "zero": zero,
        "maxq": maxq,
    }
