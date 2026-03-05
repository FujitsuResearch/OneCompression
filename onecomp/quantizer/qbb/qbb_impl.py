"""QBB (Quantized Binary Bases) quantization module

This module provides QBB quantization functionality for neural network weights.
QBB is a quantization method using multiple binary bases that represents weight matrices
as linear combinations of multiple binary matrices with scale coefficients.

Functions:
    run_qbb(layer, wbits, iters_per_basis, lr, ste_type, use_progressive_quantization, progressive_bits):
        Execute QBB quantization on a layer.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)
"""

from typing import Optional
from logging import getLogger

import torch
import torch.nn as nn
import transformers
import gc
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

logger = getLogger(__name__)


def ste_sign(x):
    """Straight-Through Estimator for sign function
    
    Forward: sign(x)
    Backward: gradient passes through if |x| <= 1, else 0
    """
    # Get sign
    y = x.sign()
    # STE: y = sign(x) in forward, but gradient is clipped identity
    y = y.detach() - x.detach() + x.clamp(-1, 1)
    return y


def identity_ste(x):
    """Identity STE - gradient passes through unchanged"""
    return x.sign()


def tanh_ste(x, k=1.0):
    """Tanh-based STE for smoother gradients
    
    Forward: sign(x)
    Backward: k * (1 - tanh^2(k*x))
    """
    y = x.sign()
    # Use tanh for gradient
    y = y.detach() - x.detach() + torch.tanh(k * x)
    return y


def qbb_layerwise(
    W: torch.Tensor,
    N: int = 4,
    iters_per_basis: int = 1000,
    lr: float = 1e-4,
    use_progressive_W: Optional[torch.Tensor] = None,
    ste_type: str = "clipped",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """QBB (Quantized Binary Bases) layer-wise quantization

    Args:
        W: Weight matrix to quantize
        N: Number of binary bases (target bits)
        iters_per_basis: Number of iterations per basis optimization
        lr: Learning rate for optimization
        ste_type: Type of Straight-Through Estimator ("clipped", "identity", "tanh")
        use_progressive_W: Optional pre-quantized weights as target (for progressive quantization)

    """
    # Target weights (use progressive quantization if provided)
    target = use_progressive_W if use_progressive_W is not None else W.clone()
    target = target.float()

    # Special case: N=1 has closed-form solution
    if N == 1:
        logger.debug("  Using closed-form solution for N=1 (single basis)")

        # Closed-form solution: B* = sign(T), α* = mean(|T|, dim=0)
        B_star = target.sign()
        B_star[target == 0] = 1  # sign(0) = +1
        alpha_star = target.abs().mean(dim=0)

        B_final = [B_star.float()]
        A_final = [alpha_star.float()]

        # Report closed-form loss
        with torch.no_grad():
            W_hat = alpha_star.unsqueeze(0) * B_star
            loss = ((target - W_hat) ** 2).sum().item()
            logger.debug(f"  Closed-form loss: {loss:.6f}")

    else:
        # Multi-basis case: use iterative optimization
        logger.debug(f"  Using iterative optimization for N={N}")

        # 1) Analytical initialization (cascade initialization from residuals)
        B = []
        alpha = []
        residual = target.clone()

        for i in range(N):
            # Binary basis from sign of residual
            Bi = residual.sign()
            # Handle sign(0) = +1 as per paper
            Bi[residual == 0] = 1

            # Per-column mean of absolute values -> shape [c_out]
            ai = residual.abs().mean(dim=0)

            B.append(Bi)
            alpha.append(ai)

            # Update residual
            residual = target - sum(a.unsqueeze(0) * b for a, b in zip(alpha, B))

        # 2) Create learnable proxies
        V = [Bi.clone().float().requires_grad_(True) for Bi in B]
        A = [ai.clone().requires_grad_(True) for ai in alpha]

        # Select STE function
        if ste_type == "clipped":
            ste_func = ste_sign
        elif ste_type == "identity":
            ste_func = identity_ste
        elif ste_type == "tanh":
            ste_func = lambda x: tanh_ste(x, k=1.0)
        else:
            raise ValueError(f"Unknown STE type: {ste_type}")

        # 3) Block coordinate descent over bases
        for i in range(N):
            logger.debug(f"  Optimizing basis {i+1}/{N}")
            # Optimize only V[i] and all A's
            params = [V[i]] + A
            opt = Adam(params, lr=lr)

            best_loss = float("inf")
            patience = 0
            max_patience = 100

            for t in range(iters_per_basis):
                # Reconstruct with STE
                W_hat = sum(A[j].unsqueeze(0) * ste_func(V[j]) for j in range(N))

                # MSE loss
                loss = ((target - W_hat) ** 2).sum()

                opt.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                clip_grad_norm_(V[i], 1.0)
                for a in A:
                    clip_grad_norm_(a, 1.0)

                opt.step()

                # Early stopping
                if t % 100 == 0:
                    loss_val = loss.item()
                    if loss_val < best_loss:
                        best_loss = loss_val
                        patience = 0
                    else:
                        patience += 1

                    if patience >= max_patience:
                        break

        # 4) Finalize binary weights and scales
        with torch.no_grad():
            B_final = []
            for v in V:
                b = v.sign()
                b[v == 0] = 1
                B_final.append(b.float())

            A_final = [a.detach().float() for a in A]

        # Clean up
        if N > 1:
            del V, A, opt

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return B_final, A_final


def run_qbb(
    layer: torch.nn.Module,
    wbits: int = 4,
    iters_per_basis: int = 1000,
    lr: float = 1e-4,
    ste_type: str = "clipped",
    use_progressive_quantization: bool = False,
    progressive_bits: int = 2,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Execute quantization using QBB.

    Performs quantization using QBB (Quantized Binary Bases).
    Represents weight matrices using multiple binary bases and optimizes each binary basis
    and scale coefficient using gradient-based optimization.

    For Conv2d layers, weights are flattened to 2D for processing.
    For transformers.Conv1D layers, weights are transposed for processing.
    All tensors are moved to CPU after processing.
    When wbits=1, a closed-form solution exists and optimization is not needed.
    Weight reconstruction: W ≈ Σ(alpha_i * quantized_weight_list[i])

    Args:
        layer (torch.nn.Module): Layer module to quantize (Linear, Conv2d, Conv1D, etc.).
        wbits (int, optional): Number of quantization bits (number of binary bases). Default is 4.
        iters_per_basis (int, optional): Number of optimization iterations per basis. Default is 1000.
        lr (float, optional): Learning rate. Used for gradient-based optimization. Default is 1e-4.
        ste_type (str, optional): Type of Straight-Through Estimator.
            Choose from "clipped", "identity", "tanh". Default is "clipped".
        use_progressive_quantization (bool, optional): Whether to use progressive quantization.
            If True, first quantizes to progressive_bits bits, then converts to binary bases.
            Default is False.
        progressive_bits (int, optional): Starting number of bits for progressive quantization.
            Used when use_progressive_quantization=True. Default is 2.

    Returns:
        dict[str, torch.Tensor | list[torch.Tensor]]: Dictionary containing quantization results
            with the following keys:
            - "dequantized_weight": Dequantized weights (original shape, original dtype, CPU).
            - "quantized_weight_list": List of quantized weights (each element is binary matrix {±1}, INT8, CPU).
            - "alpha_list": List of scale coefficients for each binary basis (FP16, CPU).
    """
    # In QBB, number of bits = number of binary bases
    N = wbits
    logger.debug(f"[QBB] Starting quantization with {wbits} bits (N={N} bases)")

    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    # Optional: Progressive quantization (quantize to higher bits first)
    progressive_W = None
    if use_progressive_quantization:
        logger.debug(f"[QBB] Using progressive quantization: FP16 -> {progressive_bits}bit -> binary")
        # Simple uniform quantization for progressive target
        scale = W.abs().max() / (2 ** (progressive_bits - 1) - 1)
        progressive_W = torch.round(W / scale) * scale

    # Run QBB quantization (with gradients enabled)
    with torch.enable_grad():
        # Run QBB quantization
        B_list, alpha_list = qbb_layerwise(
            W,
            N=N,
            iters_per_basis=iters_per_basis,
            lr=lr,
            use_progressive_W=progressive_W,
            ste_type=ste_type,
        )

    # Reconstruct quantized weights
    with torch.no_grad():
        Q = sum(alpha.unsqueeze(0) * B for alpha, B in zip(alpha_list, B_list))
        
        # Calculate and report error before transforming Q back to original shape
        error = ((W - Q) ** 2).sum().item()
        logger.debug(f"[QBB] Quantization complete. Reconstruction error: {error:.6f}")
        
        # Convert binary matrices to integer type (int8, {±1})
        B_int_list = [B.to(torch.int8) for B in B_list]

        if isinstance(layer, transformers.Conv1D):
            Q = Q.t()
            B_int_list = [B_int.t() for B_int in B_int_list]

        dequantized_weight = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype).cpu()
        quantized_weight_list = [B_int.reshape(layer.weight.shape).cpu() for B_int in B_int_list]

    alpha_list_cpu = [alpha.cpu() for alpha in alpha_list]

    del W, Q, B_list, B_int_list
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dequantized_weight": dequantized_weight,
        "quantized_weight_list": quantized_weight_list,
        "alpha_list": alpha_list_cpu,
    }

