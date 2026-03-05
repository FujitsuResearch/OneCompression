"""ARB (Alternating Refined Binarization) quantization module

This module provides ARB quantization functionality for neural network weights.
ARB is a 1-bit binary quantization method that decomposes weight matrices into
binary matrices with scale and bias terms.

Functions:
    run_arb(layer, arb_iters, split_points, verbose): Execute ARB quantization on a layer.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)
"""

from typing import Optional
import gc
import torch
import torch.nn as nn
import transformers
from logging import getLogger

logger = getLogger(__name__)


def run_arb(
    layer: torch.nn.Module,
    arb_iters: int = 15,
    split_points: int = 2,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """Execute quantization using ARB.

    Performs 1-bit quantization of weight matrices using ARB (Alternating Refined Binarization).
    Based on the BTC-LLM paper implementation, it improves quantization accuracy by alternately
    refining mean, scale, and sign to correct distribution shifts.

    Decomposes weight matrix W as W ≈ α⊙B + μ:
    - B: Binary matrix {±1}^{n×m}
    - α: Row-wise scaling coefficients ∈ R^n
    - μ: Row-wise bias (mean) ∈ R^n

    For Conv2d layers, weights are flattened to 2D for processing.
    For transformers.Conv1D layers, weights are transposed for processing.
    All tensors are moved to CPU after processing.

    Args:
        layer (torch.nn.Module): Layer module to quantize (Linear, Conv2d, Conv1D, etc.).
        arb_iters (int, optional): Number of ARB iterations. Default is 15.
        split_points (int, optional): Number of split points. Number of groups to split
            non-salient weights into. Default is 2. If 1, standard ARB (no splitting) is executed.
        verbose (bool, optional): Whether to output detailed logs. Default is False.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing quantization results with the following keys:
            - "dequantized_weight": Dequantized weights (original shape, original dtype, CPU).
            - "quantized_weight": Quantized weights (binary matrix {±1}, INT8, CPU).
            - "alpha": Row-wise scale coefficients (FP16, CPU).
            - "mu": Row-wise bias (FP16, CPU).
    """
    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    # Execute ARB quantization
    if split_points == 1:
        # Standard ARB (no splitting)
        alpha, B, mu = arb_quantize(W, iters=arb_iters, verbose=verbose)
    else:
        # ARB with split points (splitting non-salient weights)
        alpha, B, mu = arb_quantize_with_splits(
            W, iters=arb_iters, split_points=split_points, verbose=verbose
        )
    
    # Reconstruct quantized weights
    Q = dequantize(alpha, B, mu)
    
    # Convert B to integer type (int8, {±1})
    B_int = B.to(torch.int8)
    
    if isinstance(layer, transformers.Conv1D):
        Q = Q.t()
        B_int = B_int.t()
    
    # Store quantized weights on CPU
    dequantized_weight = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype).cpu()
    quantized_weight = B_int.reshape(layer.weight.shape).cpu()
    
    alpha = alpha.cpu()
    mu = mu.cpu()
    
    del W, Q, B, B_int
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "dequantized_weight": dequantized_weight,
        "quantized_weight": quantized_weight,
        "alpha": alpha,
        "mu": mu,
    }


def arb_quantize(
    W: torch.Tensor,
    iters: int = 15,
    epsilon: float = 1e-8,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard ARB quantization (naive ARB).

    Improves binary approximation via row-mean redistribution and alternating refinement.
    W ≈ α⊙B + μ where B ∈ {±1}^{n×m}, α ∈ R^n, μ ∈ R^n

    Update rules (paper Eq.(1)(2) p.4):
    - μ ← μ + (1/m)Σ_j R_{·j}
    - α ← (1/m) diag(B^T(W-μ))
    - B ← sign(W-μ)

    Args:
        W: Weight matrix [out, in]
        iters: Number of ARB iterations (default: 15)
        epsilon: Numerical stability constant
        verbose: Print iteration details

    Returns:
        alpha: Row scales [out]
        B: Binary matrix {±1}^{out, in}
        mu: Row biases [out]
    """
    dtype = W.dtype
    W = W.float()
    
    in_dim = W.shape[1]
    
    # Step 1: Initialize - use row mean as μ
    mu = W.mean(dim=1, keepdim=True)  # [out, 1]
    
    if verbose:
        logger.debug(f"[ARB] Initial mu stats: mean={mu.mean().item():.6f}, std={mu.std().item():.6f}")
    
    # Step 2: ARB iteration (alternating optimization)
    for iteration in range(iters):
        # 2.1: Update B - sign of centered weights
        W_centered = W - mu
        B = torch.sign(W_centered)
        B[B == 0] = 1  # sign(0) = +1 (paper convention)
        
        # 2.2: Update α - compute optimal scale for each row
        # α_i = (1/m) * Σ_j (B_ij * (W - μ)_ij)
        # Equivalent to (1/m)Σ_j |W-μ|_ij when B=sign(W-μ)
        alpha = ((B * W_centered).sum(dim=1)) / in_dim  # [out]
        alpha = alpha.clamp(min=epsilon)  # numerical stability
        
        # 2.3: Compute residual
        # R = W - α⊙B - μ
        R = W - alpha[:, None] * B - mu  # [out, in]
        
        # 2.4: Update μ - add row mean of residual
        mu_update = R.mean(dim=1, keepdim=True)
        mu = mu + mu_update
        
        # Progress display
        if verbose and (iteration % 5 == 0 or iteration == iters - 1):
            residual_norm = torch.norm(R, p='fro').item()
            mu_change = torch.norm(mu_update, p='fro').item()
            logger.debug(f"[ARB]   Iter {iteration:3d}: ||R||_F={residual_norm:.6f}, "
                  f"||Δμ||_F={mu_change:.6f}, "
                  f"α_mean={alpha.mean().item():.6f}")
    
    # Step 3: Final computation of B and α
    W_centered = W - mu
    B = torch.sign(W_centered)
    B[B == 0] = 1
    
    # Recompute final α based on L1 norm
    # α_i = (1/m)Σ_j |W_i - μ_i|_j
    alpha = W_centered.abs().mean(dim=1)  # [out]
    alpha = alpha.clamp(min=epsilon)
    
    # Squeeze mu back to 1D
    mu = mu.squeeze(1)
    
    # Restore original dtype
    return alpha.to(dtype), B.to(dtype), mu.to(dtype)


def arb_quantize_with_splits(
    W: torch.Tensor,
    iters: int = 15,
    split_points: int = 2,
    epsilon: float = 1e-8,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ARB quantization with split points (experimental).

    Splits non-salient weights into multiple groups and progressively
    refines μ to improve accuracy.

    - Uses group column count |G_i| as denominator
    - μ update is consistent with ARB formula: μ ← μ + (1/m)Σ(R_G)
    - Final full-column regeneration replaced with short ARB (preserves split effect)
    According to the paper, 1 split -> 2 splits improves PPL from 10.12 to 6.60.

    Args:
        W: Weight matrix [out, in]
        iters: Number of ARB iterations
        split_points: Number of groups to split (2 or 3)
        epsilon: Numerical stability constant
        verbose: Print details

    Returns:
        alpha: Row scales [out]
        B: Binary matrix {±1}^{out, in}
        mu: Row biases [out]
    """
    device = W.device
    dtype = W.dtype
    W = W.float()
    
    in_dim = W.shape[1]
    
    if split_points == 1:
        # No splitting: fall back to standard ARB
        return arb_quantize(W, iters=iters, epsilon=epsilon, verbose=verbose)
    
    # Step 1: Initial μ (row mean)
    mu = W.mean(dim=1, keepdim=True)  # [out, 1]
    
    # Step 2: Compute importance (global quantiles)
    W_centered = W - mu
    importance = W_centered.abs()  # More stable without α weighting
    
    # Compute quantiles (non-salient to salient order)
    quantiles = torch.linspace(0, 1, split_points + 1, device=device)[1:-1]
    try:
        thresholds = torch.quantile(importance.flatten(), quantiles)
    except RuntimeError as e:
        if "input tensor is too large" in str(e):
            # If importance tensor is too large, sample and compute quantiles
            importance_flat = importance.flatten()
            num_samples = 1000000  # 1M samples
            # Random sampling
            indices = torch.randperm(importance_flat.numel(), device=device)[:num_samples]
            sampled_importance = importance_flat[indices]
            thresholds = torch.quantile(sampled_importance, quantiles)
        else:
            raise e
    
    if verbose:
        logger.debug(f"[ARB] Split points: {split_points}, thresholds: {thresholds.tolist()}")
    
    # Step 3: Process groups from non-salient to salient
    prev_threshold = torch.zeros((), device=device)
    for group_idx in range(split_points):
        if group_idx < split_points - 1:
            mask = (importance > prev_threshold) & (importance <= thresholds[group_idx])
            prev_threshold = thresholds[group_idx]
        else:
            mask = importance > prev_threshold
        
        if not mask.any():
            continue
        
        # Per-row group column count |G_i|
        group_sizes = mask.sum(dim=1).clamp_min(1).float()  # [out]
        
        # Refine μ in ARB-style for this group
        group_iters = max(1, iters // split_points)
        for _ in range(group_iters):
            W_centered = W - mu
            B = torch.sign(W_centered)
            B[B == 0] = 1
            
            # B active only within the group (zero outside)
            B_group = B * mask  # [out, in]
            
            # α_G(i) = Σ_{j∈G_i} B_ij * (W-μ)_ij / |G_i|
            alpha_group = ((B_group * W_centered).sum(dim=1) / group_sizes).clamp_min(epsilon)
            
            # Group residual R_G (zero outside group)
            R_group = (W_centered - alpha_group[:, None] * B_group) * mask
            
            # μ update: μ ← μ + (1/m) * row_sum(R_G)
            # Consistent with ARB Eq.(2) (incremental group contribution)
            mu = mu + R_group.sum(dim=1, keepdim=True) / float(in_dim)
        
        if verbose:
            coverage = mask.float().mean().item()
            logger.debug(f"[ARB] Group {group_idx + 1}/{split_points}: "
                  f"{coverage * 100:.1f}% of weights, "
                  f"{group_iters} iterations")
    
    # Step 4: Final refinement (converge with short ARB over all columns)
    # At this point μ has been progressively refined, so few iterations suffice
    final_iters = max(1, iters // 3)
    if verbose:
        logger.debug(f"[ARB] Final refinement with {final_iters} iterations")
    
    alpha, B, mu_vec = arb_quantize(W, iters=final_iters, epsilon=epsilon, verbose=False)
    
    return alpha.to(dtype), B.to(dtype), mu_vec.to(dtype)


def dequantize(
    alpha: torch.Tensor,
    B: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Dequantize ARB parameters back to floating point.

    W_reconstructed = α⊙B + μ

    Args:
        alpha: Row scales [out]
        B: Binary matrix {±1}^{out, in} (also supports int8)
        mu: Row biases [out]

    Returns:
        Dequantized floating point tensor [out, in]
    """
    # Convert B stored as int8 to float
    if B.dtype == torch.int8:
        B = B.float()
    return alpha[:, None] * B + mu[:, None]

