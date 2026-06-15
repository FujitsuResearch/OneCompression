"""
MDBF (Multi-Envelope Double Binary Factorization) utility functions

- Calculate rank r from BPW (Bits Per Weight)
- Reconstruct weight matrix

Parameter breakdown (per path):

- Sign matrices: S_A (n×r), S_B (r×m) -> Binary = r(n+m) bits
- Scales: A_amp (n×l), B_amp (m×l), Q_U_amp (r×l), Q_V_amp (r×l)
           -> FP16 = 16 * (ln + lm + 2lr) bits

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import gc
import math
from logging import getLogger
from typing import Literal
import torch

logger = getLogger(__name__)


def cleanup_gpu_memory() -> None:
    """Release GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def ensure_float32(
    tensor: torch.Tensor,
    device: torch.device = None,
    clone: bool = False,
) -> torch.Tensor:
    """
    Convert a tensor to float32

    Args:
        tensor: Input tensor
        device: Target device (None to keep the original device)
        clone: If True, always return a clone

    Returns:
        float32 tensor (same object if clone=False and no conversion needed)
    """
    target_device = device if device is not None else tensor.device
    needs_dtype = tensor.dtype != torch.float32
    needs_device = tensor.device != target_device

    if not needs_dtype and not needs_device:
        return tensor.clone() if clone else tensor
    # .to() returns a new tensor, so no need to clone
    return tensor.to(device=target_device, dtype=torch.float32)


def ensure_float32_clone(tensor: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """
    Convert a tensor to float32 and clone it (wrapper for ensure_float32)

    Args:
        tensor: Input tensor
        device: Target device (None to keep the original device)

    Returns:
        Clone of the float32 tensor
    """
    return ensure_float32(tensor, device, clone=True)


def rank_from_bpw(
    n: int,
    m: int,
    b_target: float,
    l: int = 1,
    P: int = 2,
    min_rank: int = 1,
    rounding: Literal["floor", "ceil", "round"] = "floor",
    scale_bits: int = 16,
) -> int:
    """
    Calculate rank r from target BPW

    # b_eff = P * [r(n+m) + scale_bits*l*(n+m+2r)] / (nm)
    # Solving for r:
    # r = (b_target * nm / P - scale_bits*l*(n+m)) / ((n+m) + 2*scale_bits*l)

    Args:
        n: Number of rows (output dimension)
        m: Number of columns (input dimension)
        b_target: Target BPW
        l: Multi-scale rank
        P: Number of paths (1, 2, ...)
        min_rank: Minimum rank
        rounding: Rounding method
            - "floor": Round down (ensure b_target is not exceeded)
            - "ceil": Round up (prioritize approximation accuracy)
            - "round": Round to nearest (balance)
        scale_bits: Bits per stored scale element (default FP16 = 16). Must match
            the value passed to bpw_from_rank() so that the rank-selection and the
            reported BPW stay consistent.

    Returns:
        Calculated rank r
    """
    numerator = (b_target * n * m / P) - scale_bits * l * (n + m)
    denominator = (n + m) + 2 * scale_bits * l

    r_real = numerator / denominator
    max_rank = min(n, m)

    if r_real < min_rank:
        logger.warning(
            f"[MDBF] b_target={b_target} is too small for {n}x{m} matrix. "
            f"r_real={r_real:.2f}, using min_rank={min_rank}"
        )
        return min_rank

    if r_real > max_rank:
        logger.warning(
            f"[MDBF] b_target={b_target} exceeds full-rank for {n}x{m} matrix. "
            f"r_real={r_real:.2f}, clamping to {max_rank}"
        )
        return max_rank

    if rounding == "floor":
        r = int(math.floor(r_real))
    elif rounding == "ceil":
        r = int(math.ceil(r_real))
    else:
        r = int(round(r_real))

    return max(r, min_rank)


def bpw_from_rank(
    n: int,
    m: int,
    r: int,
    l: int = 1,
    P: int = 2,
    scale_bits: int = 16,
) -> float:
    """
    # Calculate effective BPW from rank r

    b_eff = P * [r(n+m) + scale_bits*l*(n + m + 2*r)] / (nm)

    Args:
        n: Number of rows (output dimension)
        m: Number of columns (input dimension)
        r: Rank
        l: Multi-scale rank
        P: Number of paths
        scale_bits: Bits per stored scale element (default FP16 = 16). Must match
            the value passed to rank_from_bpw().

    Returns:
        Effective BPW
    """
    bits_binary = r * (n + m)
    bits_scale = scale_bits * l * (n + m + 2 * r)

    total_bits = P * (bits_binary + bits_scale)
    return total_bits / (n * m)


def to_binary_sign(x: torch.Tensor) -> torch.Tensor:
    """Binarize a sign matrix {-1, +1}"""
    out = torch.sign(x)
    out[out == 0] = 1.0
    return out


def symmetrize_matrix(H: torch.Tensor) -> torch.Tensor:
    """Symmetrize a matrix: (H + H^T) / 2"""
    return (H + H.T) * 0.5


def compute_hessian_error(E: torch.Tensor, H: torch.Tensor, nsamples: int) -> float:
    """
    Calculate Hessian-weighted error: N * tr(E @ H @ E^T)

    Args:
        E: Error matrix (n, m)
        H: Hessian matrix (m, m)
        nsamples: Number of samples N

    Returns:
        N * tr(E @ H @ E^T) = N * sum((E @ H) * E)
    """
    EH = E @ H
    return float(nsamples) * torch.sum(EH * E).item()


def reconstruct_weight(
    A_sign: torch.Tensor,
    B_sign: torch.Tensor,
    A_amp: torch.Tensor,
    B_amp: torch.Tensor,
    Q_U_amp: torch.Tensor,
    Q_V_amp: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct weight matrix from parameters

    W = F @ G
    where F = S_A * (A_amp @ Q_U_amp^T)  : (n, r)
          G = S_B * (Q_V_amp @ B_amp^T)  : (r, m)

    Computational complexity: O(nlr + rlm + nrm) (faster than l^2 loop version)

    Args:
        A_sign: Sign matrix S_A (n, r) - {-1, +1}
        B_sign: Sign matrix S_B (r, m) - {-1, +1}
        A_amp: Row scale (n, l)
        B_amp: Column scale (m, l)
        Q_U_amp: Latent row scale (r, l)
        Q_V_amp: Latent column scale (r, l)

    Returns:
        Reconstructed weight matrix W (n, m)
    """
    # F = S_A * (A_amp @ Q_U_amp^T)
    amp_A = A_amp @ Q_U_amp.T
    F = A_sign * amp_A
    del amp_A

    # G = S_B * (Q_V_amp @ B_amp^T)
    amp_B = Q_V_amp @ B_amp.T
    G = B_sign * amp_B
    del amp_B

    W_recon = F @ G
    del F, G

    return W_recon
