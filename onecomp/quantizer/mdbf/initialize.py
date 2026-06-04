"""
Phase 1: MDBF Initialization

Perform low-rank decomposition and multi-scale amplitude decomposition using SVD/SVD-LLM/OSVD.

Weight representation:
    W ≈ Σ_{p=1}^{P} W^{(p)}
    W^{(p)} = F^{(p)} @ G^{(p)}
    where F = S_A * (A_amp @ Q_U_amp^T)
          G = S_B * (Q_V_amp @ B_amp^T)

Activation-aware extension:
    act_init="osvd": Initialization using Output-SVD (OSVD)
    - Objective function: min ||Y - Y_hat||_F^2 where Y = X @ W^T

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from dataclasses import dataclass
from logging import getLogger
from typing import List, Literal, Optional, Tuple

import torch

logger = getLogger(__name__)

from .utils import (
    cleanup_gpu_memory,
    ensure_float32,
    reconstruct_weight,
    to_binary_sign,
)


@dataclass
class MDBFParams:
    """MDBF parameters for a single path"""

    A_sign: torch.Tensor  # Sign matrix S_A (n, r) - {-1, +1}
    B_sign: torch.Tensor  # Sign matrix S_B (r, m) - {-1, +1}
    A_amp: torch.Tensor  # Row scale (n, l)
    B_amp: torch.Tensor  # Column scale (m, l)
    Q_U_amp: torch.Tensor  # Latent scale row side (r, l)
    Q_V_amp: torch.Tensor  # Latent scale column side (r, l)


# =============================================================================
# Low-rank decomposition
# =============================================================================


def lowrank_svd(
    W: torch.Tensor,
    r: int,
    H: Optional[torch.Tensor] = None,
    mode: Literal["svd", "svd_llm"] = "svd",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform low-rank decomposition (SVD or SVD-LLM)

    Result: W ≈ U' @ V'^T

    Args:
        W: Input matrix (n, m)
        r: Target rank
        H: Hessian matrix (m, m) - for SVD-LLM
        mode: "svd" or "svd_llm"

    Returns:
        U': (n, r), V': (m, r)
    """
    n, m = W.shape
    r = min(r, min(n, m))
    W_fp32 = ensure_float32(W)

    if mode == "svd_llm" and H is not None:
        return _lowrank_svd_llm(W_fp32, r, H, W.dtype)
    return _lowrank_svd_standard(W_fp32, r, W.dtype)


def _lowrank_svd_standard(
    W_fp32: torch.Tensor,
    r: int,
    orig_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard SVD (stabilized with random noise)"""
    # Stabilize with random noise (avoid constant addition as it introduces rank-1 bias)
    eps = 1e-6 * W_fp32.abs().max().clamp(min=1e-12)
    W_reg = W_fp32 + eps * torch.randn_like(W_fp32)
    U_r, S_r, Vh_r = torch.linalg.svd(W_reg, full_matrices=False)
    del W_reg
    U_r = U_r[:, :r]
    S_r = S_r[:r]
    Vh_r = Vh_r[:r, :]

    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))
    U_prime = U_r * sqrt_S[None, :]
    V_prime = Vh_r.T * sqrt_S[None, :]
    del U_r, S_r, Vh_r, sqrt_S, W_fp32

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return U_prime.to(orig_dtype), V_prime.to(orig_dtype)


def _lowrank_svd_llm(
    W_fp32: torch.Tensor,
    r: int,
    H: torch.Tensor,
    orig_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SVD-LLM: Decomposition using Hessian matrix
    Objective: min_{rank(W')=r} ||WX - W'X||_F^2
    """
    m = W_fp32.shape[1]
    H_fp32 = ensure_float32(H, W_fp32.device)
    H_fp32 = (H_fp32 + H_fp32.T) / 2.0  # Symmetrize

    # Add regularization
    eye = torch.eye(m, device=H_fp32.device, dtype=H_fp32.dtype)
    diag_mean = H_fp32.diag().mean().clamp(min=1e-8)
    eps = 1e-4 * diag_mean
    H_reg = H_fp32 + eps * eye
    del H_fp32

    # Cholesky decomposition
    try:
        S = torch.linalg.cholesky(H_reg)
        del H_reg
    except RuntimeError as e:
        logger.warning(f"[SVD-LLM] Cholesky failed ({e}), falling back to standard SVD")
        del H_reg, eye
        return _lowrank_svd_standard(W_fp32, r, orig_dtype)

    # W_tilde = W @ S (stabilized with random noise, avoid constant addition as it introduces rank-1 bias)
    W_tilde = torch.mm(W_fp32, S)
    eps_tilde = 1e-6 * W_tilde.abs().max().clamp(min=1e-12)
    W_tilde_reg = W_tilde + eps_tilde * torch.randn_like(W_tilde)
    U_r, S_r, Vh_r = torch.linalg.svd(W_tilde_reg, full_matrices=False)
    del W_tilde, W_tilde_reg

    U_r = U_r[:, :r]
    S_r = S_r[:r]
    Vh_r = Vh_r[:r, :]

    sqrt_S = torch.sqrt(S_r)
    U_prime = U_r * sqrt_S[None, :]
    V_prime_T = Vh_r * sqrt_S[:, None]  # (r, m) = sqrt(Σ) @ V^T
    del U_r, S_r, Vh_r, sqrt_S

    # Compute V' = S^{-T} @ (V @ sqrt(Σ))
    # W = W_tilde @ S^{-1} = U' @ V_prime_T @ S^{-1} = U' @ (S^{-T} @ V_prime_T^T)^T
    # Therefore, V' = S^{-T} @ V_prime_T^T
    try:
        # Solve S^T @ X = V_prime_T^T → X = S^{-T} @ V_prime_T^T
        V_prime = torch.linalg.solve_triangular(S.T, V_prime_T.T, upper=True)
    except RuntimeError as e:
        logger.warning(f"[SVD-LLM] solve_triangular failed ({e}), using fallback")
        try:
            S_inv = torch.linalg.solve(S, eye)
        except RuntimeError:
            S_inv = torch.linalg.lstsq(S, eye).solution
        # V' = S^{-T} @ V_prime_T^T = (V_prime_T @ S^{-1})^T
        V_prime = (V_prime_T @ S_inv).T
        del S_inv

    del S, eye, V_prime_T, W_fp32

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return U_prime.to(orig_dtype), V_prime.to(orig_dtype)


def lowrank_osvd(
    W: torch.Tensor,
    H: torch.Tensor,
    r: int,
    ridge: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Output-SVD (OSVD) based low-rank decomposition (Hessian-based)

    Objective: min tr((W - VU^T) H (W - VU^T)^T)

    Compute a low-rank approximation that minimizes the output error using the Hessian H = X^T X / N

    Args:
        W: Input matrix (n, m)
        H: Hessian matrix (m, m) = X^T X / N
        r: Target rank
        ridge: Ridge regularization

    Returns:
        U': (n, r), V': (m, r)
    """
    n, m = W.shape
    r = min(r, min(n, m))

    W_fp32 = ensure_float32(W)
    H_fp32 = ensure_float32(H)

    # Regularize and eigen decomposition of Hessian
    diag_mean = H_fp32.diag().mean().clamp(min=1e-12)
    eps = ridge * diag_mean
    H_reg = H_fp32 + eps * torch.eye(m, device=H_fp32.device, dtype=H_fp32.dtype)

    try:
        eig_vals, eig_vecs = torch.linalg.eigh(H_reg)
    except RuntimeError:
        # If eigen decomposition fails: fallback to standard SVD
        del H_reg, H_fp32
        return _lowrank_svd_standard(W, r, W.dtype)

    eig_vals = eig_vals.clamp(min=1e-12)
    sqrt_eig = torch.sqrt(eig_vals)

    # W_tilde = W @ H^{1/2} = W @ Q @ diag(sqrt(λ))
    W_tilde = W_fp32 @ eig_vecs @ torch.diag(sqrt_eig)
    del H_reg

    # Rank-r SVD of W_tilde
    eps_svd = 1e-6 * W_tilde.abs().max().clamp(min=1e-12)
    W_tilde_reg = W_tilde + eps_svd * torch.randn_like(W_tilde)
    U_w, S_w, Vh_w = torch.linalg.svd(W_tilde_reg, full_matrices=False)
    del W_tilde, W_tilde_reg

    r_eff = min(r, S_w.numel())
    U_r = U_w[:, :r_eff]
    S_r = S_w[:r_eff]
    V_r = Vh_w[:r_eff, :].T  # (m, r_eff)
    del U_w, S_w, Vh_w

    # U' = U_r @ diag(sqrt(S_r))
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))
    U_prime = U_r * sqrt_S[None, :]

    # V' = H^{-1/2} @ V_r @ diag(sqrt(S_r))
    # H^{-1/2} = Q @ diag(1/sqrt(λ)) @ Q^T
    inv_sqrt_eig = 1.0 / sqrt_eig
    V_prime = eig_vecs @ torch.diag(inv_sqrt_eig) @ eig_vecs.T @ V_r @ torch.diag(sqrt_S)

    del eig_vals, eig_vecs, sqrt_eig, inv_sqrt_eig, U_r, S_r, V_r, sqrt_S
    del H_fp32, W_fp32
    cleanup_gpu_memory()

    return U_prime.to(W.dtype), V_prime.to(W.dtype)


# =============================================================================
# Amplitude decomposition
# =============================================================================


def amplitude_rank_l_approx(U_abs: torch.Tensor, l: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank-l approximation of the amplitude matrix: |U'| ≈ A @ Q_U^T

    Args:
        U_abs: |U'| (n, r) - Non-negative matrix
        l: Multi-scale rank

    Returns:
        A: (n, l), Q_U: (r, l)
    """
    n, r = U_abs.shape
    l = min(l, min(n, r))

    U_abs_fp32 = U_abs.float().clamp(min=1e-8)
    # Regularization for non-negative matrix (add constant, avoid random values that can be negative)
    eps = 1e-6 * U_abs_fp32.max().clamp(min=1e-12)
    U_abs_reg = U_abs_fp32 + eps

    U_svd, S_svd, Vh_svd = torch.linalg.svd(U_abs_reg, full_matrices=False)
    del U_abs_fp32, U_abs_reg

    U_svd = U_svd[:, :l]
    S_svd = S_svd[:l]
    Vh_svd = Vh_svd[:l, :]

    sqrt_S = torch.sqrt(S_svd.clamp(min=1e-12))
    A = U_svd * sqrt_S[None, :]
    Q_U = Vh_svd.T * sqrt_S[None, :]
    del U_svd, S_svd, Vh_svd, sqrt_S

    # Align the sign of each column (adjust to make the product non-negative)
    col_sum = A.sum(dim=0)
    mask = col_sum < 0
    A[:, mask] = -A[:, mask]
    Q_U[:, mask] = -Q_U[:, mask]

    return A.to(U_abs.dtype), Q_U.to(U_abs.dtype)


# =============================================================================
# Path initialization
# =============================================================================


def init_single_path(
    W: torch.Tensor,
    r: int,
    l: int = 1,
    H: Optional[torch.Tensor] = None,
    mode: Literal["svd", "svd_llm"] = "svd",
    act_init: Literal["none", "osvd", "svd_llm"] = "none",
) -> MDBFParams:
    """
    Initialize MDBF parameters for a single path

    Args:
        W: Input matrix (n, m)
        r: Rank
        l: Multi-scale rank
        H: Hessian matrix (m, m) - for SVD-LLM/OSVD
        mode: SVD mode ("svd" or "svd_llm")
        act_init: Initialization mode ("none", "osvd", "svd_llm")

    Returns:
        Initialized MDBFParams
    """
    # Low-rank decomposition
    if act_init == "osvd" and H is not None:
        U_prime, V_prime = lowrank_osvd(W, H, r)
    else:
        U_prime, V_prime = lowrank_svd(W, r, H, mode)

    # Binarize sign matrices
    A_sign = to_binary_sign(U_prime)
    B_sign = to_binary_sign(V_prime.T)

    # Rank-l approximation of the amplitude
    A_amp, Q_U_amp = amplitude_rank_l_approx(U_prime.abs(), l)
    B_amp, Q_V_amp = amplitude_rank_l_approx(V_prime.abs(), l)

    del U_prime, V_prime
    cleanup_gpu_memory()

    return MDBFParams(
        A_sign=A_sign,
        B_sign=B_sign,
        A_amp=A_amp,
        B_amp=B_amp,
        Q_U_amp=Q_U_amp,
        Q_V_amp=Q_V_amp,
    )


def initialize_MDBF(
    W: torch.Tensor,
    r: int,
    l: int = 1,
    P: int = 2,
    H: Optional[torch.Tensor] = None,
    mode: Literal["svd", "svd_llm"] = "svd",
    act_init: Literal["none", "osvd", "svd_llm"] = "none",
) -> Tuple[List[MDBFParams], torch.Tensor]:
    """
    Phase 1: MDBF initialization (all paths)

    Args:
        W: Input matrix (n, m)
        r: Rank
        l: Multi-scale rank
        P: Number of paths (1=Primary, 2=Primary+Residual, ...)
        H: Hessian matrix (m, m) - for SVD-LLM/OSVD
        mode: SVD mode ("svd" or "svd_llm")
        act_init: Initialization mode ("none", "osvd", "svd_llm")

    Returns:
        all_params: MDBFParams for each path
        W_recon: Reconstructed weight matrix
    """
    n, m = W.shape
    W_float = ensure_float32(W)
    orig_norm = torch.norm(W_float, p="fro").item()

    # Display OSVD initialization
    if act_init == "osvd" and H is not None:
        logger.debug("[MDBF Init] Using OSVD initialization (Hessian-based)")

    W_residual = W_float.clone()
    W_recon = torch.zeros_like(W_float)
    all_params: List[MDBFParams] = []

    for p in range(P):
        if act_init == "osvd" and H is not None:
            params = init_single_path(W_residual, r, l, H, mode, act_init=act_init)
        else:
            params = init_single_path(W_residual, r, l, H, mode)
        all_params.append(params)

        W_p = reconstruct_weight(
            params.A_sign,
            params.B_sign,
            params.A_amp,
            params.B_amp,
            params.Q_U_amp,
            params.Q_V_amp,
        )

        if p == 0:
            error_p = torch.norm(W_float - W_p, p="fro").item()
            logger.debug(
                f"[MDBF Init] Primary path error: {error_p:.4e} (rel: {error_p/orig_norm:.4f})"
            )

        W_residual -= W_p
        W_recon += W_p
        del W_p
        cleanup_gpu_memory()

    final_error = torch.norm(W_float - W_recon, p="fro").item()
    logger.debug(
        f"[MDBF Init] Final weight error: {final_error:.4e} (rel: {final_error/orig_norm:.4f})"
    )

    del W_float, W_residual
    cleanup_gpu_memory()

    return all_params, W_recon
