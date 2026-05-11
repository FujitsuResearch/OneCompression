# -*- coding: utf-8 -*-
"""
MDBF-ADMM: ADMM optimization using Multi-Scale SVID (Phase 2)

Using the MDBFParams initialized in Phase 1 as the initial values,
minimize the residual W - Σ_{p≠p'} W^p for each path p'.


Activation-aware extension:
- Hessian-based: Use H = X^T @ X / N to efficiently minimize output error
- Objective function: min tr((W - W_hat) @ H @ (W - W_hat)^T)

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from logging import getLogger
from typing import List, Optional, Tuple

import torch

logger = getLogger(__name__)

from .initialize import MDBFParams
from .utils import (
    cleanup_gpu_memory,
    compute_hessian_error,
    ensure_float32,
    ensure_float32_clone,
    reconstruct_weight,
    symmetrize_matrix,
    to_binary_sign,
)


# =============================================================================
# Helper functions
# =============================================================================


def _tsvd_block_power(
    M: torch.Tensor,
    k: int,
    n_iter: int = 5,
    oversample: int = 4,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Approximate top-k singular value decomposition using block power iteration (subspace iteration).

    Args:
        M: (n, m) float32 recommended
        k: desired rank
        n_iter: number of power/subspace iterations (1-3 recommended)
        oversample: oversampling (2-8 recommended)
        seed: random seed (if None, use global seed)

    Returns:
        U: (n, k)
        S: (k,)
        V: (m, k)
        such that M ≈ (U * S) @ V^T
    """
    n, m = M.shape
    k_eff = min(k, n, m)
    if k_eff <= 0:
        raise ValueError("k must be >= 1")
    k0 = min(k_eff + oversample, n, m)

    # Initialize the right subspace (random)
    if seed is not None:
        gen = torch.Generator(device=M.device).manual_seed(seed)
        V = torch.randn(m, k0, device=M.device, dtype=M.dtype, generator=gen)
    else:
        V = torch.randn(m, k0, device=M.device, dtype=M.dtype)
    V, _ = torch.linalg.qr(V, mode="reduced")  # (m, k0)

    # Subspace iteration
    for _ in range(max(0, n_iter)):
        U = M @ V                              # (n, k0)
        U, _ = torch.linalg.qr(U, mode="reduced")
        V = M.T @ U                            # (m, k0)
        V, _ = torch.linalg.qr(V, mode="reduced")

    # Finally, QR decomposition of M@V to form the small matrix R
    Y = M @ V                                  # (n, k0)
    U, R = torch.linalg.qr(Y, mode="reduced")  # U:(n,k0), R:(k0,k0)
    del Y

    # SVD of the small matrix only (default driver)
    Ur, S, Vhr = torch.linalg.svd(R, full_matrices=False)
    del R

    U = U @ Ur[:, :k_eff]                      # (n, k)
    V = V @ Vhr[:k_eff, :].T                   # (m, k)
    S = S[:k_eff]

    return U, S, V


def _decompose_abs_matrix(
    M_abs: torch.Tensor,
    l: int,
    for_transpose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose the amplitude matrix |M| into rank-l components.

    Args:
        M_abs: Amplitude matrix (a, b)
        l: Rank
        for_transpose: If True, decompose M^T (for G side)

    Returns:
        (scale_left, scale_right): |M| ≈ scale_left @ scale_right^T
    """
    if for_transpose:
        M_abs = M_abs.T

    a, b = M_abs.shape
    l_eff = min(l, min(a, b))

    # Regularization for non-negative matrix (add constant, avoid negative values from random)
    eps = 1e-6 * M_abs.max().clamp(min=1e-12)
    M_abs_reg = M_abs + eps

    U, S, Vh = torch.linalg.svd(M_abs_reg, full_matrices=False)
    l_eff = min(l_eff, S.numel())
    sqrt_S = torch.sqrt(S[:l_eff].clamp(min=1e-12))
    left = U[:, :l_eff] * sqrt_S[None, :]
    right = Vh[:l_eff, :].T * sqrt_S[None, :]
    del U, S, Vh, M_abs_reg

    return left, right


def _solve_linear_system(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    L_cached: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Solve the linear system lhs @ X = rhs (using cached Cholesky decomposition)

    Args:
        lhs: Left-hand side matrix (k, k)
        rhs: Right-hand side (k, ...) or (k,)
        L_cached: Cholesky-decomposed matrix (if None, solve directly)

    Returns:
        Solution X

    Note:
        Fallback order: cholesky_solve → solve → lstsq
    """
    if L_cached is not None:
        if rhs.dim() == 1:
            return torch.cholesky_solve(rhs.unsqueeze(-1), L_cached).squeeze(-1)
        return torch.cholesky_solve(rhs, L_cached)

    # Cholesky decomposition is not cached, solve directly
    try:
        return torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        # Fallback to least squares for singular or numerically unstable matrices
        if rhs.dim() == 1:
            return torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
        return torch.linalg.lstsq(lhs, rhs).solution


# =============================================================================
# MDBF projection
# =============================================================================


def svd_abs_rank_l(
    W: torch.Tensor,
    l: int,
    seed: int = None,
) -> torch.Tensor:
    """
    MDBF (rank-l) projection: Z = sign(W) * TSVD_l(|W|)

    Fix the sign to sign(W) and apply a rank-l constraint to the amplitude.

    Args:
        W: Input matrix
        l: Multi-scale rank
        seed: Random seed (if None, use global seed; if integer, deterministic)
    """
    S = torch.sign(W)
    S[S == 0] = 1.0

    if l == 1:
        # Power iteration for rank-1 approximation (randn initialization)
        # Number of iterations is unified with block power iteration for l>=2 (n_iter=5)
        W_abs = W.abs().float()
        n, m_dim = W_abs.shape

        # Random initialization (robust for degenerate eigenvalues)
        # Deterministic initialization when seed is specified (for reproducibility)
        if seed is not None:
            gen = torch.Generator(device=W.device).manual_seed(seed)
            a = torch.randn(n, device=W.device, dtype=torch.float32, generator=gen)
            m_vec = torch.randn(m_dim, device=W.device, dtype=torch.float32, generator=gen)
        else:
            a = torch.randn(n, device=W.device, dtype=torch.float32)
            m_vec = torch.randn(m_dim, device=W.device, dtype=torch.float32)
        a = a / (torch.norm(a) + 1e-12)
        m_vec = m_vec / (torch.norm(m_vec) + 1e-12)

        for _ in range(5):
            a = W_abs @ m_vec
            a.div_(a.norm().clamp(min=1e-12))
            m_vec = W_abs.T @ a
            m_vec.div_(m_vec.norm().clamp(min=1e-12))

        sigma = torch.dot(a, W_abs @ m_vec).clamp(min=1e-12)
        root_sigma = torch.sqrt(sigma)
        a = a * root_sigma
        m_vec = m_vec * root_sigma

        amp = torch.outer(a, m_vec)
        del W_abs, a, m_vec
    else:
        # Block power iteration for acceleration
        W_abs = W.abs().float()
        n_dim, m_dim = W_abs.shape
        kmax = min(n_dim, m_dim)

        if l >= kmax:
            amp = W_abs
        else:
            l_eff = max(1, min(l, kmax - 1))
            # Block power iteration for acceleration (QR and GEMM based, so no regularization needed)
            U_l, S_l, V_l = _tsvd_block_power(W_abs, k=l_eff, seed=seed)
            amp = (U_l * S_l[None, :]) @ V_l.T
            del U_l, S_l, V_l
        del W_abs

    Z = S * amp.to(W.dtype)
    del S, amp
    return Z


# =============================================================================
# Parameter Conversion
# =============================================================================


def _params_to_factor_matrices(
    params: MDBFParams,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct factor matrices (F, G) from MDBFParams

    F = S_A * (A_amp @ Q_U_amp^T)
    G = S_B * (Q_V_amp @ B_amp^T)
    """
    A_sign = params.A_sign.to(device).float()
    B_sign = params.B_sign.to(device).float()
    A_amp = params.A_amp.to(device).float()
    B_amp = params.B_amp.to(device).float()
    Q_U_amp = params.Q_U_amp.to(device).float()
    Q_V_amp = params.Q_V_amp.to(device).float()

    amp_A = A_amp @ Q_U_amp.T
    F = A_sign * amp_A

    amp_B = Q_V_amp @ B_amp.T
    G = B_sign * amp_B

    del A_sign, B_sign, A_amp, B_amp, Q_U_amp, Q_V_amp, amp_A, amp_B
    return F, G


def _factor_matrices_to_params(
    F: torch.Tensor,
    G: torch.Tensor,
    l: int,
    dtype: torch.dtype,
) -> MDBFParams:
    """
    Extract MDBFParams from factor matrices (F, G)

    Note:
        F, G are after ADMM projection, so |F|, |G| are already rank-l.
    """
    A_sign = to_binary_sign(F)
    B_sign = to_binary_sign(G)

    F_abs = F.abs().float()
    G_abs = G.abs().float()

    # |F| ≈ A_amp @ Q_U_amp^T
    A_amp, Q_U_amp = _decompose_abs_matrix(F_abs, l)
    del F_abs

    # |G| ≈ Q_V_amp @ B_amp^T (decompose G^T)
    B_amp, Q_V_amp = _decompose_abs_matrix(G_abs, l, for_transpose=True)
    del G_abs

    return MDBFParams(
        A_sign=A_sign.to(dtype),
        B_sign=B_sign.to(dtype),
        A_amp=A_amp.to(dtype),
        B_amp=B_amp.to(dtype),
        Q_U_amp=Q_U_amp.to(dtype),
        Q_V_amp=Q_V_amp.to(dtype),
    )


# =============================================================================
# One-Side Fixed ADMM
# =============================================================================


def _admm_optimize_one_side(
    Fixed: torch.Tensor,
    W_target: torch.Tensor,
    Z_init: torch.Tensor,
    U_init: torch.Tensor,
    l: int,
    inner_iters: int = 3,
    reg: float = 0.03,
    rho_start: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-side fixed ADMM optimization (DBF-compatible fixed rho mode)

    min ||W_target - Fixed @ Z||_F^2 + λ*||Z||_F^2  s.t. Z in MDBF(l)

    Here, λ = reg * mean(diag(Fixed^T @ Fixed))
    (Regularization proportional to the mean of the diagonal elements of Fixed^T @ Fixed)

    Args:
        Fixed: Fixed-side matrix
        W_target: Target matrix
        Z_init, U_init: Initial values
        l: Multi-scale rank
        inner_iters: Number of inner iterations
        reg: Regularization coefficient (multiplied by the mean of the diagonal elements of Fixed^T @ Fixed)
        rho_start: Initial rho

    Returns:
        (Z, U): Optimized variables
    """
    device = Fixed.device
    Fixed = ensure_float32(Fixed)
    W_target = ensure_float32(W_target)
    Z = ensure_float32_clone(Z_init)
    U = ensure_float32_clone(U_init)

    k = Fixed.shape[1]
    I = torch.eye(k, device=device, dtype=torch.float32)

    XX = Fixed.T @ Fixed
    if reg > 0:
        XX = XX + I * (XX.diag().mean() * reg)
    XY = Fixed.T @ W_target

    Z, U = _admm_fixed_rho_loop(XX, XY, I, Z, U, l, inner_iters, rho_start)

    del XX, XY, I
    return Z.float(), U.float()


def _admm_fixed_rho_loop(
    XX: torch.Tensor,
    XY: torch.Tensor,
    I: torch.Tensor,
    Z: torch.Tensor,
    U: torch.Tensor,
    l: int,
    inner_iters: int,
    rho_start: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ADMM inner loop in fixed rho mode"""
    rho = 1.0
    lhs_rho = XX + rho * I
    lhs_rho_start = XX + rho_start * I

    try:
        L_rho = torch.linalg.cholesky(lhs_rho)
    except RuntimeError:
        L_rho = None

    try:
        L_rho_start = torch.linalg.cholesky(lhs_rho_start)
    except RuntimeError:
        L_rho_start = None

    # Initial B update (rho_start)
    rhs = XY + rho_start * (Z - U)
    B = _solve_linear_system(lhs_rho_start, rhs, L_rho_start)

    # Remaining iterations (rho=1.0)
    for _ in range(inner_iters - 1):
        Z = svd_abs_rank_l(B + U, l)
        U = U + (B - Z)
        rhs = XY + rho * (Z - U)
        B = _solve_linear_system(lhs_rho, rhs, L_rho)

    # Final Z, U update
    Z = svd_abs_rank_l(B + U, l)
    U = U + (B - Z)

    del lhs_rho, lhs_rho_start
    if L_rho is not None:
        del L_rho
    if L_rho_start is not None:
        del L_rho_start

    return Z, U


# =============================================================================
# Single-Path ADMM Optimization
# =============================================================================


def _admm_refine_single_path(
    W_target: torch.Tensor,
    F_init: torch.Tensor,
    G_init: torch.Tensor,
    l: int,
    iters: int = 50,
    inner_iters: int = 3,
    reg: float = 0.03,
    path_idx: int = 0,
    H: Optional[torch.Tensor] = None,
    nsamples: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ADMM optimization for a single path (alternating F<->G optimization, DBF-compatible fixed rho mode)

    min ||W_target - F @ G||_F^2  s.t. F,G in MDBF(l)
    """
    device = W_target.device
    n, r = F_init.shape
    _, m = G_init.shape

    # F is stored in transposed form (r, n)
    Zf_T = ensure_float32_clone(F_init.T)
    Uf_T = torch.zeros(r, n, device=device, dtype=torch.float32)
    Zg = ensure_float32_clone(G_init)
    Ug = torch.zeros_like(Zg)

    W_float = ensure_float32(W_target)
    orig_norm = torch.norm(W_float, p='fro').item() + 1e-12
    W_float_T = W_float.T

    # For output error display (H is assumed to be symmetrized by the caller)
    H_float = None
    orig_output_err = None
    if H is not None:
        H_float = ensure_float32(H, device)
        WH = W_float @ H_float
        orig_output_err = (nsamples * torch.sum(WH * W_float)).item() + 1e-12

    # ===== Scale Normalization Strategy =====
    # In alternating optimization of W ≈ F @ G, there is freedom in the scale of F and G.
    # (F * c) @ (G / c) = F @ G, so the scale needs to be concentrated on one side.
    #
    # Strategy:
    # 1. When updating F: Normalize each row of G → Scale is absorbed into F
    # 2. When updating G: Normalize each column of F → Scale is absorbed into G
    # 3. Final output: Normalize each column of F before returning → Scale is concentrated in G
    #
    # This ensures that the final F @ G approximates the original W, with F's column norms equal to 1.

    for itt in range(iters):
        rho_start = 0.03 + (1.0 - 0.03) * min(1.0, itt / max(1, iters - 3)) ** 3

        # F update (G fixed): Normalize each row of G
        # Optimization problem: min ||W^T - G_normalized @ F^T||_F^2 s.t. F in MDBF(l)
        mid_norm_g = torch.norm(Zg, dim=1) + 1e-12
        Zg_normalized = Zg / mid_norm_g[:, None]

        Zf_T, Uf_T = _admm_optimize_one_side(
            Fixed=Zg_normalized.T,
            W_target=W_float_T,
            Z_init=Zf_T,
            U_init=Uf_T,
            l=l,
            inner_iters=inner_iters,
            reg=reg,
            rho_start=rho_start,
        )

        # G update (F fixed): Normalize each column of F
        # Optimization problem: min ||W - F_normalized @ G||_F^2 s.t. G in MDBF(l)
        mid_norm_f = torch.norm(Zf_T, dim=1) + 1e-12
        Zf_normalized = Zf_T.T / mid_norm_f[None, :]

        Zg, Ug = _admm_optimize_one_side(
            Fixed=Zf_normalized,
            W_target=W_float,
            Z_init=Zg,
            U_init=Ug,
            l=l,
            inner_iters=inner_iters,
            reg=reg,
            rho_start=rho_start,
        )

        # Log output
        if (itt % max(10, iters // 5) == 0 or itt == iters - 1):
            mid_norm_final = torch.norm(Zf_T, dim=1) + 1e-12
            W_recon = (Zf_T.T / mid_norm_final[None, :]) @ Zg
            E = W_float - W_recon

            if H_float is not None:
                output_err = compute_hessian_error(E, H_float, nsamples)
                logger.debug(f"[MDBF-ADMM] Path {path_idx+1}, Outer Step {itt:3d}: "
                             f"output_error = {output_err:.4e} (rel: {output_err/orig_output_err:.4f}), "
                             f"rho_start={rho_start:.3f}")
            else:
                err = torch.norm(E, p='fro').item()
                logger.debug(f"[MDBF-ADMM] Path {path_idx+1}, Outer Step {itt:3d}: "
                             f"weight_error = {err:.4e} (rel: {err/orig_norm:.4f}), "
                             f"rho_start={rho_start:.3f}")
            del W_recon, E

    del W_float, W_float_T, Uf_T, Ug
    if H_float is not None:
        del H_float

    # Final scale normalization: Set the column norms of F to 1 and concentrate the scale in G
    # This ensures that W ≈ F_normalized @ G holds, and F becomes a normalized factor matrix
    mid_norm_final = torch.norm(Zf_T, dim=1) + 1e-12
    Zf_normalized = Zf_T.T / mid_norm_final[None, :]

    return Zf_normalized, Zg


# =============================================================================
# Standard ADMM Optimization
# =============================================================================


def optimize_MDBF_admm(
    W_original: torch.Tensor,
    params_list: List[MDBFParams],
    l: int,
    iters: int = 260,
    inner_iters: int = 3,
    reg: float = 0.03,
    verbose: bool = True,
    H: Optional[torch.Tensor] = None,
    nsamples: int = 1,
) -> Tuple[List[MDBFParams], torch.Tensor]:
    """
    Refine MDBF parameters using ADMM optimization (Phase 2, DBF-compatible fixed rho mode)

    Minimize the residual W - Σ_{p≠p'} W^p for each path p'
    """
    device = W_original.device
    dtype = W_original.dtype
    n, m = W_original.shape
    P = len(params_list)

    logger.debug(f"[MDBF-ADMM] outer_iters={iters}, inner_iters={inner_iters}, "
                 f"l={l}, P={P}, reg={reg}")

    W_float = ensure_float32(W_original)
    orig_norm = torch.norm(W_float, p='fro').item() + 1e-12

    # Hessian preprocessing
    H_float = None
    orig_output_err = None
    if H is not None:
        H_float = symmetrize_matrix(ensure_float32(H, device))
        WH = W_float @ H_float
        orig_output_err = (nsamples * torch.sum(WH * W_float)).item() + 1e-12

    # Initial error
    W_init_recon = torch.zeros(n, m, device=device, dtype=torch.float32)
    for p in params_list:
        W_p = reconstruct_weight(
            p.A_sign.to(device), p.B_sign.to(device),
            p.A_amp.to(device), p.B_amp.to(device),
            p.Q_U_amp.to(device), p.Q_V_amp.to(device),
        )
        W_init_recon += W_p
        del W_p

    E_init = W_float - W_init_recon
    if H_float is not None:
        init_error = compute_hessian_error(E_init, H_float, nsamples)
        logger.debug(f"[MDBF-ADMM] Initial output_error: {init_error:.4e} "
                     f"(rel: {init_error/orig_output_err:.4f})")
    else:
        init_error = torch.norm(E_init, p='fro').item()
        logger.debug(f"[MDBF-ADMM] Initial weight_error: {init_error:.4e} "
                     f"(rel: {init_error/orig_norm:.4f})")
    del W_init_recon, E_init

    # Initialize factor matrices
    factor_list = [_params_to_factor_matrices(p, device) for p in params_list]

    # Optimize each path
    optimized_factors = []
    for p_idx in range(P):
        logger.debug(f"[MDBF-ADMM] Optimizing path {p_idx+1}/{P}...")

        # Residual
        W_target = W_float.clone()
        for other_idx in range(P):
            if other_idx != p_idx:
                F_other, G_other = (optimized_factors[other_idx]
                                    if other_idx < len(optimized_factors)
                                    else factor_list[other_idx])
                W_target -= F_other @ G_other

        F_init, G_init = factor_list[p_idx]
        F_opt, G_opt = _admm_refine_single_path(
            W_target=W_target,
            F_init=F_init,
            G_init=G_init,
            l=l,
            iters=iters,
            inner_iters=inner_iters,
            reg=reg,
            path_idx=p_idx,
            H=H_float,
            nsamples=nsamples,
        )
        optimized_factors.append((F_opt, G_opt))
        del W_target, F_init, G_init

    # Final reconstruction
    W_recon = torch.zeros(n, m, device=device, dtype=torch.float32)
    for F_opt, G_opt in optimized_factors:
        W_recon += F_opt @ G_opt

    E_final = W_float - W_recon
    if H_float is not None:
        final_error = compute_hessian_error(E_final, H_float, nsamples)
        improvement = (init_error - final_error) / init_error * 100
        logger.debug(f"[MDBF-ADMM] Final output_error: {final_error:.4e} "
                     f"(rel: {final_error/orig_output_err:.4f})")
        logger.debug(f"[MDBF-ADMM] Improvement: {improvement:+.2f}%")
    else:
        final_error = torch.norm(E_final, p='fro').item()
        improvement = (init_error - final_error) / init_error * 100
        logger.debug(f"[MDBF-ADMM] Final weight_error: {final_error:.4e} "
                     f"(rel: {final_error/orig_norm:.4f})")
        logger.debug(f"[MDBF-ADMM] Improvement: {improvement:+.2f}%")
    del E_final

    # Convert to MDBFParams
    optimized_params = [
        _factor_matrices_to_params(F_opt, G_opt, l, dtype)
        for F_opt, G_opt in optimized_factors
    ]

    W_recon_params = torch.zeros(n, m, device=device, dtype=torch.float32)
    for p in optimized_params:
        W_p = reconstruct_weight(
            p.A_sign.to(device), p.B_sign.to(device),
            p.A_amp.to(device), p.B_amp.to(device),
            p.Q_U_amp.to(device), p.Q_V_amp.to(device),
        )
        W_recon_params += W_p
        del W_p
    E_params = W_float - W_recon_params
    if H_float is not None:
        params_error = compute_hessian_error(E_params, H_float, nsamples)
        logger.debug(f"[MDBF-ADMM] Error (via params): {params_error:.4e} "
                     f"(rel: {params_error/orig_output_err:.4f})")
    else:
        params_error = torch.norm(E_params, p='fro').item()
        logger.debug(f"[MDBF-ADMM] Error (via params): {params_error:.4e} "
                     f"(rel: {params_error/orig_norm:.4f})")
    del W_recon_params, E_params

    del W_float
    if H_float is not None:
        del H_float
    for F, G in factor_list:
        del F, G
    for F, G in optimized_factors:
        del F, G
    cleanup_gpu_memory()

    return optimized_params, W_recon.to(dtype)


# =============================================================================
# Hessian-based Activation-Aware ADMM
# =============================================================================


def _admm_refine_single_path_hessian(
    H: torch.Tensor,
    nsamples: int,
    W_target: torch.Tensor,
    F_init: torch.Tensor,
    G_init: torch.Tensor,
    l: int,
    iters: int = 50,
    inner_iters: int = 3,
    reg: float = 0.03,
    path_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hessian-based Activation-aware version: Single-path ADMM optimization

    Objective function: min tr((W - VU^T) @ H @ (W - VU^T)^T)  s.t. U,V in MDBF(l)

    Stabilization modifications:
    - Regularize H consistently (H_use = H + eps * I) - Prevent drift in the null space direction
    - Balance gauge (equalize the norms of V and U) - Prevent extreme scaling
    - Always add a small ridge to U/V updates - Suppress divergence
    - Periodically check the error and save the best result - Handle non-monotonic convergence
    """
    device = H.device
    m = H.shape[0]
    n, r = F_init.shape
    N = float(nsamples)

    # V = F (n, r), U = G^T (m, r)
    V = ensure_float32_clone(F_init)
    U = ensure_float32_clone(G_init.T)

    # Track the best result (Problem 3 & 5)
    best_V = V.clone()
    best_U = U.clone()
    best_err = float('inf')
    check_interval = max(10, iters // 10)  # Check every 10 iterations or 1/10 of total

    LamU = torch.zeros_like(U)
    GamV = torch.zeros_like(V)
    I_r = torch.eye(r, device=device, dtype=torch.float32)
    I_m = torch.eye(m, device=device, dtype=torch.float32)

    W_float = ensure_float32(W_target)
    H0 = symmetrize_matrix(ensure_float32(H))

    # Regularize H consistently (use the same H_use for both eigh and update equations)
    # This prevents drift in the null space direction of H
    H_diag_mean = H0.diag().mean().clamp(min=1e-12)
    eps_H = 1e-3 * H_diag_mean  # Fixed regularization coefficient
    H_use = H0 + eps_H * I_m
    del H0

    # Precompute the eigen decomposition of the Hessian
    H_eig_vals, H_eig_vecs = torch.linalg.eigh(H_use)
    H_eig_vals = H_eig_vals.clamp(min=1e-12)

    W_norm = torch.norm(W_float, p='fro').item() + 1e-12
    rho_scale = N * H_diag_mean

    for itt in range(iters):
        rho_base = 0.03 + (1.0 - 0.03) * min(1.0, itt / max(1, iters - 3)) ** 3
        rho = rho_base * rho_scale
        rho_over_N = rho / N

        # ===== V update (U fixed) =====
        # Normal equation: V (U^T H_use U + (ρ/N + λ_v) I) = W H_use U + ρ/N (Z_V - Γ_V)
        HU = H_use @ U
        UtHU = U.T @ HU
        WHU = W_float @ HU

        # Always add a small ridge to suppress divergence
        lambda_v = reg * UtHU.diag().mean().clamp(min=1e-12)
        lhs_v = UtHU + (rho_over_N + lambda_v) * I_r

        L_v = None
        try:
            L_v = torch.linalg.cholesky(lhs_v)
        except RuntimeError:
            L_v = None

        for _ in range(inner_iters):
            rhs = WHU + rho_over_N * (V - GamV)
            if L_v is not None:
                Vtilde = torch.cholesky_solve(rhs.T, L_v).T
            else:
                try:
                    Vtilde = torch.linalg.solve(lhs_v, rhs.T).T
                except RuntimeError:
                    Vtilde = torch.linalg.lstsq(lhs_v, rhs.T).solution.T

            V = svd_abs_rank_l(Vtilde + GamV, l)
            GamV = GamV + (Vtilde - V)

        # Release intermediate tensors (Problem 10)
        del lhs_v, HU, UtHU, WHU
        if L_v is not None:
            del L_v

        # NaN/Inf check: If detected, use the best result and exit the loop (Problem 3)
        if not torch.isfinite(V).all() or not torch.isfinite(U).all():
            bad_v = (~torch.isfinite(V)).sum().item()
            bad_u = (~torch.isfinite(U)).sum().item()
            logger.warning(f"[MDBF-ADMM] WARNING: Diverged at step {itt} "
                           f"(V: {bad_v}, U: {bad_u} NaN/Inf). Using best result.")
            V = best_V
            U = best_U
            break

        # ===== U update (V fixed) =====
        # Eigen decomposition of V^T V is required
        VtV = V.T @ V
        VtV = (VtV + VtV.T) * 0.5  # Ensure symmetry
        r_dim = VtV.size(0)
        diag_mean_vtv = VtV.diag().mean().clamp(min=1e-12)
        eps_damp = 1e-4 * diag_mean_vtv
        I_VtV = torch.eye(r_dim, device=VtV.device, dtype=VtV.dtype)
        VtV_reg = VtV + eps_damp * I_VtV
        del VtV

        eig_success = False
        try:
            sigma, Q = torch.linalg.eigh(VtV_reg)
            eig_success = True
        except RuntimeError:
            pass

        if not eig_success:
            # Retry with stronger damping
            eps_damp_strong = 1e-2 * diag_mean_vtv
            VtV_reg_strong = VtV_reg + (eps_damp_strong - eps_damp) * I_VtV
            try:
                sigma, Q = torch.linalg.eigh(VtV_reg_strong)
                eig_success = True
            except RuntimeError:
                pass
            del VtV_reg_strong

        if not eig_success:
            # Fallback: block power method
            k_eff = min(l * 25, r_dim)
            try:
                Q_k, sigma_k, _ = _tsvd_block_power(VtV_reg, k=k_eff, n_iter=5, oversample=4)
                sigma = torch.full((r_dim,), eps_damp, device=V.device, dtype=V.dtype)
                sigma[:k_eff] = sigma_k
                del sigma_k
                if k_eff < r_dim:
                    Q_rest = torch.randn(r_dim, r_dim - k_eff, device=V.device, dtype=V.dtype)
                    Q_rest = Q_rest - Q_k @ (Q_k.T @ Q_rest)
                    Q_rest, _ = torch.linalg.qr(Q_rest, mode='reduced')
                    Q = torch.cat([Q_k, Q_rest], dim=1)
                    del Q_rest
                else:
                    Q = Q_k
                del Q_k
                eig_success = True
            except RuntimeError:
                # Final fallback: identity
                sigma = torch.full((r_dim,), eps_damp, device=V.device, dtype=V.dtype)
                Q = torch.eye(r_dim, device=V.device, dtype=V.dtype)
                eig_success = True

        del VtV_reg, I_VtV
        sigma = sigma.clamp(min=1e-12)

        WtV = W_float.T @ V
        HWtV = H_use @ WtV
        del WtV

        # Add regularization to the U side as well
        Hscale = H_eig_vals.mean().clamp(min=1e-12)
        lambda_u = reg * Hscale

        for _ in range(inner_iters):
            RHS = HWtV + rho_over_N * (U - LamU)
            B = RHS @ Q
            QtB = H_eig_vecs.T @ B
            denom = sigma[None, :] * H_eig_vals[:, None] + rho_over_N + lambda_u
            denom = denom.clamp(min=1e-12)
            UQ = H_eig_vecs @ (QtB / denom)
            Utilde = UQ @ Q.T

            U = svd_abs_rank_l(Utilde + LamU, l)
            LamU = LamU + (Utilde - U)

        # Release intermediate tensors (Problem 10)
        del HWtV, Q, sigma

        # NaN/Inf check: If detected, use the best result and exit the loop (Problem 3)
        if not torch.isfinite(U).all():
            bad_u = (~torch.isfinite(U)).sum().item()
            logger.warning(f"[MDBF-ADMM] WARNING: U diverged at step {itt} "
                           f"({bad_u} NaN/Inf). Using best result.")
            V = best_V
            U = best_U
            break

        # Balance gauge fixing (once per outer iteration)
        # Instead of unit norm, balance the norms of V and U
        # d_j = sqrt(||V_j|| / ||U_j||) to evenly distribute the scale
        with torch.no_grad():
            v_norm = torch.linalg.norm(V, dim=0).clamp(min=1e-6)
            u_norm = torch.linalg.norm(U, dim=0).clamp(min=1e-6)
            d = torch.sqrt(v_norm / u_norm)
            # Prohibit extreme scale transformations (effective for preventing divergence)
            d = d.clamp(min=1e-2, max=1e2)

            V = V / d[None, :]
            U = U * d[None, :]
            GamV = GamV / d[None, :]
            LamU = LamU * d[None, :]

        # Periodically check the error and save the best result (Problem 5)
        if itt % check_interval == 0 or itt == iters - 1:
            with torch.no_grad():
                W_recon = V @ U.T
                E = W_float - W_recon
                current_err = compute_hessian_error(E, H_use, N)

                if current_err < best_err:
                    best_err = current_err
                    best_V = V.clone()
                    best_U = U.clone()

                weight_err = torch.norm(E, p='fro').item()
                logger.debug(f"[MDBF-ADMM] Path {path_idx+1}, Step {itt:3d}: "
                             f"output_err={current_err:.4e}, weight_err={weight_err:.4e} "
                             f"(rel: {weight_err/W_norm:.4f}), rho_base={rho_base:.4f}"
                             f"{' *' if current_err <= best_err else ''}")

    # Final result: use the best result
    F_opt = best_V
    G_opt = best_U.T

    del H_eig_vals, H_eig_vecs, LamU, GamV, H_use, W_float, I_r, I_m
    del V, U, best_V, best_U
    cleanup_gpu_memory()

    return F_opt, G_opt


def optimize_MDBF_admm_hessian(
    W_original: torch.Tensor,
    params_list: List[MDBFParams],
    l: int,
    H: torch.Tensor,
    nsamples: int,
    iters: int = 260,
    inner_iters: int = 3,
    reg: float = 0.03,
) -> Tuple[List[MDBFParams], torch.Tensor]:
    """
    Hessian-based Activation-aware ADMM optimization (Phase 2) - P=1 only

    Objective function: N * tr((W - W_hat) @ H @ (W - W_hat)^T)
    """
    device = W_original.device
    dtype = W_original.dtype
    n, m = W_original.shape

    if len(params_list) != 1:
        raise ValueError(f"optimize_MDBF_admm_hessian only supports P=1, got P={len(params_list)}")

    if H.shape != (m, m):
        raise ValueError(f"H shape must be ({m}, {m}), got {H.shape}")

    H = symmetrize_matrix(ensure_float32(H, device))

    logger.debug(f"[MDBF-ADMM] iters={iters}, inner={inner_iters}, l={l}, N={nsamples}")

    W_float = ensure_float32(W_original)
    orig_norm = torch.norm(W_float, p='fro').item() + 1e-12

    def compute_hess_err(W_hat):
        return compute_hessian_error(W_float - W_hat, H, nsamples)

    # Initial reconstruction and error
    p = params_list[0]
    W_init = reconstruct_weight(
        p.A_sign.to(device), p.B_sign.to(device),
        p.A_amp.to(device), p.B_amp.to(device),
        p.Q_U_amp.to(device), p.Q_V_amp.to(device),
    )

    init_weight_err = torch.norm(W_float - W_init, p='fro').item()
    init_hess_err = compute_hess_err(W_init)

    logger.debug(f"[MDBF-ADMM] Initial: output_err={init_hess_err:.4e}, "
                 f"weight_err={init_weight_err:.4e} (rel={init_weight_err/orig_norm:.4f})")
    del W_init

    # ADMM optimization
    F_init, G_init = _params_to_factor_matrices(p, device)
    F_opt, G_opt = _admm_refine_single_path_hessian(
        H=H,
        nsamples=nsamples,
        W_target=W_float,
        F_init=F_init,
        G_init=G_init,
        l=l,
        iters=iters,
        inner_iters=inner_iters,
        reg=reg,
        path_idx=0,
    )

    W_recon = F_opt @ G_opt
    final_weight_err = torch.norm(W_float - W_recon, p='fro').item()
    final_hess_err = compute_hess_err(W_recon)

    weight_impr = (init_weight_err - final_weight_err) / init_weight_err * 100
    output_impr = (init_hess_err - final_hess_err) / (init_hess_err + 1e-12) * 100
    logger.debug(f"[MDBF-ADMM] Final: output_err={final_hess_err:.4e}, weight_err={final_weight_err:.4e}")
    logger.debug(f"[MDBF-ADMM] Output improvement: {output_impr:+.2f}%, Weight improvement: {weight_impr:+.2f}%")

    del W_float, H, F_init, G_init

    opt_params = _factor_matrices_to_params(F_opt, G_opt, l, dtype)

    del F_opt, G_opt
    cleanup_gpu_memory()

    return [opt_params], W_recon.to(dtype)
