# -*- coding: utf-8 -*-
"""
Middle phase functions for MDBF (Matrix-extended Double Binary Factorization)

Main features:
1. Closed-form optimization of intermediate matrix M (dense) or UV^T (low-rank) for residuals
2. Optional fine-tuning via gradient methods
3. Numerically stable Cholesky decomposition and SVD implementations
"""

import gc  # ! Free memory
from typing import Optional, Tuple

import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch
from torch import optim


def _safe_cholesky(A: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
    """
    Numerically stable Cholesky decomposition implementation.

    Features:
    - Processing in float32 for numerical precision
    - Stepwise regularization to avoid decomposition failure
    - Eigendecomposition as final fallback

    Args:
        A: Positive definite symmetric matrix (or nearly so)
        lam: Initial regularization parameter

    Returns:
        L: Lower triangular matrix (A ≈ L @ L.T)
    """
    # Process in float32 for numerical precision (save original dtype)
    odtype = A.dtype
    dev = A.device
    A = A.to(torch.float32)

    # Compute diagonal mean (used as regularization reference)
    diag_mean = A.diag().mean().clamp(min=1e-8)
    I = torch.eye(A.shape[0], device=dev, dtype=torch.float32)

    # Prepare initial regularization matrix
    A_reg = A + lam * I * diag_mean if lam > 0 else A.clone()

    # Try Cholesky decomposition with progressively stronger regularization
    # Start with small regularization and increase as needed
    reg_factors = [0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    for factor in reg_factors:
        try:
            A_try = A_reg + factor * I * diag_mean if factor > 0 else A_reg
            L = torch.linalg.cholesky(A_try)

            # Warn if large regularization was needed
            if factor > 1e-4:
                logger.debug(
                    f"[WARNING] Cholesky succeeded with regularization factor: {factor}"
                )

            return L.to(dtype=odtype)
        except Exception:
            continue  # Try next regularization level

    # Last resort: correction via eigendecomposition
    logger.debug(
        f"[WARNING] Cholesky failed even with strong regularization, using eigendecomposition"
    )
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        # Replace negative eigenvalues with small positive values
        eigenvalues = eigenvalues.clamp(min=1e-6 * diag_mean)
        # Reconstruct
        A_fixed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        L = torch.linalg.cholesky(A_fixed)
        return L.to(dtype=odtype)
    except Exception as e:
        logger.debug(f"[ERROR] Even eigendecomposition-based Cholesky failed: {e}")
        # Last resort: approximate as diagonal matrix
        return torch.diag(torch.sqrt(A.diag().clamp(min=1e-6))).to(
            dtype=odtype
        )


def _solve_sylvester_two_sides(
    GA: torch.Tensor, GB: torch.Tensor, R: torch.Tensor, lam: float = 1e-3
) -> torch.Tensor:
    """
    Solve Sylvester equation GA * X * GB = R.

    Efficiently solved using Cholesky decomposition:
    1. GA = LA @ LA.T, GB = LB @ LB.T
    2. X = LA^{-1} @ R @ LB^{-T}
    """
    orig_dtype = R.dtype

    # _safe_cholesky internally converts to float32
    LA = _safe_cholesky(GA, lam)
    LB = _safe_cholesky(GB, lam)

    # Ensure float32 for cholesky_solve
    if R.dtype != torch.float32:
        R = R.to(torch.float32)
        LA = LA.to(torch.float32)
        LB = LB.to(torch.float32)

    X = torch.cholesky_solve(R, LA)  # GA^{-1} R
    M = torch.cholesky_solve(X.T, LB).T  # (GA^{-1} R) GB^{-1}
    return M.to(dtype=orig_dtype)


def _compose_dense_bal(
    A: torch.Tensor,
    d: torch.Tensor,
    M: torch.Tensor,
    B: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
) -> torch.Tensor:
    """
    Dense matrix composition in balanced space: W = Da * A * (diag(d) + M) * B * Db

    For memory efficiency, compute without explicitly creating diag(d):
    W = Da * A * M * B * Db + Da * A * diag(d) * B * Db
      = Atil @ M @ Btil + Atil @ (d[:, None] * Btil)

    where Atil = Da[:, None] * A, Btil = B * Db[None, :]
    """
    # Unify all to same dtype
    dtype = A.dtype

    # dtype conversion (if needed)
    if d.dtype != dtype:
        d = d.to(dtype=dtype)
    if M.dtype != dtype:
        M = M.to(dtype=dtype)
    if B.dtype != dtype:
        B = B.to(dtype=dtype)
    if Da.dtype != dtype:
        Da = Da.to(dtype=dtype)
    if Db.dtype != dtype:
        Db = Db.to(dtype=dtype)

    Atil = Da[:, None] * A
    Btil = B * Db[None, :]

    return Atil @ M @ Btil + Atil @ (d[:, None] * Btil)


def _compose_lowrank_bal(
    A: torch.Tensor,
    d: torch.Tensor,
    U: Optional[torch.Tensor],
    V: Optional[torch.Tensor],
    B: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
) -> torch.Tensor:
    """
    Low-rank composition in balanced space: W = Da * A * (diag(d) + U*V^T) * B * Db

    Decomposed into two terms for efficient computation:
    1. Diagonal term: Da * A * diag(d) * B * Db
    2. Low-rank term: Da * A * U * V^T * B * Db (when U, V exist)
    """
    # Unify all to same dtype (using A's dtype as reference)
    dtype = A.dtype
    # device = A.device

    # dtype conversion (if needed)
    if B.dtype != dtype:
        B = B.to(dtype=dtype)
    if d.dtype != dtype:
        d = d.to(dtype=dtype)
    if Da.dtype != dtype:
        Da = Da.to(dtype=dtype)
    if Db.dtype != dtype:
        Db = Db.to(dtype=dtype)

    Atil = Da[:, None] * A
    Btil = B * Db[None, :]
    W1 = Atil @ (d[:, None] * Btil)
    if U is not None and U.numel() and V is not None and V.numel():
        if U.dtype != dtype:
            U = U.to(dtype=dtype)
        if V.dtype != dtype:
            V = V.to(dtype=dtype)
        W1 = W1 + (Atil @ U) @ (V.T @ Btil)

    return W1


@torch.no_grad()
def dense_M_closed_form_given_d_stable(
    A: torch.Tensor,
    B: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    W_bal: torch.Tensor,
    d: torch.Tensor,
    lam: float = 1e-6,
) -> torch.Tensor:
    """
    Compute closed-form solution for dense matrix M with numerical stability (diagonal d fixed).

    Solves Sylvester equation: (A^T*Da^2*A) * M * (B*Db^2*B^T) = A^T*Da*R*Db*B^T
    where R = W_bal - Da*A*diag(d)*B*Db is the residual matrix.

    For numerical stability:
    - All linear algebra operations in float32
    - Uses Cholesky decomposition (avoids direct inverse computation)
    - Appropriate regularization parameter lam
    """
    # Save original dtype
    orig_dtype = A.dtype
    # Cholesky decomposition requires float32 or higher
    dtype = torch.float32
    device = A.device

    # Convert to float32 (for Cholesky decomposition)
    A = A.to(dtype=dtype)
    B = B.to(dtype=dtype)
    Da = Da.to(dtype=dtype)
    Db = Db.to(dtype=dtype)
    W_bal = W_bal.to(dtype=dtype)
    d = d.to(dtype=dtype)

    Atil = Da[:, None] * A
    Btil = B * Db[None, :]
    R0 = W_bal - Atil @ (d[:, None] * Btil)

    GA = Atil.T @ Atil
    GB = Btil @ Btil.T

    # Debug shape checks
    assert (
        Atil.shape[0] == W_bal.shape[0]
    ), f"Atil shape mismatch: {Atil.shape} vs W_bal {W_bal.shape}"
    assert (
        Btil.shape[1] == W_bal.shape[1]
    ), f"Btil shape mismatch: {Btil.shape} vs W_bal {W_bal.shape}"
    assert GA.shape[0] == GA.shape[1], f"GA not square: {GA.shape}"
    assert GB.shape[0] == GB.shape[1], f"GB not square: {GB.shape}"

    # Add regularization term (adaptive regularization)
    I = torch.eye(GA.shape[0], device=device, dtype=dtype)

    # Adjust regularization based on diagonal element statistics
    diag_A = GA.diag()
    diag_B = GB.diag()
    scaleA = diag_A.mean().clamp_min(1e-6)
    scaleB = diag_B.mean().clamp_min(1e-6)

    # Estimate condition number (max/min of diagonal elements)
    cond_A = diag_A.max() / diag_A.clamp_min(1e-8).min()
    cond_B = diag_B.max() / diag_B.clamp_min(1e-8).min()

    # Strengthen regularization only when condition number is very poor
    if cond_A > 1e8 or cond_B > 1e8:
        logger.debug(
            f"    [INFO] Poor condition number detected: cond_A={cond_A:.2e}, cond_B={cond_B:.2e}"
        )
        lam = max(lam, 1e-3)

    GA_reg = GA + lam * I * scaleA
    GB_reg = GB + lam * I * scaleB

    # Use safe Cholesky decomposition
    LA = _safe_cholesky(GA_reg, lam)
    LB = _safe_cholesky(GB_reg, lam)

    # Compute M = GA^{-1} A^T R0 B^T GB^{-1}
    # Reformulate to avoid huge intermediate matrix (k,m)
    # Since R0 = W_bal - A*diag(d)*B:
    # A^T*R0*B^T = A^T*W_bal*B^T - GA*diag(d)*GB

    # More efficient computation (reduce intermediate matrix size)
    # Block computation for large matrices
    k = Atil.shape[1]
    # n, m = W_bal.shape

    if W_bal.shape[0] > 2048 or W_bal.shape[1] > 2048:
        # Block computation (for memory efficiency)
        RHS = torch.zeros((k, k), device=device, dtype=dtype)
        block_size = max(1, min(512, W_bal.shape[1] // 4))
        for j0 in range(0, W_bal.shape[1], block_size):
            j1 = min(W_bal.shape[1], j0 + block_size)
            W_block = W_bal[:, j0:j1]
            Btil_block = Btil[:, j0:j1]
            RHS += Atil.T @ (W_block @ Btil_block.T)
    else:
        # Direct computation for small matrices
        Tmp = W_bal @ Btil.T  # (n, k)
        RHS = Atil.T @ Tmp  # (k, k)
    RHS = RHS - GA @ torch.diag(d) @ GB  # completes in (k, k)

    # RHS is already float32, pass directly to cholesky_solve
    S1 = torch.cholesky_solve(RHS, LA)
    M = torch.cholesky_solve(S1.T, LB).T

    # Check M norm (prevent divergence)
    M_norm = torch.norm(M, "fro").item()

    # NaN check
    if torch.isnan(M).any() or torch.isinf(M).any() or not np.isfinite(M_norm):
        logger.debug(
            f"    [WARNING] NaN/Inf detected in M, falling back to zero matrix"
        )
        M = torch.zeros_like(M)
        return M.to(dtype=orig_dtype)

    if M_norm > 1e5:  # Strengthen regularization only when exceeding 100k
        logger.debug(
            f"    [INFO] Large M norm detected: {M_norm:.2e}, applying stronger regularization"
        )

        # Initialize for reference outside loop
        M_norm_new = M_norm

        # Progressively strengthen regularization
        for reg_factor in [10, 100, 1000]:
            GA_reg = GA + reg_factor * lam * I * scaleA
            GB_reg = GB + reg_factor * lam * I * scaleB

            try:
                LA = _safe_cholesky(GA_reg, lam)
                LB = _safe_cholesky(GB_reg, lam)
                S1 = torch.cholesky_solve(RHS, LA)
                M = torch.cholesky_solve(S1.T, LB).T
                M_norm_new = torch.norm(M, "fro").item()

                if (
                    torch.isnan(M).any()
                    or torch.isinf(M).any()
                    or not np.isfinite(M_norm_new)
                ):
                    continue

                logger.debug(
                    f"    [INFO] New M norm after {reg_factor}x regularization: {M_norm_new:.2e}"
                )

                # End if improved
                if M_norm_new < M_norm * 0.9:  # More than 10% improvement
                    break

            except Exception as e:
                logger.debug(
                    f"    [WARNING] Failed with {reg_factor}x regularization: {e}"
                )
                continue

        # Fall back to zero matrix only when extremely large
        if M_norm_new > 1e5:  # Only when exceeding 100k
            logger.debug(
                f"    [WARNING] M norm extremely large ({M_norm_new:.2e}), falling back to zero matrix"
            )
            M = torch.zeros_like(M)
        else:
            # Continue if M norm is within acceptable range even without improvement
            M_norm = M_norm_new

    #! Free memory
    del A, B, Da, Db, W_bal, d
    del (
        Atil,
        Btil,
        R0,
        GA,
        GB,
        I,
        diag_A,
        diag_B,
        scaleA,
        scaleB,
        cond_A,
        cond_B,
    )
    del GA_reg, GB_reg, LA, LB, RHS
    if "W_block" in locals():
        del W_block
    if "Btil_block" in locals():
        del Btil_block
    if "Tmp" in locals():
        del Tmp
    del S1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Return to original dtype
    return M.to(dtype=orig_dtype)


# dense_M_closed_form_given_d removed (unified with _stable version)


def uv_closed_form_given_d(
    A: torch.Tensor,
    B: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    W_bal: torch.Tensor,
    d: torch.Tensor,
    rank: int,
    lam: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute closed-form solution for low-rank decomposition U, V using whitened SVD (diagonal d fixed).

    Steps:
    1. Compute residual R = W_bal - Da*A*diag(d)*B*Db
    2. Whitening transform: S = (A^T*Da^2*A)^{-1/2} * A^T*Da*R*Db*B^T * (B*Db^2*B^T)^{-1/2}
    3. SVD decomposition: S = U_s * Σ * V_s^T
    4. Take rank-r approximation and map back to original space

    All linear algebra operations in float32 for numerical stability.
    """
    if rank <= 0:
        k = A.shape[1]
        z = torch.zeros((k, 0), device=A.device, dtype=A.dtype)
        return z, z.clone()

    dev, odtype = A.device, A.dtype
    A32 = (Da[:, None] * A).to(torch.float32)  # \tilde A
    B32 = (B * Db[None, :]).to(torch.float32)  # \tilde B
    W32 = W_bal.to(torch.float32)
    d32 = d.to(torch.float32)

    # R = W - \tilde A diag(d) \tilde B
    GA = A32.T @ A32
    GB = B32 @ B32.T

    # T = \tilde A^T R \tilde B^T = \tilde A^T W \tilde B^T - GA diag(d) GB
    # Block computation for large matrices
    n, m = W32.shape
    if m > 8192:  # For very large m
        # Compute W @ B32.T in column blocks
        T = torch.zeros(
            (A32.shape[1], B32.shape[0]), device=dev, dtype=torch.float32
        )
        block_size = 2048
        for j0 in range(0, m, block_size):
            j1 = min(m, j0 + block_size)
            W_block = W32[:, j0:j1]
            B_block = B32[:, j0:j1]
            T += A32.T @ (W_block @ B_block.T)
    else:
        T = A32.T @ (W32 @ B32.T)

    T = T - GA @ torch.diag(d32) @ GB

    # Cholesky factors: GA = LA LA^T, GB = LB LB^T
    lam32 = max(lam, 1e-4) if rank > 4 else max(lam, 1e-5)
    LA = _safe_cholesky(GA, lam32).to(torch.float32)
    LB = _safe_cholesky(GB, lam32).to(torch.float32)

    # Whitening: S = LA^{-1} T LB^{-T}
    S1 = torch.linalg.solve_triangular(LA, T, upper=False)  # LA^{-1} T
    S = torch.linalg.solve_triangular(
        LB, S1.T, upper=False
    ).T  # (LA^{-1} T) LB^{-T}

    # SVD
    try:
        if 0 < rank < S.shape[0] // 2:
            q = min(S.shape[0], max(rank + 8, int(1.5 * rank)))
            U_s, S_s, V_s = torch.svd_lowrank(S, q=q)
            Vh_s = V_s.T
        else:
            U_s, S_s, Vh_s = torch.linalg.svd(S, full_matrices=False)
    except (torch.linalg.LinAlgError, RuntimeError):
        # Fallback via eig on S S^T
        SST = S @ S.T
        evals, U_s = torch.linalg.eigh(SST)
        evals = evals.clamp(min=0)
        S_s = torch.sqrt(evals)
        idx = torch.argsort(S_s, descending=True)
        S_s, U_s = S_s[idx], U_s[:, idx]
        Vh_s = (torch.diag(1.0 / S_s.clamp(min=1e-12)) @ U_s.T @ S).mH

    # Automatically trim effective rank (drop very small singular values)
    r_max = min(rank, S_s.numel())
    if r_max > 0:
        cutoff = max(1e-7, 1e-3 * S_s[0].item())
        r = int((S_s[:r_max] >= cutoff).sum().item())
    else:
        r = 0

    if r == 0:
        z = torch.zeros((A.shape[1], 0), device=dev, dtype=odtype)
        return z, z.clone()

    U_r = U_s[:, :r]
    V_r = (Vh_s.mH if hasattr(Vh_s, "mH") else Vh_s.T)[:, :r]
    sqrtSig = torch.sqrt(S_s[:r].clamp(min=1e-12))

    # Unwhitening:
    # U = LA^{-T} U_r sqrtSig,    V = LB^{-T} V_r sqrtSig
    try:
        Uw = torch.linalg.solve_triangular(
            LA.T, U_r, upper=True
        )  # LA^{-T} U_r
        Vw = torch.linalg.solve_triangular(
            LB.T, V_r, upper=True
        )  # LB^{-T} V_r
    except RuntimeError:
        # Fallback: explicit inverses (rare)
        Uw = torch.linalg.inv(LA.T) @ U_r
        Vw = torch.linalg.inv(LB.T) @ V_r

    U = (Uw * sqrtSig[None, :]).to(device=dev, dtype=odtype)
    V = (Vw * sqrtSig[None, :]).to(device=dev, dtype=odtype)

    # Optional: small-r debug
    if r <= 8:
        AtilU = A32 @ U.to(torch.float32)  # n×r
        VBtil = V.to(torch.float32).T @ B32  # r×m
        # resid = torch.norm(
        #     (W32 - A32 @ (d32[:, None] * B32)) - AtilU @ VBtil, p="fro"
        # )
        # denom = torch.norm(
        #     W32 - A32 @ (d32[:, None] * B32), p="fro"
        # ).clamp_min(1e-12)
        # Suppress verbose low-rank init output
        # logger.debug(f"    [Low-rank init] rank={r}, residual={(resid/denom).item():.1%}, "
        #       f"singular values: max={S_s[0].item():.2e}, min={S_s[r-1].item():.2e}")

    #! Free memory
    del A, B, Da, Db, W_bal, d
    del A32, B32, W32, d32, GA, GB, T
    if "W_block" in locals():
        del W_block
    if "B_block" in locals():
        del B_block
    del LA, LB, S1, S
    if "U_s" in locals():
        del U_s
    if "S_s" in locals():
        del S_s
    if "V_s" in locals():
        del V_s
    if "Vh_s" in locals():
        del Vh_s
    if "SST" in locals():
        del SST
    if "evals" in locals():
        del evals
    del U_r, V_r, Uw, Vw
    if "AtilU" in locals():
        del AtilU
    if "VBtil" in locals():
        del VBtil
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return U, V


def update_d_hadamard(
    A: torch.Tensor,
    B: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    W_bal: torch.Tensor,
    U: Optional[torch.Tensor],
    V: Optional[torch.Tensor],
    lam: float = 1e-3,
) -> torch.Tensor:
    """
    Update diagonal components d using Hadamard product (U, V fixed).

    Optimization: min_d ||W_bal - Da*A*(diag(d)+U*V^T)*B*Db||^2

    Closed-form solution: d = (H + λI)^{-1} * g
    where:
    - H[i,j] = (A^T*Da^2*A)[i,j] * (B*Db^2*B^T)[i,j] (Hadamard product)
    - g[i] = <(Da*A)[:,i], R*Db*B^T[:,i]> (inner product with residual)
    """
    orig_dtype = A.dtype

    Atil = Da[:, None] * A
    Btil = B * Db[None, :]
    GA = Atil.T @ Atil
    GB = Btil @ Btil.T
    H = GA * GB
    R2 = W_bal - Atil @ (U @ V.T) @ Btil
    g = torch.diagonal(Atil.T @ R2 @ Btil.T)

    # _safe_cholesky internally converts to float32
    H = H + torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * (
        lam * H.diag().mean().clamp_min(1e-6)
    )
    L = _safe_cholesky(H, 0.0)

    # Ensure float32 for cholesky_solve
    if g.dtype != torch.float32:
        g = g.to(torch.float32)
        L = L.to(torch.float32)

    d = torch.cholesky_solve(g.unsqueeze(1), L).squeeze(1)
    return d.to(dtype=orig_dtype)


@torch.no_grad()
def update_d_hadamard_dense(
    A: torch.Tensor,
    B: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    W_bal: torch.Tensor,
    M: torch.Tensor,
    lam: float = 1e-3,
) -> torch.Tensor:
    """
    Dense version: Update diagonal components d using Hadamard product (M fixed).

    Optimization: min_d ||W_bal - Da*A*(diag(d)+M)*B*Db||^2

    Closed-form solution: d = (H + λI)^{-1} * g
    """
    orig_dtype = A.dtype

    Atil = Da[:, None] * A
    Btil = B * Db[None, :]
    GA = Atil.T @ Atil
    GB = Btil @ Btil.T
    H = GA * GB
    R = W_bal - Atil @ (M @ Btil)
    g = torch.diagonal(Atil.T @ R @ Btil.T)

    # _safe_cholesky internally converts to float32
    H = H + torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * (
        lam * H.diag().mean().clamp_min(1e-6)
    )
    L = _safe_cholesky(H, 0.0)

    # Ensure float32 for cholesky_solve
    if g.dtype != torch.float32:
        g = g.to(torch.float32)
        L = L.to(torch.float32)

    d = torch.cholesky_solve(g.unsqueeze(1), L).squeeze(1)
    return d.to(dtype=orig_dtype)


def middle_refine_dense_grad(
    W_bal: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    M: torch.Tensor,
    steps: int = 50,
    lr: float = 5e-2,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fine-tune dense matrix M and related parameters via gradient descent (A, B fixed).

    Optimization targets: Da, Db, d, M (continuous parameters only)
    Objective: ||W_bal - Da*A*(diag(d)+M)*B*Db||_F^2

    Args:
        steps: Number of gradient descent steps
        lr: Learning rate
        eps: Minimum value constraint for Da, Db

    Returns:
        Optimized (Da, Db, d, M)
    """
    # Preserve original dtype
    orig_dtype = A.dtype

    # Explicitly enable gradient computation with torch.enable_grad()
    with torch.enable_grad():
        # Upcast all to fp32 for optimization
        W_bal32 = W_bal.detach().to(torch.float32)
        A32 = A.detach().to(torch.float32)
        B32 = B.detach().to(torch.float32)

        # Optimization target parameters (initialized in fp32)
        d_p = d.clone().to(torch.float32).requires_grad_(True)
        M_p = M.clone().to(torch.float32).requires_grad_(True)
        Da_p = Da.clone().to(torch.float32).requires_grad_(True)
        Db_p = Db.clone().to(torch.float32).requires_grad_(True)

        # Initialize optimizer
        opt = optim.Adam([d_p, M_p, Da_p, Db_p], lr=lr)

        best_error = float("inf")
        best_params = None

        for step in range(steps):
            opt.zero_grad()

            # Forward computation (fp32)
            W_hat = _compose_dense_bal(A32, d_p, M_p, B32, Da_p, Db_p)
            loss = torch.norm(W_bal32 - W_hat, p="fro") ** 2

            loss.backward()
            opt.step()

            with torch.no_grad():
                Da_p.clamp_(min=eps)
                Db_p.clamp_(min=eps)

                # Update best_params every step
                error = torch.sqrt(loss).item()
                if error < best_error:
                    best_error = error
                    best_params = (
                        Da_p.clone(),
                        Db_p.clone(),
                        d_p.clone(),
                        M_p.clone(),
                    )

            if step % 50 == 0:
                logger.debug(f"    Step {step:3d}: error = {best_error:.4e}")

        # Use best parameters
        if best_params is None:
            best_params = (Da_p, Db_p, d_p, M_p)

    # Return to original dtype before returning
    Da_best, Db_best, d_best, M_best = best_params

    #! Free memory
    del W_bal, A, B, d, Da, Db, M
    del W_bal32, A32, B32, d_p, M_p, Da_p, Db_p
    del opt, W_hat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return (
        Da_best.detach().to(orig_dtype),
        Db_best.detach().to(orig_dtype),
        d_best.detach().to(orig_dtype),
        M_best.detach().to(orig_dtype),
    )


def middle_refine_lowrank_grad(
    W_bal: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    U: Optional[torch.Tensor],
    V: Optional[torch.Tensor],
    steps: int = 50,
    lr: float = 5e-2,
    eps: float = 1e-6,
) -> Tuple:
    """
    Fine-tune low-rank U, V and related parameters via gradient descent (A, B fixed).

    Optimization targets: Da, Db, d, U, V (continuous parameters only)
    Objective: ||W_bal - Da*A*(diag(d)+U*V^T)*B*Db||_F^2

    Args:
        steps: Number of gradient descent steps
        lr: Learning rate
        eps: Minimum value constraint for Da, Db

    Returns:
        Optimized (Da, Db, d, U, V)
    """
    # Skip if rank=0
    if U is None or U.numel() == 0 or V is None or V.numel() == 0:
        logger.debug(f"    Skipping gradient refinement (rank=0 or empty U/V)")
        return Da, Db, d, U, V

    # Preserve original dtype
    orig_dtype = A.dtype

    # Explicitly enable gradient computation with torch.enable_grad()
    with torch.enable_grad():
        # Upcast all to fp32 for optimization
        W_bal32 = W_bal.detach().to(torch.float32)
        A32 = A.detach().to(torch.float32)
        B32 = B.detach().to(torch.float32)

        # Variables to optimize as parameters (initialized in fp32)
        d_p = d.clone().to(torch.float32).requires_grad_(True)
        U_p = U.clone().to(torch.float32).requires_grad_(True)
        V_p = V.clone().to(torch.float32).requires_grad_(True)
        Da_p = Da.clone().to(torch.float32).requires_grad_(True)
        Db_p = Db.clone().to(torch.float32).requires_grad_(True)

        # Initialize optimizer
        opt = optim.Adam([d_p, U_p, V_p, Da_p, Db_p], lr=lr)

        best_error = float("inf")
        best_params = None

        for step in range(steps):
            opt.zero_grad()

            # Forward computation (fp32)
            W_hat = _compose_lowrank_bal(A32, d_p, U_p, V_p, B32, Da_p, Db_p)
            loss = torch.norm(W_bal32 - W_hat, p="fro") ** 2

            # Backward computation
            loss.backward()

            # Parameter update
            opt.step()

            # Apply constraints
            with torch.no_grad():
                Da_p.clamp_(min=eps)
                Db_p.clamp_(min=eps)

                # Update best_params every step
                error = torch.sqrt(loss).item()
                if error < best_error:
                    best_error = error
                    best_params = (
                        Da_p.clone(),
                        Db_p.clone(),
                        d_p.clone(),
                        U_p.clone(),
                        V_p.clone(),
                    )

            # Record and display error
            if step % 50 == 0:
                logger.debug(f"    Step {step:3d}: error = {best_error:.4e}")

        # Use best parameters
        if best_params is None:
            best_params = (Da_p, Db_p, d_p, U_p, V_p)

    # Return to original dtype before returning
    Da_best, Db_best, d_best, U_best, V_best = best_params

    #! Free memory
    del W_bal, A, B, d, Da, Db, U, V
    del W_bal32, A32, B32, d_p, U_p, V_p, Da_p, Db_p
    del opt, W_hat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return (
        Da_best.detach().to(orig_dtype),
        Db_best.detach().to(orig_dtype),
        d_best.detach().to(orig_dtype),
        U_best.detach().to(orig_dtype),
        V_best.detach().to(orig_dtype),
    )


def update_Da_Db_closed_form(
    A: torch.Tensor,
    B: torch.Tensor,
    Mprime: torch.Tensor,
    W_bal: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    every: int = 10,
    step: int = 0,
    eps: float = 1e-6,
    block_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form update of diagonal scalings Da, Db (memory-efficient with blocking).

    Optimization: min_{Da,Db} ||W_bal - Da*A*Mprime*B*Db||^2

    Closed-form solution (alternating optimization):
    1. Da[i] = <W_bal[i,:], X[i,:]*Db> / ||X[i,:]*Db||^2
    2. Db[j] = <W_bal[:,j], Da*X[:,j]> / ||Da*X[:,j]||^2
    where X = A*Mprime*B

    Args:
        every: Update every N steps (reduce computation cost)
        step: Current step number (-1 forces execution)
        block_size: Block size (tradeoff with memory usage)
    """
    if every <= 0 or (step % every != 0 and step != -1):
        return Da, Db

    n, m = W_bal.shape
    # k = A.shape[1]
    device = A.device
    dtype = A.dtype

    # Pre-compute left side
    L = A @ Mprime  # (n,k)

    # --- Da update (column blocking) ---
    num_da = torch.zeros(n, device=device, dtype=dtype)
    den_da = torch.zeros(n, device=device, dtype=dtype)

    for j0 in range(0, m, block_size):
        j1 = min(m, j0 + block_size)
        Bj = B[:, j0:j1]  # (k, block)
        Wj = W_bal[:, j0:j1]  # (n, block)
        Yj = (L @ Bj) * Db[None, j0:j1]  # (n, block)
        num_da += (Yj * Wj).sum(dim=1)
        den_da += (Yj * Yj).sum(dim=1)

    Da = (num_da / den_da.clamp_min(1e-12)).clamp_min(eps)

    # --- Db update (also with blocking) ---
    Xr = Da[:, None] * L  # (n,k)
    num_db = torch.empty(m, device=device, dtype=dtype)
    den_db = torch.empty(m, device=device, dtype=dtype)

    for j0 in range(0, m, block_size):
        j1 = min(m, j0 + block_size)
        Bj = B[:, j0:j1]  # (k, block)
        Wj = W_bal[:, j0:j1]  # (n, block)
        Xj = Xr @ Bj  # (n, block)
        num_db[j0:j1] = (Xj * Wj).sum(dim=0)
        den_db[j0:j1] = (Xj * Xj).sum(dim=0)

    Db = (num_db / den_db.clamp_min(1e-12)).clamp_min(eps)

    return Da, Db


def update_Da_Db_closed_form_factored(
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    M: Optional[torch.Tensor],
    W_bal: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    every: int = 10,
    step: int = 0,
    eps: float = 1e-6,
    block_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form update of diagonal scalings Da, Db (without creating Mprime).

    Optimization: min_{Da,Db} ||W_bal - Da*A*(diag(d)+M)*B*Db||^2

    Args:
        d: Diagonal elements (k,)
        M: Dense matrix (k×k) or None
        every: Update every N steps
        step: Current step number
        block_size: Block size
    """
    if every <= 0 or (step % every != 0 and step != -1):
        return Da, Db

    n, m = W_bal.shape
    device, dtype = A.device, A.dtype

    # L = A*(diag(d)+M) = (A*d) + (A@M)  (B not yet multiplied)
    L = A * d[None, :]  # (n,k)
    if M is not None and M.numel() > 0:
        L = L + (A @ M)

    # --- Da update (with blocking) ---
    num_da = torch.zeros(n, device=device, dtype=dtype)
    den_da = torch.zeros(n, device=device, dtype=dtype)

    for j0 in range(0, m, block_size):
        j1 = min(m, j0 + block_size)
        Bj = B[:, j0:j1]  # (k, block)
        Wj = W_bal[:, j0:j1]  # (n, block)
        Yj = (L @ Bj) * Db[None, j0:j1]  # (n, block)
        num_da += (Yj * Wj).sum(dim=1)
        den_da += (Yj * Yj).sum(dim=1)

    Da = (num_da / den_da.clamp_min(1e-12)).clamp_min(eps)

    # --- Db update (also with blocking) ---
    Xr = Da[:, None] * L  # (n,k)
    num_db = torch.empty(m, device=device, dtype=dtype)
    den_db = torch.empty(m, device=device, dtype=dtype)

    for j0 in range(0, m, block_size):
        j1 = min(m, j0 + block_size)
        Bj = B[:, j0:j1]  # (k, block)
        Wj = W_bal[:, j0:j1]  # (n, block)
        Xj = Xr @ Bj  # (n, block)
        num_db[j0:j1] = (Xj * Wj).sum(dim=0)
        den_db[j0:j1] = (Xj * Xj).sum(dim=0)

    Db = (num_db / den_db.clamp_min(1e-12)).clamp_min(eps)

    return Da, Db


def update_Da_Db_closed_form_lowrank(
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    U: Optional[torch.Tensor],
    V: Optional[torch.Tensor],
    W_bal: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    every: int = 10,
    step: int = 0,
    eps: float = 1e-6,
    block_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Low-rank UV^T version of Da, Db update (without explicitly creating UV^T).

    Optimization problem: min_{Da,Db} ||W_bal - Da*A*(diag(d)+U*V^T)*B*Db||^2

    Args:
        d: Diagonal elements (k,)
        U, V: Low-rank decomposition (k×r)
        every: Update every N steps
        step: Current step number
        block_size: Block size
    """
    if every <= 0 or (step % every != 0 and step != -1):
        return Da, Db

    n, m = W_bal.shape
    device, dtype = A.device, A.dtype

    # L = A*(diag(d)+U*V^T) = (A*d) + (A@U)@V^T
    L = A * d[None, :]  # (n,k)
    if U is not None and U.numel() > 0 and V is not None and V.numel() > 0:
        L = L + (A @ U) @ V.T  # (n,k)

    # --- Da update (with blocking) ---
    num_da = torch.zeros(n, device=device, dtype=dtype)
    den_da = torch.zeros(n, device=device, dtype=dtype)

    for j0 in range(0, m, block_size):
        j1 = min(m, j0 + block_size)
        Bj = B[:, j0:j1]  # (k, block)
        Wj = W_bal[:, j0:j1]  # (n, block)
        Yj = (L @ Bj) * Db[None, j0:j1]  # (n, block)
        num_da += (Yj * Wj).sum(dim=1)
        den_da += (Yj * Yj).sum(dim=1)

    Da = (num_da / den_da.clamp_min(1e-12)).clamp_min(eps)

    # --- Db update (also with blocking) ---
    Xr = Da[:, None] * L  # (n,k)
    num_db = torch.empty(m, device=device, dtype=dtype)
    den_db = torch.empty(m, device=device, dtype=dtype)

    for j0 in range(0, m, block_size):
        j1 = min(m, j0 + block_size)
        Bj = B[:, j0:j1]  # (k, block)
        Wj = W_bal[:, j0:j1]  # (n, block)
        Xj = Xr @ Bj  # (n, block)
        num_db[j0:j1] = (Xj * Wj).sum(dim=0)
        den_db[j0:j1] = (Xj * Xj).sum(dim=0)

    Db = (num_db / den_db.clamp_min(1e-12)).clamp_min(eps)

    return Da, Db


def _orthonormal_cols(X: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize column vectors (returns Q matrix from QR decomposition).

    Always executed in float32 for numerical stability.
    """
    dev, odtype = X.device, X.dtype
    Q, _ = torch.linalg.qr(X.to(torch.float32), mode="reduced")
    return Q.to(device=dev, dtype=odtype)


@torch.no_grad()
def compute_S_and_perm(
    A: torch.Tensor,
    B: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    W_bal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the middle matrix S and determine the optimal permutation order.

    Theoretical background:
    In the matrix factorization W ≈ Da*A*M*B*Db, appropriate permutation
    enhances the diagonal dominance of M and improves the accuracy of low-rank approximation.

    Steps:
    1. QR decomposition of Atil = Da*A, Btil = B*Db: Atil = QA*RA, Btil^T = QB*RB
    2. Compute the middle matrix S = QA^T * W_bal * QB
    3. Permute using the Fiedler vector (2nd eigenvector of graph Laplacian of S)

    Diagnostic information is also output:
    - offdiag_ratio: Relative magnitude of off-diagonal components
    - tailE(r): Singular value energy beyond rank-r (indicator of low-rank property)

    Returns:
        S: (k, k) Middle matrix
        perm: (k,) Optimal permutation indices
    """
    # Construct Atil, Btil (safely in fp32)
    Atil = (Da[:, None] * A).to(torch.float32)  # (n,k)
    Btil = (B * Db[None, :]).to(torch.float32)  # (k,m)

    # Orthogonal basis (reduced QR)
    QA = _orthonormal_cols(Atil)  # (n,k)
    QB = _orthonormal_cols(Btil.T)  # (m,k) - QR decomposition of Btil^T

    # Shape verification
    assert (
        QA.shape[1] == QB.shape[1]
    ), f"k mismatch: QA shape={QA.shape} vs QB shape={QB.shape}"

    # Middle matrix
    S = QA.T @ W_bal.to(torch.float32) @ QB  # (k,n)@(n,m)@(m,k) → (k,k)

    # When k=1, Fiedler vector cannot be obtained, so use identity order
    k = S.shape[0]
    if k < 2:
        perm = torch.arange(k, device=S.device)
        off = S - torch.diag(torch.diag(S))
        rho_off = (off.norm() / S.norm().clamp_min(1e-12)).item() ** 2
        logger.debug(
            f"[S diagnosis] offdiag_ratio={rho_off:.3f} (k=1, no permutation needed)"
        )
        return S, perm

    # Seriation: permute using Fiedler vector (symmetrize and use proximity weights)
    Wg = (S.abs() + S.abs().T) * 0.5
    Dg = torch.diag(Wg.sum(dim=1))
    Lg = Dg - Wg
    _, evecs = torch.linalg.eigh(Lg)
    fied = evecs[:, 1]  # 2nd eigenvector
    perm = torch.argsort(fied)

    # Compute diagnostic information
    off = S - torch.diag(torch.diag(S))
    rho_off = (off.norm() / S.norm().clamp_min(1e-12)).item() ** 2

    # Evaluate low-rank property
    with torch.no_grad():
        try:
            s = torch.linalg.svdvals(off)

            def tail_energy(r):
                if r >= len(s):
                    return 0.0
                return float(
                    (s[r:] ** 2).sum() / (s**2).sum().clamp_min(1e-12)
                )

            logger.debug(
                f"[S diagnosis] offdiag_ratio={rho_off:.3f}, tailE(r=4)={tail_energy(4):.3f}, tailE(r=8)={tail_energy(8):.3f}"
            )

            # Automatic selection hints
            if rho_off < 0.1:
                logger.debug(
                    f"  → Suggestion: rank=0 (standard DBF) should be sufficient"
                )
            elif tail_energy(4) < 0.1:
                logger.debug(f"  → Suggestion: low-rank with r≤4 is recommended")
            elif tail_energy(8) < 0.1:
                logger.debug(f"  → Suggestion: low-rank with r≤8 is recommended")
            else:
                logger.debug(f"  → Suggestion: dense M or higher rank may be needed")
        except:
            logger.debug(f"[S diagnosis] offdiag_ratio={rho_off:.3f}")

    return S, perm
