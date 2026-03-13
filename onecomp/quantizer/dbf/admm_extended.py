# -*- coding: utf-8 -*-
"""Extended ADMM functions for MDBF (Matrix-extended Double Binary Factorization)

Implements coordinated optimization of all parameters via extended ADMM.
Extends the basic DBF ADMM (which only optimizes A, B) to simultaneously optimize:
- A, B: Binary matrices (SVID constraint)
- M (dense matrix) or U, V (low-rank): Intermediate matrix
- Da, Db: Diagonal scaling
- d: Diagonal component (for low-rank case)

Alternating optimization is performed at each outer iteration.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import gc  # ! Free memory
from typing import Optional, Tuple

import logging

logger = logging.getLogger(__name__)
import torch

from .dbf_original import find_other2

# from .middle import update_d_hadamard_dense  # Dense version d update
# from .middle import update_Da_Db_closed_form  # Da, Db scaling update
# from .middle import _compose_dense_bal  # Dense matrix composition (currently used only)
from .middle import _compose_lowrank_bal  # Low-rank composition (currently used only)
from .middle import dense_M_closed_form_given_d_stable  # Dense M closed-form solution
from .middle import update_d_hadamard  # Diagonal d update
from .middle import update_Da_Db_closed_form_lowrank  # Low-rank efficient implementation
from .middle import uv_closed_form_given_d  # Low-rank U, V closed-form solution
from .middle import update_Da_Db_closed_form_factored  # Efficient version without building Mprime


@torch.no_grad()
def _frob_error_dense_fast(
    W_norm2: torch.Tensor,
    GA: torch.Tensor,
    GB: torch.Tensor,
    T: torch.Tensor,
    d: torch.Tensor,
    M: torch.Tensor,
) -> float:
    """
    Fast Frobenius error computation using only k×k matrices.

    Efficiently computes ||W - Da*A*(diag(d)+M)*B*Db||_F^2 by expansion.
    """
    S = GA * GB  # Hadamard product
    dd = (d * (S @ d)).sum()

    # ||A M B||^2 = tr(M^T GA M GB) = tr((GA M)(GB M)^T)
    GAm = GA @ M
    GBm = GB @ M
    mm = (GAm * GBm).sum()

    # Cross terms
    GAd = GA * d[None, :]
    cross_MD = 2.0 * (M * (GAd @ GB)).sum()

    # <W, A(M+diag(d))B> = tr(T^T M) + tr(diag(T) diag(d))
    cross_W = (T.T * M).sum() + (T.diag() * d).sum()

    err2 = W_norm2 + mm + dd + cross_MD - 2.0 * cross_W
    return torch.sqrt(err2.clamp_min(0.0)).item()


@torch.no_grad()
def _frob_error_lowrank_fast(
    W_norm2: torch.Tensor,
    GA: torch.Tensor,
    GB: torch.Tensor,
    T: torch.Tensor,
    d: torch.Tensor,
    U: Optional[torch.Tensor],
    V: Optional[torch.Tensor],
) -> float:
    """
    Low-rank version: Fast Frobenius error computation using only k×k matrices.

    Efficiently computes ||W - Da*A*(diag(d)+U*V^T)*B*Db||_F^2 by expansion.
    """
    S = GA * GB  # Hadamard product
    dd = (d * (S @ d)).sum()

    if U is None or U.numel() == 0 or V is None or V.numel() == 0:
        # U, V are empty
        cross_W = (T.diag() * d).sum()
        err2 = W_norm2 + dd - 2.0 * cross_W
    else:
        # ||A U V^T B||^2 = tr((U^T GA U)(V^T GB V))
        GAU = GA @ U  # k×r
        GBV = GB @ V  # k×r
        mm = torch.trace((U.T @ GAU) @ (V.T @ GBV))  # Trace of r×r product

        # Cross term: 2 * sum_i d_i * inner product of [(GA U)(GB V)]_{i,:}
        cross_MD = 2.0 * ((d[:, None] * GAU) * GBV).sum()

        # Cross term with W
        cross_W = (T.diag() * d).sum() + (U * (T @ V)).sum()

        err2 = W_norm2 + dd + mm + cross_MD - 2.0 * cross_W

    return torch.sqrt(err2.clamp_min(0.0)).item()


def _admm_update_Z_given_left(
    left_mat: torch.Tensor,
    W: torch.Tensor,
    Z: torch.Tensor,
    U: torch.Tensor,
    reg: float,
    rho_start: float,
    iters: int,
    use_adaptive_rho: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Execute ADMM update with SVID constraint, fixing the left-side matrix.

    Optimization problem: min_Z ||W - left_mat @ Z||^2 s.t. Z in SVID

    Args:
        left_mat: Fixed left-side matrix.
        W: Target matrix.
        Z: Matrix to update (SVID constraint).
        U: ADMM auxiliary variable.
        reg: Regularization parameter.
        rho_start: Initial penalty parameter.
        iters: Number of inner iterations.
        use_adaptive_rho: Whether to use adaptive ρ adjustment.

    Returns:
        Z_new: Updated matrix.
        U_new: Updated auxiliary variable.
    """
    # Save original dtype
    orig_dtype = Z.dtype

    # Compute in fp32 (for numerical stability)
    left_mat_fp32 = left_mat.to(torch.float32)
    W_fp32 = W.to(torch.float32)
    Z_fp32 = Z.to(torch.float32)
    U_fp32 = U.to(torch.float32)

    Z_new, U_new = find_other2(
        left_mat_fp32,
        W_fp32,
        Z_fp32,
        U_fp32,
        reg=reg,
        rho_start=rho_start,
        iters=iters,
        use_adaptive_rho=use_adaptive_rho,
    )

    # Return to original dtype
    return Z_new.to(orig_dtype), U_new.to(orig_dtype)


def _gauge_balance_dense_Mprime(
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    M: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple:
    """
    Gauge balance adjustment for dense matrix M version (normalization for numerical stability).

    Gauge invariance: W = A*(diag(d)+M)*B = (A*G)*((1/G)*(diag(d)+M)*(1/H))*(H*B)
    The same W can be represented for any diagonal matrices G, H.

    Using this degree of freedom, normalize column/row norms of A, B to 1,
    and absorb scales into d, M for improved numerical stability.

    Steps:
    1. Compute column/row norms of A, B
    2. Normalize A, B (norm=1)
    3. Absorb scales into d, M
    """
    # Column/row norms
    col = A.norm(dim=0).clamp_min(eps)  # (k,)
    row = B.norm(dim=1).clamp_min(eps)  # (k,)

    # Gauge: Normalize A, B
    A = A / col
    B = (B.T / row).T

    # Scale M' = diag(d) + M from left and right → distribute to d and M
    d = d * col * row  # Diagonal update
    M = (col[:, None] * M) * row[None, :]  # Off-diagonal update

    return A, B, d, M


# Line search feature removed (to reduce computation cost)


def _gauge_balance_lowrank(
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    U: Optional[torch.Tensor],
    V: Optional[torch.Tensor],
    eps: float = 1e-12,
) -> Tuple:
    """
    Gauge balance adjustment for low-rank version (normalization for numerical stability).

    Same as dense version: normalize column/row norms of A, B to 1,
    and absorb scales into d, U, V.

    Note: Scale adjustment is applied only when U, V exist.
    """
    col = A.norm(dim=0).clamp_min(eps)
    A = A / col
    d = d * col
    if U is not None and U.numel():
        U = col[:, None] * U

    row = B.norm(dim=1).clamp_min(eps)
    B = (B.T / row).T
    d = d * row
    if V is not None and V.numel():
        V = V * row[:, None]

    return A, B, d, U, V


def extended_admm_dense(
    W_bal: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    *,
    outer_iters: int = 120,
    inner_iters: int = 3,
    reg: float = 3e-2,
    use_adaptive_rho: bool = True,
    scale_update_every: int = 10,
    M_init: Optional[torch.Tensor] = None,
    Da_init: Optional[torch.Tensor] = None,
    Db_init: Optional[torch.Tensor] = None,
    # convergence_tol=1e-6,
    # patience=20,
) -> Tuple:
    """
    Extended ADMM for dense matrix M version (coordinated optimization of all parameters).

    Target: W_bal ≈ Da * A * (diag(d) + M) * B * Db

    Each outer iteration alternately optimizes:
    1. A update: ADMM + SVID constraint (B fixed)
    2. B update: ADMM + SVID constraint (A fixed)
    3. Gauge balance adjustment (improve numerical stability)
    4. M update: Closed-form solution
    5. Da, Db update: Closed-form solution (executed periodically)

    Features:
    - Adaptive ρ adjustment (balancing primal/dual residuals)
    - Best parameter saving
    - Various techniques for numerical stability

    Args:
        outer_iters: Number of outer iterations.
        inner_iters: Number of inner ADMM iterations (for A, B updates).
        scale_update_every: Update Da, Db every this many steps.

    Returns:
        Optimized (A, B, d, M, Da, Db).
    """
    device = W_bal.device
    dtype = W_bal.dtype
    n, m = W_bal.shape
    k = A.shape[1]

    # Parameter initialization
    A = A.clone()
    B = B.clone()
    d = d.clone()
    M = torch.zeros((k, k), device=device, dtype=dtype) if M_init is None else M_init.clone()
    Da = torch.ones(n, device=device, dtype=dtype) if Da_init is None else Da_init.clone()
    Db = torch.ones(m, device=device, dtype=dtype) if Db_init is None else Db_init.clone()

    # ADMM auxiliary variables
    UA = torch.zeros_like(A)
    UB = torch.zeros_like(B)

    # Convergence monitoring variables
    prev_error = float("inf")
    no_improve_count = 0
    best_error = float("inf")
    best_params = None

    # Extended ADMM Dense outputs only concise logs

    # Initial ρ (adaptively adjusted in outer loop)
    rho = 1.0
    rho_min, rho_max = 0.1, 10.0  # ρ bounds

    # Save previous values for ρ adjustment
    A_prev = A.clone()
    B_prev = B.clone()

    # Pre-computation for fast error calculation
    W_norm2 = torch.norm(W_bal, "fro").pow(2)

    # Reduce error calculation frequency (10 steps for debug, 20-50 for production)
    error_check_every = min(20, outer_iters // 5) if outer_iters > 50 else 10

    for t in range(outer_iters):
        # === Alternating optimization steps ===
        # Current decomposition: W_bal ≈ Da * A * (diag(d) + M) * B * Db

        # [Step 1] A update (B, M, d, Da, Db fixed)
        # Optimization: min_A ||W_bal - Da*A*(diag(d)+M)*B*Db||^2 s.t. A in SVID
        # Reformulated: min_A ||W_bal/Da - A*C||^2 where C = (diag(d)+M)*B*Db
        # Compute reciprocal in same dtype to prevent dtype promotion
        eps_Da = Da.new_tensor(1e-12)
        invDa = Da.clamp_min(eps_Da).reciprocal()
        W_eff_A_T = invDa[:, None] * W_bal
        # Avoid explicit diag(d) in C computation
        Btil = B * Db[None, :]
        C = (M @ Btil) + d[:, None] * Btil  # k×m
        A_T, UA_T = _admm_update_Z_given_left(
            C.T,
            W_eff_A_T.T,
            A.T,
            UA.T,
            reg,
            rho,
            inner_iters,
            use_adaptive_rho,
        )
        A = A_T.T.to(dtype)  # dtype guard
        UA = UA_T.T.to(dtype)  # dtype guard

        # [Step 2] B update (A, M, d, Da, Db fixed)
        # Optimization: min_B ||W_bal - Da*A*(diag(d)+M)*B*Db||^2 s.t. B in SVID
        # Reformulated: min_B ||W_bal/Db - D*B||^2 where D = Da*A*(diag(d)+M)
        # Compute reciprocal in same dtype to prevent dtype promotion
        eps_Db = Db.new_tensor(1e-12)
        invDb = Db.clamp_min(eps_Db).reciprocal()
        W_eff_B = W_bal * invDb[None, :]
        # Avoid explicit diag(d) in D computation
        Atil = Da[:, None] * A
        D = (Atil @ M) + Atil * d[None, :]  # (n×k)
        B, UB = _admm_update_Z_given_left(
            D, W_eff_B, B, UB, reg, rho, inner_iters, use_adaptive_rho
        )
        B = B.to(dtype)  # dtype guard
        UB = UB.to(dtype)  # dtype guard

        # [Step 3] Gauge balance adjustment (improve numerical stability)
        # Normalize A, B column/row norms to 1, absorb scales into d, M
        # Reduce frequency to save computation time (every 10 steps)
        if t % 10 == 0:
            A, B, d, M = _gauge_balance_dense_Mprime(A, B, d, M)

        # dtype barrier: unify all parameters to W_bal.dtype
        A = A.to(dtype)
        B = B.to(dtype)
        d = d.to(dtype)
        M = M.to(dtype)
        Da = Da.to(dtype)
        Db = Db.to(dtype)

        # Outer ADMM ρ adjustment (evaluated every 5 steps)
        # Based on Boyd et al. (2011) adaptive ρ adjustment strategy
        if t > 0 and t % 5 == 0:
            # Primal residual: relative ||A^(k+1) - A^(k)|| (indicates slow convergence)
            r_primal = max(
                torch.norm(A - A_prev, "fro") / torch.norm(A, "fro").clamp(min=1e-12),
                torch.norm(B - B_prev, "fro") / torch.norm(B, "fro").clamp(min=1e-12),
            ).item()

            # Dual residual: relative ||U|| (indicates constraint violation magnitude)
            s_dual = max(
                torch.norm(UA, "fro") / torch.norm(A, "fro").clamp(min=1e-12),
                torch.norm(UB, "fro") / torch.norm(B, "fro").clamp(min=1e-12),
            ).item()

            # Adjust ρ based on primal/dual residual balance (safe division)
            # ratio1 = r_primal / max(s_dual, 1e-12)
            # ratio2 = s_dual / max(r_primal, 1e-12)

            if r_primal > 10 * s_dual:
                # Primal residual is large → increase ρ (enforce constraints more strongly)
                rho = min(rho * 2.0, rho_max)
                # Suppress rho adjustment messages
                # if t % 50 == 0:
                #     logger.debug(f"    Increasing rho to {rho:.2f} (r_p/s_d = {ratio1:.1f})")
            elif s_dual > 10 * r_primal:
                # Dual residual is large → decrease ρ (emphasize objective function more)
                rho = max(rho * 0.5, rho_min)
                # Suppress rho adjustment messages
                # if t % 50 == 0:
                #     logger.debug(f"    Decreasing rho to {rho:.2f} (s_d/r_p = {ratio2:.1f})")

        # Update for next comparison
        if t % 5 == 4:  # When next step is ρ adjustment timing
            A_prev = A.clone()
            B_prev = B.clone()

        # Reset auxiliary variables only after gauge balance adjustment
        if t % 10 == 0:
            UA.zero_()
            UB.zero_()

        # [Step 4] M update (A, B, d, Da, Db fixed)
        # Compute closed-form solution
        M = dense_M_closed_form_given_d_stable(A, B, Da, Db, W_bal, d, lam=1e-6).to(dtype)

        # [Step 4.5] d update (A, B, M, Da, Db fixed)
        # Update d even in Dense version (improve numerical stability)
        # Note: d update in Dense version is computationally expensive; enable as needed
        # d = update_d_hadamard_dense(A, B, Da, Db, W_bal, M, lam=1e-3).to(dtype)

        # [Step 5] Da, Db update (A, B, M, d fixed)
        # Execute every scale_update_every steps to reduce computation cost
        # Use factored version without explicitly creating Mprime
        Da_new, Db_new = update_Da_Db_closed_form_factored(
            A, B, d, M, W_bal, Da, Db, every=scale_update_every, step=t
        )
        Da_new = Da_new.to(dtype)
        Db_new = Db_new.to(dtype)  # just in case

        # Adaptive damping (prevent abrupt changes)
        # Conservative initially (η=0.05), aggressive later (η→0.5)
        eta = min(0.5, 0.05 + 0.45 * (t / max(1, outer_iters - 1)))
        Da = ((1 - eta) * Da + eta * Da_new).clamp_min(1e-6)
        Db = ((1 - eta) * Db + eta * Db_new).clamp_min(1e-6)

        # Periodic convergence check (thinned execution to reduce computation cost)
        if t % error_check_every == 0 or t == outer_iters - 1:
            # Compute current reconstruction error (fast version)
            Atil = Da[:, None] * A
            Btil = B * Db[None, :]
            GA = Atil.T @ Atil
            GB = Btil @ Btil.T

            # T computation: compute simply at once
            # Consider blocking only if memory is insufficient
            T = Atil.T @ (W_bal @ Btil.T)

            err = _frob_error_dense_fast(W_norm2, GA, GB, T, d, M)

            # NaN check
            if not torch.isfinite(torch.tensor(err)):
                logger.debug(
                    f"    [WARNING] NaN/Inf detected in error computation at step {t}, using previous error"
                )
                err = prev_error if prev_error != float("inf") else 1e10

            # Update best parameters (if error improved)
            if err < best_error:
                best_error = err
                best_params = (
                    A.clone(),
                    B.clone(),
                    d.clone(),
                    M.clone(),
                    Da.clone(),
                    Db.clone(),
                )
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Progress display (reduced to every 100 steps)
            if t % 100 == 0 or t == outer_iters - 1:
                if t == 0:
                    logger.debug(f"  Step {t:3d}: error = {err:.4e}")
                else:
                    rel_change = abs(prev_error - err) / (prev_error + 1e-12)
                    logger.debug(
                        f"  Step {t:3d}: error = {err:.4e}, rel_change = {rel_change:.2e}"
                    )

            # Convergence check (always run at least 3/4 of specified iterations, relaxed criteria)
            if t >= (3 * outer_iters) // 4 and abs(prev_error - err) < 1e-5 * (prev_error + 1e-12):
                logger.debug(f"  Converged at step {t} (after minimum {(3*outer_iters)//4} steps)")
                break

            # Early stopping disabled (run all iterations for stability)
            # if t >= outer_iters // 2 and no_improve_count >= 20:
            #     logger.debug(f"  Early stopping at step {t} (after minimum {outer_iters//2} steps)")
            #     if best_params is not None:
            #         A, B, d, M, Da, Db = best_params
            #     break

            prev_error = err

    # Return best parameters (if better than final parameters)
    if best_params is not None:
        # best_error is already recorded, so only compute final error
        Atil = Da[:, None] * A
        Btil = B * Db[None, :]
        GA = Atil.T @ Atil
        GB = Btil @ Btil.T

        # T computation: compute simply at once
        T = Atil.T @ (W_bal @ Btil.T)

        err_curr = _frob_error_dense_fast(W_norm2, GA, GB, T, d, M)

        if best_error < err_curr:
            A, B, d, M, Da, Db = best_params
            logger.debug(f"  Using best parameters with error: {best_error:.4e}")

    #! Free memory
    del W_bal, M_init, Da_init, Db_init
    del UA, UB, A_prev, B_prev, W_norm2
    del eps_Da, invDa, W_eff_A_T, Btil, C, A_T, UA_T
    del eps_Db, invDb, W_eff_B, Atil, D
    del Da_new, Db_new
    del GA, GB, T
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return A, B, d, M, Da, Db


def extended_admm_lowrank(
    W_bal: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    rank: int,
    *,
    outer_iters: int = 120,
    inner_iters: int = 3,
    reg: float = 3e-2,
    use_adaptive_rho: bool = True,
    scale_update_every: int = 10,
    U_init: Optional[torch.Tensor] = None,
    V_init: Optional[torch.Tensor] = None,
    Da_init: Optional[torch.Tensor] = None,
    Db_init: Optional[torch.Tensor] = None,
    # convergence_tol=1e-6,
    # patience=20,
) -> Tuple:
    """
    Low-rank version of extended ADMM (co-optimization of all parameters).

    Objective: W_bal ≈ Da * A * (diag(d) + U*V^T) * B * Db

    Differences from Dense version:
    - Uses low-rank decomposition U*V^T instead of M (rank << k)
    - Includes d in optimization targets (Hadamard update)
    - U, V updates use closed-form solutions

    Alternating optimization in each outer iteration:
    1. A update: ADMM + SVID constraint
    2. B update: ADMM + SVID constraint
    3. Gauge balance adjustment (including d, U, V)
    4. U, V update: closed-form solution
    5. d update: closed-form solution using Hadamard product
    6. Da, Db update: closed-form solution (thinned execution)

    Returns:
        Optimized (A, B, d, U, V, Da, Db)
    """
    device = W_bal.device
    dtype = W_bal.dtype
    n, m = W_bal.shape
    k = A.shape[1]  # intermediate dimension

    # Parameter initialization
    A = A.clone()
    B = B.clone()
    d = d.clone()  # initialized by warmup, fixed in ADMM

    # U, V initialization (use Middle phase results, or zero-initialize)
    if U_init is not None and U_init.shape == (k, max(0, rank)):
        U = U_init.clone()
        # Suppress initialization messages
        # logger.debug(f"  Using Middle U init: shape {U.shape}")
    else:
        U = torch.zeros((k, max(0, rank)), device=device, dtype=dtype)
        # logger.debug(f"  Zero U init: shape {U.shape}")

    if V_init is not None and V_init.shape == (k, max(0, rank)):
        V = V_init.clone()
        # logger.debug(f"  Using Middle V init: shape {V.shape}")
    else:
        V = torch.zeros((k, max(0, rank)), device=device, dtype=dtype)
        # logger.debug(f"  Zero V init: shape {V.shape}")

    Da = torch.ones(n, device=device, dtype=dtype) if Da_init is None else Da_init.clone()
    Db = torch.ones(m, device=device, dtype=dtype) if Db_init is None else Db_init.clone()

    # ADMM auxiliary variables
    UA = torch.zeros_like(A)
    UB = torch.zeros_like(B)

    # Convergence monitoring variables
    prev_error = float("inf")
    no_improve_count = 0
    best_error = float("inf")
    best_params = None

    # Extended ADMM Low-rank outputs only concise logs

    # Initial ρ (start smaller for low-rank version)
    rho = 0.5  # Start with smaller ρ for low-rank
    # rho_min, rho_max = 0.05, 5.0  # Adjust ρ bounds

    # Record initial error (for improvement rate display)
    # initial_error = torch.norm(
    #     W_bal - _compose_lowrank_bal(A, d, U, V, B, Da, Db), p="fro"
    # ).item()

    # Pre-compute W_bal norm (for fast error calculation)
    W_norm2 = torch.norm(W_bal, p="fro").pow(2).item()

    for t in range(outer_iters):
        # === Alternating optimization steps (low-rank version) ===
        # Current decomposition: W_bal ≈ Da * A * (diag(d) + U @ V.T) * B * Db

        # Current Mprime (compute only when needed, avoid diag(d))
        # Mprime is not computed here as C/D computation uses element-wise products

        # A-update (ADMM+SVID)
        # Compute reciprocal in same dtype to prevent dtype promotion
        eps_Da = Da.new_tensor(1e-12)
        invDa = Da.clamp_min(eps_Da).reciprocal()
        W_eff_A_T = invDa[:, None] * W_bal
        # Avoid explicit diag(d) in C computation
        Btil = B * Db[None, :]
        if U is not None and U.numel() > 0 and V is not None and V.numel() > 0:
            C = (U @ (V.T @ Btil)) + d[:, None] * Btil  # k×m, U@V.T part also computed efficiently
        else:
            C = d[:, None] * Btil  # k×m
        A_T, UA_T = _admm_update_Z_given_left(
            C.T,
            W_eff_A_T.T,
            A.T,
            UA.T,
            reg,
            rho,
            inner_iters,
            use_adaptive_rho,
        )
        A = A_T.T.to(dtype)  # dtype guard
        UA = UA_T.T.to(dtype)  # dtype guard

        # B-update (ADMM+SVID)
        # Compute reciprocal in same dtype to prevent dtype promotion
        eps_Db = Db.new_tensor(1e-12)
        invDb = Db.clamp_min(eps_Db).reciprocal()
        W_eff_B = W_bal * invDb[None, :]
        # Avoid explicit diag(d) in D computation
        Atil = Da[:, None] * A
        if U is not None and U.numel() > 0 and V is not None and V.numel() > 0:
            D = (Atil @ U) @ V.T + Atil * d[None, :]  # (n×k), U@V.T part also computed efficiently
        else:
            D = Atil * d[None, :]  # (n×k)
        B, UB = _admm_update_Z_given_left(
            D, W_eff_B, B, UB, reg, rho, inner_iters, use_adaptive_rho
        )
        B = B.to(dtype)  # dtype guard
        UB = UB.to(dtype)  # dtype guard

        # Gauge balance adjustment (limited to every 10 steps for stability)
        if t % 10 == 0:
            A, B, d, U, V = _gauge_balance_lowrank(A, B, d, U, V)
            # Reset auxiliary variables only after gauge balance adjustment
            UA.zero_()
            UB.zero_()

        # dtype barrier: unify all parameters to W_bal.dtype
        A = A.to(dtype)
        B = B.to(dtype)
        d = d.to(dtype)
        Da = Da.to(dtype)
        Db = Db.to(dtype)
        if U is not None:
            U = U.to(dtype)
        if V is not None:
            V = V.to(dtype)

        # [Step 4] U, V update (A, B, d, Da, Db fixed)
        # Compute optimal low-rank approximation using whitened SVD
        # Strengthen regularization for numerical stability
        U_new, V_new = uv_closed_form_given_d(A, B, Da, Db, W_bal, d, rank, lam=1e-4)
        if U_new is not None and V_new is not None:
            # Apply adaptive damping to U, V updates (more conservative)
            uv_eta = min(
                0.3, 0.05 + 0.25 * (t / max(1, outer_iters - 1))
            )  # even more conservative
            if U is not None and V is not None:
                U = ((1 - uv_eta) * U + uv_eta * U_new).to(dtype)
                V = ((1 - uv_eta) * V + uv_eta * V_new).to(dtype)
            else:
                U = U_new.to(dtype)
                V = V_new.to(dtype)
        elif U is None or V is None:
            U = U_new.to(dtype) if U_new is not None else U
            V = V_new.to(dtype) if V_new is not None else V

        # [Step 5] d update (A, B, U, V, Da, Db fixed)
        # Closed-form solution using Hadamard product
        d = update_d_hadamard(A, B, Da, Db, W_bal, U, V, lam=1e-3).to(dtype)

        # Da/Db update (with adaptive damping)
        # Process efficiently without creating Mprime
        if U is not None and U.numel() > 0 and V is not None and V.numel() > 0:
            # Low-rank version factored update (without explicitly creating UV^T)
            Da_new, Db_new = update_Da_Db_closed_form_lowrank(
                A, B, d, U, V, W_bal, Da, Db, every=scale_update_every, step=t
            )
        else:
            # rank=0 case
            Da_new, Db_new = update_Da_Db_closed_form_factored(
                A, B, d, None, W_bal, Da, Db, every=scale_update_every, step=t
            )
        Da_new = Da_new.to(dtype)
        Db_new = Db_new.to(dtype)  # just in case

        # Adaptive damping: even more conservative for low-rank version
        eta = min(0.3, 0.02 + 0.28 * (t / max(1, outer_iters - 1)))  # more conservative
        Da = ((1 - eta) * Da + eta * Da_new).clamp_min(1e-6)
        Db = ((1 - eta) * Db + eta * Db_new).clamp_min(1e-6)

        # Convergence check and progress display (frequency adjusted for stability)
        if t % 20 == 0 or t == outer_iters - 1:
            # Fast k×k error computation
            Atil = Da[:, None] * A
            Btil = B * Db[None, :]
            GA = Atil.T @ Atil
            GB = Btil @ Btil.T
            T = Atil.T @ (W_bal @ Btil.T)
            err = _frob_error_lowrank_fast(W_norm2, GA, GB, T, d, U, V)

            # Save best parameters
            if err < best_error:
                best_error = err
                best_params = (
                    A.clone(),
                    B.clone(),
                    d.clone(),
                    U.clone(),
                    V.clone(),
                    Da.clone(),
                    Db.clone(),
                )
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Progress display (reduced to every 100 steps)
            if t % 100 == 0 or t == outer_iters - 1:
                if t == 0:
                    logger.debug(f"  Step {t:3d}: error = {err:.4e}")
                else:
                    rel_change = abs(prev_error - err) / (prev_error + 1e-12)
                    logger.debug(
                        f"  Step {t:3d}: error = {err:.4e}, rel_change = {rel_change:.2e}"
                    )

            # Convergence check (always run at least 3/4 of specified iterations, relaxed criteria)
            if t >= (3 * outer_iters) // 4 and abs(prev_error - err) < 1e-5 * (prev_error + 1e-12):
                logger.debug(f"  Converged at step {t} (after minimum {(3*outer_iters)//4} steps)")
                break

            # Early stopping disabled (run all iterations for stability)
            # if t >= outer_iters // 2 and no_improve_count >= 20:
            #     logger.debug(f"  Early stopping at step {t} (after minimum {outer_iters//2} steps)")
            #     if best_params is not None:
            #         A, B, d, U, V, Da, Db = best_params
            #     break

            prev_error = err

    # Return best parameters (if better than final parameters)
    if best_params is not None:
        A_best, B_best, d_best, U_best, V_best, Da_best, Db_best = best_params
        err_best = torch.norm(
            W_bal - _compose_lowrank_bal(A_best, d_best, U_best, V_best, B_best, Da_best, Db_best),
            p="fro",
        ).item()
        err_curr = torch.norm(W_bal - _compose_lowrank_bal(A, d, U, V, B, Da, Db), p="fro").item()
        if err_best < err_curr:
            A, B, d, U, V, Da, Db = best_params
            logger.debug(f"  Using best parameters with error: {best_error:.4e}")

    # Display final improvement rate (concisely)
    # final_error = torch.norm(
    #     W_bal - _compose_lowrank_bal(A, d, U, V, B, Da, Db), p="fro"
    # ).item()
    # improvement = (initial_error - final_error) / initial_error * 100
    # Suppress improvement messages for cleaner output
    # if improvement > 0:
    #     logger.debug(f"  Improvement from initial: {improvement:.1f}%")
    # else:
    #     logger.debug(f"  Warning: No improvement from initial error ({initial_error:.4e} -> {final_error:.4e})")

    #! Free memory
    del W_bal, U_init, V_init, Da_init, Db_init
    del UA, UB, W_norm2
    del eps_Da, invDa, W_eff_A_T, Btil, C, A_T, UA_T
    del eps_Db, invDb, W_eff_B, Atil, D
    del U_new, V_new, Da_new, Db_new
    del GA, GB, T
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return A, B, d, U, V, Da, Db
