# -*- coding: utf-8 -*-
"""


Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

- Ported from QEP-dev codebase (2025/12/05)

============================================================

Original DBF (Double Binary Factorization) implementation

Implementation of the basic DBF algorithm. Decomposes weight matrices into
products of binary matrices and scaling factors:
    W ≈ A * diag(mid) * B
where A, B ∈ {-1, 0, 1} are sparse sign matrices satisfying SVID constraints.

Optimized using ADMM (Alternating Direction Method of Multipliers).
"""

import gc
from logging import getLogger
from typing import Dict, Optional, Tuple

import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch
import transformers

from .balance import balance_track


def _power_rank1_abs_scaled(
    A: torch.Tensor, nit: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute the largest absolute singular value and corresponding vectors via power iteration.

    Computes the largest singular value σ of |A| and corresponding left/right singular vectors a, m.
    Separates scale: σ = a^T |A| m, returns a, m scaled by √σ.

    Args:
        A: Input matrix.
        nit: Number of iterations.

    Returns:
        a: Left singular vector * √σ.
        m: Right singular vector * √σ.
        sigma: Largest singular value.
    """
    a = torch.randn(A.shape[0], device=A.device, dtype=A.dtype)
    m = torch.randn(A.shape[1], device=A.device, dtype=A.dtype)
    for _ in range(nit):
        a = A.abs() @ m
        a = a / (torch.norm(a) + 1e-12)
        m = A.abs().t() @ a
        m = m / (torch.norm(m) + 1e-12)
    sigma = float(a.t() @ (A.abs() @ m))
    sigma = max(sigma, 1e-12)
    root_sigma = torch.sqrt(
        torch.tensor(sigma, device=A.device, dtype=A.dtype)
    )
    a = a * root_sigma
    m = m * root_sigma
    return a, m, sigma


def svd_abs(W: torch.Tensor) -> torch.Tensor:
    """
    Rank-1 approximation via SVID (Sign-Value-Independent Decomposition).

    Decomposes matrix W into:
        W ≈ diag(a) @ sign(W) @ diag(m)
    where a, m are singular vectors corresponding to the largest singular value of |W|.

    Computed efficiently without explicitly creating diagonal matrices for memory efficiency.

    Returns:
        Result efficiently computed as (a[:, None] * sign(W)) * m[None, :].
    """
    S = torch.sign(W)
    S[S == 0] = 1.0
    a, m, _ = _power_rank1_abs_scaled(W, nit=10)
    return (a[:, None] * S) * (m[None, :])


def clear_dbf_meta(weight_results: Dict[str, torch.Tensor]) -> None:
    """
    Delete DBF/MDBF-related metadata (used during error handling).

    Deletes the following attributes:
    - DBF-related: dbf_A, dbf_B, dbf_mid, is_dbf_quantized
    - MDBF-related: all mdbf_* attributes
    - Weight metadata: A, B, mid
    """
    # DBF-related
    attr_list = ["dbf_A", "dbf_B", "dbf_mid", "is_dbf_quantized"]
    # MDBF-related
    attr_list += [
        "mdbf_A",
        "mdbf_B",
        "mdbf_mid",
        "mdbf_d",
        "mdbf_U",
        "mdbf_V",
        "mdbf_Da",
        "mdbf_Db",
        "mdbf_Dr",
        "mdbf_Dc",
        "is_mdbf_quantized",
        "mdbf_use_dense",
        "mdbf_rank",
    ]
    # Weight metadata
    attr_list += ["A", "B", "mid"]
    for attr in attr_list:
        if attr in weight_results:
            del weight_results[attr]
        elif attr in weight_results.get("weight", {}):
            del weight_results["weight"][attr]


def _compute_sparsity(W: torch.Tensor, target_bits: float) -> float:
    """
    Estimate optimal sparse density from target bit-width.

    Determines the sparse density (proportion of non-zero elements)
    to achieve the target bit-width, considering DBF compression ratio.
    Uses a linear search approach.

    Bit count calculation considering entropy coding:
    - Each non-zero element: 2 bits (sign + existence)
    - Mask information: compression based on entropy

    Args:
        W: Input matrix (n, m).
        target_bits: Average bits per element.

    Returns:
        Estimated sparse density (0.05 to 0.95).
    """
    n, m = W.shape
    sm, lg = min(n, m), max(n, m)
    mid_dim = max(1, int(1.5 * sm * lg / (sm + lg)))
    lim = float(target_bits)
    sp = 0.1
    for density in np.linspace(0.05, 0.95, 451):
        total_pars = sm * lg
        mask_size = sm * mid_dim + mid_dim * lg
        total_ones2 = sm * lg * density
        p = total_ones2 / max(mask_size, 1)
        if 0.0 < p < 1.0:
            ent = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        else:
            ent = 0.0
        total_size_bits = total_ones2 * 2.0 + mask_size * ent
        bpp = total_size_bits / max(total_pars, 1)
        if bpp <= lim + 1e-12:
            sp = float(density)
    return sp


def find_other2(
    A: torch.Tensor,
    W: torch.Tensor,
    Z: torch.Tensor,
    U: torch.Tensor,
    reg: float = 0.0,
    rho_start: float = 1.0,
    iters: int = 3,
    use_adaptive_rho: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ADMM sub-solver: optimize Z under SVID constraint with A fixed.

    Optimization problem:
        minimize   ||W - A*Z||_F^2 + reg*||Z||_F^2
        subject to Z ∈ SVID (sign-value independent decomposition)

    Split into two sub-problems via ADMM:
    1. B-update: Quadratic problem (closed-form solution)
    2. Z-update: SVID projection (svd_abs function)

    Args:
        A: Fixed left-side matrix.
        W: Target matrix.
        Z: Initial solution (satisfying SVID constraint).
        U: ADMM auxiliary variable.
        reg: Regularization parameter.
        rho_start: Initial penalty parameter.
        iters: Number of inner iterations.
        use_adaptive_rho: Whether to use adaptive ρ adjustment.

    Returns:
        Z: Updated Z (satisfying SVID constraint).
        U: Updated auxiliary variable.
    """
    # Upcast to float32 for numerical precision
    A = A.float()
    W = W.float()
    Z = Z.float()
    U = U.float()

    # Pre-compute A^T*A and A^T*W (avoid recomputing each iteration)
    XX = A.T @ A  # (k, k) matrix
    if reg > 0:
        XX += torch.eye(XX.shape[0], device=XX.device, dtype=XX.dtype) * (
            XX.diag().mean() * reg
        )
    XY = A.T @ W  # (k, m) matrix

    if use_adaptive_rho:
        # Use adaptive ρ adjustment (Boyd et al. 2011 method)
        rho = rho_start
        Z_prev = Z.clone()
        I = torch.eye(XX.shape[0], device=XX.device, dtype=XX.dtype)

        for _ in range(iters):
            Z_prev = Z.clone()

            # B-update: (A^T*A + ρ*I)*B = A^T*W + ρ*(Z - U)
            lhs = XX + rho * I
            rhs = XY + rho * (Z - U)

            # Prefer Cholesky decomposition, fall back to general solver on failure
            try:
                L = torch.linalg.cholesky(lhs)
                B = (
                    torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
                    if rhs.dim() == 1
                    else torch.cholesky_solve(rhs, L)
                )
            except Exception:
                B = torch.linalg.solve(lhs, rhs)

            # Z-update: SVID projection
            Z = svd_abs(B + U)

            # U-update: Update dual variable
            U = U + (B - Z)

            # Adaptive ρ adjustment (except last iteration)
            if _ < iters - 1:
                # Compute primal/dual residuals
                r_primal = torch.norm(B - Z, "fro")  # Constraint violation
                s_dual = rho * torch.norm(Z - Z_prev, "fro")  # Dual change

                # Adjust ρ based on residual balance
                mu = 10.0  # Adjustment threshold
                if r_primal > mu * s_dual:
                    rho *= 2.0
                    U /= 2.0  # Primal too large → increase ρ
                elif s_dual > mu * r_primal:
                    rho /= 2.0
                    U *= 2.0  # Dual too large → decrease ρ

                # Clamp ρ range
                rho = float(
                    torch.clamp(torch.tensor(rho, device=A.device), 1e-6, 1e6)
                )
    else:
        # Use fixed ρ (pre-compute inverse for speedup)
        rho = 1.0
        XXinv = torch.inverse(
            XX + torch.eye(XX.shape[0], device=XX.device) * rho
        )
        XXinv_start = torch.inverse(
            XX + torch.eye(XX.shape[0], device=XX.device) * rho_start
        )

        # First B-update
        B = XXinv_start @ (XY + rho_start * (Z - U))

        # ADMM iterations
        for _ in range(iters - 1):
            Z = svd_abs(B + U)
            U = U + (B - Z)
            B = XXinv @ (XY + rho * (Z - U))

        # Final Z-update
        Z = svd_abs(B + U)
        U = U + (B - Z)
    return Z.float(), U.float()


def _factorizef(
    W: torch.Tensor,
    i_norm: torch.Tensor,
    o_norm: torch.Tensor,
    iters: int,
    reg: float,
    target_bits: float = 1.0,
    use_adaptive_rho: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DBF decomposition via ADMM iterations (for n < m case).

    Decomposes weight matrix W (n×m) into:
        W ≈ A * diag(mid) * B
    where:
    - A ∈ {-1,0,1}^{n×k}: Sign matrix with sparsity asp
    - B ∈ {-1,0,1}^{k×m}: Sparse sign matrix
    - mid: Intermediate scaling vector

    Args:
        W: Input matrix (n×m, n<m).
        i_norm, o_norm: Input/output norms (for preconditioning).
        iters: Number of ADMM iterations.
        reg: Regularization parameter.
        target_bits: Target bit-width (used for determining k).
        use_adaptive_rho: Whether to use adaptive ρ adjustment.

    Returns:
        W2: Reconstructed matrix.
        Az, Bz: Decomposed matrices (scale removed).
        1/mid: Inverse scale.
    """

    logger = getLogger(__name__)

    # Transpose and process when n >= m (for efficiency)
    if W.shape[0] >= W.shape[1]:
        return _factorizeT(
            W.T,
            i_norm,
            o_norm,
            iters,
            reg,
            target_bits,
            use_adaptive_rho,
        )

    # Number of non-zero elements (unused)
    # nza = int(W.shape[0] ** 2 * asp)
    # nzb = int(W.numel() * sp - nza)

    logger.debug("[DBF] _factorizef: Starting ADMM with %s iterations", iters)
    # sys.stdout.flush()

    # Preconditioning with input/output norms (improves numerical stability)
    norm = i_norm.sqrt() + 1e-8  # Input scaling
    norm_o = (o_norm.sqrt() + 1e-8).unsqueeze(1)  # Output scaling
    Wn = W * norm * norm_o  # Scaled matrix
    mid = int(
        target_bits * (W.shape[0] * W.shape[1]) / (W.shape[0] + W.shape[1])
    )
    mid = max(1, min(mid, W.shape[0], W.shape[1]))
    Az = torch.randn((W.shape[0], mid), device=W.device)
    Au = torch.zeros_like(Az)
    Bz = torch.randn((mid, W.shape[1]), device=W.device)
    Bu = torch.zeros_like(Bz)
    final_mid = None
    for itt in range(iters):
        rho_start = (
            0.03 + (1.0 - 0.03) * min(1.0, itt / max(1, iters - 3)) ** 3
        )
        mid_norm = Bz.norm(dim=1) + 1e-12
        Az_T, Au_T = find_other2(
            Bz.T / mid_norm,
            Wn.T,
            Az.T,
            Au.T,
            reg=reg,
            rho_start=rho_start,
            iters=3,
            use_adaptive_rho=use_adaptive_rho,
        )
        Az, Au = Az_T.T, Au_T.T
        mid_norm = Az.norm(dim=0) + 1e-12
        Bz, Bu = find_other2(
            Az / mid_norm,
            Wn,
            Bz,
            Bu,
            reg=reg,
            rho_start=rho_start,
            iters=3,
            use_adaptive_rho=use_adaptive_rho,
        )
        if itt % max(10, min(50, iters // 5)) == 0 or itt == iters - 1:
            try:
                current_reconstruction = (Az / mid_norm).matmul(Bz)
                frobenius_error = torch.norm(
                    Wn - current_reconstruction, "fro"
                ).item()
                logger.debug(
                    "  Step %.3d: recon error = %.4e", itt, frobenius_error
                )
            except Exception:
                pass
        if itt == iters - 1:
            final_mid = mid_norm

    # Undo preconditioning
    W2 = (Az / norm_o).matmul(Bz / norm)  # Reconstructed matrix
    return W2, Az / norm_o, Bz / norm, 1.0 / final_mid  # Return inverse scale


def _factorizeT(
    W: torch.Tensor,
    i_norm: torch.Tensor,
    o_norm: torch.Tensor,
    iters: int,
    reg: float,
    target_bits: float = 1.0,
    use_adaptive_rho: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DBF decomposition via ADMM iterations (for n >= m case).

    Same processing as _factorizef, adjusted for transposed matrices.
    Note that the scaling order differs.
    """

    logger = getLogger(__name__)

    # nza = int(W.shape[0] ** 2 * asp)
    # nzb = int(W.numel() * sp - nza)
    norm = i_norm.sqrt().unsqueeze(1) + 1e-8
    norm_o = o_norm.sqrt() + 1e-8
    Wn = W * norm * norm_o  # Scaled matrix

    # Compute intermediate dimension k (based on target bit-width)
    # k ≈ target_bits * (n*m) / (n+m)
    mid = int(
        target_bits * (W.shape[0] * W.shape[1]) / (W.shape[0] + W.shape[1])
    )
    mid = max(1, min(mid, W.shape[0], W.shape[1]))  # 1 ≤ k ≤ min(n,m)

    # Initialize ADMM variables
    Az = torch.randn(
        (W.shape[0], mid), device=W.device
    )  # Initial A (random)
    Au = torch.zeros_like(Az)  # Auxiliary variable for A
    Bz = torch.randn(
        (mid, W.shape[1]), device=W.device
    )  # Initial B (random)
    Bu = torch.zeros_like(Bz)  # Auxiliary variable for B
    final_mid = None  # Final intermediate scale

    # ADMM main loop
    logger.debug("[DBF] Starting main ADMM loop...")
    # sys.stdout.flush()
    for itt in range(iters):
        # ρ warmup (start small, gradually increase)
        rho_start = (
            0.03 + (1.0 - 0.03) * min(1.0, itt / max(1, iters - 3)) ** 3
        )

        # === A update (B fixed) ===
        # Normalize by intermediate scale
        mid_norm = Bz.norm(dim=1) + 1e-12
        # Solve transposed problem (for efficiency)
        Az_T, Au_T = find_other2(
            Bz.T / mid_norm,
            Wn.T,
            Az.T,
            Au.T,
            reg=reg,
            rho_start=rho_start,
            iters=3,
            use_adaptive_rho=use_adaptive_rho,
        )
        Az, Au = Az_T.T, Au_T.T

        # === B update (A fixed) ===
        # Normalize by intermediate scale
        mid_norm = Az.norm(dim=0) + 1e-12
        Bz, Bu = find_other2(
            Az / mid_norm,
            Wn,
            Bz,
            Bu,
            reg=reg,
            rho_start=rho_start,
            iters=3,
            use_adaptive_rho=use_adaptive_rho,
        )

        # Progress display (every 50 steps or last step)
        if itt % max(10, min(50, iters // 5)) == 0 or itt == iters - 1:
            try:
                current_reconstruction = (Az / mid_norm).matmul(Bz)
                frobenius_error = torch.norm(
                    Wn - current_reconstruction, "fro"
                ).item()
                logger.debug(
                    "  Step %.3d: recon error = %.4e", itt, frobenius_error
                )
            except Exception:
                pass

        # Save the final scale
        if itt == iters - 1:
            final_mid = mid_norm
    W2 = (Az / norm).matmul(Bz / norm_o)
    return W2.T, (Bz / norm_o).T, (Az / norm).T, 1.0 / final_mid


def _factorize(
    W: torch.Tensor,
    i_norm: torch.Tensor,
    o_norm: torch.Tensor,
    iters: int = 260,
    reg: float = 3e-2,
    target_bits: float = 1.0,
    use_adaptive_rho: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main DBF decomposition processing.

    1. Execute decomposition via _factorizef or _factorizeT
    2. Balance A,B norms for improved numerical stability
    3. Measure and display actual sparsity
    """

    logger = getLogger(__name__)
    logger.debug(
        "[DBF] _factorize called with iters=%s, W.shape=%s", iters, W.shape
    )
    # sys.stdout.flush()

    # Execute decomposition
    _, Ab, Bb, mid = _factorizef(
        W, i_norm, o_norm, iters, reg, target_bits, use_adaptive_rho
    )

    # Balance A,B norms (for numerical stability)
    Ac = Ab
    Bc = Bb
    An = Ac.norm() + 1e-12
    Bn = Bc.norm() + 1e-12
    Ac = Ac * (Bn / An).sqrt()  # Adjust A scale
    Bc = Bc * (An / Bn).sqrt()  # Adjust B scale

    # Reconstruction and sparsity measurement
    W3 = (Ac * mid).matmul(Bc)
    actual_sparsity = ((Ac != 0).sum() + (Bc != 0).sum()).item() / (
        Ac.numel() + Bc.numel()
    )
    logger.debug("[DBF] Actual sparsity: %.4f", actual_sparsity)

    return W3, Ac, Bc, mid


def run_dbf_original(
    hessian: Optional[torch.Tensor],
    layer: torch.nn.Module,
    target_bits: float = 1.0,
    iters: int = 260,
    reg: float = 3e-2,
    use_balancing: bool = True,
    balance_iters: int = 40,
    balance_alpha: float = 1.0,
    balance_mode: str = "l1",
    use_adaptive_rho: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Execute basic DBF (Double Binary Factorization).

    Decomposes the layer's weight matrix into:
        W ≈ A * diag(mid) * B
    where A, B are sparse sign matrices satisfying SVID constraints.

    Processing flow:
    1. Optionally apply matrix balancing
    2. Compute sparsity and decompose via ADMM
    3. Transpose processing for Conv1D compatibility
    4. Save metadata to layer

    Args:
        hessian: Hessian matrix (may contain Hessian information).
        layer: Layer to quantize.
        target_bits: Average bits per element.
        iters: Number of ADMM iterations.
        reg: Regularization parameter.
        use_balancing: Whether to use matrix balancing.
        balance_iters: Number of balancing iterations.
        balance_alpha: Target norm for balancing.
        balance_mode: Balancing mode ('l1' or 'l2').
        use_adaptive_rho: Whether to use adaptive ρ adjustment in ADMM.
    """

    logger = getLogger(__name__)

    with torch.no_grad():
        # Backup original weights (for restoration on error)
        weight_backup = layer.weight.data.detach().clone()

        # Convert weights to float32 for processing
        W = layer.weight.data.clone().float()

        # Transpose for Conv1D (transformers library convention)
        is_conv1d = isinstance(layer, transformers.Conv1D)
        if is_conv1d:
            W = W.t()

        # Save original for final error calculation
        W_original = W.clone()

        # === Optional: matrix balancing ===
        balance_Dr = None
        balance_Dc = None
        if use_balancing:
            # Balancing for numerical stability
            W_balanced, balance_hist = balance_track(
                W, its=balance_iters, alpha=balance_alpha, mode=balance_mode
            )
            balance_Dr = balance_hist["Dr"]
            balance_Dc = balance_hist["Dc"]
            W = W_balanced

            # Free memory
            del W_balanced, balance_hist
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # === Set input/output norms ===
        if use_balancing:
            # Use unit norms when balancing is applied
            i_norm = torch.ones(W.shape[1], device=W.device)
            o_norm = torch.ones(W.shape[0], device=W.device)
        else:
            # Without balancing, obtain from layer or Hessian
            i_norm = getattr(layer, "i_norm", None)
            o_norm = getattr(layer, "o_norm", None)

            # When input norm is not available
            if i_norm is None:
                if hessian is not None:
                    # Use Hessian diagonal elements
                    i_norm = torch.diag(hessian).clamp(min=1e-8)
                else:
                    # Default to unit norms
                    i_norm = torch.ones(W.shape[1], device=W.device)

            # Default to unit norms when output norm is not available
            if o_norm is None:
                o_norm = torch.ones(W.shape[0], device=W.device)

        # === Execute DBF decomposition ===
        # Compute sparsity
        sp = _compute_sparsity(W, target_bits)
        asp = (
            sp / 2 if W.shape[0] == W.shape[1] else sp
        )  # Halve A's sparsity for square matrices

        logger.debug(
            "[DBF] Configuration: target_bits=%.2f, sp=%.3f, asp=%.3f",
            target_bits,
            sp,
            asp,
        )
        logger.debug(
            "[DBF] Using ADMM with %s ρ",
            "adaptive" if use_adaptive_rho else "fixed",
        )
        # sys.stdout.flush()  # Force output flush

        # Decomposition via ADMM
        W_fact, A_bal, B_bal, mid = _factorize(
            W,
            i_norm,
            o_norm,
            iters=iters,
            reg=reg,
            target_bits=target_bits,
            use_adaptive_rho=use_adaptive_rho,
        )

        # === Inverse balancing transform ===
        if use_balancing and balance_Dr is not None and balance_Dc is not None:
            # Inverse balance transform: W = Dr^{-1} * W_bal * Dc^{-1}
            invDr = 1.0 / (balance_Dr + 1e-12)
            invDc = 1.0 / (balance_Dc + 1e-12)
            W_fact = W_fact * invDr[:, None] * invDc[None, :]
            A_bal = invDr[:, None] * A_bal
            B_bal = B_bal * invDc[None, :]

            # Free memory
            del balance_Dr, balance_Dc, invDr, invDc

        # === Transpose processing for Conv1D compatibility ===
        if is_conv1d:
            # For Conv1D, W^T = B^T * mid * A^T, so swap A and B
            W_out = W_fact.t()
            A_store = B_bal.t()
            B_store = A_bal.t()
            mid_store = mid
        else:
            # Normal case
            W_out = W_fact
            A_store = A_bal
            B_store = B_bal
            mid_store = mid

        # === A,B norm balancing ===
        # Equalize Frobenius norms of A and B for numerical stability
        An = torch.norm(A_store, p="fro") + 1e-12
        Bn = torch.norm(B_store, p="fro") + 1e-12
        balance_factor = (Bn / An).sqrt()
        A_store = A_store * balance_factor
        B_store = B_store / balance_factor

        # === Final reconstruction ===
        if is_conv1d:
            # Conv1D: W = (B^T * mid * A^T)^T  
            W_balanced = (B_store.t() * mid_store).matmul(A_store.t()).t()
        else:
            # Normal: W = A * mid * B
            W_balanced = (A_store * mid_store).matmul(B_store)
        W_out = W_balanced

        ## === Update layer weights ===
        dequantized_weight = (
            W_out.reshape(layer.weight.shape)
            .to(layer.weight.data.dtype)
            .contiguous()
        )

        # === Error check: detect NaN/Inf ===
        if (
            torch.isnan(dequantized_weight).any()
            or torch.isinf(dequantized_weight).any()
        ):
            logger.warning(
                "[DBF] ERROR: NaN/Inf in weights. Reverting to original weights."
            )
            dequantized_weight = weight_backup.contiguous()
            clear_dbf_meta(dequantized_weight)
            logger.warning(
                "[DBF] ERROR Details: NaN count=%s, Inf count=%s",
                torch.isnan(W_out).sum().item(),
                torch.isinf(W_out).sum().item(),
            )
            return dequantized_weight

        # === Compute reconstruction error ===
        W_compare = W_balanced.t() if is_conv1d else W_balanced
        err = (W_compare - W_original).square().sum().item()  # Absolute error
        fro_norm = torch.norm(W_original, p="fro").item() + 1e-12
        rel_err = (
            torch.norm(W_compare - W_original, p="fro").item() / fro_norm
        )  # Relative error
        logger.debug(
            "[DBF] Final Reconstruction error: abs=%.4e, rel=%.4e",
            err,
            rel_err,
        )

        # === Save metadata ===
        dev = layer.weight.device
        dtype = layer.weight.dtype

        # Save DBF decomposition results
        dbf_A = A_store.to(dev, dtype=dtype)
        dbf_B = B_store.to(dev, dtype=dtype)
        dbf_mid = (
            mid_store
            if isinstance(mid_store, torch.Tensor)
            else torch.tensor(mid_store, device=dev, dtype=dtype)
        ).to(dtype)
        is_dbf_quantized = True

        # Attach metadata to weight object (for compatibility)
        weight_A = dbf_A
        weight_B = dbf_B
        weight_mid = dbf_mid

        # === Metadata verification ===
        # Verify by reconstructing from saved metadata
        if is_conv1d:
            W_meta = (dbf_B.t() * dbf_mid).matmul(dbf_A.t()).t()
        else:
            W_meta = (dbf_A * dbf_mid).matmul(dbf_B)

        # Check if within tolerance
        if not torch.allclose(
            W_meta, dequantized_weight, rtol=5e-3, atol=5e-3
        ):
            logger.warning(
                "[DBF] WARNING: metadata (A,B,mid) does not reconst"
                "ruct dequantized_weight within tolerance."
            )
            recon_error = torch.norm(W_meta - dequantized_weight, p="fro") / (
                torch.norm(dequantized_weight, p="fro") + 1e-12
            )
            logger.warning(
                "[DBF] Reconstruction relative error: %.4e", recon_error
            )
            if recon_error > 0.01:
                logger.warning(
                    "[DBF] Large error detected: A=%s, B=%s",
                    layer.dbf_A.shape,
                    layer.dbf_B.shape,
                )

        # === Memory cleanup ===
        del W, W_original, W_fact, A_bal, B_bal
        #! Free memory - additional cleanup below
        del (
            weight_backup,
            W_out,
            W_compare,
            W_meta,
            A_store,
            B_store,
            mid_store,
            mid,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Additional code
        weight_results = {
            "dequantized_weight": dequantized_weight,
            "dbf_A": weight_A,
            "dbf_B": weight_B,
            "dbf_mid": weight_mid,
            "is_dbf_quantized": is_dbf_quantized,
        }
        return weight_results
