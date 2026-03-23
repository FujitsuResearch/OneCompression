"""OneBit implementation.

This module provides the run_onebit function for OneBit quantization.

Functions:
    run_onebit(hessian, layer, iters, use_importance_scaling, use_balancing,
                balance_iters, balance_alpha): Execute OneBit quantization on a layer.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import gc

import torch
import transformers
import logging

logger = logging.getLogger(__name__)

from onecomp.quantizer.dbf.balance import balance_track


def run_onebit(
    hessian,
    layer,
    iters: int = 10,
    use_importance_scaling: bool = True,
    use_balancing: bool = True,
    balance_iters: int = 40,
    balance_alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Decompose the weight matrix with OneBit quantization.

    Decomposes W as W ≈ a ⊙ W_±1 ⊙ b^T, where:
      - W_±1: Sign matrix (sign(W))
      - a, b: Scaling vectors (rank-1 factors)

    Args:
        hessian (torch.Tensor): Hessian matrix.
        layer (torch.nn.Module): Target layer.
        iters (int): Optimization iterations.
        use_importance_scaling (bool): Whether to use importance scaling.
        use_balancing (bool): Whether to apply weight balancing.
        balance_iters (int): Balancing iterations.
        balance_alpha (float): Balancing alpha.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing quantization results with the following keys:
            - "dequantized_weight": Dequantized weights (original shape, original dtype, CPU).
            - "a": Scaling vector a.
            - "b": Scaling vector b.
            - "sign": Sign matrix sign(W).

    """
    # Step 1: Get the weight matrix
    W = layer.weight.data.clone().float()
    if isinstance(layer, transformers.Conv1D):
        W = W.t()

    # Save the original weight matrix
    W_original = W.clone() if use_balancing else W

    # Step 2: Weight balancing (optional)
    balance_Dr = None
    balance_Dc = None
    if use_balancing:
        logger.debug(f"[OneBit] Applying weight balancing \
                (mode=l1, iterations={balance_iters}, alpha={balance_alpha})")
        W_balanced, balance_hist = balance_track(
            W, its=balance_iters, alpha=balance_alpha, mode="l1"
        )
        balance_Dr = balance_hist["Dr"]
        balance_Dc = balance_hist["Dc"]

        # Check convergence
        final_kkt_row = (
            balance_hist["kkt_row"][-1] if len(balance_hist["kkt_row"]) > 0 else float("inf")
        )
        final_kkt_col = (
            balance_hist["kkt_col"][-1] if len(balance_hist["kkt_col"]) > 0 else float("inf")
        )
        logger.debug(f"[OneBit] Weight balancing completed: \
                KKT row={final_kkt_row:.2e}, KKT col={final_kkt_col:.2e}")

        W = W_balanced
        del W_balanced, balance_hist
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 3: Importance scaling
    if use_balancing:
        logger.debug("[OneBit] Weight balancing enabled, skipping i_norm/o_norm scaling")
        W_scaled = W
        norm_i = torch.ones(W.shape[1], device=W.device)
        norm_o = torch.ones(W.shape[0], device=W.device)
        use_importance_scaling = False
    elif use_importance_scaling:
        i_norm = getattr(layer, "i_norm", None)
        o_norm = getattr(layer, "o_norm", None)

        if i_norm is None:
            if hessian is not None:
                i_norm = torch.diag(hessian).clamp(min=1e-8)
                logger.debug("[OneBit] Using Hessian diagonal as i_norm approximation")
            else:
                i_norm = torch.ones(W.shape[1], device=W.device)
                logger.debug(
                    "[OneBit] Warning: No i_norm or Hessian available, using dummy values"
                )
        else:
            logger.debug(f"[OneBit] Using pre-collected i_norm (shape: {i_norm.shape})")

        if o_norm is None:
            o_norm = torch.ones(W.shape[0], device=W.device)
            logger.debug("[OneBit] Warning: o_norm not available, using dummy values")
        else:
            logger.debug(f"[OneBit] Using pre-collected o_norm (shape: {o_norm.shape})")

        norm_i = i_norm.sqrt().clamp(min=1e-8)
        norm_o = o_norm.sqrt().clamp(min=1e-8)

        W_scaled = W * norm_o.unsqueeze(1) * norm_i.unsqueeze(0)
    else:
        W_scaled = W
        norm_i = torch.ones(W.shape[1], device=W.device)
        norm_o = torch.ones(W.shape[0], device=W.device)

    # Step 4: Sign-Value-Independent Decomposition (SVID)
    W_sign = torch.sign(W_scaled)
    W_sign[W_sign == 0] = 1

    W_abs = W_scaled.abs()

    # Step 5: Compute the exact solution
    logger.debug("[OneBit] Computing exact solution via SVD of absolute value matrix")

    try:
        if W_abs.shape[0] * W_abs.shape[1] < 1e6:
            U, S, Vh = torch.linalg.svd(W_abs, full_matrices=False)
            sigma_max = S[0]
            u_max = U[:, 0].abs()
            v_max = Vh[0, :].abs()
        else:
            logger.debug("[OneBit] Using power method for large matrix")
            u_max, sigma_max, v_max = power_iteration(W_abs, num_iters=max(iters, 30))
            u_max = u_max.abs()
            v_max = v_max.abs()

        # Scale the exact solution
        scale = torch.sqrt(sigma_max + 1e-12)
        a = u_max * scale
        b = v_max * scale

        # Gauge normalization
        a_norm = torch.norm(a) + 1e-12
        b_norm = torch.norm(b) + 1e-12
        balance = torch.sqrt(b_norm / a_norm)
        a = a * balance
        b = b / balance

        logger.debug(f"[OneBit] SVD solution: \
                σ_max={sigma_max:.4e}, ||a||={torch.norm(a):.4e}, ||b||={torch.norm(b):.4e}")

    except Exception as e:
        logger.debug(f"[OneBit] SVD failed: {e}, falling back to power method")
        a, b, _ = _power_rank1_abs_scaled(W_abs, nit=iters)

    # Step 6: Inverse importance scaling
    if use_importance_scaling and not use_balancing:
        a = a / norm_o
        b = b / norm_i

    # Step 7: Reconstruction
    W_reconstructed = a.unsqueeze(1) * W_sign * b.unsqueeze(0)

    # Step 8: Inverse weight balancing
    if use_balancing and balance_Dr is not None and balance_Dc is not None:
        logger.debug("[OneBit] Applying inverse weight balancing transformation")
        # Correct inverse transform: W_orig = Dr^{-1} * W_balanced * Dc^{-1}
        # Since W_balanced = Dr * W * Dc, W = W_balanced / (Dr * Dc)
        W_reconstructed = W_reconstructed / (balance_Dr[:, None] * balance_Dc[None, :])
        del balance_Dr, balance_Dc

    # Step 9: Update weights
    if isinstance(layer, transformers.Conv1D):
        W_reconstructed = W_reconstructed.t()

    dequantized_weight = (
        W_reconstructed.reshape(layer.weight.shape).to(layer.weight.data.dtype).cpu()
    )

    # Save decomposition results
    weight_a = a.cpu()
    weight_b = b.cpu()
    weight_sign = W_sign.cpu()

    # Check error
    err = (W_reconstructed - W_original).square().sum().item()
    logger.debug(f"[OneBit] Final Reconstruction error: {err:.4e}")

    # Free memory
    if use_balancing:
        del W_original
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Numerical stability check
    if err > 1e4:
        logger.debug(f"[OneBit] WARNING: High reconstruction error detected: {err:.4e}")

    if torch.isnan(W_reconstructed).any() or torch.isinf(W_reconstructed).any():
        logger.debug("[OneBit] ERROR: NaN or Inf detected in quantized weights!")
        return False

    if not use_balancing:
        del W
    gc.collect()
    torch.cuda.empty_cache()
    del W_reconstructed, a, b, W_sign

    weight_results = {
        "dequantized_weight": dequantized_weight,
        "a": weight_a,
        "b": weight_b,
        "sign": weight_sign,
    }

    return weight_results


def power_iteration(
    A: torch.Tensor, num_iters: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Power iteration for approximate SVD."""
    n = A.shape[1]
    v = torch.randn(n, device=A.device)
    v = v / torch.norm(v)

    for itt in range(num_iters):
        # Show progress every 10 steps (OneBit typically uses a small number of iterations)
        if itt % 10 == 0 or itt == num_iters - 1:
            current_error = torch.norm(
                A - torch.outer(u if "u" in locals() else torch.mv(A, v), v),
                "fro",
            ).item()
            logger.debug(f"[OneBit Power Iteration {itt} STEP] ||A-ûv̂||_F = {current_error:.4e}")

        u = torch.mv(A, v)
        u_norm = torch.norm(u)
        if u_norm == 0:
            break
        u = u / u_norm

        v = torch.mv(A.t(), u)
        v_norm = torch.norm(v)
        if v_norm == 0:
            break
        v = v / v_norm

    sigma = torch.norm(torch.mv(A, v))
    u = torch.mv(A, v) / sigma if sigma > 0 else torch.zeros_like(torch.mv(A, v))
    return u, sigma, v


def _power_rank1_abs_scaled(
    Z: torch.Tensor, nit: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank-1 approximation of |Z| with scale-separated power method."""
    A = Z.abs()
    _, k = A.shape
    m = torch.randn(k, device=A.device, dtype=A.dtype).abs()
    m = m / (torch.norm(m) + 1e-12)

    for _ in range(nit):
        a = A @ m
        a = a / (torch.norm(a) + 1e-12)
        m = A.t() @ a
        m = m / (torch.norm(m) + 1e-12)

    sigma = float(a.t() @ (A @ m))
    sigma = max(sigma, 1e-12)
    a = a * torch.sqrt(torch.tensor(sigma, device=A.device, dtype=A.dtype))
    m = m * torch.sqrt(torch.tensor(sigma, device=A.device, dtype=A.dtype))

    return a, m, sigma
