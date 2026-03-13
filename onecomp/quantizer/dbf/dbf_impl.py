"""DBF (Double Binary Factorization) quantization module.

DBF : W ≈ A * diag(d) * B

Functions:
    run_dbf( ... ): Execute DBF quantization on a layer.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import logging
import torch
import torch.nn as nn

from .dbf_original import clear_dbf_meta, run_dbf_original

logger = logging.getLogger(__name__)


def _get_dbf_meta_in_op_space(weight_results):
    """
    Extract DBF metadata from quantization results and normalize for op space.

    Handle dbf_original behavior:
    - Conv1D: dbf_original computes A,B on W.t() and swaps A,B on save.
    - Otherwise: return A,B as-is.

    Returns:
        A0, B0, d0: DBF metadata adjusted for op space.
    """
    A0 = weight_results.get("dbf_A", None)
    B0 = weight_results.get("dbf_B", None)
    d0 = weight_results.get("dbf_mid", None)

    # Guard for warmup failure.
    if A0 is None or B0 is None or d0 is None:
        raise ValueError("DBF metadata not found in layer. Warmup may have failed.")

    # Clone while preserving dtype.
    A0 = A0.clone()
    B0 = B0.clone()
    d0 = d0.clone()

    # Return as-is for Conv1D (dbf_original already handles the swap).
    return A0, B0, d0


def power_iteration(A, num_iters=5):
    """Power iteration for SVD approximation"""
    n = A.shape[1]
    v = torch.randn(n, device=A.device)
    v = v / torch.norm(v)

    for _ in range(num_iters):
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
    u = torch.mv(A, v) / (sigma + 1e-12)
    return u, sigma, v


def svd_abs2(W):
    """SVD decomposition for absolute value with sign matrix"""
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    return u * s, Sg, v


# ============================================================
# Main: run_dbf
# ============================================================


def run_dbf(  # pylint: disable=too-many-positional-arguments
    hessian: torch.Tensor,
    layer: torch.nn.Module,
    target_bits: float = 1.5,
    iters: int = 600,
    reg: float = 3e-2,
    # percdamp: float = 0.01,    # Kept for backward compatibility (unused)
    use_balancing: bool = True,  # Used only in DBF-only path
    balance_iters: int = 40,
    balance_alpha: float = 1.0,
    balance_mode: str = "l1",
    use_adaptive_rho: bool = True,
) -> dict:
    """Run the integrated DBF pipeline.

    Args:
        hessian (torch.Tensor): Hessian matrix.
        layer (torch.nn.Module): Target layer.
        target_bits (float): Target bit-width.
        iters (int): DBF iterations.
        reg (float): Regularization parameter.
        use_balancing (bool): Whether to use balancing (DBF-only mode).
        balance_iters (int): Balancing iterations.
        balance_alpha (float): Balancing alpha.
        balance_mode (str): Balancing mode ("l1" or "l2").
        use_adaptive_rho (bool): Whether to adapt ADMM rho.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing quantization results with the following keys:
            - "dbf_Da": Input scaling matrix.
            - "dbf_A": Binary weight matrix A.
            - "dbf_mid": Scale matrix (diagonal).
            - "dbf_B": Binary weight matrix B.
            - "dbf_Db": Output scaling matrix.
            - "is_dbf_quantized": Whether DBF quantization was applied.
    """

    # Enable TF32 for Ampere+ GPUs.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    # === DBF-only path ===
    # Call the original DBF implementation directly.
    results_3_stage = run_dbf_original(
        hessian=hessian,
        layer=layer,
        target_bits=target_bits,
        iters=iters,
        reg=reg,
        use_balancing=use_balancing,
        balance_iters=balance_iters,
        balance_alpha=balance_alpha,
        balance_mode=balance_mode,
        use_adaptive_rho=use_adaptive_rho,
    )

    dbf_A, dbf_B, dbf_mid = _get_dbf_meta_in_op_space(results_3_stage)

    # DBF factorization: W ≈ A × diag(mid) × B
    # A: mid_dim × in_dim, B: out_dim × mid_dim

    # SVD on A (input side).
    u_A, binary_A, v_A = svd_abs2(dbf_A.float())
    # SVD on B (output side).
    u_B, binary_B, v_B = svd_abs2(dbf_B.float())

    # Middle scaling
    # If we binarize W = A × diag(mid) × B:
    # W = diag(u_A) @ binary_A @ diag(v_A * mid * u_B) @
    #     binary_B @ diag(v_B)
    # So the middle scaling is v_A * mid * u_B.
    if isinstance(dbf_mid, torch.Tensor) and dbf_mid.numel() > 1:
        # mid is a vector (typical case).
        scaling2 = v_A * dbf_mid * u_B
    else:
        # mid is a scalar.
        mid_val = dbf_mid.item() if isinstance(dbf_mid, torch.Tensor) else dbf_mid
        scaling2 = v_A * mid_val * u_B

    # Save as five-stage representation.
    results_5_stage = {
        "dequantized_weight": results_3_stage["dequantized_weight"],
        # Stage 0: Input scaling (right singular vector of A).
        "dbf_Da": nn.Parameter(u_A.to(dtype=torch.float16, device="cpu"), requires_grad=False),
        # Stage 1: Binary A matrix.
        "dbf_A": binary_A.to(dtype=torch.float16, device="cpu"),
        # Stage 2: Middle scaling.
        "dbf_mid": nn.Parameter(
            scaling2.to(dtype=torch.float16, device="cpu"), requires_grad=False
        ),
        # Stage 3: Binary B matrix.
        "dbf_B": binary_B.to(dtype=torch.float16, device="cpu"),
        # Stage 4: Output scaling (left singular vector of B).
        "dbf_Db": nn.Parameter(v_B.to(dtype=torch.float16, device="cpu"), requires_grad=False),
        "is_dbf_quantized": True,
    }

    # Temporary: keep metadata for PPL/Acc evaluation.
    if False:
        clear_dbf_meta(results_3_stage)

    return results_5_stage
