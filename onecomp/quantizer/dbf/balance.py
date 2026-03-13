"""Weight balancing utilities for DBF quantization.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from typing import Any, Dict, Literal, Tuple

import logging

logger = logging.getLogger(__name__)
import torch


def balance_track(
    W: torch.Tensor,
    its: int = 30,
    alpha: float = 1.0,
    beta: float = None,
    eps: float = 1e-12,
    mode: Literal["l2", "l1"] = "l1",  # Default to l1
    gauge_fix: bool = True,
    tol: float = None,
    stop_window: int = 3,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Weight matrix balancing (L1/L2 diagonal scaling) to transform W into:
        Wb = Dr * W * Dc

    Parameters
    ----------
    W : (n, m) torch.Tensor
        Input matrix. Assumed real-valued. No gradient required (with no_grad).
    its : int
        Maximum number of iterations.
    alpha : float
        Target row norm value.
        - For mode='l2': target value for row ℓ2 norm squared (backward compatible)
        - For mode='l1': target value for row ℓ1 norm
    beta : float or None
        Target column norm value. If None, automatically determined from n*alpha = m*beta.
        (Same consistency relation regardless of mode)
    eps : float
        Small value for numerical stability (e.g., avoiding division by zero).
    mode : {'l2','l1'}
        'l2' for ℓ2-balancing (convex Sinkhorn-ℓ2), 'l1' for ℓ1-balancing (Sinkhorn).
        Default is 'l1'.
    gauge_fix : bool
        Apply gauge invariance correction (Dr, Dc) -> (c*Dr, (1/c)*Dc) at each iteration to prevent divergence.
    tol : float or None
        Convergence tolerance. Early stopping if KKT residuals for both rows and columns
        remain below tol for stop_window consecutive iterations.
    stop_window : int
        Number of consecutive iterations for early stopping.

    Returns
    -------
    Wb : (n, m) torch.Tensor
        Balanced matrix Wb = Dr * W * Dc
    history : dict
        Dictionary with keys:
        - "Dr" : (n,) Row-side diagonal scaling (final value)
        - "Dc" : (m,) Column-side diagonal scaling (final value)
        - "kkt_r" : (num_iters,) Row-side KKT residual at each iteration
        - "kkt_c" : (num_iters,) Column-side KKT residual at each iteration
        - "kkt_row" : Alias for "kkt_r" (backward compatibility)
        - "kkt_col" : Alias for "kkt_c" (backward compatibility)
        - "obj" : (num_iters,) Objective function value at each iteration

    Balancing Methods
    -----------------
    mode='l2' (L2-balancing):
        Finds adaptive diagonal scalings Dr, Dc such that Wb = Dr W Dc satisfies:
        - Each row's ℓ2 norm squared = alpha
        - Each column's ℓ2 norm squared = beta
        (approximately). Mathematically, solves the minimization problem:
          minimize_{Dr,Dc > 0}  ||Dr W Dc||_F^2
          subject to:
            ||Dr_i W||_2^2 = alpha  for all i (rows)
            ||W Dc_j||_2^2 = beta   for all j (cols)
        via alternating updates. Convex problem with closed-form Dr, Dc updates.

    mode='l1' (L1-balancing, classical Sinkhorn):
        Finds adaptive diagonal scalings Dr, Dc such that Wb = Dr W Dc satisfies:
        - Each row's ℓ1 norm = alpha
        - Each column's ℓ1 norm = beta
        (approximately). This is a generalization of doubly stochastic matrix scaling.
        Solved via alternating update iteration scheme (Sinkhorn iteration).

    Notes
    -----
    - When tol is set to None (default), 1e-3 is used.
    - When gauge_fix=True, the scales of Dr, Dc are adjusted to prevent numerical divergence.
      This stabilizes convergence, but the final scales have solution ambiguity.
    """
    n, m = W.shape
    device = W.device
    dtype = W.dtype

    # Initialize diagonal scaling
    Dr_vec = torch.ones(n, device=device, dtype=dtype)
    Dc_vec = torch.ones(m, device=device, dtype=dtype)

    # If beta is not specified, auto-set from sum preservation condition
    if beta is None:
        beta = n * alpha / m

    # Record iteration history
    kkt_r_history = []
    kkt_c_history = []
    obj_history = []

    # For convergence check
    if tol is None:
        tol = 1e-3
    consecutive_below_tol = 0

    # Objective function definition (meaningful only for mode='l2')
    def f_obj(Dr_vec: torch.Tensor, Dc_vec: torch.Tensor) -> float:
        """Objective: ||Dr W Dc||_F^2 for mode='l2'."""
        if mode == "l2":
            Wb = Dr_vec[:, None] * W * Dc_vec[None, :]
            return torch.sum(Wb**2).item()
        return 0.0

    # KKT residual and statistics computation
    def compute_stats(Dr_vec: torch.Tensor, Dc_vec: torch.Tensor) -> Tuple[float, float]:
        """Compute KKT residuals and statistics."""
        Wb = Dr_vec[:, None] * W * Dc_vec[None, :]

        if mode == "l2":
            # Row/column ℓ2 norm squared
            row_norms = torch.sum(Wb**2, dim=1)
            col_norms = torch.sum(Wb**2, dim=0)
            # KKT residual: |row_norm^2 - alpha| and |col_norm^2 - beta|
            kkt_r = torch.abs(row_norms - alpha).max().item()
            kkt_c = torch.abs(col_norms - beta).max().item()
        else:  # mode == "l1"
            # Row/column ℓ1 norm
            row_norms = torch.sum(torch.abs(Wb), dim=1)
            col_norms = torch.sum(torch.abs(Wb), dim=0)
            # KKT residual: |row_norm - alpha| and |col_norm - beta|
            kkt_r = torch.abs(row_norms - alpha).max().item()
            kkt_c = torch.abs(col_norms - beta).max().item()

        return kkt_r, kkt_c

    # Balancing iterations
    for _ in range(its):
        if mode == "l2":
            # ==== L2-balancing (convex Sinkhorn-L2) ====
            # Row update: Dr_i ← Dr_i * sqrt(alpha / ||Dr_i W||_2^2)
            Wb = Dr_vec[:, None] * W * Dc_vec[None, :]
            row_norms_sq = torch.sum(Wb**2, dim=1).clamp(min=eps)
            Dr_vec = Dr_vec * torch.sqrt(alpha / row_norms_sq)

            # Column update: Dc_j ← Dc_j * sqrt(beta / ||W Dc_j||_2^2)
            Wb = Dr_vec[:, None] * W * Dc_vec[None, :]
            col_norms_sq = torch.sum(Wb**2, dim=0).clamp(min=eps)
            Dc_vec = Dc_vec * torch.sqrt(beta / col_norms_sq)

        else:  # mode == "l1"
            # ==== L1-balancing (classical Sinkhorn) ====
            # Row update: Dr_i ← Dr_i * (alpha / ||Dr_i W||_1)
            Wb = Dr_vec[:, None] * W * Dc_vec[None, :]
            row_norms = torch.sum(torch.abs(Wb), dim=1).clamp(min=eps)
            Dr_vec = Dr_vec * (alpha / row_norms)

            # Column update: Dc_j ← Dc_j * (beta / ||W Dc_j||_1)
            Wb = Dr_vec[:, None] * W * Dc_vec[None, :]
            col_norms = torch.sum(torch.abs(Wb), dim=0).clamp(min=eps)
            Dc_vec = Dc_vec * (beta / col_norms)

        # Gauge correction (prevent scale divergence)
        if gauge_fix:
            # Equalize geometric means of Dr, Dc
            gauge = torch.sqrt(Dr_vec.mean() / Dc_vec.mean())
            Dr_vec = Dr_vec / gauge
            Dc_vec = Dc_vec * gauge

        # Compute and record statistics
        kkt_r, kkt_c = compute_stats(Dr_vec, Dc_vec)
        obj_val = f_obj(Dr_vec, Dc_vec)

        kkt_r_history.append(kkt_r)
        kkt_c_history.append(kkt_c)
        obj_history.append(obj_val)

        # Convergence check
        if kkt_r < tol and kkt_c < tol:
            consecutive_below_tol += 1
            if consecutive_below_tol >= stop_window:
                # Early stopping
                break
        else:
            consecutive_below_tol = 0

        # For debugging (as needed)
        # if it % 10 == 0:
        #     logger.debug(f"[Balance] iter {it}: kkt_r={kkt_r:.2e}, kkt_c={kkt_c:.2e}")

    # Final balanced matrix
    Wb = Dr_vec[:, None] * W * Dc_vec[None, :]

    # Debug information (comment out as needed)
    # Convergence diagnostics (optional)
    # def spread(v: torch.Tensor, mask: torch.Tensor) -> float:
    #     """Compute spread (max/min ratio) of positive values."""
    #     v_pos = v[mask]
    #     if len(v_pos) == 0:
    #         return 1.0
    #     return (v_pos.max() / v_pos.min()).item()

    # abs_W = torch.abs(W)
    # row_mask = torch.sum(abs_W, dim=1) > eps
    # col_mask = torch.sum(abs_W, dim=0) > eps
    # logger.debug(f"[Balance] Final Dr spread: {spread(Dr_vec, row_mask):.2f}")
    # logger.debug(f"[Balance] Final Dc spread: {spread(Dc_vec, col_mask):.2f}")

    # Create history dictionary
    history = {
        "Dr": Dr_vec,
        "Dc": Dc_vec,
        "kkt_r": torch.tensor(kkt_r_history),
        "kkt_c": torch.tensor(kkt_c_history),
        "kkt_row": torch.tensor(kkt_r_history),  # backward compatibility
        "kkt_col": torch.tensor(kkt_c_history),  # backward compatibility
        "obj": torch.tensor(obj_history),
    }

    return Wb, history
