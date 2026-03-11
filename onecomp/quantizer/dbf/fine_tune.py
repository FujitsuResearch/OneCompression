"""Fine-tuning module for DBF/MDBF quantization

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import gc  # ! Free memory
from typing import Optional, Tuple

import logging
logger = logging.getLogger(__name__)
import torch
from torch import optim

from .middle import _compose_dense_bal, _compose_lowrank_bal


def fine_tune_dense(
    W_bal: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    M: torch.Tensor,
    Da: torch.Tensor,
    Db: torch.Tensor,
    steps: int = 80,
    lr: float = 1e-4,
    eps: float = 1e-6,
    patience: int = 10,
    wd: float = 1e-5,
    clip: float = 0.5,
) -> Tuple:
    """
    Fine-tuning for dense matrix M version (stable optimization with fp32 master).

    Optimizes only d, M, Da, Db while keeping A, B fixed.
    Uses log parameterization and fp32 optimization to prevent scale degeneracy.

    Objective: ||W_bal - Da*A*(diag(d)+M)*B*Db||_F^2

    Args:
        W_bal: Target matrix in balance space.
        A, B: Factor matrices (fixed).
        d: Diagonal elements.
        M: Dense matrix.
        Da, Db: Scale factors.
        steps: Number of optimization steps.
        lr: Learning rate.
        eps: Minimum value for Da, Db.
        patience: Patience parameter for early stopping.
        wd: Regularization coefficient.
        clip: Gradient clipping.

    Returns:
        d, M, Da, Db: Optimized parameters.
    """
    # Optimize with fp32 master (save original dtype)
    odtype = A.dtype
    W32 = W_bal.detach().to(torch.float32)
    A32 = A.detach().to(torch.float32)
    B32 = B.detach().to(torch.float32)

    with torch.enable_grad():
        # Initialize parameters in fp32
        d_p = d.detach().to(torch.float32).requires_grad_(True)
        M_p = M.detach().to(torch.float32).requires_grad_(True)
        Da_p = Da.detach().to(torch.float32).requires_grad_(True)
        Db_p = Db.detach().to(torch.float32).requires_grad_(True)

        # Optimization setup
        opt = optim.Adam([d_p, M_p, Da_p, Db_p], lr=lr)

        # Best parameter tracking
        best = (
            d_p.detach().clone(),
            M_p.detach().clone(),
            Da_p.detach().clone(),
            Db_p.detach().clone(),
        )
        best_loss = float("inf")
        bad_count = 0
        initial_loss = None

        for step in range(steps):
            opt.zero_grad(set_to_none=True)

            # Forward computation in fp32
            W_hat = _compose_dense_bal(A32, d_p, M_p, B32, Da_p, Db_p)

            # Simple loss function (light regularization)
            loss = (W32 - W_hat).pow(2).sum()

            # Regularization (prevent scale degeneracy and M explosion)
            # Stronger regularization to prevent parameter divergence
            reg_scale = (Da_p - 1.0).square().mean() + (
                Db_p - 1.0
            ).square().mean()
            reg_param = M_p.square().mean() + d_p.square().mean()  # Also regularize d
            reg = reg_scale + 1e-4 * reg_param  # Stronger regularization
            loss = loss + wd * reg

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([d_p, M_p, Da_p, Db_p], clip)

            opt.step()

            with torch.no_grad():
                # Apply Da, Db constraints
                Da_p.clamp_(min=eps)
                Db_p.clamp_(min=eps)

                cur = loss.sqrt().item()

                # Record initial loss
                if initial_loss is None:
                    initial_loss = cur

                # Update best solution and deterioration count
                if cur < best_loss * 0.999:  # More than 0.1% improvement
                    best_loss = cur
                    best = (
                        d_p.detach().clone(),
                        M_p.detach().clone(),
                        Da_p.detach().clone(),
                        Db_p.detach().clone(),
                    )
                    bad_count = 0
                else:
                    bad_count += 1

                # Early stopping (stop after patience deteriorations once 50% improvement from initial loss)
                if cur < initial_loss * 0.5 and bad_count >= patience:
                    logger.debug(
                        f"    Early stopping at step {step} (no improvement for {patience} steps)"
                    )
                    break

                if step == 0 or step == steps - 1 or (step + 1) % 50 == 0:
                    # Display error in balance space (consistent with optimization)
                    logger.debug(f"    Fine-tune step {step:3d}: loss = {cur:.4e}")

        # Use best parameters
        d_p, M_p, Da_p, Db_p = best

    #! Free memory
    del W_bal, A, B, d, M, Da, Db
    del W32, A32, B32, opt, best, W_hat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Return in original dtype
    return (d_p.to(odtype), M_p.to(odtype), Da_p.to(odtype), Db_p.to(odtype))


def fine_tune_lowrank(
    W_bal: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    d: torch.Tensor,
    U: Optional[torch.Tensor],
    V: Optional[torch.Tensor],
    Da: torch.Tensor,
    Db: torch.Tensor,
    steps: int = 80,
    lr: float = 1e-4,
    eps: float = 1e-6,
    patience: int = 10,
    wd: float = 1e-5,
    clip: float = 0.5,
) -> Tuple:
    """
    Fine-tuning for low-rank UV^T version (stable optimization with fp32 master).

    Optimizes only d, U, V, Da, Db while keeping A, B fixed.
    Uses log parameterization and fp32 optimization to prevent scale degeneracy.

    Objective: ||W_bal - Da*A*(diag(d)+U*V^T)*B*Db||_F^2

    Args:
        W_bal: Target matrix in balance space.
        A, B: Factor matrices (fixed).
        d: Diagonal elements.
        U, V: Low-rank decomposition.
        Da, Db: Scale factors.
        steps: Number of optimization steps.
        lr: Learning rate.
        eps: Minimum value for Da, Db.
        patience: Patience parameter for early stopping.
        wd: Regularization coefficient.
        clip: Gradient clipping.

    Returns:
        d, U, V, Da, Db: Optimized parameters.
    """
    # Skip when rank=0
    if U is None or U.numel() == 0 or V is None or V.numel() == 0:
        logger.debug("    Skipping fine-tune (rank=0 or empty U/V)")
        return d, U, V, Da, Db

    # Optimize with fp32 master (save original dtype)
    odtype = A.dtype
    W32 = W_bal.detach().to(torch.float32)
    A32 = A.detach().to(torch.float32)
    B32 = B.detach().to(torch.float32)

    with torch.enable_grad():
        # Initialize parameters in fp32
        d_p = d.detach().to(torch.float32).requires_grad_(True)
        U_p = U.detach().to(torch.float32).requires_grad_(True)
        V_p = V.detach().to(torch.float32).requires_grad_(True)
        Da_p = Da.detach().to(torch.float32).requires_grad_(True)
        Db_p = Db.detach().to(torch.float32).requires_grad_(True)

        # Optimization setup
        opt = optim.Adam([d_p, U_p, V_p, Da_p, Db_p], lr=lr)

        # Best parameter tracking
        best = (
            d_p.detach().clone(),
            U_p.detach().clone(),
            V_p.detach().clone(),
            Da_p.detach().clone(),
            Db_p.detach().clone(),
        )
        best_loss = float("inf")
        bad_count = 0
        initial_loss = None

        for step in range(steps):
            opt.zero_grad(set_to_none=True)

            # Forward computation in fp32
            W_hat = _compose_lowrank_bal(A32, d_p, U_p, V_p, B32, Da_p, Db_p)

            # Simple loss function (light regularization)
            loss = (W32 - W_hat).pow(2).sum()

            # Regularization (prevent scale degeneracy and U/V explosion)
            # Stronger regularization to prevent parameter divergence
            reg_scale = (Da_p - 1.0).square().mean() + (
                Db_p - 1.0
            ).square().mean()
            reg_param = (
                U_p.square().mean() + V_p.square().mean() + d_p.square().mean()
            )  # Also regularize d
            reg = reg_scale + 1e-4 * reg_param  # Stronger regularization
            loss = loss + wd * reg

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([d_p, U_p, V_p, Da_p, Db_p], clip)

            opt.step()

            with torch.no_grad():
                # Apply Da, Db constraints
                Da_p.clamp_(min=eps)
                Db_p.clamp_(min=eps)

                cur = loss.sqrt().item()

                # Record initial loss
                if initial_loss is None:
                    initial_loss = cur

                # Update best solution and deterioration count
                if cur < best_loss * 0.999:  # More than 0.1% improvement
                    best_loss = cur
                    best = (
                        d_p.detach().clone(),
                        U_p.detach().clone(),
                        V_p.detach().clone(),
                        Da_p.detach().clone(),
                        Db_p.detach().clone(),
                    )
                    bad_count = 0
                else:
                    bad_count += 1

                # Early stopping (stop after patience deteriorations once 50% improvement from initial loss)
                if cur < initial_loss * 0.5 and bad_count >= patience:
                    logger.debug(
                        f"    Early stopping at step {step} (no improvement for {patience} steps)"
                    )
                    break

                if step == 0 or step == steps - 1 or (step + 1) % 50 == 0:
                    # Display error in balance space (consistent with optimization)
                    logger.debug(f"    Fine-tune step {step:3d}: loss = {cur:.4e}")

        # Use best parameters
        d_p, U_p, V_p, Da_p, Db_p = best

    #! Free memory
    del W_bal, A, B, d, U, V, Da, Db
    del W32, A32, B32, opt, best, W_hat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Return in original dtype
    return (
        d_p.to(odtype),
        U_p.to(odtype),
        V_p.to(odtype),
        Da_p.to(odtype),
        Db_p.to(odtype),
    )
