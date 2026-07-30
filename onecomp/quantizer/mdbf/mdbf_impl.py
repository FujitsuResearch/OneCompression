"""
MDBF (Multi-Envelope Double Binary Factorization) OneComp wrapper

Convert QEP-DEV's run_MDBF() to match OneComp's calling convention.

Changes:

- Instead of using a helper object, directly receive the Hessian tensor and input tensor
- Remove weight overwriting (side effects) and return the result as a dict

The core algorithm (initialize / admm / gradient_refine) remains unchanged.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from logging import getLogger
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import transformers

logger = getLogger(__name__)

from .initialize import MDBFParams, initialize_MDBF
from .utils import (
    DEFAULT_L,
    DEFAULT_P,
    DEFAULT_SCALE_BITS,
    bpw_from_rank,
    cleanup_gpu_memory,
    rank_from_bpw,
)


def _move_MDBF_params_to_cpu(params_list: List[MDBFParams]) -> List[MDBFParams]:
    """Move MDBFParams to CPU"""
    return [
        MDBFParams(
            A_sign=p.A_sign.cpu(),
            B_sign=p.B_sign.cpu(),
            A_amp=p.A_amp.cpu(),
            B_amp=p.B_amp.cpu(),
            Q_U_amp=p.Q_U_amp.cpu(),
            Q_V_amp=p.Q_V_amp.cpu(),
        )
        for p in params_list
    ]


def run_mdbf(
    hessian: Optional[torch.Tensor],
    module: nn.Module,
    input=None,
    target_bits: float = 1.0,
    l: int = DEFAULT_L,
    P: int = DEFAULT_P,
    svd_mode: Literal["svd", "svd_llm"] = "svd",
    use_admm: bool = False,
    admm_iters: int = 260,
    admm_inner_iters: int = 3,
    admm_reg: float = 0.03,
    use_gradient_refine: bool = False,
    gradient_iters: int = 1000,
    gradient_lr: float = 0.01,
    activation_aware: bool = False,
    act_init: Literal["none", "osvd", "svd_llm"] = "osvd",
    nsamples: Optional[int] = None,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> dict:
    """
    Run MDBF quantization (OneComp convention)

    Phase 1: Initialization (SVD decomposition + binarization + Multi-scale amplitude decomposition)
    Phase 2: ADMM optimization (optional)
    Phase 3: Gradient-based amplitude optimization (optional)

    Args:
        hessian: Precomputed Hessian matrix (passed from the base class)
        module: Layer to be quantized
        input: Input activations during calibration (passed from the base class)
        target_bits: Target BPW
        l: Multi-scale rank
        P: Number of paths
        svd_mode: SVD mode ("svd" or "svd_llm")
        use_admm: Use ADMM optimization
        admm_iters: Number of ADMM outer iterations
        admm_inner_iters: Number of ADMM inner iterations
        admm_reg: Regularization parameter
        use_gradient_refine: Use gradient-based amplitude optimization
        gradient_iters: Number of gradient optimization iterations
        gradient_lr: Gradient optimization learning rate
        activation_aware: Activation-aware mode (P=1 only)
        act_init: Initialization mode
        nsamples: Number of tokens used for Hessian calculation. If None, fallback to 1.
        scale_bits: Bit-width used to account for the FP16 amplitude scales when
            sizing the rank and reporting BPW (accounting only; does not change
            the stored dtype). Defaults to DEFAULT_SCALE_BITS (16, FP16).

    Returns:
        dict with keys:
            "mdbf_params": List[MDBFParams]  — Quantization parameters for P paths (CPU)
            "W_recon": torch.Tensor           — Quantized weights (FP16, CPU)
            "actual_bpw": float               — Actual achieved BPW
            "r": int                          — Rank used
            "is_mdbf_quantized": bool
            "actual_activation_aware": bool   — Whether activation-aware mode was actually used
    """
    is_conv1d = isinstance(module, transformers.Conv1D)

    W = module.weight.data.clone()
    if is_conv1d:
        W = W.t()

    n, m = W.shape
    device = W.device
    dtype = W.dtype

    # Hessian for SVD-LLM
    H_svd = None
    if svd_mode == "svd_llm":
        H_svd = hessian.clone().to(device) if hessian is not None else None
        if H_svd is not None:
            logger.debug(f"[MDBF] Using Hessian for SVD-LLM (shape: {H_svd.shape})")
        else:
            logger.debug("[MDBF] No Hessian, falling back to SVD")
            svd_mode = "svd"

    # Activation-aware settings
    use_hessian_mode = False
    _nsamples = nsamples if nsamples is not None else 1
    H_act = None

    if activation_aware:
        if P != 1:
            logger.warning("[MDBF] activation_aware=True but P!=1; fallback to non-aware.")
            activation_aware = False
        else:
            H_act = hessian.clone().to(device) if hessian is not None else None
            if H_act is not None:
                use_hessian_mode = True
                logger.debug(f"[MDBF] Activation-aware mode (Hessian-based): nsamples={_nsamples}")
            else:
                logger.warning(
                    "[MDBF] activation_aware=True but H not found; fallback to non-aware."
                )
                activation_aware = False

    # Prepare act_X (not used in Hessian-based mode)
    act_X = None
    if activation_aware and use_hessian_mode:
        act_X = None  # act_X is not used
    elif input is not None:
        inp = input[0] if isinstance(input, (list, tuple)) else input
        act_X = inp.reshape(-1, m).to(device=device, dtype=torch.float32)

    # Rank calculation
    r = rank_from_bpw(n, m, target_bits, l, P, scale_bits=scale_bits)
    actual_bpw = bpw_from_rank(n, m, r, l, P, scale_bits=scale_bits)

    logger.debug(f"[MDBF] n={n}, m={m}, target_bpw={target_bits:.2f}, actual_bpw={actual_bpw:.2f}")
    logger.debug(f"[MDBF] r={r}, l={l}, P={P}, mode={svd_mode}, use_admm={use_admm}")

    # Phase 1: Initialization
    init_act_init = act_init if (activation_aware and use_hessian_mode) else "none"
    if init_act_init == "osvd" and H_act is not None:
        init_H = H_act
    elif svd_mode == "svd_llm" and H_svd is not None:
        init_H = H_svd
    else:
        init_H = None

    all_params, W_recon = initialize_MDBF(W, r, l, P, init_H, svd_mode, act_init=init_act_init)

    if H_svd is not None:
        del H_svd
    cleanup_gpu_memory()

    # Phase 2: ADMM optimization
    if use_admm and admm_iters > 0:
        if activation_aware and use_hessian_mode and H_act is not None:
            from .admm import optimize_MDBF_admm_hessian

            logger.debug(
                f"[MDBF] ADMM (Activation-Aware, Hessian-based): "
                f"outer={admm_iters}, inner={admm_inner_iters}, reg={admm_reg}"
            )

            all_params, W_recon = optimize_MDBF_admm_hessian(
                W_original=W,
                params_list=all_params,
                l=l,
                H=H_act,
                nsamples=_nsamples,
                iters=admm_iters,
                inner_iters=admm_inner_iters,
                reg=admm_reg,
            )
        else:
            from .admm import optimize_MDBF_admm

            H_for_display = hessian.clone().to(device) if hessian is not None else None

            logger.debug(
                f"[MDBF] ADMM: outer={admm_iters}, inner={admm_inner_iters}, " f"reg={admm_reg}"
            )

            all_params, W_recon = optimize_MDBF_admm(
                W_original=W,
                params_list=all_params,
                l=l,
                iters=admm_iters,
                inner_iters=admm_inner_iters,
                reg=admm_reg,
                H=H_for_display,
                nsamples=_nsamples,
            )

            if H_for_display is not None:
                del H_for_display

        actual_r = all_params[0].A_sign.shape[1]
        actual_P = len(all_params)
        actual_bpw = bpw_from_rank(n, m, actual_r, l, actual_P, scale_bits=scale_bits)
        r = actual_r
        P = actual_P

        logger.debug(f"[MDBF] After ADMM: r={r}, P={P}, actual_bpw={actual_bpw:.3f}")

    # Phase 3: Gradient-based amplitude optimization
    if use_gradient_refine and gradient_iters > 0:
        from .gradient_refine import refine_amplitude_gradient

        grad_activation_aware = activation_aware and P == 1
        grad_H = hessian.clone().to(device) if hessian is not None else None

        if grad_activation_aware and grad_H is None:
            grad_activation_aware = False

        mode_str = " (ActAware)" if grad_activation_aware else ""
        logger.debug(f"[MDBF] Gradient Refine{mode_str}: iters={gradient_iters}, lr={gradient_lr}")

        all_params, W_recon = refine_amplitude_gradient(
            W_original=W,
            params_list=all_params,
            l=l,
            lr=gradient_lr,
            iters=gradient_iters,
            activation_aware=grad_activation_aware,
            H=grad_H,
            nsamples=_nsamples,
        )

        if grad_H is not None:
            del grad_H

        logger.debug(f"[MDBF] After Gradient Refine: r={r}, P={P}")

    del W
    cleanup_gpu_memory()

    # Move results to CPU and return
    all_params_cpu = _move_MDBF_params_to_cpu(all_params)
    del all_params

    W_recon_cpu = W_recon.to(torch.float16).cpu()
    del W_recon

    if act_X is not None:
        del act_X
    if H_act is not None:
        del H_act

    cleanup_gpu_memory()

    return {
        "mdbf_params": all_params_cpu,
        "W_recon": W_recon_cpu,
        "actual_bpw": actual_bpw,
        "r": r,
        "is_mdbf_quantized": True,
        # Return the actually-used activation_aware flag under a distinct key
        "actual_activation_aware": activation_aware,
    }
