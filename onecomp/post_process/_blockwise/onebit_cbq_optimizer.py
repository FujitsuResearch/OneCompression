"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

"""

"""Cross-Block Quantisation (CBQ) for OneBit.

Jointly optimises scaling vectors (a, b) and sign matrices of two adjacent
OneBit-quantised blocks (window K=2).  Uses state_dict snapshots for rollback.

Follows the same structure as dbf_cbq_optimizer.py / gptq_cbq_optimizer.py:
  - Two-layer forward window
  - SmoothSign STE for sign optimisation
  - Periodic hard evaluation with best-state tracking
  - Final rollback safety net against initial error
"""

import copy
from logging import getLogger
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import layer_kwargs_to_device
from .onebit_block_optimizer import (
    _smooth_sign_ste,
    _find_onebit_modules,
    _layer_output,
)

logger = getLogger(__name__)


def optimize_onebit_cross_block(
    layer_i: nn.Module,
    layer_j: nn.Module,
    inps,
    target_outputs_j,
    layer_kwargs: dict,
    lr: float = 1e-4,
    epochs: int = 10,
    dev: torch.device = None,
    optimize_sign: bool = True,
    k_smooth: float = 100.0,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """Jointly optimise two adjacent OneBit blocks. Returns (initial_mse, final_mse)."""
    if dev is None:
        dev = next(layer_i.parameters()).device

    kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)
    n = len(inps)

    def _forward_window(inp_gpu):
        mid = _layer_output(layer_i, inp_gpu, kw_gpu)
        if mid.dim() == 2:
            mid = mid.unsqueeze(0)
        return _layer_output(layer_j, mid, kw_gpu)

    def _eval_mse():
        layer_i.eval()
        layer_j.eval()
        # Match input dtype to the layers' current dtype — see the
        # corresponding comment in dbf_cbq_optimizer._eval_mse for the full
        # rationale (nn.LayerNorm dtype mismatch on GPT-NeoX / OPT).
        _dtype = next(layer_i.parameters()).dtype
        with torch.no_grad():
            total = 0.0
            for j in range(n):
                inp_gpu = inps[j].unsqueeze(0).to(dev, dtype=_dtype)
                out = _forward_window(inp_gpu)
                tgt = target_outputs_j[j].to(dev, dtype=_dtype)
                total += F.mse_loss(out.float(), tgt.float()).item()
                del inp_gpu, out, tgt
        return total / max(n, 1)

    initial_error = _eval_mse()
    logger.info("[CBQ-OneBit] Initial window MSE: %.6f", initial_error)

    all_onebit = _find_onebit_modules(layer_i) + _find_onebit_modules(layer_j)
    if not all_onebit:
        del kw_gpu
        return initial_error, initial_error

    # --- Snapshot for rollback ---
    snap_i = copy.deepcopy(layer_i.state_dict())
    snap_j = copy.deepcopy(layer_j.state_dict())

    # Promote to float32 — see onebit_block_optimizer.py for rationale.
    original_dtype = next(layer_i.parameters()).dtype
    layer_i.float()
    layer_j.float()

    # --- Collect params ---
    scaling_params = []
    sign_params = []

    for name, mod in all_onebit:
        mod.a.requires_grad_(True)
        scaling_params.append(mod.a)
        mod.b.requires_grad_(True)
        scaling_params.append(mod.b)

        if optimize_sign and mod.sign_matrix is not None:
            sign_weight = mod.sign_matrix.float().detach().clone().to(dev)
            sign_weight.requires_grad_(True)
            sign_params.append((name, mod, sign_weight))

    all_params = scaling_params + [sp[2] for sp in sign_params]
    if not all_params:
        layer_i.to(original_dtype)
        layer_j.to(original_dtype)
        del kw_gpu
        return initial_error, initial_error

    optimizer = torch.optim.Adam(all_params, lr=lr)
    logger.info(
        "[CBQ-OneBit] %d OneBit modules, %d scaling, %d sign",
        len(all_onebit),
        len(scaling_params),
        len(sign_params),
    )

    original_sign_mats = {}
    if optimize_sign:
        for name, mod, _sw in sign_params:
            original_sign_mats[name] = mod.sign_matrix.clone()

    from ...quantizer.onebit.onebit_layer import my_pack

    best_eval_mse = initial_error
    best_snap_i = {}
    best_snap_j = {}
    eval_interval = max(1, epochs // 4)

    def _hard_eval_window():
        """Hard-quantize sign matrices, evaluate window MSE (leaves hard state)."""
        if optimize_sign:
            with torch.no_grad():
                for _name, mod, sign_weight in sign_params:
                    sq = sign_weight.data.sign()
                    sq[sq == 0] = 1
                    mod.sign_matrix = sq.to(torch.int8)
                    mod.sign_packed = my_pack(sq)
        return _eval_mse()

    # --- Training ---
    for epoch in range(epochs):
        layer_i.train()
        layer_j.train()
        total_loss = 0.0
        for j in range(n):
            optimizer.zero_grad()

            saved_mats = []
            if optimize_sign:
                for _mod_name, mod, sign_weight in sign_params:
                    saved_mats.append(mod.sign_matrix)
                    mod.sign_matrix = _smooth_sign_ste(sign_weight, k=k_smooth)

            inp_gpu = inps[j].unsqueeze(0).detach().to(dev).float()
            tgt = target_outputs_j[j].detach().to(dev).float()
            out = _forward_window(inp_gpu)
            loss = F.mse_loss(out, tgt)
            loss.backward()

            if optimize_sign:
                for idx, (_mod_name, mod, _sw) in enumerate(sign_params):
                    mod.sign_matrix = saved_mats[idx]

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
            optimizer.step()
            total_loss += loss.item()
            del inp_gpu, tgt, out, loss

        avg_soft = total_loss / max(n, 1)

        # --- Periodic hard evaluation ---
        do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
        if do_eval:
            eval_mse = _hard_eval_window()
            if eval_mse < best_eval_mse:
                best_eval_mse = eval_mse
                best_snap_i = copy.deepcopy(layer_i.state_dict())
                best_snap_j = copy.deepcopy(layer_j.state_dict())
            if optimize_sign:
                for name, mod, _sw in sign_params:
                    mod.sign_matrix = original_sign_mats[name]
            logger.info(
                "  [CBQ-OneBit] Epoch %d/%d: train=%.6f, eval=%.6f (best: %.6f)",
                epoch + 1,
                epochs,
                avg_soft,
                eval_mse,
                best_eval_mse,
            )
        elif (epoch + 1) % max(1, epochs // 4) == 0:
            logger.info(
                "  [CBQ-OneBit] Epoch %d/%d: train=%.6f",
                epoch + 1,
                epochs,
                avg_soft,
            )

    # --- Restore best state ---
    if best_snap_i:
        layer_i.load_state_dict(best_snap_i)
        layer_j.load_state_dict(best_snap_j)
    elif optimize_sign:
        with torch.no_grad():
            for _name, mod, sign_weight in sign_params:
                sq = sign_weight.data.sign()
                sq[sq == 0] = 1
                mod.sign_matrix = sq.to(torch.int8)
                mod.sign_packed = my_pack(sq)

    # --- Disable gradients ---
    for _name, mod in all_onebit:
        mod.a.requires_grad_(False)
        mod.b.requires_grad_(False)

    # Restore original dtype for final evaluation and state serialisation
    layer_i.to(original_dtype)
    layer_j.to(original_dtype)

    final_error = _eval_mse()

    if final_error >= initial_error:
        logger.info(
            "[CBQ-OneBit] No improvement (%.6f >= %.6f), reverting",
            final_error,
            initial_error,
        )
        layer_i.load_state_dict(snap_i)
        layer_j.load_state_dict(snap_j)
        final_error = initial_error

    delta = initial_error - final_error
    pct = (delta / max(initial_error, 1e-10)) * 100
    logger.info(
        "[CBQ-OneBit] Final window MSE: %.6f (delta: %.6f, %+.1f%%)",
        final_error,
        delta,
        pct,
    )

    del kw_gpu, snap_i, snap_j
    return initial_error, final_error
