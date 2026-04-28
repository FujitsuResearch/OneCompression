"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

"""

"""Cross-Block Quantisation (CBQ) for DBF.

Ported from qep-dev/src/blockwise_quantization/method/blockwise_dbf/
cbq_dbf_optimizer.py.

Jointly optimises scaling vectors and binary matrices of two adjacent
DBF-quantised blocks (window K=2).  Uses state_dict snapshots for rollback.

Same naming adaptations as dbf_block_optimizer.py:
    DoubleBinaryLinear with scaling0/2/4 and binary_multiplication1/3.
    BitLinearPacked.scale is frozen (qep-dev compatible).
"""

import copy
from logging import getLogger
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import layer_kwargs_to_device
from .dbf_block_optimizer import (
    _smooth_sign_ste,
    _find_dbf_modules,
    _layer_output,
    _get_binary_parts,
)

logger = getLogger(__name__)


def optimize_dbf_cross_block(
    layer_i: nn.Module,
    layer_j: nn.Module,
    inps,
    target_outputs_j,
    layer_kwargs: dict,
    lr: float = 1e-4,
    epochs: int = 10,
    dev: torch.device = None,
    optimize_binary: bool = True,
    k_smooth: float = 100.0,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """Jointly optimise two adjacent DBF blocks. Returns (initial_mse, final_mse)."""
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
        # Match input dtype to the layers' current dtype.
        # This function is called both BEFORE float32 promotion (layers in
        # original float16) and AFTER dtype restoration.  Hardcoding .float()
        # would create a dtype mismatch with nn.LayerNorm in architectures
        # like GPT-NeoX and OPT that do not auto-cast internally (unlike
        # Llama's RMSNorm).  MSE is still computed in float32 for precision.
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
    logger.info("[CBQ-DBF] Initial window MSE: %.6f", initial_error)

    all_dbf = _find_dbf_modules(layer_i) + _find_dbf_modules(layer_j)
    if not all_dbf:
        del kw_gpu
        return initial_error, initial_error

    # --- Freeze BitLinearPacked.scale ---
    for _name, mod in all_dbf:
        for attr in ("binary_multiplication1", "binary_multiplication3"):
            blp = getattr(mod, attr, None)
            if blp is not None and hasattr(blp, "scale"):
                blp.scale.requires_grad_(False)

    # --- Snapshot for rollback ---
    snap_i = copy.deepcopy(layer_i.state_dict())
    snap_j = copy.deepcopy(layer_j.state_dict())

    # Promote to float32 — see dbf_block_optimizer.py for rationale.
    original_dtype = next(layer_i.parameters()).dtype
    layer_i.float()
    layer_j.float()

    # --- Collect params ---
    scaling_params = []
    binary_params = []

    for _name, mod in all_dbf:
        for attr in ("scaling0", "scaling2", "scaling4"):
            p = getattr(mod, attr, None)
            if p is not None:
                p.requires_grad_(True)
                scaling_params.append(p)
        if optimize_binary:
            for blp, bit_mat, shape in _get_binary_parts(mod):
                bw = bit_mat.float().detach().clone().to(dev)
                bw.requires_grad_(True)
                binary_params.append((blp, bw))

    all_params = scaling_params + [bp[1] for bp in binary_params]
    if not all_params:
        layer_i.to(original_dtype)
        layer_j.to(original_dtype)
        del kw_gpu
        return initial_error, initial_error

    optimizer = torch.optim.Adam(all_params, lr=lr)
    logger.info(
        "[CBQ-DBF] %d DBF modules, %d scaling, %d binary",
        len(all_dbf),
        len(scaling_params),
        len(binary_params),
    )

    original_bit_mats = {id(blp): blp.bit_mat.clone() for blp, _ in binary_params}

    from ...quantizer.dbf.dbf_layer import pack_binary

    best_eval_mse = initial_error
    best_snap_i = {}
    best_snap_j = {}
    eval_interval = max(1, epochs // 4)

    def _hard_eval_window():
        """Hard-quantize binary, evaluate window MSE (leaves hard state)."""
        if optimize_binary:
            with torch.no_grad():
                for blp, bw in binary_params:
                    bq = bw.data.sign()
                    bq[bq == 0] = 1
                    blp.bit_mat = bq.to(torch.int8)
                    blp.bp = pack_binary(bq)
        return _eval_mse()

    # --- Training ---
    for epoch in range(epochs):
        layer_i.train()
        layer_j.train()
        total_loss = 0.0
        for j in range(n):
            optimizer.zero_grad()

            if optimize_binary:
                for blp, bw in binary_params:
                    blp.bit_mat = _smooth_sign_ste(bw, k=k_smooth)

            inp_gpu = inps[j].unsqueeze(0).detach().to(dev).float()
            tgt = target_outputs_j[j].detach().to(dev).float()
            out = _forward_window(inp_gpu)
            loss = F.mse_loss(out, tgt)
            loss.backward()

            if optimize_binary:
                for blp, _ in binary_params:
                    blp.bit_mat = original_bit_mats[id(blp)]

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
            optimizer.step()
            total_loss += loss.item()
            del inp_gpu, tgt, out, loss

        avg_soft = total_loss / max(n, 1)

        # --- Periodic hard evaluation (like GPTQ CBQ) ---
        do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
        if do_eval:
            eval_mse = _hard_eval_window()
            if eval_mse < best_eval_mse:
                best_eval_mse = eval_mse
                best_snap_i = copy.deepcopy(layer_i.state_dict())
                best_snap_j = copy.deepcopy(layer_j.state_dict())
            if optimize_binary:
                for blp, _ in binary_params:
                    blp.bit_mat = original_bit_mats[id(blp)]
            logger.info(
                "  [CBQ-DBF] Epoch %d/%d: train=%.6f, eval=%.6f (best: %.6f)",
                epoch + 1,
                epochs,
                avg_soft,
                eval_mse,
                best_eval_mse,
            )
        elif (epoch + 1) % max(1, epochs // 4) == 0:
            logger.info(
                "  [CBQ-DBF] Epoch %d/%d: train=%.6f",
                epoch + 1,
                epochs,
                avg_soft,
            )

    # --- Restore best state ---
    if best_snap_i:
        layer_i.load_state_dict(best_snap_i)
        layer_j.load_state_dict(best_snap_j)
    elif optimize_binary:
        with torch.no_grad():
            for blp, bw in binary_params:
                bq = bw.data.sign()
                bq[bq == 0] = 1
                blp.bit_mat = bq.to(torch.int8)
                blp.bp = pack_binary(bq)

    # --- Disable gradients ---
    for _name, mod in all_dbf:
        for attr in ("scaling0", "scaling2", "scaling4"):
            p = getattr(mod, attr, None)
            if p is not None:
                p.requires_grad_(False)

    # Restore original dtype for final evaluation and state serialisation
    layer_i.to(original_dtype)
    layer_j.to(original_dtype)

    final_error = _eval_mse()

    if final_error >= initial_error:
        logger.info(
            "[CBQ-DBF] No improvement (%.6f >= %.6f), reverting",
            final_error,
            initial_error,
        )
        layer_i.load_state_dict(snap_i)
        layer_j.load_state_dict(snap_j)
        final_error = initial_error

    delta = initial_error - final_error
    pct = (delta / max(initial_error, 1e-10)) * 100
    logger.info(
        "[CBQ-DBF] Final window MSE: %.6f (delta: %.6f, %+.1f%%)",
        final_error,
        delta,
        pct,
    )

    del kw_gpu, snap_i, snap_j
    return initial_error, final_error
