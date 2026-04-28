"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

"""

"""Cross-Block Quantisation (CBQ) for GPTQ.

Ported from qep-dev/src/blockwise_quantization/method/blockwise_gptq/
cbq_optimizer.py.

Jointly optimises two adjacent GPTQ blocks (window K=2) to reduce
greedy error accumulation from Phase 1.  Supports intweight STE, PSTA,
cosine LR, directional loss, and sample shuffling.

Same attribute adaptations as gptq_block_optimizer.py:
    mod.scales / mod.qzeros / mod.wbits / mod.qweight / mod._weight_is_packed
"""

import random
from logging import getLogger
from types import MethodType
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import layer_kwargs_to_device
from .gptq_block_optimizer import (
    _find_gptq_modules,
    _get_float_zeros,
    _make_differentiable_forward,
    _layer_output,
    _cosine_warmup_lr,
    _get_int_weights,
    _save_initial_state,
    _restore_state,
)

logger = getLogger(__name__)


def optimize_gptq_cross_block(
    layer_i: nn.Module,
    layer_j: nn.Module,
    inps,
    target_outputs_j,
    layer_kwargs: dict,
    lr: float = 5e-5,
    epochs: int = 10,
    dev: torch.device = None,
    grad_clip: float = 1.0,
    optimize_intweight: bool = False,
    intweight_lr: float = 1e-4,
    use_cosine_schedule: bool = False,
    warmup_ratio: float = 0.1,
    cosine_loss_weight: float = 0.0,
    ste_k_schedule: str = "fixed",
    ste_k_min: float = 2.0,
    ste_k_max: float = 20.0,
    shuffle_samples: bool = False,
    **kwargs,
) -> Tuple[float, float]:
    """Jointly optimise two adjacent GPTQ blocks. Returns (initial_mse, final_mse)."""
    if dev is None:
        dev = next(layer_i.parameters()).device

    gptq_modules_i = _find_gptq_modules(layer_i)
    gptq_modules_j = _find_gptq_modules(layer_j)
    all_gptq = gptq_modules_i + gptq_modules_j

    if not all_gptq:
        return 0.0, 0.0

    for _name, mod in all_gptq:
        fmt = getattr(mod, "checkpoint_format", "gptq")
        if fmt != "gptq":
            raise ValueError(
                f"blockwise PTQ currently supports only checkpoint_format='gptq' (v1), "
                f"but {_name} has '{fmt}'"
            )

    kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)

    def _forward_window(inp_gpu):
        mid = _layer_output(layer_i, inp_gpu, kw_gpu)
        if mid.dim() == 2:
            mid = mid.unsqueeze(0)
        return _layer_output(layer_j, mid, kw_gpu)

    n = len(inps)

    # --- Initial MSE ---
    with torch.no_grad():
        total_init = 0.0
        for j in range(n):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            out = _forward_window(inp_gpu)
            tgt = target_outputs_j[j].to(dev)
            total_init += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, out, tgt
        initial_error = total_init / max(n, 1)

    logger.info("[CBQ-GPTQ] Initial window MSE: %.6f", initial_error)

    # --- Promote scale/zero to nn.Parameter ---
    original_forwards = {}
    params_sz = []
    params_iw = []

    for name, mod in all_gptq:
        mod._opt_scales = nn.Parameter(mod.scales.clone().float().to(dev))
        mod._opt_zeros = nn.Parameter(_get_float_zeros(mod).to(dev))
        params_sz.extend([mod._opt_scales, mod._opt_zeros])

        if optimize_intweight:
            int_weights = _get_int_weights(mod)
            mod._opt_intweight = nn.Parameter(int_weights.float().to(dev))
            mod._ste_k = ste_k_min if ste_k_schedule == "progressive" else 10.0
            params_iw.append(mod._opt_intweight)

        original_forwards[id(mod)] = mod.forward
        mod.forward = MethodType(
            _make_differentiable_forward(mod, use_intweight_param=optimize_intweight),
            mod,
        )

    param_groups = [{"params": params_sz, "lr": lr}]
    if params_iw:
        param_groups.append({"params": params_iw, "lr": intweight_lr})

    initial_state_i = _save_initial_state(gptq_modules_i)
    initial_state_j = _save_initial_state(gptq_modules_j)

    optimizer = torch.optim.Adam(param_groups)

    best_eval_mse = initial_error
    best_state_i = None
    best_state_j = None
    eval_interval = max(1, epochs // 4)
    total_steps = epochs * n
    global_step = 0

    diff_forwards = {id(mod): mod.forward for _name, mod in all_gptq}

    from .gptq_block_optimizer import _write_back_params

    def _eval_window_mse():
        _write_back_params(all_gptq, optimize_intweight)
        for _name, mod in all_gptq:
            mod.forward = original_forwards[id(mod)]
        layer_i.eval()
        layer_j.eval()
        with torch.no_grad():
            total = 0.0
            for j in range(n):
                inp_gpu = inps[j].unsqueeze(0).to(dev)
                out = _forward_window(inp_gpu)
                tgt = target_outputs_j[j].to(dev)
                total += F.mse_loss(out.float(), tgt.float()).item()
                del inp_gpu, out, tgt
        for _name, mod in all_gptq:
            mod.forward = diff_forwards[id(mod)]
        return total / max(n, 1)

    # --- Training loop ---
    for epoch in range(epochs):
        layer_i.train()
        layer_j.train()
        total_loss = 0.0

        if ste_k_schedule == "progressive" and optimize_intweight:
            cur_k = ste_k_min + (ste_k_max - ste_k_min) * epoch / max(epochs - 1, 1)
            for _name, mod in all_gptq:
                if hasattr(mod, "_ste_k"):
                    mod._ste_k = cur_k

        sample_order = list(range(n))
        if shuffle_samples:
            random.shuffle(sample_order)

        all_params = params_sz + params_iw
        for j in sample_order:
            if use_cosine_schedule:
                cur_lr_sz = _cosine_warmup_lr(lr, global_step, total_steps, warmup_ratio)
                cur_lr_iw = _cosine_warmup_lr(
                    intweight_lr,
                    global_step,
                    total_steps,
                    warmup_ratio,
                )
                optimizer.param_groups[0]["lr"] = cur_lr_sz
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = cur_lr_iw

            inp_gpu = inps[j].unsqueeze(0).detach().to(dev)
            tgt = target_outputs_j[j].detach().to(dev)

            optimizer.zero_grad()
            out = _forward_window(inp_gpu)
            loss = F.mse_loss(out.float(), tgt.float())
            if cosine_loss_weight > 0:
                out_flat = out.float().reshape(-1, out.size(-1))
                tgt_flat = tgt.float().reshape(-1, tgt.size(-1))
                cos_sim = F.cosine_similarity(out_flat, tgt_flat, dim=-1).mean()
                loss = loss + cosine_loss_weight * (1.0 - cos_sim)
            loss.backward()
            if grad_clip > 0 and all_params:
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
            optimizer.step()
            total_loss += loss.item()
            del out, loss, inp_gpu, tgt
            global_step += 1

        avg_loss = total_loss / max(n, 1)

        do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
        if do_eval:
            eval_mse = _eval_window_mse()
            if eval_mse < best_eval_mse:
                best_eval_mse = eval_mse
                best_state_i = _save_initial_state(gptq_modules_i)
                best_state_j = _save_initial_state(gptq_modules_j)
            logger.info(
                "  [CBQ-GPTQ] Epoch %d/%d: train=%.6f, eval=%.6f (best: %.6f)",
                epoch + 1,
                epochs,
                avg_loss,
                eval_mse,
                best_eval_mse,
            )
        elif (epoch + 1) % max(1, epochs // 4) == 0:
            logger.info(
                "  [CBQ-GPTQ] Epoch %d/%d: train=%.6f",
                epoch + 1,
                epochs,
                avg_loss,
            )

    # --- Restore best or rollback ---
    for _name, mod in all_gptq:
        mod.forward = original_forwards[id(mod)]
        for attr in ("_opt_scales", "_opt_zeros", "_opt_intweight", "_ste_k"):
            if hasattr(mod, attr):
                delattr(mod, attr)

    _restore_state(gptq_modules_i, best_state_i if best_state_i else initial_state_i)
    _restore_state(gptq_modules_j, best_state_j if best_state_j else initial_state_j)

    # --- Final evaluation ---
    layer_i.eval()
    layer_j.eval()
    with torch.no_grad():
        total_final = 0.0
        for j in range(n):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            out = _forward_window(inp_gpu)
            tgt = target_outputs_j[j].to(dev)
            total_final += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, out, tgt
        final_error = total_final / max(n, 1)

    if final_error >= initial_error:
        logger.info("[CBQ-GPTQ] No improvement, rolling back")
        _restore_state(gptq_modules_i, initial_state_i)
        _restore_state(gptq_modules_j, initial_state_j)
        final_error = initial_error

    delta = initial_error - final_error
    pct = (delta / max(initial_error, 1e-10)) * 100
    logger.info(
        "[CBQ-GPTQ] Final window MSE: %.6f (delta: %.6f, %+.1f%%)",
        final_error,
        delta,
        pct,
    )

    del kw_gpu
    return initial_error, final_error
