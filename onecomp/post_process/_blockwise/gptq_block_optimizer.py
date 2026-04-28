"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

"""

"""GPTQ block-wise optimiser.

Ported from qep-dev/src/blockwise_quantization/method/blockwise_gptq/
gptq_block_optimizer.py.

onecomp-lab GPTQLinear attributes (AutoGPTQ-compatible, export/v0-5-0+):
    mod.scales           : (num_groups, out_features) float16
    mod.qzeros           : packed INT32 (v1 = stored with -1 offset)
    mod.qweight          : packed INT32 (AutoGPTQ format)
    mod._weight_is_packed: bool
    mod.wbits            : int
    mod.g_idx            : (in_features,) int32
    mod.in_features      : int
    mod.out_features     : int
    mod.checkpoint_format: "gptq" (v1) or "gptq_v2"

Key design decisions (same as qep-dev):
  - Promotes scale/zero to float32 nn.Parameter for gradient-based optimisation.
  - Replaces GPTQLinear.forward with a differentiable version that computes
    dequantisation in float32 (prevents float16 overflow in attention backward).
  - Evaluates in "hard" mode (original forward) at checkpoints.
  - Tracks best state and rolls back if no improvement.
"""

import math
import random
from logging import getLogger
from types import MethodType
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import layer_kwargs_to_device

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# STE helper (same as qep-dev)
# ---------------------------------------------------------------------------


def _smooth_ste_round(
    x: torch.Tensor,
    min_val: int,
    max_val: int,
    k: float = 10.0,
) -> torch.Tensor:
    """Forward: round(clamp(x)).  Backward: gradient via sigmoid soft-rounding."""
    x_clamped = x.clamp(min_val, max_val)
    hard = x_clamped.round()
    frac = x_clamped - x_clamped.floor()
    soft = x_clamped.floor() + torch.sigmoid(k * (frac - 0.5))
    return soft + (hard - soft).detach()


# ---------------------------------------------------------------------------
# GPTQLinear helpers (adapted for onecomp-lab naming)
# ---------------------------------------------------------------------------


def _find_gptq_modules(layer: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return all GPTQLinear modules in *layer*."""
    from ...quantizer.gptq.gptq_layer import GPTQLinear

    return [(name, mod) for name, mod in layer.named_modules() if isinstance(mod, GPTQLinear)]


def _get_float_zeros(mod: nn.Module) -> torch.Tensor:
    """Read qzeros from GPTQLinear and return float32 zero points.

    Reverses the v1 packing: unpack → +1 → float32.
    """
    from ...quantizer.gptq.gptq_layer import unpack_zeros

    _v1 = getattr(mod, "checkpoint_format", "gptq") != "gptq_v2"
    if mod._weight_is_packed:
        zeros = unpack_zeros(mod.qzeros, mod.wbits, mod.out_features)
    else:
        zeros = mod.qzeros
    if _v1:
        zeros = (zeros + 1) & ((1 << mod.wbits) - 1)
    return zeros.float()


def _get_int_weights(mod: nn.Module) -> torch.Tensor:
    """Get unpacked integer weights from a GPTQLinear module.

    Handles both packed (_weight_is_packed=True) and unpacked modes.
    Returns shape (out_features, in_features), dtype int32.
    """
    if mod._weight_is_packed:
        from ...quantizer.gptq.gptq_layer import unpack_int_weights

        return unpack_int_weights(
            mod.qweight,
            mod.wbits,
            (mod.out_features, mod.in_features),
        )
    return mod.qweight


def _set_int_weights(mod: nn.Module, int_weights: torch.Tensor) -> None:
    """Write integer weights back to GPTQLinear buffers."""
    if mod._weight_is_packed:
        from ...quantizer.gptq.gptq_layer import pack_int_weights

        packed = pack_int_weights(int_weights, mod.wbits)
        mod.qweight.copy_(packed.to(mod.qweight.device))
    else:
        mod.qweight.copy_(int_weights.to(mod.qweight.device))


# ---------------------------------------------------------------------------
# Differentiable forward (float32, core of the optimizer)
# ---------------------------------------------------------------------------


def _make_differentiable_forward(mod: nn.Module, use_intweight_param: bool = False):
    """Build a differentiable forward routing gradients through _opt_scales/_opt_zeros.

    All computation is done in float32 to prevent float16 overflow in attention
    backward passes.  Output is cast back to the input dtype.

    qep-dev equivalent: _make_differentiable_forward() in gptq_block_optimizer.py
    """

    def differentiable_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).float()

        if use_intweight_param and hasattr(self, "_opt_intweight"):
            intweight_float = _smooth_ste_round(
                self._opt_intweight,
                min_val=0,
                max_val=(1 << self.wbits) - 1,
                k=getattr(self, "_ste_k", 10.0),
            )
        else:
            int_weights = _get_int_weights(self)
            intweight_float = int_weights.detach().float()

        scales = self._opt_scales  # (num_groups, out_features)
        zeros = self._opt_zeros  # (num_groups, out_features)

        # g_idx is None when groupsize == -1 (per-channel quantisation)
        if self.g_idx is not None:
            scales_expanded = scales[self.g_idx, :].T  # (out_features, in_features)
            zeros_expanded = zeros[self.g_idx, :].T  # (out_features, in_features)
        else:
            scales_expanded = scales.T.expand(self.out_features, self.in_features)
            zeros_expanded = zeros.T.expand(self.out_features, self.in_features)

        weight_dequant = (intweight_float - zeros_expanded) * scales_expanded
        bias_f = self.bias.float() if self.bias is not None else None
        out = F.linear(x_2d, weight_dequant, bias_f)
        out = out.reshape(*orig_shape[:-1], self.out_features)
        return out.to(x.dtype)

    return differentiable_forward


# ---------------------------------------------------------------------------
# Block-level MSE evaluation (hard forward, same as qep-dev _compute_block_mse)
# ---------------------------------------------------------------------------


def _compute_block_mse(layer, inps, target_outputs, layer_kwargs, dev):
    kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)
    with torch.no_grad():
        total_error = 0.0
        for j in range(len(inps)):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            raw = layer(inp_gpu, **kw_gpu)
            out = raw[0] if isinstance(raw, tuple) else raw
            tgt = target_outputs[j].to(dev)
            if out.dim() == 3 and out.size(0) == 1:
                out = out.squeeze(0)
            total_error += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, out, tgt
        return total_error / max(len(inps), 1)


def _layer_output(layer, inp_gpu, kw_gpu):
    raw = layer(inp_gpu, **kw_gpu)
    out = raw[0] if isinstance(raw, tuple) else raw
    if out.dim() == 3 and out.size(0) == 1:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Write-back from float32 parameters to module buffers
# ---------------------------------------------------------------------------


def _write_back_params(gptq_modules, optimize_intweight):
    """Copy optimised float32 params back to module buffers for hard evaluation.

    scales: float32 → FP16, write to mod.scales
    zeros:  float32 → round → INT32 → v1 offset (−1) → pack → mod.qzeros
    """
    from ...quantizer.gptq.gptq_layer import pack_zeros

    for _name, mod in gptq_modules:
        mod.scales.copy_(mod._opt_scales.data.to(mod.scales.dtype))
        zero_int = mod._opt_zeros.data.round().to(torch.int32) - 1
        if mod._weight_is_packed:
            mod.qzeros.copy_(pack_zeros(zero_int, mod.wbits).to(mod.qzeros.device))
        else:
            mod.qzeros.copy_(zero_int.to(mod.qzeros.device))
        if optimize_intweight and hasattr(mod, "_opt_intweight"):
            iw = mod._opt_intweight.data.round().clamp(0, (1 << mod.wbits) - 1).to(torch.int32)
            _set_int_weights(mod, iw)


# ---------------------------------------------------------------------------
# Cosine schedule with warmup (same as qep-dev)
# ---------------------------------------------------------------------------


def _cosine_warmup_lr(
    base_lr,
    step,
    total_steps,
    warmup_ratio=0.1,
    min_lr_ratio=0.01,
):
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Save / restore initial state
# ---------------------------------------------------------------------------


def _save_initial_state(gptq_modules):
    state = {}
    for name, mod in gptq_modules:
        state[name] = {
            "scales": mod.scales.data.clone(),
            "qzeros": mod.qzeros.data.clone(),
            "qweight": mod.qweight.data.clone(),
        }
    return state


def _restore_state(gptq_modules, state_dict):
    for name, mod in gptq_modules:
        s = state_dict[name]
        mod.scales.copy_(s["scales"])
        mod.qzeros.copy_(s["qzeros"])
        mod.qweight.copy_(s["qweight"])


# ===========================================================================
# Main entry point
# ===========================================================================


def optimize_gptq_block(
    layer: nn.Module,
    inps,
    target_outputs,
    layer_kwargs: dict,
    lr: float = 1e-3,
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
    """Optimise GPTQ scales/zeros (+ optional intweight via Smooth STE) per block.

    Returns (initial_mse, final_mse).  Rolls back if no improvement.

    This is the onecomp-lab port of qep-dev's optimize_gptq_block().  The key
    difference from the generic optimizer is that the forward pass is replaced
    with a float32 differentiable version that optimises scale/zero directly
    (not LayerNorm weights).
    """
    if dev is None:
        dev = next(layer.parameters()).device

    gptq_modules = _find_gptq_modules(layer)
    if not gptq_modules:
        logger.info("[GPTQ Block-wise] No GPTQLinear modules found — skipping")
        error = _compute_block_mse(layer, inps, target_outputs, layer_kwargs, dev)
        return error, error

    for _name, mod in gptq_modules:
        fmt = getattr(mod, "checkpoint_format", "gptq")
        if fmt != "gptq":
            raise ValueError(
                f"blockwise PTQ currently supports only checkpoint_format='gptq' (v1), "
                f"but {_name} has '{fmt}'"
            )

    logger.info("[GPTQ Block-wise] Found %d GPTQLinear modules", len(gptq_modules))

    initial_error = _compute_block_mse(layer, inps, target_outputs, layer_kwargs, dev)
    logger.info("[GPTQ Block-wise] Initial MSE: %.6f", initial_error)

    # --- Save initial state for rollback ---
    initial_state = _save_initial_state(gptq_modules)

    # --- Promote scale/zero to nn.Parameter (float32) for optimisation ---
    original_forwards = {}
    params_scales_zeros = []
    params_intweight = []

    for name, mod in gptq_modules:
        mod._opt_scales = nn.Parameter(mod.scales.clone().float().to(dev))
        mod._opt_zeros = nn.Parameter(_get_float_zeros(mod).to(dev))
        params_scales_zeros.extend([mod._opt_scales, mod._opt_zeros])

        if optimize_intweight:
            int_weights = _get_int_weights(mod)
            mod._opt_intweight = nn.Parameter(int_weights.float().to(dev))
            mod._ste_k = ste_k_min if ste_k_schedule == "progressive" else 10.0
            params_intweight.append(mod._opt_intweight)

        original_forwards[name] = mod.forward
        mod.forward = MethodType(
            _make_differentiable_forward(mod, use_intweight_param=optimize_intweight),
            mod,
        )

    param_groups = [{"params": params_scales_zeros, "lr": lr}]
    if params_intweight:
        param_groups.append({"params": params_intweight, "lr": intweight_lr})

    optimizer = torch.optim.Adam(param_groups)

    n_samples = len(inps)
    kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)
    total_steps = epochs * n_samples

    best_eval_mse = initial_error
    best_state = {}
    eval_interval = max(1, epochs // 4)
    diff_forwards = {name: mod.forward for name, mod in gptq_modules}
    global_step = 0

    try:
        # --- Training loop ---
        for epoch in range(epochs):
            layer.train()
            total_loss = 0.0

            if ste_k_schedule == "progressive" and optimize_intweight:
                cur_k = ste_k_min + (ste_k_max - ste_k_min) * epoch / max(epochs - 1, 1)
                for _name, mod in gptq_modules:
                    if hasattr(mod, "_ste_k"):
                        mod._ste_k = cur_k

            sample_order = list(range(n_samples))
            if shuffle_samples:
                random.shuffle(sample_order)

            all_params = params_scales_zeros + params_intweight
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
                target_gpu = target_outputs[j].detach().to(dev)

                optimizer.zero_grad()
                out = _layer_output(layer, inp_gpu, kw_gpu)
                loss = F.mse_loss(out.float(), target_gpu.float())
                if cosine_loss_weight > 0:
                    out_flat = out.float().reshape(-1, out.size(-1))
                    tgt_flat = target_gpu.float().reshape(-1, target_gpu.size(-1))
                    cos_sim = F.cosine_similarity(out_flat, tgt_flat, dim=-1).mean()
                    loss = loss + cosine_loss_weight * (1.0 - cos_sim)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()
                total_loss += loss.item()
                del out, loss, inp_gpu, target_gpu
                global_step += 1

            avg_loss = total_loss / n_samples

            # --- Periodic evaluation in hard mode ---
            do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
            if do_eval:
                _write_back_params(gptq_modules, optimize_intweight)
                for name, mod in gptq_modules:
                    mod.forward = original_forwards[name]
                layer.eval()
                eval_mse = _compute_block_mse(
                    layer,
                    inps,
                    target_outputs,
                    layer_kwargs,
                    dev,
                )
                for name, mod in gptq_modules:
                    mod.forward = diff_forwards[name]

                if eval_mse < best_eval_mse:
                    best_eval_mse = eval_mse
                    best_state = _save_initial_state(gptq_modules)

                logger.info(
                    "  [GPTQ] Epoch %d/%d: train_loss=%.6f, eval_mse=%.6f (best: %.6f)",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    eval_mse,
                    best_eval_mse,
                )
            elif (epoch + 1) % max(1, epochs // 4) == 0:
                logger.info(
                    "  [GPTQ] Epoch %d/%d: train_loss=%.6f",
                    epoch + 1,
                    epochs,
                    avg_loss,
                )

        # --- Restore best state (or initial if no improvement) ---
        restore_src = best_state if best_state else initial_state
        _restore_state(gptq_modules, restore_src)
    finally:
        # --- Cleanup: restore original forwards and remove temporary attributes ---
        for name, mod in gptq_modules:
            mod.forward = original_forwards[name]
            for attr in ("_opt_scales", "_opt_zeros", "_opt_intweight", "_ste_k"):
                if hasattr(mod, attr):
                    delattr(mod, attr)

    # --- Final evaluation ---
    layer.eval()
    final_error = _compute_block_mse(layer, inps, target_outputs, layer_kwargs, dev)

    if final_error >= initial_error:
        logger.info(
            "[GPTQ Block-wise] No improvement (%.6f >= %.6f), rolling back",
            final_error,
            initial_error,
        )
        _restore_state(gptq_modules, initial_state)
        final_error = initial_error

    delta = initial_error - final_error
    pct = (delta / max(initial_error, 1e-10)) * 100
    logger.info(
        "[GPTQ Block-wise] Final MSE: %.6f (delta: %.6f, %+.1f%%)",
        final_error,
        delta,
        pct,
    )

    del kw_gpu
    return initial_error, final_error
