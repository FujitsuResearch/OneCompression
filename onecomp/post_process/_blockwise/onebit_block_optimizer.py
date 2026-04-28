"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

"""

"""OneBit block-wise optimiser.

Ported from qep-dev/src/blockwise_quantization/method/blockwise_onebit/
onebit_block_optimizer.py with the following adaptations:

    qep-dev OneBitLinear             onecomp-lab OneBitLinear
    ────────────────────             ──────────────────────────
    mod.a  (varies)                  mod.a  (buffer, register_buffer)
    mod.b  (varies)                  mod.b  (buffer, register_buffer)
    mod.sign_matrix                  mod.sign_matrix  (buffer, int8)
    (no sign_packed)                 mod.sign_packed  (buffer, uint8) — must update
    -                                my_pack() for repacking sign_packed

Key differences from qep-dev:
  - a/b are buffers (not Parameters), so we manually set requires_grad.
  - After hard-quantising sign_matrix, sign_packed must also be updated
    via my_pack() (onecomp-lab stores both representations).
  - Periodic hard evaluation with best-state tracking (GPTQ/DBF parity).
  - Rollback uses state_dict snapshots.
"""

import copy
import gc
from logging import getLogger
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import layer_kwargs_to_device

logger = getLogger(__name__)


def _smooth_sign_ste(x: torch.Tensor, k: float = 100.0) -> torch.Tensor:
    """SmoothSign STE: forward=sign(x), backward=d/dx tanh(kx)."""
    y = x.sign()
    y[y == 0] = 1
    return y.detach() - x.detach() + torch.tanh(k * x)


def _find_onebit_modules(layer: nn.Module) -> List[Tuple[str, nn.Module]]:
    from ...quantizer.onebit.onebit_layer import OneBitLinear

    return [(name, mod) for name, mod in layer.named_modules() if isinstance(mod, OneBitLinear)]


def _layer_output(layer, inp_gpu, kw_gpu):
    raw = layer(inp_gpu, **kw_gpu)
    out = raw[0] if isinstance(raw, tuple) else raw
    if out.dim() == 3 and out.size(0) == 1:
        out = out.squeeze(0)
    return out


def optimize_onebit_block(
    layer: nn.Module,
    inps,
    target_outputs,
    layer_kwargs: dict,
    lr: float = 1e-4,
    epochs: int = 4,
    dev: torch.device = None,
    optimize_sign: bool = True,
    k_smooth: float = 100.0,
    **kwargs,
) -> Tuple[float, float]:
    """Optimise a single OneBit block. Returns (initial_mse, final_mse).

    Optimises scaling vectors (a, b) via Adam and sign matrices via
    SmoothSign STE to minimise block-level MSE against the FP16 teacher.
    Includes rollback guarantee.
    """
    if dev is None:
        dev = next(layer.parameters()).device

    kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)

    onebit_modules = _find_onebit_modules(layer)
    if not onebit_modules:
        logger.info("[OneBit Block-wise] No OneBitLinear modules found")
        del kw_gpu
        return 0.0, 0.0

    logger.info("[OneBit Block-wise] %d OneBitLinear modules", len(onebit_modules))

    # --- Initial MSE ---
    layer.eval()
    with torch.no_grad():
        initial_error = 0.0
        for j in range(len(inps)):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            out = _layer_output(layer, inp_gpu, kw_gpu)
            tgt = target_outputs[j].to(dev)
            initial_error += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, out, tgt
        initial_error /= max(len(inps), 1)

    logger.info("[OneBit Block-wise] Initial MSE: %.6f", initial_error)

    from ...quantizer.onebit.onebit_layer import my_pack

    # --- Save initial state for rollback (state_dict snapshot) ---
    initial_snap = copy.deepcopy(layer.state_dict())

    # Promote to float32 for numerically stable Adam optimisation.
    # OneBitLinear.forward casts params via .to(x.dtype), so both the
    # layer AND inputs must be float32 to keep the entire computation
    # graph in float32.  Without this, Adam's eps (1e-8) underflows to
    # 0 in float16, causing division-by-zero → NaN.
    original_dtype = next(layer.parameters()).dtype
    layer.float()

    # --- Collect optimisable parameters ---
    scaling_params = []
    sign_params = []

    for name, mod in onebit_modules:
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
        logger.info("[OneBit Block-wise] No optimisable parameters found")
        layer.to(original_dtype)
        del kw_gpu, initial_snap
        return initial_error, initial_error

    optimizer = torch.optim.Adam(all_params, lr=lr)
    logger.info(
        "[OneBit Block-wise] Params: %d scaling, %d sign, k=%.1f",
        len(scaling_params),
        len(sign_params),
        k_smooth,
    )

    # --- Prepare for hard evaluation (GPTQ/DBF-style best tracking) ---
    original_sign_mats = {}
    if optimize_sign:
        for name, mod, _sw in sign_params:
            original_sign_mats[name] = mod.sign_matrix.clone()

    best_eval_mse = initial_error
    best_snap = {}
    eval_interval = max(1, epochs // 4)

    def _hard_eval_mse():
        """Hard-quantize sign matrices, evaluate MSE (leaves hard state)."""
        if optimize_sign:
            with torch.no_grad():
                for _name, mod, sign_weight in sign_params:
                    sq = sign_weight.data.sign()
                    sq[sq == 0] = 1
                    mod.sign_matrix = sq.to(torch.int8)
                    mod.sign_packed = my_pack(sq)
        layer.eval()
        with torch.no_grad():
            total = 0.0
            for j in range(len(inps)):
                inp_gpu = inps[j].unsqueeze(0).to(dev).float()
                out = _layer_output(layer, inp_gpu, kw_gpu)
                tgt = target_outputs[j].to(dev).float()
                total += F.mse_loss(out, tgt).item()
                del inp_gpu, out, tgt
        return total / max(len(inps), 1)

    try:
        # --- Training loop ---
        for epoch in range(epochs):
            layer.train()
            total_loss = 0.0

            for j in range(len(inps)):
                optimizer.zero_grad()

                saved_mats = []
                if optimize_sign:
                    for _mod_name, mod, sign_weight in sign_params:
                        saved_mats.append(mod.sign_matrix)
                        mod.sign_matrix = _smooth_sign_ste(sign_weight, k=k_smooth)

                inp_gpu = inps[j].unsqueeze(0).detach().to(dev).float()
                tgt = target_outputs[j].detach().to(dev).float()
                out = _layer_output(layer, inp_gpu, kw_gpu)
                loss = F.mse_loss(out, tgt)
                loss.backward()

                if optimize_sign:
                    for idx, (_mod_name, mod, _sw) in enumerate(sign_params):
                        mod.sign_matrix = saved_mats[idx]

                optimizer.step()
                total_loss += loss.item()
                del inp_gpu, tgt, out, loss

            avg_loss = total_loss / max(len(inps), 1)

            # --- Periodic hard evaluation (like GPTQ/DBF block optimiser) ---
            do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
            if do_eval:
                eval_mse = _hard_eval_mse()
                if eval_mse < best_eval_mse:
                    best_eval_mse = eval_mse
                    best_snap = copy.deepcopy(layer.state_dict())
                if optimize_sign:
                    for name, mod, _sw in sign_params:
                        mod.sign_matrix = original_sign_mats[name]
                logger.info(
                    "  [OneBit] Epoch %d/%d: train=%.6f, eval=%.6f (best: %.6f)",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    eval_mse,
                    best_eval_mse,
                )
            elif (epoch + 1) % max(1, epochs // 4) == 0 or epoch == 0:
                logger.info(
                    "  [OneBit] Epoch %d/%d: train=%.6f",
                    epoch + 1,
                    epochs,
                    avg_loss,
                )

        # --- Restore best state ---
        if best_snap:
            layer.load_state_dict(best_snap)
        elif optimize_sign:
            with torch.no_grad():
                for _mod_name, mod, sign_weight in sign_params:
                    sq = sign_weight.data.sign()
                    sq[sq == 0] = 1
                    mod.sign_matrix = sq.to(torch.int8)
                    mod.sign_packed = my_pack(sq)
    finally:
        # --- Cleanup ---
        for _name, mod in onebit_modules:
            mod.a.requires_grad_(False)
            mod.b.requires_grad_(False)

        # Restore original dtype for final evaluation and state serialisation
        layer.to(original_dtype)

    # --- Final MSE ---
    layer.eval()
    with torch.no_grad():
        final_error = 0.0
        for j in range(len(inps)):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            out = _layer_output(layer, inp_gpu, kw_gpu)
            tgt = target_outputs[j].to(dev)
            final_error += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, tgt, out
        final_error /= max(len(inps), 1)

    # --- Rollback if no improvement ---
    if final_error >= initial_error:
        logger.info(
            "[OneBit Block-wise] No improvement (%.6f >= %.6f), rolling back",
            final_error,
            initial_error,
        )
        layer.load_state_dict(initial_snap)
        final_error = initial_error

    delta = initial_error - final_error
    pct = (delta / max(initial_error, 1e-10)) * 100
    logger.info(
        "[OneBit Block-wise] Final MSE: %.6f (delta: %.6f, %+.1f%%)",
        final_error,
        delta,
        pct,
    )

    del kw_gpu, initial_snap
    gc.collect()
    return initial_error, final_error
