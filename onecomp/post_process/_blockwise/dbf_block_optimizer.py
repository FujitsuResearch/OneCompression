"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

"""

"""DBF block-wise optimiser.

Ported from qep-dev/src/blockwise_quantization/method/blockwise_dbf/
dbf_block_optimizer.py with the following adaptations:

    qep-dev (5-stage nn.Sequential)    onecomp-lab (DoubleBinaryLinear)
    ──────────────────────────────     ──────────────────────────────────
    dbf_seq[0].w                       mod.scaling0  (nn.Parameter, FP16)
    dbf_seq[1].bit_mat                 mod.binary_multiplication1.bit_mat
    dbf_seq[1].bp                      mod.binary_multiplication1.bp
    dbf_seq[1].shape                   mod.binary_multiplication1.shape
    dbf_seq[2].w                       mod.scaling2
    dbf_seq[3].bit_mat                 mod.binary_multiplication3.bit_mat
    dbf_seq[3].bp                      mod.binary_multiplication3.bp
    dbf_seq[4].w                       mod.scaling4
    _is_dbf_sequential(mod)            isinstance(mod, DoubleBinaryLinear)
    _pack_binary_to_uint8(bq)          pack_binary(bq) from dbf_layer.py

Key differences from qep-dev:
  - onecomp-lab uses a single DoubleBinaryLinear class (not nn.Sequential).
  - BitLinearPacked.scale (nn.Parameter, init=1.0) is frozen during
    optimisation to match qep-dev behaviour.
  - pack_binary is imported from dbf_layer.py instead of being defined here.
"""

import copy
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


def _find_dbf_modules(layer: nn.Module) -> List[Tuple[str, nn.Module]]:
    from ...quantizer.dbf.dbf_layer import DoubleBinaryLinear

    return [
        (name, mod) for name, mod in layer.named_modules() if isinstance(mod, DoubleBinaryLinear)
    ]


def _layer_output(layer, inp_gpu, kw_gpu):
    raw = layer(inp_gpu, **kw_gpu)
    out = raw[0] if isinstance(raw, tuple) else raw
    if out.dim() == 3 and out.size(0) == 1:
        out = out.squeeze(0)
    return out


def _get_binary_parts(mod):
    """Return list of (BitLinearPacked, bit_mat_tensor, shape) for stages 1 & 3.

    onecomp-lab equivalent of qep-dev's iteration over dbf_seq[1] and dbf_seq[3].
    """
    parts = []
    for attr in ("binary_multiplication1", "binary_multiplication3"):
        blp = getattr(mod, attr, None)
        if blp is None:
            continue
        bit_mat = getattr(blp, "bit_mat", None)
        if bit_mat is None:
            continue
        parts.append((blp, bit_mat, blp.shape))
    return parts


def optimize_dbf_block(
    layer: nn.Module,
    inps,
    target_outputs,
    layer_kwargs: dict,
    lr: float = 1e-4,
    epochs: int = 4,
    dev: torch.device = None,
    optimize_binary: bool = True,
    k_smooth: float = 100.0,
    **kwargs,
) -> Tuple[float, float]:
    """Optimise DBF scaling vectors (+ optional binary via SmoothSign STE).

    Returns (initial_mse, final_mse).
    """
    if dev is None:
        dev = next(layer.parameters()).device

    dbf_modules = _find_dbf_modules(layer)
    if not dbf_modules:
        logger.info("[DBF Block-wise] No DoubleBinaryLinear modules found")
        return 0.0, 0.0

    logger.info(
        "[DBF Block-wise] %d DBF modules, optimize_binary=%s",
        len(dbf_modules),
        optimize_binary,
    )

    kw_gpu = layer_kwargs_to_device(layer_kwargs, dev)

    # --- Freeze BitLinearPacked.scale (qep-dev has no such parameter) ---
    for _name, mod in dbf_modules:
        for attr in ("binary_multiplication1", "binary_multiplication3"):
            blp = getattr(mod, attr, None)
            if blp is not None and hasattr(blp, "scale"):
                blp.scale.requires_grad_(False)

    # --- Initial MSE ---
    layer.eval()
    with torch.no_grad():
        initial_error = 0.0
        for j in range(len(inps)):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            out = _layer_output(layer, inp_gpu, kw_gpu)
            tgt = target_outputs[j].to(dev)
            initial_error += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, tgt, out
        initial_error /= max(len(inps), 1)

    logger.info("[DBF Block-wise] Initial MSE: %.6f", initial_error)

    # --- Save initial state for rollback ---
    initial_snap = copy.deepcopy(layer.state_dict())

    # Promote to float32 for numerically stable Adam optimisation.
    # GPTQ already operates in float32 (via explicit nn.Parameter copies);
    # DBF/OneBit forward casts params via .to(x.dtype), so both the layer
    # AND inputs must be float32 to keep the entire computation graph —
    # forward, backward, and Adam moment accumulation — in float32.
    # Without this, Adam's eps (1e-8) underflows to 0 in float16,
    # causing division-by-zero → NaN.
    original_dtype = next(layer.parameters()).dtype
    layer.float()

    # --- Collect scaling params (stages 0, 2, 4) ---
    scaling_params = []
    for _name, mod in dbf_modules:
        for attr in ("scaling0", "scaling2", "scaling4"):
            p = getattr(mod, attr, None)
            if p is not None:
                p.requires_grad_(True)
                scaling_params.append(p)

    # --- Collect binary params (stages 1, 3) ---
    binary_params = []
    if optimize_binary:
        for _name, mod in dbf_modules:
            for blp, bit_mat, _shape in _get_binary_parts(mod):
                bw = bit_mat.float().detach().clone().to(dev)
                bw.requires_grad_(True)
                binary_params.append((blp, bw))

    all_params = scaling_params + [bp[1] for bp in binary_params]
    if not all_params:
        logger.info("[DBF Block-wise] No optimizable parameters")
        layer.to(original_dtype)
        del kw_gpu, initial_snap
        return initial_error, initial_error

    optimizer = torch.optim.Adam(all_params, lr=lr)
    logger.info(
        "[DBF Block-wise] Params: %d scaling, %d binary, k=%.1f",
        len(scaling_params),
        len(binary_params),
        k_smooth,
    )

    # --- Prepare for hard evaluation (GPTQ-style best tracking) ---
    from ...quantizer.dbf.dbf_layer import pack_binary

    original_bit_mats = {}
    if optimize_binary:
        for blp, _bw in binary_params:
            original_bit_mats[id(blp)] = blp.bit_mat.clone()

    best_eval_mse = initial_error
    best_snap = {}
    eval_interval = max(1, epochs // 4)

    def _hard_eval_mse():
        """Hard-quantize binary, evaluate MSE (leaves bit_mat in hard state)."""
        if optimize_binary:
            with torch.no_grad():
                for blp, bw in binary_params:
                    bq = bw.data.sign()
                    bq[bq == 0] = 1
                    blp.bit_mat = bq.to(torch.int8)
                    blp.bp = pack_binary(bq)
        layer.eval()
        with torch.no_grad():
            total = 0.0
            for j in range(len(inps)):
                inp_gpu = inps[j].unsqueeze(0).to(dev).float()
                out = _layer_output(layer, inp_gpu, kw_gpu)
                tgt = target_outputs[j].to(dev).float()
                total += F.mse_loss(out, tgt).item()
                del inp_gpu, tgt, out
        return total / max(len(inps), 1)

    try:
        # --- Training loop ---
        for epoch in range(epochs):
            layer.train()
            total_loss = 0.0
            for j in range(len(inps)):
                optimizer.zero_grad()

                saved_mats = []
                if optimize_binary:
                    for blp, bw in binary_params:
                        saved_mats.append(blp.bit_mat)
                        blp.bit_mat = _smooth_sign_ste(bw, k=k_smooth)

                inp_gpu = inps[j].unsqueeze(0).detach().to(dev).float()
                tgt = target_outputs[j].detach().to(dev).float()
                out = _layer_output(layer, inp_gpu, kw_gpu)
                loss = F.mse_loss(out, tgt)
                loss.backward()

                if optimize_binary:
                    for idx, (blp, _) in enumerate(binary_params):
                        blp.bit_mat = saved_mats[idx]

                optimizer.step()
                total_loss += loss.item()
                del inp_gpu, tgt, out, loss

            avg_loss = total_loss / max(len(inps), 1)

            # --- Periodic hard evaluation (like GPTQ block optimizer) ---
            do_eval = ((epoch + 1) % eval_interval == 0) or (epoch == epochs - 1)
            if do_eval:
                eval_mse = _hard_eval_mse()
                if eval_mse < best_eval_mse:
                    best_eval_mse = eval_mse
                    best_snap = copy.deepcopy(layer.state_dict())
                if optimize_binary:
                    for blp, _ in binary_params:
                        blp.bit_mat = original_bit_mats[id(blp)]
                logger.info(
                    "  [DBF] Epoch %d/%d: train=%.6f, eval=%.6f (best: %.6f)",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    eval_mse,
                    best_eval_mse,
                )
            elif (epoch + 1) % max(1, epochs // 4) == 0 or epoch == 0:
                logger.info(
                    "  [DBF] Epoch %d/%d: train=%.6f",
                    epoch + 1,
                    epochs,
                    avg_loss,
                )

        # --- Restore best state ---
        if best_snap:
            layer.load_state_dict(best_snap)
        elif optimize_binary:
            with torch.no_grad():
                for blp, bw in binary_params:
                    bq = bw.data.sign()
                    bq[bq == 0] = 1
                    blp.bit_mat = bq.to(torch.int8)
                    blp.bp = pack_binary(bq)
    finally:
        # --- Disable gradients on scaling params ---
        for _name, mod in dbf_modules:
            for attr in ("scaling0", "scaling2", "scaling4"):
                p = getattr(mod, attr, None)
                if p is not None:
                    p.requires_grad_(False)

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
        logger.info("[DBF Block-wise] No improvement, rolling back")
        layer.load_state_dict(initial_snap)
        final_error = initial_error

    delta = initial_error - final_error
    pct = (delta / max(initial_error, 1e-10)) * 100
    logger.info(
        "[DBF Block-wise] Final MSE: %.6f (delta: %.6f, %+.1f%%)",
        final_error,
        delta,
        pct,
    )

    del kw_gpu, initial_snap
    return initial_error, final_error
