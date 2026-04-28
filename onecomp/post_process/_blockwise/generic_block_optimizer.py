"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

"""

"""Generic (fallback) block-wise optimiser.

Optimises all floating-point nn.Parameters in the block via Adam + MSE
distillation against the FP16 teacher output.

Ported from qep-dev's optimize_generic_block() with minor adaptations:
  - Uses helpers from this package instead of qep-dev imports.
  - Works for any quantiser type as a no-knowledge baseline.

Key difference from qep-dev:
  transformers 5.x passes complex objects (DynamicCache, etc.) in
  layer_kwargs.  _sanitize_layer_kwargs strips them to prevent
  computation-graph leaks across training iterations.
"""

from logging import getLogger
from typing import Tuple

import torch
import torch.nn.functional as F

from .helpers import layer_kwargs_to_device

logger = getLogger(__name__)


def _sanitize_layer_kwargs(layer_kwargs: dict) -> dict:
    """Clone tensors and drop cache objects from layer_kwargs.

    transformers >=5.x passes DynamicCache / StaticCache objects that
    accumulate graph-connected KV state across forward calls.  We drop
    them because block-level optimisation does not need KV caching.
    All other values (including unknown types) are kept as-is.
    """
    try:
        from transformers.cache_utils import Cache as _CacheBase
    except ImportError:
        _CacheBase = None

    clean = {}
    for k, v in layer_kwargs.items():
        if k in ("past_key_value", "past_key_values"):
            continue
        if _CacheBase is not None and isinstance(v, _CacheBase):
            continue
        if isinstance(v, torch.Tensor):
            clean[k] = v.detach().clone()
        elif isinstance(v, tuple) and v and isinstance(v[0], torch.Tensor):
            clean[k] = tuple(t.detach().clone() for t in v)
        else:
            clean[k] = v
    return clean


def optimize_generic_block(
    layer,
    inps,
    target_outputs,
    layer_kwargs: dict,
    lr: float = 1e-4,
    epochs: int = 10,
    dev: torch.device = None,
) -> Tuple[float, float]:
    """Optimise a single block using all float Parameters.

    Returns (initial_mse, final_mse).
    """
    if dev is None:
        dev = next(layer.parameters()).device

    safe_kw = _sanitize_layer_kwargs(layer_kwargs)
    kw_gpu = layer_kwargs_to_device(safe_kw, dev)

    def _forward(inp_gpu):
        raw = layer(inp_gpu, **kw_gpu)
        out = raw[0] if isinstance(raw, tuple) else raw
        if out.dim() == 3 and out.size(0) == 1:
            out = out.squeeze(0)
        return out

    # --- Initial MSE ---
    with torch.no_grad():
        initial_error = 0.0
        for j in range(len(inps)):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            out = _forward(inp_gpu)
            tgt = target_outputs[j].to(dev)
            initial_error += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, out, tgt
        initial_error /= max(len(inps), 1)

    # --- Collect optimisable parameters ---
    params_to_optimize = []
    for _name, param in layer.named_parameters():
        if param.dtype.is_floating_point:
            param.requires_grad_(True)
            params_to_optimize.append(param)
        else:
            param.requires_grad_(False)

    if not params_to_optimize:
        logger.info("[Generic] No optimisable parameters found — skipping")
        return initial_error, initial_error

    logger.info(
        "[Generic] Optimising %d parameters, lr=%g, epochs=%d",
        len(params_to_optimize),
        lr,
        epochs,
    )
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    # --- Training loop ---
    for epoch in range(epochs):
        layer.train()
        total_loss = 0.0
        for j in range(len(inps)):
            optimizer.zero_grad()
            inp_gpu = inps[j].unsqueeze(0).detach().to(dev)
            tgt = target_outputs[j].detach().to(dev)
            out = _forward(inp_gpu)
            loss = F.mse_loss(out.float(), tgt.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del inp_gpu, tgt, out, loss
        avg_loss = total_loss / max(len(inps), 1)
        if (epoch + 1) % max(1, epochs // 4) == 0 or epoch == epochs - 1:
            logger.info(
                "  [Generic] Epoch %d/%d: loss = %.6f",
                epoch + 1,
                epochs,
                avg_loss,
            )

    # --- Final MSE ---
    layer.eval()
    with torch.no_grad():
        final_error = 0.0
        for j in range(len(inps)):
            inp_gpu = inps[j].unsqueeze(0).to(dev)
            tgt = target_outputs[j].to(dev)
            out = _forward(inp_gpu)
            final_error += F.mse_loss(out.float(), tgt.float()).item()
            del inp_gpu, tgt, out
        final_error /= max(len(inps), 1)

    for param in params_to_optimize:
        param.requires_grad_(False)

    del kw_gpu
    return initial_error, final_error
