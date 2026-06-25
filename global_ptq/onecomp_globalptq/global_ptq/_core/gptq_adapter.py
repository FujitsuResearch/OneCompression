"""GPTQ differentiable forward and parameter management for global PTQ.

Makes GPTQLinear layers trainable by replacing their forward pass with a
differentiable version that exposes ``scales`` and ``zeros`` (and optionally
integer weights) as ``nn.Parameter`` objects so that gradients can flow
through the dequantization arithmetic.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from typing import Dict, List, Tuple

from .helpers import smooth_ste_round


# ---------------------------------------------------------------------------
# Internal: unpack helpers
# ---------------------------------------------------------------------------

def _get_int_weights(mod: nn.Module) -> torch.Tensor:
    """Return integer weight matrix (out_features, in_features) as INT32."""
    if not getattr(mod, "_weight_is_packed", False):
        return mod.qweight

    from onecomp.quantizer.gptq.gptq_layer import unpack_int_weights
    return unpack_int_weights(
        mod.qweight, mod.wbits, (mod.out_features, mod.in_features),
    )


def _get_float_zeros(mod: nn.Module) -> torch.Tensor:
    """Return zero points (num_groups, out_features) as float, offset-corrected."""
    is_v1 = getattr(mod, "checkpoint_format", "gptq") != "gptq_v2"
    if not getattr(mod, "_weight_is_packed", False):
        raw = mod.qzeros
    else:
        from onecomp.quantizer.gptq.gptq_layer import unpack_zeros
        raw = unpack_zeros(mod.qzeros, mod.wbits, mod.out_features)
    return (raw + 1).float() if is_v1 else raw.float()


def _pack_and_write_back_zeros(mod: nn.Module, zeros_float: torch.Tensor) -> None:
    """Write optimised float zeros back to GPTQLinear buffers."""
    is_v1 = getattr(mod, "checkpoint_format", "gptq") != "gptq_v2"
    zero_int = zeros_float.round().to(torch.int32)
    if is_v1:
        zero_int = zero_int - 1

    if not getattr(mod, "_weight_is_packed", False):
        mod.qzeros.copy_(zero_int)
    else:
        from onecomp.quantizer.gptq.gptq_layer import pack_zeros
        mod.qzeros.copy_(pack_zeros(zero_int, mod.wbits).to(mod.qzeros.device))


# ---------------------------------------------------------------------------
# Finding GPTQ modules
# ---------------------------------------------------------------------------

def find_gptq_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return all ``GPTQLinear`` modules as ``(name, module)`` pairs."""
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    from .helpers import find_target_modules
    return find_target_modules(model, GPTQLinear)


# ---------------------------------------------------------------------------
# Differentiable forward factory
# ---------------------------------------------------------------------------

def _make_differentiable_forward(
    mod: nn.Module,
    use_intweight_param: bool = False,
):
    """Build a differentiable ``forward`` for a ``GPTQLinear`` module.

    The returned function dequantises weights using ``_opt_scales`` and
    ``_opt_zeros`` (and optionally ``_opt_intweight``) which are
    ``nn.Parameter`` objects attached by :func:`setup_gptq_differentiable`.
    """

    def differentiable_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        # Upcast to float32 for numerically stable dequantisation arithmetic.
        # The output is cast back to the input dtype at the end of this function,
        # so mixed-precision (bf16/fp16) autocast still applies to the rest of
        # the model graph.
        x_2d = x.reshape(-1, self.in_features).float()

        if use_intweight_param and hasattr(self, "_opt_intweight"):
            intweight_float = smooth_ste_round(
                self._opt_intweight.float(),
                min_val=0,
                max_val=(1 << self.wbits) - 1,
                k=getattr(self, "_ste_k", 10.0),
            )
        else:
            intweight_float = _get_int_weights(self).detach().float()

        scales = self._opt_scales.float()
        zeros = self._opt_zeros.float()

        scale_expanded = scales[self.g_idx, :].T
        zero_expanded = zeros[self.g_idx, :].T

        weight_dequant = scale_expanded * (intweight_float - zero_expanded)

        bias_f = self.bias.float() if self.bias is not None else None
        out = F.linear(x_2d, weight_dequant, bias_f)
        return out.reshape(*orig_shape[:-1], self.out_features).to(x.dtype)

    return differentiable_forward


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def setup_gptq_differentiable(
    gptq_modules: List[Tuple[str, nn.Module]],
    dev: torch.device,
    optimize_intweight: bool = False,
    ste_k: float = 10.0,
) -> Tuple[Dict[str, object], List[nn.Parameter], List[nn.Parameter]]:
    """Replace each GPTQLinear's forward with a differentiable version.

    Returns:
        (original_forwards, scaling_params, intweight_params)
    """
    original_forwards: Dict[str, object] = {}
    scaling_params: List[nn.Parameter] = []
    intweight_params: List[nn.Parameter] = []

    for name, mod in gptq_modules:
        mod._opt_scales = nn.Parameter(mod.scales.clone().float().to(dev))
        mod._opt_zeros = nn.Parameter(_get_float_zeros(mod).to(dev))
        scaling_params.extend([mod._opt_scales, mod._opt_zeros])

        if optimize_intweight:
            iw = _get_int_weights(mod)
            mod._opt_intweight = nn.Parameter(iw.float().to(dev))
            mod._ste_k = ste_k
            intweight_params.append(mod._opt_intweight)

        original_forwards[name] = mod.forward
        mod.forward = MethodType(
            _make_differentiable_forward(mod, use_intweight_param=optimize_intweight),
            mod,
        )

    return original_forwards, scaling_params, intweight_params


def write_back_gptq_params(
    gptq_modules: List[Tuple[str, nn.Module]],
    optimize_intweight: bool = False,
) -> None:
    """Copy optimised ``_opt_*`` parameters back into GPTQLinear buffers."""
    for _name, mod in gptq_modules:
        mod.scales.copy_(mod._opt_scales.data.to(mod.scales.dtype))
        _pack_and_write_back_zeros(mod, mod._opt_zeros.data)

        if optimize_intweight and hasattr(mod, "_opt_intweight"):
            iw = (
                mod._opt_intweight.data
                .round()
                .clamp(0, (1 << mod.wbits) - 1)
                .to(torch.int32)
            )
            if not getattr(mod, "_weight_is_packed", False):
                mod.qweight.copy_(iw)
            else:
                from onecomp.quantizer.gptq.gptq_layer import pack_int_weights
                mod.qweight.copy_(pack_int_weights(iw, mod.wbits).to(mod.qweight.device))


def restore_gptq_original(
    gptq_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
    cleanup: bool = False,
) -> None:
    """Restore the original (non-differentiable) forward methods.

    Args:
        cleanup: If ``True``, also remove ``_opt_scales`` / ``_opt_zeros``
            / ``_opt_intweight`` so they do not bloat ``state_dict()`` or
            appear in ``parameters()`` after Global PTQ.  Must be
            ``False`` during training when
            :func:`setup_gptq_forwards_only` will be called again.
    """
    for name, mod in gptq_modules:
        if name in original_forwards:
            # Restore original forward.  We use __dict__.pop to safely remove
            # the instance-level 'forward' method (MethodType) and fall back
            # to the class-level forward, or restore the saved bound method.
            mod.__dict__.pop("forward", None)
            if not hasattr(mod, "forward") or mod.forward != original_forwards[name]:
                mod.forward = original_forwards[name]

        if cleanup:
            for attr in ("_opt_scales", "_opt_zeros", "_opt_intweight", "_ste_k"):
                if hasattr(mod, attr):
                    delattr(mod, attr)


def setup_gptq_forwards_only(
    gptq_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
    optimize_intweight: bool = False,
) -> None:
    """Re-bind differentiable forwards (keeping existing ``_opt_*`` params)."""
    for name, mod in gptq_modules:
        if name not in original_forwards:
            continue
        mod.forward = MethodType(
            _make_differentiable_forward(mod, use_intweight_param=optimize_intweight),
            mod,
        )


def save_gptq_state(gptq_modules: List[Tuple[str, nn.Module]]) -> Dict:
    """Snapshot ``scales``, ``qzeros``, ``qweight`` for later rollback."""
    return {
        name: {k: getattr(mod, k).data.clone() for k in ("scales", "qzeros", "qweight")}
        for name, mod in gptq_modules
    }


def load_gptq_state(
    gptq_modules: List[Tuple[str, nn.Module]],
    state: Dict,
) -> None:
    """Restore a previously saved snapshot."""
    for name, mod in gptq_modules:
        if name not in state:
            continue
        for k in ("scales", "qzeros", "qweight"):
            getattr(mod, k).copy_(state[name][k])
