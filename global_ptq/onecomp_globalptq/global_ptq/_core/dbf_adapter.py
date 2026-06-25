"""DBF differentiable parameter management for global PTQ.

Makes DoubleBinaryLinear layers trainable by exposing scaling parameters
(scaling0, scaling2, scaling4) as optimisable tensors, and optionally
enabling differentiable binary weight optimisation via smooth sign STE.

The approach mirrors the GPTQ adapter: the original ``forward`` method is
replaced with a differentiable version whose computation graph connects
the loss back to every trainable parameter.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

import torch
import torch.nn as nn
from types import MethodType
from typing import Dict, List, Tuple

from .helpers import smooth_sign_ste


_SCALING_ATTRS = ("scaling0", "scaling2", "scaling4")
_BINARY_ATTRS = ("bp1", "bp3")
_BINARY_SHAPES = ("_bp1_shape", "_bp3_shape")
_OPT_BINARY_ATTRS = ("_opt_bp1", "_opt_bp3")

# Round STE (GPTQ) operates on integer weights in [0, 2^bits-1] where k=100
# gives sharp transitions at integer boundaries.  Sign STE (DBF) operates on
# values near ±1, where k must be much smaller to avoid gradient saturation
# in tanh(k*x).  k=2 yields gradient ~0.14 at x=±1.
_BINARY_STE_K = 2.0


# ---------------------------------------------------------------------------
# Finding DBF modules
# ---------------------------------------------------------------------------


def find_dbf_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return all ``DoubleBinaryLinear`` modules as ``(name, module)`` pairs."""
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear
    from .helpers import find_target_modules

    return find_target_modules(model, DoubleBinaryLinear)


# ---------------------------------------------------------------------------
# Differentiable forward
# ---------------------------------------------------------------------------


def _make_dbf_differentiable_forward():
    """Build a differentiable ``forward`` for a ``DoubleBinaryLinear``.

    When ``_opt_bp1`` / ``_opt_bp3`` exist on the module the forward
    passes through :func:`smooth_sign_ste` so that gradients flow to the
    float binary weight tensors.  Otherwise it falls back to the packed
    buffer path (scaling-only optimisation).

    The sign-STE sharpness *k* is read from ``self._binary_ste_k``
    (set during :func:`setup_dbf_differentiable`).
    """

    def differentiable_forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        k = getattr(self, "_binary_ste_k", _BINARY_STE_K)

        x = x * self.scaling0.to(dtype)

        if hasattr(self, "_opt_bp1"):
            b1 = smooth_sign_ste(self._opt_bp1, k=k)
            x = x.matmul(b1.to(dtype).t())
        else:
            b1 = self._unpack_bp(self.bp1, self._bp1_shape)
            x = x.matmul(b1.to(dtype).t())

        x = x * self.scaling2.to(dtype)

        if hasattr(self, "_opt_bp3"):
            b3 = smooth_sign_ste(self._opt_bp3, k=k)
            x = x.matmul(b3.to(dtype).t())
        else:
            b3 = self._unpack_bp(self.bp3, self._bp3_shape)
            x = x.matmul(b3.to(dtype).t())

        x = x * self.scaling4.to(dtype)
        if self.bias is not None:
            x = x + self.bias.to(dtype)
        return x

    return differentiable_forward


# ---------------------------------------------------------------------------
# Parameter setup
# ---------------------------------------------------------------------------


def setup_dbf_differentiable(
    dbf_modules: List[Tuple[str, nn.Module]],
    optimize_binary: bool = False,
) -> Tuple[Dict[str, object], List[torch.Tensor], List[torch.Tensor]]:
    """Make DBF scaling (and optionally binary) parameters trainable.

    The original ``forward`` method of every module is replaced with a
    differentiable version.  Scaling parameters are promoted to float32.
    Binary STE sharpness is set to :data:`_BINARY_STE_K` (see module
    docstring for rationale).

    Returns:
        ``(original_forwards, scaling_params, binary_params)``

        *original_forwards* maps module name to the original forward
        method (for :func:`restore_dbf_original`).
        *scaling_params* is a flat list of float32 ``nn.Parameter``.
        *binary_params* is a flat list of float ``Tensor`` (empty when
        *optimize_binary* is ``False``).
    """
    from onecomp.quantizer.dbf.dbf_layer import unpack_binary

    original_forwards: Dict[str, object] = {}
    scaling_params: List[torch.Tensor] = []
    binary_params: List[torch.Tensor] = []

    for name, mod in dbf_modules:
        for attr in _SCALING_ATTRS:
            param = getattr(mod, attr)
            fp32 = param.data.detach().clone().float()
            new_param = nn.Parameter(fp32, requires_grad=True)
            setattr(mod, attr, new_param)
            scaling_params.append(new_param)

        if optimize_binary:
            for bp_attr, shape_attr, opt_attr in zip(
                _BINARY_ATTRS, _BINARY_SHAPES, _OPT_BINARY_ATTRS,
            ):
                bp_packed = getattr(mod, bp_attr)
                shape = getattr(mod, shape_attr)
                unpacked = unpack_binary(bp_packed)[: shape[0] * shape[1]].reshape(shape)
                bw = unpacked.float().detach().clone()
                bw.requires_grad = True
                setattr(mod, opt_attr, bw)
                binary_params.append(bw)

        original_forwards[name] = mod.forward
        mod._binary_ste_k = _BINARY_STE_K
        mod.forward = MethodType(_make_dbf_differentiable_forward(), mod)

    return original_forwards, scaling_params, binary_params


# ---------------------------------------------------------------------------
# Forward restore / re-install
# ---------------------------------------------------------------------------


def restore_dbf_original(
    dbf_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
    cleanup: bool = False,
) -> None:
    """Restore every module's original ``forward`` method."""
    for name, mod in dbf_modules:
        if name in original_forwards:
            mod.__dict__.pop("forward", None)
            if not hasattr(mod, "forward") or mod.forward != original_forwards[name]:
                mod.forward = original_forwards[name]
        
        if cleanup:
            for attr in ("_opt_bp1", "_opt_bp3", "_binary_ste_k"):
                if hasattr(mod, attr):
                    delattr(mod, attr)


def setup_dbf_forwards_only(
    dbf_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
) -> None:
    """Re-install differentiable forwards for continued training after eval."""
    for name, mod in dbf_modules:
        if name not in original_forwards:
            original_forwards[name] = mod.forward
        mod.forward = MethodType(_make_dbf_differentiable_forward(), mod)


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def write_back_dbf_binary(dbf_modules: List[Tuple[str, nn.Module]]) -> None:
    """Write optimised float binary weights back to packed uint8 buffers."""
    from onecomp.quantizer.dbf.dbf_layer import pack_binary

    with torch.no_grad():
        for _name, mod in dbf_modules:
            for bp_attr, shape_attr, opt_attr in zip(
                _BINARY_ATTRS, _BINARY_SHAPES, _OPT_BINARY_ATTRS,
            ):
                if not hasattr(mod, opt_attr):
                    continue
                weight = getattr(mod, opt_attr)
                q = weight.sign()
                q[q == 0] = 1
                shape = getattr(mod, shape_attr)
                repacked = pack_binary(q.to(torch.int8).reshape(shape))
                buf = getattr(mod, bp_attr)
                buf.copy_(repacked.to(buf.device))


def write_back_dbf_scaling(dbf_modules: List[Tuple[str, nn.Module]]) -> None:
    """Convert float32 optimisation params back to float16 for inference."""
    with torch.no_grad():
        for _name, mod in dbf_modules:
            for attr in _SCALING_ATTRS:
                param = getattr(mod, attr)
                setattr(mod, attr, nn.Parameter(param.data.half(), requires_grad=False))


# ---------------------------------------------------------------------------
# State save / load (for rollback)
# ---------------------------------------------------------------------------


def save_dbf_state(dbf_modules: List[Tuple[str, nn.Module]]) -> Dict:
    """Snapshot scaling parameters and packed binary buffers."""
    state: Dict[str, dict] = {}
    for name, mod in dbf_modules:
        d: dict = {}
        for attr in _SCALING_ATTRS:
            d[attr] = getattr(mod, attr).data.clone()
        for attr in _BINARY_ATTRS:
            d[attr] = getattr(mod, attr).clone()
        state[name] = d
    return state


def load_dbf_state(
    dbf_modules: List[Tuple[str, nn.Module]],
    state: Dict,
) -> None:
    """Restore a previously saved snapshot."""
    for name, mod in dbf_modules:
        if name not in state:
            continue
        for attr in _SCALING_ATTRS:
            if attr in state[name]:
                getattr(mod, attr).data.copy_(state[name][attr])
        for attr in _BINARY_ATTRS:
            if attr in state[name]:
                getattr(mod, attr).copy_(state[name][attr])
