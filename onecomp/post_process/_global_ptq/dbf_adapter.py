"""DBF differentiable parameter management for global PTQ.

Makes DoubleBinaryLinear layers trainable by exposing scaling parameters
(scaling0, scaling2, scaling4) as optimisable tensors.

The approach mirrors the GPTQ adapter: the original ``forward`` method is
replaced with a differentiable version whose computation graph connects
the loss back to every trainable parameter.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

from types import MethodType
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

_SCALING_ATTRS = ("scaling0", "scaling2", "scaling4")


# ---------------------------------------------------------------------------
# Finding DBF modules
# ---------------------------------------------------------------------------


def find_dbf_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return all ``DoubleBinaryLinear`` modules as ``(name, module)`` pairs."""
    from ...quantizer.dbf.dbf_layer import DoubleBinaryLinear

    return [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, DoubleBinaryLinear)
    ]


# ---------------------------------------------------------------------------
# Differentiable forward
# ---------------------------------------------------------------------------


def _make_dbf_differentiable_forward():
    """Build a differentiable ``forward`` for a ``DoubleBinaryLinear``.

    Binary weights are read from the packed buffers (no gradient).
    Scaling parameters (``scaling0``, ``scaling2``, ``scaling4``) are
    ``nn.Parameter`` objects so that gradients flow through the scaling
    arithmetic.
    """

    def differentiable_forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype

        x = x * self.scaling0.to(dtype)

        b1 = self._unpack_bp(self.bp1, self._bp1_shape)
        x = x.matmul(b1.to(dtype).t())

        x = x * self.scaling2.to(dtype)

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
) -> Tuple[Dict[str, object], List[torch.Tensor]]:
    """Make DBF scaling parameters trainable.

    The original ``forward`` method of every module is replaced with a
    differentiable version.  Scaling parameters are promoted to float32.

    Returns:
        ``(original_forwards, scaling_params)``

        *original_forwards* maps module name to the original forward
        method (for :func:`restore_dbf_original`).
        *scaling_params* is a flat list of float32 ``nn.Parameter``.
    """
    original_forwards: Dict[str, object] = {}
    scaling_params: List[torch.Tensor] = []

    for name, mod in dbf_modules:
        for attr in _SCALING_ATTRS:
            param = getattr(mod, attr)
            fp32 = param.data.detach().clone().float()
            new_param = nn.Parameter(fp32, requires_grad=True)
            setattr(mod, attr, new_param)
            scaling_params.append(new_param)

        original_forwards[name] = mod.forward
        mod.forward = MethodType(_make_dbf_differentiable_forward(), mod)

    return original_forwards, scaling_params


# ---------------------------------------------------------------------------
# Forward restore / re-install
# ---------------------------------------------------------------------------


def restore_dbf_original(
    dbf_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
) -> None:
    """Restore every module's original ``forward`` method."""
    for name, mod in dbf_modules:
        if name in original_forwards:
            mod.forward = original_forwards[name]


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
    """Snapshot scaling parameters for later rollback."""
    state: Dict[str, dict] = {}
    for name, mod in dbf_modules:
        d: dict = {}
        for attr in _SCALING_ATTRS:
            d[attr] = getattr(mod, attr).data.clone()
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
