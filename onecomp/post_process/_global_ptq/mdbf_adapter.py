"""
Differentiable parameter management for MDBF global PTQ.

Copyright 2025-2026 Fujitsu Ltd.
"""

from types import MethodType
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .helpers import smooth_sign_ste

_AMP_ATTRS = ("A_amp", "B_amp", "Q_U_amp", "Q_V_amp")
_BINARY_SIGN_NAMES = ("A", "B")
_BINARY_STE_K = 2.0


def find_mdbf_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return all MDBF modules as ``(name, module)`` pairs."""
    from ...quantizer.mdbf.mdbf_layer import MultipathMDBFLinear

    return [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, MultipathMDBFLinear)
    ]


def _make_mdbf_differentiable_forward():
    from ...quantizer.mdbf.mdbf_layer import unpack_binary

    def differentiable_forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        sharpness = getattr(self, "_binary_ste_k", _BINARY_STE_K)
        output = None
        for path in self.paths:
            amplitudes = {
                attr: getattr(path, f"_opt_{attr}", getattr(path, attr)).to(dtype)
                for attr in _AMP_ATTRS
            }
            if hasattr(path, "_opt_A_sign"):
                a_sign = smooth_sign_ste(path._opt_A_sign, k=sharpness).to(dtype)
            else:
                a_sign = unpack_binary(path._packed_sign("A", x.device), (path.n, path.r)).to(
                    dtype
                )
            if hasattr(path, "_opt_B_sign"):
                b_sign = smooth_sign_ste(path._opt_B_sign, k=sharpness).to(dtype)
            else:
                b_sign = unpack_binary(path._packed_sign("B", x.device), (path.r, path.m)).to(
                    dtype
                )

            factor_a = a_sign * (amplitudes["A_amp"] @ amplitudes["Q_U_amp"].T)
            factor_b = b_sign * (amplitudes["Q_V_amp"] @ amplitudes["B_amp"].T)
            path_output = x @ factor_b.T @ factor_a.T
            output = path_output if output is None else output + path_output

        if self.bias is not None:
            output = output + self.bias.to(dtype)
        return output

    return differentiable_forward


def setup_mdbf_differentiable(
    mdbf_modules: List[Tuple[str, nn.Module]],
    optimize_binary: bool = False,
    ste_k: float = _BINARY_STE_K,
) -> Tuple[Dict[str, object], List[torch.Tensor], List[torch.Tensor]]:
    """Expose MDBF amplitudes and optionally binary signs for optimisation."""
    from ...quantizer.mdbf.mdbf_layer import unpack_binary

    original_forwards: Dict[str, object] = {}
    amplitude_params: List[torch.Tensor] = []
    binary_params: List[torch.Tensor] = []
    try:
        for name, module in mdbf_modules:
            original_forwards[name] = module.forward
            module._global_ptq_original_forward = module.forward
            for path in module.paths:
                for attr in _AMP_ATTRS:
                    parameter = nn.Parameter(getattr(path, attr).data.detach().clone().float())
                    setattr(path, f"_opt_{attr}", parameter)
                    amplitude_params.append(parameter)
                if optimize_binary:
                    for sign in _BINARY_SIGN_NAMES:
                        shape = (path.n, path.r) if sign == "A" else (path.r, path.m)
                        packed_key = f"{sign}_sign_packed"
                        packed = path._buffers.get(packed_key)
                        if packed is None:
                            packed = path._packed_cpu.get(sign)
                        if packed is None:
                            continue
                        unpacked = (
                            unpack_binary(packed.to(path.A_amp.device), shape)
                            .float()
                            .detach()
                            .clone()
                        )
                        parameter = nn.Parameter(unpacked)
                        setattr(path, f"_opt_{sign}_sign", parameter)
                        binary_params.append(parameter)
            module._binary_ste_k = ste_k
            module.forward = MethodType(_make_mdbf_differentiable_forward(), module)
    except Exception:
        restore_mdbf_original(mdbf_modules, original_forwards, cleanup=True)
        raise
    return original_forwards, amplitude_params, binary_params


def restore_mdbf_original(
    mdbf_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
    cleanup: bool = False,
) -> None:
    """Restore original forwards and optionally remove optimisation parameters."""
    for name, module in mdbf_modules:
        original_forward = original_forwards.get(
            name,
            getattr(module, "_global_ptq_original_forward", None),
        )
        if original_forward is not None:
            module.__dict__.pop("forward", None)
            module.forward = original_forward
        if cleanup:
            if hasattr(module, "_binary_ste_k"):
                delattr(module, "_binary_ste_k")
            if hasattr(module, "_global_ptq_original_forward"):
                delattr(module, "_global_ptq_original_forward")
            for path in module.paths:
                for attr in _AMP_ATTRS + tuple(f"{sign}_sign" for sign in _BINARY_SIGN_NAMES):
                    opt_attr = f"_opt_{attr}"
                    if hasattr(path, opt_attr):
                        delattr(path, opt_attr)


def setup_mdbf_forwards_only(
    mdbf_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
) -> None:
    """Re-install differentiable forwards after an evaluation pass."""
    for name, module in mdbf_modules:
        if name not in original_forwards:
            original_forwards[name] = module.forward
        module.forward = MethodType(_make_mdbf_differentiable_forward(), module)


def _refresh_gemlite_sign_kernels(path: nn.Module) -> None:
    if not getattr(path, "use_gemlite", False) and not getattr(path, "_gemlite_layers", None):
        return
    enable_gemlite = getattr(path, "enable_gemlite", None)
    if enable_gemlite is None:
        return
    path._gemlite_layers = {}
    path.use_gemlite = False
    enable_gemlite(device=path.A_amp.device, force=True)


def write_back_mdbf_binary(mdbf_modules: List[Tuple[str, nn.Module]]) -> None:
    """Write optimised float sign matrices back to packed buffers."""
    from ...quantizer.mdbf.mdbf_layer import pack_binary

    with torch.no_grad():
        for _name, module in mdbf_modules:
            for path in module.paths:
                changed = False
                for sign in _BINARY_SIGN_NAMES:
                    opt_attr = f"_opt_{sign}_sign"
                    if not hasattr(path, opt_attr):
                        continue
                    values = getattr(path, opt_attr).sign()
                    values[values == 0] = 1
                    packed, _ = pack_binary(values.to(torch.int8))
                    packed_key = f"{sign}_sign_packed"
                    if packed_key in path._buffers:
                        path._buffers[packed_key].copy_(packed)
                    elif sign in path._packed_cpu:
                        path._packed_cpu[sign].copy_(packed.cpu())
                    changed = True
                if changed:
                    _refresh_gemlite_sign_kernels(path)


def write_back_mdbf_amp(mdbf_modules: List[Tuple[str, nn.Module]]) -> None:
    """Copy optimised float32 amplitudes back to inference buffers."""
    with torch.no_grad():
        for _name, module in mdbf_modules:
            for path in module.paths:
                for attr in _AMP_ATTRS:
                    opt_attr = f"_opt_{attr}"
                    if hasattr(path, opt_attr):
                        getattr(path, attr).copy_(getattr(path, opt_attr).data.half())


def save_mdbf_state(mdbf_modules: List[Tuple[str, nn.Module]]) -> Dict:
    """Snapshot MDBF amplitudes and packed signs."""
    state: Dict[str, dict] = {}
    for name, module in mdbf_modules:
        paths_state = {}
        for index, path in enumerate(module.paths):
            values = {attr: getattr(path, attr).data.clone() for attr in _AMP_ATTRS}
            for sign in _BINARY_SIGN_NAMES:
                packed_key = f"{sign}_sign_packed"
                if packed_key in path._buffers:
                    values[packed_key] = path._buffers[packed_key].clone()
                elif sign in path._packed_cpu:
                    values[packed_key] = path._packed_cpu[sign].clone()
            paths_state[index] = values
        state[name] = paths_state
    return state


def load_mdbf_state(mdbf_modules: List[Tuple[str, nn.Module]], state: Dict) -> None:
    """Restore a previously saved MDBF snapshot."""
    with torch.no_grad():
        for name, module in mdbf_modules:
            if name not in state:
                continue
            for index, path in enumerate(module.paths):
                values = state[name].get(index, {})
                for attr in _AMP_ATTRS:
                    if attr in values:
                        getattr(path, attr).copy_(values[attr])
                signs_changed = False
                for sign in _BINARY_SIGN_NAMES:
                    packed_key = f"{sign}_sign_packed"
                    if packed_key not in values:
                        continue
                    if packed_key in path._buffers:
                        path._buffers[packed_key].copy_(values[packed_key])
                    elif sign in path._packed_cpu:
                        path._packed_cpu[sign].copy_(values[packed_key].cpu())
                    signs_changed = True
                if signs_changed:
                    _refresh_gemlite_sign_kernels(path)
