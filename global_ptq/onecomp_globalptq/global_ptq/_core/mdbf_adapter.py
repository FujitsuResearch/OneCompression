"""MDBF differentiable parameter management for global PTQ.

Makes MultipathMDBFLinear layers trainable by exposing amplitude parameters
(A_amp, B_amp, Q_U_amp, Q_V_amp) per path as optimisable tensors, and
optionally enabling differentiable binary-sign optimisation via smooth sign STE.

Architecture recap (per MDBFLinear path):
    F = A_sign * (A_amp @ Q_U_amp^T)   shape: (n, r)
    G = B_sign * (Q_V_amp @ B_amp^T)   shape: (r, m)
    y = x @ G^T @ F^T

Trainable (continuous):
    A_amp, B_amp, Q_U_amp, Q_V_amp  — amplitude/scale factors per path.
Trainable (discrete, opt-in):
    A_sign, B_sign  — ±1 binary factor matrices per path, via sign STE.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Keiji Kimura

"""

from types import MethodType
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .helpers import smooth_sign_ste

_AMP_ATTRS = ("A_amp", "B_amp", "Q_U_amp", "Q_V_amp")
_BINARY_SIGN_NAMES = ("A", "B")

# Sharpness for sign-STE: same rationale as DBF adapter (values near ±1,
# tanh saturation avoided by using k=2 instead of the GPTQ default k=100).
_BINARY_STE_K = 2.0


# ---------------------------------------------------------------------------
# Finding MDBF modules
# ---------------------------------------------------------------------------


def find_mdbf_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return all ``MultipathMDBFLinear`` modules as ``(name, module)`` pairs."""
    from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear

    from .helpers import find_target_modules

    return find_target_modules(model, MultipathMDBFLinear)


# ---------------------------------------------------------------------------
# Differentiable forward (per MDBFLinear path)
# ---------------------------------------------------------------------------


def _make_mdbf_differentiable_forward():
    """Build a differentiable ``forward`` for a ``MultipathMDBFLinear``.

    Each path's computation graph is reconstructed from the (possibly
    optimisable) amplitude parameters and, when ``_opt_A_sign_{p}`` /
    ``_opt_B_sign_{p}`` exist, through :func:`smooth_sign_ste` so that
    gradients flow to the float sign-weight tensors as well.
    """
    from onecomp.quantizer.mdbf.mdbf_layer import unpack_binary

    def differentiable_forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        k = getattr(self, "_binary_ste_k", _BINARY_STE_K)

        y = None
        for path in self.paths:
            # ---- amplitude parameters (always trainable) ----
            A_amp = getattr(path, "_opt_A_amp", path.A_amp).to(dtype)
            B_amp = getattr(path, "_opt_B_amp", path.B_amp).to(dtype)
            Q_U_amp = getattr(path, "_opt_Q_U_amp", path.Q_U_amp).to(dtype)
            Q_V_amp = getattr(path, "_opt_Q_V_amp", path.Q_V_amp).to(dtype)

            # ---- sign matrices (STE or packed buffer) ----
            if hasattr(path, "_opt_A_sign"):
                A_sign = smooth_sign_ste(path._opt_A_sign, k=k).to(dtype)
            else:
                A_sign = unpack_binary(path._packed_sign("A", x.device), (path.n, path.r)).to(
                    dtype
                )

            if hasattr(path, "_opt_B_sign"):
                B_sign = smooth_sign_ste(path._opt_B_sign, k=k).to(dtype)
            else:
                B_sign = unpack_binary(path._packed_sign("B", x.device), (path.r, path.m)).to(
                    dtype
                )

            # F = A_sign * (A_amp @ Q_U_amp^T)  shape: (n, r)
            F = A_sign * (A_amp @ Q_U_amp.T)
            # G = B_sign * (Q_V_amp @ B_amp^T)  shape: (r, m)
            G = B_sign * (Q_V_amp @ B_amp.T)

            # y += x @ G^T @ F^T
            path_out = x @ G.T @ F.T

            y = path_out if y is None else y + path_out

        if self.bias is not None:
            y = y + self.bias.to(dtype)
        return y

    return differentiable_forward


# ---------------------------------------------------------------------------
# Parameter setup
# ---------------------------------------------------------------------------


def setup_mdbf_differentiable(
    mdbf_modules: List[Tuple[str, nn.Module]],
    optimize_binary: bool = False,
    ste_k: float = _BINARY_STE_K,
) -> Tuple[Dict[str, object], List[torch.Tensor], List[torch.Tensor]]:
    """Make MDBF amplitude (and optionally sign) parameters trainable.

    For each ``MultipathMDBFLinear`` module, the original ``forward`` is
    replaced with a differentiable version.  Amplitude parameters
    (A_amp, B_amp, Q_U_amp, Q_V_amp) of every path are promoted to
    float32 ``nn.Parameter`` objects stored as ``_opt_*`` attributes on the
    individual ``MDBFLinear`` path modules.

    Args:
        mdbf_modules: List of ``(name, module)`` pairs from
            :func:`find_mdbf_modules`.
        optimize_binary: When True, also expose unpacked ±1 sign matrices
            as float tensors with ``requires_grad=True`` so that gradients
            flow through :func:`smooth_sign_ste`.
        ste_k: Sharpness for binary sign STE (``tanh(k*x)`` backward).
            Default is :data:`_BINARY_STE_K` (2.0).

    Returns:
        ``(original_forwards, amp_params, binary_params)``

        *original_forwards* maps module name → original forward (for restore).
        *amp_params* is a flat list of float32 ``nn.Parameter`` objects.
        *binary_params* is a flat list of float tensors (empty when
        *optimize_binary* is ``False``).
    """
    from onecomp.quantizer.mdbf.mdbf_layer import unpack_binary

    original_forwards: Dict[str, object] = {}
    amp_params: List[torch.Tensor] = []
    binary_params: List[torch.Tensor] = []

    for name, mod in mdbf_modules:
        for path in mod.paths:
            # ---- continuous amplitude parameters ----
            for attr in _AMP_ATTRS:
                buf = getattr(path, attr)
                fp32 = buf.data.detach().clone().float()
                new_param = nn.Parameter(fp32, requires_grad=True)
                setattr(path, f"_opt_{attr}", new_param)
                amp_params.append(new_param)

            # ---- discrete sign parameters (optional) ----
            if optimize_binary:
                for which in _BINARY_SIGN_NAMES:
                    shape = (path.n, path.r) if which == "A" else (path.r, path.m)
                    packed_key = f"{which}_sign_packed"
                    packed = path._buffers.get(packed_key)
                    if packed is None:
                        # GemLite mode: stashed on CPU
                        packed = path._packed_cpu.get(which)
                    if packed is None:
                        continue
                    unpacked = unpack_binary(packed, shape).float().detach().clone()
                    new_param = nn.Parameter(unpacked, requires_grad=True)
                    setattr(path, f"_opt_{which}_sign", new_param)
                    binary_params.append(new_param)

        original_forwards[name] = mod.forward
        mod._binary_ste_k = float(ste_k)
        mod.forward = MethodType(_make_mdbf_differentiable_forward(), mod)

    return original_forwards, amp_params, binary_params


# ---------------------------------------------------------------------------
# Forward restore / re-install
# ---------------------------------------------------------------------------


def restore_mdbf_original(
    mdbf_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
    cleanup: bool = False,
) -> None:
    """Restore every module's original ``forward`` method."""
    for name, mod in mdbf_modules:
        if name in original_forwards:
            mod.__dict__.pop("forward", None)
            if not hasattr(mod, "forward") or mod.forward != original_forwards[name]:
                mod.forward = original_forwards[name]

        if cleanup:
            if hasattr(mod, "_binary_ste_k"):
                delattr(mod, "_binary_ste_k")
            for path in mod.paths:
                for attr in _AMP_ATTRS:
                    opt_attr = f"_opt_{attr}"
                    if hasattr(path, opt_attr):
                        delattr(path, opt_attr)
                for which in _BINARY_SIGN_NAMES:
                    opt_attr = f"_opt_{which}_sign"
                    if hasattr(path, opt_attr):
                        delattr(path, opt_attr)


def setup_mdbf_forwards_only(
    mdbf_modules: List[Tuple[str, nn.Module]],
    original_forwards: Dict[str, object],
) -> None:
    """Re-install differentiable forwards for continued training after eval."""
    for name, mod in mdbf_modules:
        if name not in original_forwards:
            original_forwards[name] = mod.forward
        mod.forward = MethodType(_make_mdbf_differentiable_forward(), mod)


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


def write_back_mdbf_binary(mdbf_modules: List[Tuple[str, nn.Module]]) -> None:
    """Write optimised float sign tensors back to packed uint8 buffers."""
    from onecomp.quantizer.mdbf.mdbf_layer import pack_binary

    with torch.no_grad():
        for _name, mod in mdbf_modules:
            for path in mod.paths:
                for which in _BINARY_SIGN_NAMES:
                    opt_attr = f"_opt_{which}_sign"
                    if not hasattr(path, opt_attr):
                        continue
                    w = getattr(path, opt_attr)
                    q = w.sign()
                    q[q == 0] = 1
                    shape = (path.n, path.r) if which == "A" else (path.r, path.m)
                    packed, _ = pack_binary(q.to(torch.int8).reshape(shape))
                    buf_key = f"{which}_sign_packed"
                    if buf_key in path._buffers:
                        path._buffers[buf_key].copy_(packed.to(path._buffers[buf_key].device))
                    elif which in path._packed_cpu:
                        path._packed_cpu[which].copy_(packed.cpu())


def write_back_mdbf_amp(mdbf_modules: List[Tuple[str, nn.Module]]) -> None:
    """Copy float32 optimised amp params back to fp16 buffers for inference."""
    with torch.no_grad():
        for _name, mod in mdbf_modules:
            for path in mod.paths:
                for attr in _AMP_ATTRS:
                    opt_attr = f"_opt_{attr}"
                    if not hasattr(path, opt_attr):
                        continue
                    opt_param = getattr(path, opt_attr)
                    buf = getattr(path, attr)
                    buf.copy_(opt_param.data.half())


# ---------------------------------------------------------------------------
# State save / load (for rollback)
# ---------------------------------------------------------------------------


def save_mdbf_state(mdbf_modules: List[Tuple[str, nn.Module]]) -> Dict:
    """Snapshot amplitude buffers and packed sign buffers."""
    state: Dict[str, dict] = {}
    for name, mod in mdbf_modules:
        paths_state = {}
        for p, path in enumerate(mod.paths):
            d: dict = {}
            for attr in _AMP_ATTRS:
                d[attr] = getattr(path, attr).data.clone()
            for which in _BINARY_SIGN_NAMES:
                buf_key = f"{which}_sign_packed"
                if buf_key in path._buffers:
                    d[buf_key] = path._buffers[buf_key].clone()
                elif which in path._packed_cpu:
                    d[buf_key] = path._packed_cpu[which].clone()
            paths_state[p] = d
        state[name] = paths_state
    return state


def load_mdbf_state(
    mdbf_modules: List[Tuple[str, nn.Module]],
    state: Dict,
) -> None:
    """Restore a previously saved snapshot."""
    with torch.no_grad():
        for name, mod in mdbf_modules:
            if name not in state:
                continue
            paths_state = state[name]
            for p, path in enumerate(mod.paths):
                if p not in paths_state:
                    continue
                d = paths_state[p]
                for attr in _AMP_ATTRS:
                    if attr in d:
                        getattr(path, attr).copy_(d[attr])
                for which in _BINARY_SIGN_NAMES:
                    buf_key = f"{which}_sign_packed"
                    if buf_key not in d:
                        continue
                    if buf_key in path._buffers:
                        path._buffers[buf_key].copy_(d[buf_key])
                    elif which in path._packed_cpu:
                        path._packed_cpu[which].copy_(d[buf_key].cpu())
