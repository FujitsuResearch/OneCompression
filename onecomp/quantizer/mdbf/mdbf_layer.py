"""
MDBF (Multi-scale Double Binary Factorization) Layer implementation

Constructs an efficient inference layer from MDBF parameters.
Based on the DBF implementation, achieves bit-packing and memory efficiency.

Structure:
- MDBFLinear: MDBF inference layer for a single pass
- MultipathMDBFLinear: MDBF inference layer for P passes
- Packing/Unpacking: Compresses sign matrices to 1-bit

Weight representation:
    W ≈ Σ_{p=1}^{P} W^{(p)}
    W^{(p)} = F^{(p)} @ G^{(p)}
    where F = S_A * (A_amp @ Q_U_amp^T)
          G = S_B * (Q_V_amp @ B_amp^T)

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .initialize import MDBFParams

# Optional GemLite integration (mirror of dbf/dbf_layer.py)
try:
    from onecomp.quantizer.gemlite import create_gemlite_linear, is_gemlite_available

    HAS_GEMLITE_SUPPORT = True
except ImportError:
    HAS_GEMLITE_SUPPORT = False

# =============================================================================
# Bit-packing/Unpacking
# =============================================================================


def pack_binary(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Convert ±1 to {0,1} and pack into uint8 with 8:1 ratio. Pad the end with +1.

    Args:
        x: Tensor with ±1 values (any shape)

    Returns:
        (packed, original_shape): Packed uint8 tensor and original shape
    """
    original_shape = x.shape
    flat = (x.flatten() >= 0).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = F.pad(flat, (0, pad), value=1)

    out = torch.zeros((flat.numel() // 8,), device=flat.device, dtype=torch.uint8)
    for i in range(8):
        out += flat[i::8] << (7 - i)
    return out, original_shape


def unpack_binary(packed: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Expand uint8 to {−1,+1} int8 and reshape.

    Args:
        packed: Packed uint8 tensor
        original_shape: Original shape

    Returns:
        ±1 int8 tensor
    """
    numel = 1
    for dim in original_shape:
        numel *= dim

    out = torch.zeros((packed.shape[0], 8), device=packed.device, dtype=torch.int8)
    for i in range(8):
        out[:, i] = (packed >> (7 - i)) & 1
    return out.flatten()[:numel].reshape(original_shape) * 2 - 1


# =============================================================================
# MDBFLinear Layer (1-pass)
# =============================================================================


class MDBFLinear(nn.Module):
    """
    1-pass MDBF inference layer (always on-the-fly unpack).

    W^{(p)} = F @ G
    where F = S_A * (A_amp @ Q_U_amp^T)
          G = S_B * (Q_V_amp @ B_amp^T)

    Inference: y = x @ W^T = x @ G^T @ F^T

    GemLite acceleration (mirror of dbf/dbf_layer.py):
        The two heavy matmuls are against the ±1 sign matrices B_sign (r, m) and
        A_sign (n, r). The rank-l amplitudes are separable per scale, so

            y = Σ_k (((x * B_amp[:,k]) @ B_sign^T) * Q_V_amp[:,k] * Q_U_amp[:,k]) ...

        i.e. each sign matmul can be replaced by a 1-bit GemLite kernel and the
        amplitudes applied as element-wise scalings around it. GemLite is enabled
        per sign matrix (it requires the matmul's in_features to be a multiple of
        the group size), falling back to on-the-fly unpack otherwise.
    """

    def __init__(
        self,
        params: MDBFParams,
        device: Optional[torch.device] = None,
        use_gemlite: Optional[bool] = None,
    ):
        super().__init__()

        n, r = params.A_sign.shape
        r2, m = params.B_sign.shape
        assert r == r2, f"Rank mismatch: A_sign has rank {r}, B_sign has rank {r2}"

        self.n = n
        self.m = m
        self.r = r
        self.l = params.A_amp.shape[1]

        # Packed sign matrices
        A_sign_packed, _ = pack_binary(params.A_sign)
        B_sign_packed, _ = pack_binary(params.B_sign)

        self.register_buffer("A_sign_packed", A_sign_packed)
        self.register_buffer("B_sign_packed", B_sign_packed)
        self.register_buffer("_A_sign_shape", torch.tensor([n, r], dtype=torch.int64))
        self.register_buffer("_B_sign_shape", torch.tensor([r, m], dtype=torch.int64))

        # Scales (FP16)
        self.register_buffer("A_amp", params.A_amp.half())
        self.register_buffer("B_amp", params.B_amp.half())
        self.register_buffer("Q_U_amp", params.Q_U_amp.half())
        self.register_buffer("Q_V_amp", params.Q_V_amp.half())

        # Optional GemLite kernels for the ±1 sign matmuls.
        # Stored in a plain dict (not a submodule) so they stay out of state_dict,
        # matching DoubleBinaryLinear._gemlite_layers.
        self._gemlite_layers: dict = {}
        # CPU stash of packed sign buffers freed from GPU once GemLite serves them
        # (see _free_packed_sign). Plain dict -> invisible to .to()/state_dict.
        self._packed_cpu: dict = {}
        self.use_gemlite = False
        self._build_gemlite(params.A_sign, params.B_sign, device, use_gemlite)

    def _build_gemlite(
        self,
        A_sign: torch.Tensor,
        B_sign: torch.Tensor,
        device: Optional[torch.device],
        use_gemlite: Optional[bool],
    ) -> None:
        """Build 1-bit GemLite kernels for B_sign (r, m) and A_sign (n, r).

        Note on the multi-scale rank l:
            The dense path folds the rank-l amplitude into a single (n, r)/(r, m)
            matrix, so its cost is independent of l. The GemLite path keeps the
            sign matrices pure ±1 and applies the l amplitude scales outside, so it
            issues l separate 1-bit matmuls per sign matrix (cost grows ~O(l)).
            Measured on H100 (in=out=4096, r=512): GemLite is ~1.5x faster than
            dense at l=1 but ~0.76x (l=2) / ~0.37x (l=4), i.e. a net slowdown.
            Therefore, in auto mode (use_gemlite=None) GemLite is only enabled for
            l == 1. Pass use_gemlite=True to force it regardless of l.
        """
        forced = use_gemlite is True
        if use_gemlite is None:
            use_gemlite = HAS_GEMLITE_SUPPORT and is_gemlite_available()
        if not (use_gemlite and HAS_GEMLITE_SUPPORT):
            return
        if self.l > 1 and not forced:
            # Auto mode: skip GemLite for l>1 (it would be slower than dense).
            return

        device_obj = torch.device(device) if device is not None else A_sign.device
        # Stage 1 matmul: x @ B_sign^T  (in_features = m)
        gemlite_B = create_gemlite_linear(B_sign, nbits=1, device=device_obj)
        # Stage 2 matmul: u @ A_sign^T  (in_features = r)
        gemlite_A = create_gemlite_linear(A_sign, nbits=1, device=device_obj)
        if gemlite_B is not None:
            self._gemlite_layers["B"] = gemlite_B
        if gemlite_A is not None:
            self._gemlite_layers["A"] = gemlite_A
        self.use_gemlite = len(self._gemlite_layers) > 0

        # Free the redundant GPU packed-sign buffer for any matrix now served by
        # GemLite (GemLite keeps its own 1-bit W_q), halving in-memory weight.
        for which in self._gemlite_layers:
            self._free_packed_sign(which)

    def _free_packed_sign(self, which: str) -> None:
        """Move a packed sign buffer off-GPU once GemLite serves that matmul.

        The buffer is un-registered (so ``.to(device)`` won't pull it back to GPU)
        and stashed on CPU; ``_save_to_state_dict`` re-emits it, so the saved
        state_dict matches the dense layout exactly.
        """
        buf_name = f"{which}_sign_packed"
        buf = self._buffers.get(buf_name)
        if buf is not None:
            self._packed_cpu[which] = buf.detach().to("cpu")
            del self._buffers[buf_name]

    def _packed_sign(self, which: str, device: torch.device) -> torch.Tensor:
        """Return the packed (uint8) sign tensor on *device* (buffer or CPU stash)."""
        buf = self._buffers.get(f"{which}_sign_packed")
        if buf is None:
            buf = self._packed_cpu[which]
        return buf.to(device)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Re-emit any packed-sign buffers freed from GPU in GemLite mode so the
        # on-disk format is identical to the dense (non-GemLite) layout.
        for which in ("A", "B"):
            key = f"{prefix}{which}_sign_packed"
            if key not in destination and which in getattr(self, "_packed_cpu", {}):
                v = self._packed_cpu[which]
                destination[key] = v if keep_vars else v.detach()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Reconstruct dimensions from shape buffers during loading"""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

        if hasattr(self, "_A_sign_shape") and self._A_sign_shape is not None:
            A_shape = tuple(self._A_sign_shape.tolist())
            B_shape = tuple(self._B_sign_shape.tolist())

            self.n, self.r = A_shape
            _, self.m = B_shape
            self.l = self.A_amp.shape[1] if hasattr(self, "A_amp") else 1

    def _get_factor_matrices(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute factor matrices F, G (always on-the-fly unpack)"""
        dev = self.A_amp.device
        A_sign = unpack_binary(self._packed_sign("A", dev), (self.n, self.r)).to(dtype)
        B_sign = unpack_binary(self._packed_sign("B", dev), (self.r, self.m)).to(dtype)

        amp_A = self.A_amp.to(dtype) @ self.Q_U_amp.to(dtype).T
        F = A_sign * amp_A

        amp_B = self.Q_V_amp.to(dtype) @ self.B_amp.to(dtype).T
        G = B_sign * amp_B

        return F, G

    def _apply_sign(self, x: torch.Tensor, which: str) -> torch.Tensor:
        """Multiply by a ±1 sign matrix, via GemLite when available.

        which="B": x @ B_sign^T  (B_sign is (r, m))
        which="A": x @ A_sign^T  (A_sign is (n, r))
        """
        gemlite = self._gemlite_layers.get(which)
        if gemlite is not None:
            return gemlite(x)
        if which == "B":
            B_sign = unpack_binary(self._packed_sign("B", x.device), (self.r, self.m)).to(x.dtype)
            return x @ B_sign.T
        A_sign = unpack_binary(self._packed_sign("A", x.device), (self.n, self.r)).to(x.dtype)
        return x @ A_sign.T

    def _forward_gemlite(self, x: torch.Tensor) -> torch.Tensor:
        """Per-scale forward using GemLite 1-bit kernels for the sign matmuls.

        Exact reformulation of y = x @ G^T @ F^T with
            G = B_sign * (Q_V_amp @ B_amp^T),  F = A_sign * (A_amp @ Q_U_amp^T):

            u = Σ_k ((x * B_amp[:,k]) @ B_sign^T) * Q_V_amp[:,k]
            y = Σ_k ((u * Q_U_amp[:,k]) @ A_sign^T) * A_amp[:,k]
        """
        dtype = x.dtype
        B_amp = self.B_amp.to(dtype)  # (m, l)
        Q_V_amp = self.Q_V_amp.to(dtype)  # (r, l)
        A_amp = self.A_amp.to(dtype)  # (n, l)
        Q_U_amp = self.Q_U_amp.to(dtype)  # (r, l)

        # Stage 1: u = x @ G^T  -> (..., r)
        u = None
        for k in range(self.l):
            bk = self._apply_sign(x * B_amp[:, k], "B")
            contrib = bk * Q_V_amp[:, k]
            u = contrib if u is None else u + contrib

        # Stage 2: y = u @ F^T  -> (..., n)
        y = None
        for k in range(self.l):
            ak = self._apply_sign(u * Q_U_amp[:, k], "A")
            contrib = ak * A_amp[:, k]
            y = contrib if y is None else y + contrib

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gemlite:
            return self._forward_gemlite(x)
        F, G = self._get_factor_matrices(x.dtype)
        y = x @ G.T
        y = y @ F.T
        return y

    def get_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct weight matrix"""
        F, G = self._get_factor_matrices(dtype)
        return F @ G

    def enable_gemlite(
        self, device: Optional[torch.device] = None, force: bool = False
    ) -> bool:
        """Build GemLite kernels from the packed sign buffers (e.g. after load).

        Returns True if at least one sign matmul is GemLite-accelerated.
        """
        if self.use_gemlite and self._gemlite_layers:
            return True
        dev = self.A_amp.device
        A_sign = unpack_binary(self._packed_sign("A", dev), (self.n, self.r))
        B_sign = unpack_binary(self._packed_sign("B", dev), (self.r, self.m))
        self._build_gemlite(A_sign, B_sign, device, True if force else None)
        return self.use_gemlite


# =============================================================================
# MultipathMDBFLinear Layer (P-pass)
# =============================================================================


class MultipathMDBFLinear(nn.Module):
    """P-pass MDBF inference layer: W ≈ Σ_{p=1}^{P} W^{(p)}"""

    def __init__(
        self,
        params_list: List[MDBFParams],
        bias: Optional[torch.Tensor] = None,
        device=None,
        use_gemlite: Optional[bool] = None,
    ):
        super().__init__()

        if len(params_list) == 0:
            raise ValueError("params_list must not be empty")

        self.P = len(params_list)
        self.n = params_list[0].A_sign.shape[0]
        self.m = params_list[0].B_sign.shape[1]

        self.paths = nn.ModuleList(
            [MDBFLinear(params, device=device, use_gemlite=use_gemlite) for params in params_list]
        )

        if bias is not None:
            self.register_buffer("bias", bias.clone().to(torch.float16))
        else:
            self.bias = None

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.paths[0](x)
        for i in range(1, self.P):
            y = y + self.paths[i](x)

        if self.bias is not None:
            y = y + self.bias.to(x.dtype)

        return y

    def get_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct weight matrix"""
        W = self.paths[0].get_weight(dtype)
        for i in range(1, self.P):
            W = W + self.paths[i].get_weight(dtype)
        return W

    def enable_gemlite(
        self, device: Optional[torch.device] = None, force: bool = False
    ) -> bool:
        """Enable GemLite on every path (e.g. after ``from_saved_state``).

        Returns True if any path ended up GemLite-accelerated.
        """
        enabled = False
        for path in self.paths:
            if path.enable_gemlite(device=device, force=force):
                enabled = True
        return enabled

    @classmethod
    def from_quantization_result(
        cls,
        result,
        bias=None,
        device=None,
        use_gemlite=None,
    ) -> "MultipathMDBFLinear":
        """Build MultipathMDBFLinear from MDBFResult.

        Args:
            result: MDBFResult from quantizer.
            bias: Optional bias tensor (from original Linear).
            device: Device to place the layer on.
            use_gemlite: GemLite acceleration (None=auto, True/False=force).

        Returns:
            MultipathMDBFLinear instance.
        """
        params_list = result.get_MDBF_params_list()
        return cls(params_list=params_list, bias=bias, device=device, use_gemlite=use_gemlite)

    @classmethod
    def from_saved_state(
        cls,
        layer_state_dict: dict,
        in_features: int,
        out_features: int,
        empty: bool = False,
        target_bits: float = None,
    ) -> "MultipathMDBFLinear":
        """Build MultipathMDBFLinear from saved state_dict tensors.

        P (number of paths) and l (multi-scale rank) are inferred from the
        state_dict (path index keys and A_amp.shape[1] respectively), so
        callers do not need to pass them.

        Args:
            layer_state_dict: Sub-state_dict for this layer.
                Keys follow the pattern:
                  paths.{p}.A_sign_packed, paths.{p}.B_sign_packed,
                  paths.{p}._A_sign_shape, paths.{p}._B_sign_shape,
                  paths.{p}.A_amp, paths.{p}.B_amp,
                  paths.{p}.Q_U_amp, paths.{p}.Q_V_amp
                  bias (optional)
            in_features: Input feature size (m).
            out_features: Output feature size (n).
            empty: If True, create zero-initialized tensors (for
                "replace then load_state_dict" flow).
            target_bits: Nominal bit-width (from config).

        Returns:
            MultipathMDBFLinear instance.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.target_bits = target_bits
        self.n = out_features
        self.m = in_features

        def _t(k):
            t = layer_state_dict[k]
            return torch.zeros_like(t) if empty else t

        # Detect P from state_dict keys
        path_indices = set()
        for key in layer_state_dict:
            if key.startswith("paths."):
                parts = key.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    path_indices.add(int(parts[1]))
        if not path_indices:
            raise ValueError(
                "MultipathMDBFLinear.from_saved_state: no `paths.{p}.*` keys "
                "found in layer_state_dict."
            )
        self.P = max(path_indices) + 1

        paths = nn.ModuleList()
        for p_idx in range(self.P):
            path_prefix = f"paths.{p_idx}."

            path_layer = MDBFLinear.__new__(MDBFLinear)
            nn.Module.__init__(path_layer)

            # Recover rank r from Q_U_amp's shape ((r, l)).
            # (DBF analog: mid_dim = layer_state_dict["scaling2"].numel().)
            r = layer_state_dict[f"{path_prefix}Q_U_amp"].shape[0]
            path_layer.n = out_features
            path_layer.r = r
            path_layer.m = in_features
            shape_A = (out_features, r)
            shape_B = (r, in_features)

            path_layer.register_buffer("A_sign_packed", _t(f"{path_prefix}A_sign_packed"))
            path_layer.register_buffer("B_sign_packed", _t(f"{path_prefix}B_sign_packed"))
            path_layer.register_buffer("_A_sign_shape", torch.tensor(shape_A, dtype=torch.int64))
            path_layer.register_buffer("_B_sign_shape", torch.tensor(shape_B, dtype=torch.int64))
            path_layer.register_buffer("A_amp", _t(f"{path_prefix}A_amp"))
            path_layer.register_buffer("B_amp", _t(f"{path_prefix}B_amp"))
            path_layer.register_buffer("Q_U_amp", _t(f"{path_prefix}Q_U_amp"))
            path_layer.register_buffer("Q_V_amp", _t(f"{path_prefix}Q_V_amp"))

            path_layer.l = path_layer.A_amp.shape[1]

            # GemLite disabled on load (mirror DoubleBinaryLinear.from_saved_state);
            # can be enabled later via enable_gemlite().
            path_layer.use_gemlite = False
            path_layer._gemlite_layers = {}
            path_layer._packed_cpu = {}

            paths.append(path_layer)

        self.paths = paths

        bias = layer_state_dict.get("bias")
        if bias is not None:
            self.register_buffer("bias", torch.zeros_like(bias) if empty else bias)
        else:
            self.bias = None

        return self
