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
        out += (flat[i::8] << (7 - i))
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
    return (out.flatten()[:numel].reshape(original_shape) * 2 - 1)


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
    """

    def __init__(self, params: MDBFParams):
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        """Reconstruct dimensions from shape buffers during loading"""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

        if hasattr(self, '_A_sign_shape') and self._A_sign_shape is not None:
            A_shape = tuple(self._A_sign_shape.tolist())
            B_shape = tuple(self._B_sign_shape.tolist())

            self.n, self.r = A_shape
            _, self.m = B_shape
            self.l = self.A_amp.shape[1] if hasattr(self, 'A_amp') else 1

    def _get_factor_matrices(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute factor matrices F, G (always on-the-fly unpack)"""
        A_sign = unpack_binary(self.A_sign_packed, (self.n, self.r)).to(dtype)
        B_sign = unpack_binary(self.B_sign_packed, (self.r, self.m)).to(dtype)

        amp_A = self.A_amp.to(dtype) @ self.Q_U_amp.to(dtype).T
        F = A_sign * amp_A

        amp_B = self.Q_V_amp.to(dtype) @ self.B_amp.to(dtype).T
        G = B_sign * amp_B

        return F, G

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F, G = self._get_factor_matrices(x.dtype)
        y = x @ G.T
        y = y @ F.T
        return y

    def get_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct weight matrix"""
        F, G = self._get_factor_matrices(dtype)
        return F @ G


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
    ):
        super().__init__()

        if len(params_list) == 0:
            raise ValueError("params_list must not be empty")

        self.P = len(params_list)
        self.n = params_list[0].A_sign.shape[0]
        self.m = params_list[0].B_sign.shape[1]

        self.paths = nn.ModuleList([
            MDBFLinear(params)
            for params in params_list
        ])

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

    @classmethod
    def from_quantization_result(
        cls,
        result,
        bias=None,
        device=None,
    ) -> "MultipathMDBFLinear":
        """Build MultipathMDBFLinear from MDBFResult.

        Args:
            result: MDBFResult from quantizer.
            bias: Optional bias tensor (from original Linear).
            device: Device to place the layer on.

        Returns:
            MultipathMDBFLinear instance.
        """
        params_list = result.get_MDBF_params_list()
        return cls(params_list=params_list, bias=bias, device=device)

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
            path_layer.register_buffer("_A_sign_shape",
                                        torch.tensor(shape_A, dtype=torch.int64))
            path_layer.register_buffer("_B_sign_shape",
                                        torch.tensor(shape_B, dtype=torch.int64))
            path_layer.register_buffer("A_amp", _t(f"{path_prefix}A_amp"))
            path_layer.register_buffer("B_amp", _t(f"{path_prefix}B_amp"))
            path_layer.register_buffer("Q_U_amp", _t(f"{path_prefix}Q_U_amp"))
            path_layer.register_buffer("Q_V_amp", _t(f"{path_prefix}Q_V_amp"))

            path_layer.l = path_layer.A_amp.shape[1]

            paths.append(path_layer)

        self.paths = paths

        bias = layer_state_dict.get("bias")
        if bias is not None:
            self.register_buffer("bias", torch.zeros_like(bias) if empty else bias)
        else:
            self.bias = None

        return self
