"""
DBF (Double Binary Factorization) layer implementation.

Efficient inference while keeping compatibility with the reference implementation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Optional GemLite integration
try:
    from onecomp.quantizer.gemlite import create_gemlite_linear, is_gemlite_available

    HAS_GEMLITE_SUPPORT = True
except ImportError:
    HAS_GEMLITE_SUPPORT = False


# ========================================
# Bit packing / unpacking
# ========================================


def pack_binary(x: torch.Tensor) -> torch.Tensor:
    """Convert ±1 to {0,1} and pack 8:1 into uint8. Pad trailing with +1."""
    # Allowed input: {-1, +1} (int/float). Zero is treated as +1.
    flat = (x.flatten() >= 0).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        # Pad with +1 (=1), near-identity for multiplication
        flat = F.pad(flat, (0, pad), value=1)
    out = torch.zeros((flat.numel() // 8,), device=flat.device, dtype=torch.uint8)
    # Aggregate by bit position
    for i in range(8):
        out += flat[i::8] << (7 - i)
    return out


def unpack_binary(x: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 to int8 {−1,+1} (8x); slice to needed size downstream."""
    out = torch.zeros((x.shape[0], 8), device=x.device, dtype=torch.int8)
    for i in range(8):
        out[:, i] = (x >> (7 - i)) & 1
    return out.flatten() * 2 - 1


# ========================================
# Basic components
# ========================================


class BitLinearPacked(nn.Module):
    """Packed binary matrix × input linear (fallback without GemLite).
    Unpacks bp on every forward pass.
    """

    def __init__(self, b: torch.Tensor):
        super().__init__()
        if b.ndim == 2:
            self.shape = tuple(b.shape)
            self._numel = b.numel()
            bp = pack_binary(b)
        else:
            raise ValueError("BitLinearPacked: expected 2D ±1 tensor.")

        self.register_buffer("bp", bp)

    def forward(self, x):
        # Unpack, slice to needed size, reshape, and matmul
        bit_mat = unpack_binary(self.bp)[: self._numel].reshape(self.shape)
        return x.matmul(bit_mat.to(x.dtype).t())


# ========================================
# DoubleBinaryLinear layer
# ========================================


class DoubleBinaryLinear(nn.Module):
    """DBF inference layer (5-stage implementation).

    Build 5-stage DoubleBinaryLinear from DBF decomposition result.

    - Stage 0: Input scaling  (v_B)
    - Stage 1: Binary B
    - Stage 2: Middle scaling (v_A * mid * u_B)
    - Stage 3: Binary A
    - Stage 4: Output scaling (u_A)
    Binarize W ≈ A × diag(mid) × B:
    W ≈ diag(u_A) @ binary_A @ diag(v_A * mid * u_B) @ binary_B @ diag(v_B)

    Option: GemLite acceleration
    - use_gemlite=True: Use GemLite (3-5x faster when available)
    - use_gemlite=False: PyTorch implementation (default, no extra deps)
    - use_gemlite=None: Auto (use if available)

    Args:
        dbf_Da: Scaling vector paired with A (out_dim,)
        dbf_A: Binary A matrix (out_dim, mid_dim)
        dbf_mid: Middle scaling vector (mid_dim,)
        dbf_B: Binary B matrix (mid_dim, in_dim)
        dbf_Db: Scaling vector paired with B (in_dim,)
        bias: Optional bias tensor (from original Linear)
        device: Device
        use_gemlite: GemLite flag (None=auto)
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        dbf_Da: torch.Tensor,
        dbf_A: torch.Tensor,
        dbf_mid: torch.Tensor,
        dbf_B: torch.Tensor,
        dbf_Db: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        use_gemlite: Optional[bool] = None,
    ):
        super().__init__()
        # Stage 0: Input scaling
        self.scaling0 = nn.Parameter(dbf_Db.detach().to(torch.float16), requires_grad=False)
        # Stage 2: Middle scaling
        mid = dbf_mid.flatten() if dbf_mid.numel() > 1 else dbf_mid
        self.scaling2 = nn.Parameter(mid.detach().to(torch.float16), requires_grad=False)
        # Stage 4: Output scaling
        self.scaling4 = nn.Parameter(dbf_Da.detach().to(torch.float16), requires_grad=False)

        self._bp1_shape = tuple(dbf_B.shape)
        self._bp3_shape = tuple(dbf_A.shape)
        self.register_buffer("bp1", pack_binary(dbf_B))
        self.register_buffer("bp3", pack_binary(dbf_A))

        if use_gemlite is None:
            use_gemlite = HAS_GEMLITE_SUPPORT and is_gemlite_available()

        self._gemlite_layers: dict = {}
        self.use_gemlite = False
        if use_gemlite and HAS_GEMLITE_SUPPORT:
            device_obj = torch.device(device) if device else torch.device("cuda")
            gemlite1 = create_gemlite_linear(dbf_B, nbits=1, device=device_obj)
            gemlite3 = create_gemlite_linear(dbf_A, nbits=1, device=device_obj)
            if gemlite1 is not None and gemlite3 is not None:
                self._gemlite_layers["1"] = gemlite1
                self._gemlite_layers["3"] = gemlite3
                self.use_gemlite = True

        # Bias (from original Linear, if any)
        if bias is not None:
            self.register_buffer("bias", bias.clone().to(torch.float16))
        else:
            self.bias = None

        if device is not None:
            self.to(device)

    def _unpack_bp(self, bp: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Unpack a packed binary buffer to {-1,+1} matrix."""
        return unpack_binary(bp)[: shape[0] * shape[1]].reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """5-stage forward pass."""
        x = x * self.scaling0.to(x.dtype)
        if self.use_gemlite:
            x = self._gemlite_layers["1"](x)
        else:
            b1 = self._unpack_bp(self.bp1, self._bp1_shape)
            x = x.matmul(b1.to(x.dtype).t())
        x = x * self.scaling2.to(x.dtype)
        if self.use_gemlite:
            x = self._gemlite_layers["3"](x)
        else:
            b3 = self._unpack_bp(self.bp3, self._bp3_shape)
            x = x.matmul(b3.to(x.dtype).t())
        x = x * self.scaling4.to(x.dtype)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x

    @classmethod
    def from_quantization_result(cls, result, bias=None, device=None, use_gemlite=None):
        """
        Build DoubleBinaryLinear from DBFResult (5-stage format).

        Expects result with dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db.
        Args:
            result: DBFResult from quantizer.results[name]
            bias: Optional bias tensor (from original Linear)
            device: Device to place the layer on
            use_gemlite: Use GemLite acceleration (None=auto, True/False=force)

        Returns:
            DoubleBinaryLinear instance
        """
        return cls(
            dbf_Da=result.dbf_Da,
            dbf_A=result.dbf_A,
            dbf_mid=result.dbf_mid,
            dbf_B=result.dbf_B,
            dbf_Db=result.dbf_Db,
            bias=bias,
            device=device,
            use_gemlite=use_gemlite,
        )

    @classmethod
    def from_saved_state(
        cls,
        layer_state_dict: dict,
        in_features: int,
        out_features: int,
        empty: bool = False,
        target_bits: Optional[float] = None,
    ):
        """Build DoubleBinaryLinear from saved state_dict tensors.

        Args:
            layer_state_dict: Sub-state_dict for this layer
                (keys: scaling0, scaling2, scaling4, bp1, bp3, bias).
            in_features: Input feature size.
            out_features: Output feature size.
            empty: If True, create zero params/buffers of the same shape (for "replace then
                load_state_dict" flow). If False, use tensors from layer_state_dict directly.
            target_bits: Nominal bit-width for this layer (from config; for mixed_dbf).
                Stored on the module for inspection; not used for forward yet.

        Returns:
            DoubleBinaryLinear instance.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.target_bits = target_bits

        def _p(k):
            t = layer_state_dict[k]
            return torch.zeros_like(t) if empty else t

        self.scaling0 = nn.Parameter(_p("scaling0"), requires_grad=False)
        self.scaling2 = nn.Parameter(_p("scaling2"), requires_grad=False)
        self.scaling4 = nn.Parameter(_p("scaling4"), requires_grad=False)

        mid_dim = layer_state_dict["scaling2"].numel()
        self._bp1_shape = (mid_dim, in_features)
        self._bp3_shape = (out_features, mid_dim)
        self.register_buffer("bp1", _p("bp1"))
        self.register_buffer("bp3", _p("bp3"))

        bias = layer_state_dict.get("bias")
        if bias is not None:
            self.register_buffer("bias", torch.zeros_like(bias) if empty else bias)
        else:
            self.bias = None

        self.use_gemlite = False
        self._gemlite_layers = {}

        return self
