"""
DBF (Double Binary Factorization) layer implementation.

Efficient inference while keeping compatibility with the reference implementation.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

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
        out += (flat[i::8] << (7 - i))
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
       If preunpack=True, unpack once at init and keep in memory (fast, more memory).
    """
    def __init__(self, b: torch.Tensor, preunpack: bool = True):
        super().__init__()
        if b.ndim == 2:
            self.shape = tuple(b.shape)
            self._numel = b.numel()
            bp = pack_binary(b)
        else:
            raise ValueError("BitLinearPacked: expected 2D ±1 tensor.")

        self.register_buffer("bp", bp)
        self.register_parameter("scale", nn.Parameter(torch.ones(1)))

        # Speed option: unpack once at init and keep
        if preunpack:
            unpacked = unpack_binary(self.bp)[: self._numel].reshape(self.shape).to(torch.int8)
            self.register_buffer("bit_mat", unpacked)
        else:
            self.register_buffer("bit_mat", None)

    def forward(self, x):
        if self.bit_mat is None:
            # Slice to needed size then reshape
            bit_mat = unpack_binary(self.bp)[: self._numel].reshape(self.shape)
        else:
            bit_mat = self.bit_mat
        # Convert int8 to compute dtype and matmul
        # Ensure dtype conversion to avoid BFloat16/float32 mismatch
        weight_matrix = (bit_mat.to(x.dtype) * self.scale.to(x.dtype)).t()
        return x.matmul(weight_matrix)


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

    def __init__(
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
        self.scaling0 = nn.Parameter(
            dbf_Db.detach().to(torch.float16), requires_grad=False
        )
        # Stage 2: Middle scaling
        mid = dbf_mid.flatten() if dbf_mid.numel() > 1 else dbf_mid
        self.scaling2 = nn.Parameter(
            mid.detach().to(torch.float16), requires_grad=False
        )
        # Stage 4: Output scaling
        self.scaling4 = nn.Parameter(
            dbf_Da.detach().to(torch.float16), requires_grad=False
        )

        if use_gemlite is None:
            use_gemlite = HAS_GEMLITE_SUPPORT and is_gemlite_available()

        if use_gemlite and HAS_GEMLITE_SUPPORT:
            device_obj = torch.device(device) if device else torch.device("cuda")
            gemlite1 = create_gemlite_linear(dbf_B, nbits=1, device=device_obj)
            gemlite3 = create_gemlite_linear(dbf_A, nbits=1, device=device_obj)
            if gemlite1 is not None and gemlite3 is not None:
                self.binary_multiplication1 = gemlite1
                self.binary_multiplication3 = gemlite3
                self.using_gemlite = True
            else:
                self.binary_multiplication1 = BitLinearPacked(dbf_B, preunpack=True)
                self.binary_multiplication3 = BitLinearPacked(dbf_A, preunpack=True)
                self.using_gemlite = False
        else:
            # TODO: Verify later (wtfr)
            # TODO: Temporarily modified to match the received model format
            self.register_buffer("bp1", pack_binary(dbf_B))
            self.register_buffer("bp3", pack_binary(dbf_A))
            self.using_gemlite = False

        # Bias (from original Linear, if any)
        if bias is not None:
            self.register_buffer('bias', bias.clone().to(torch.float16))
        else:
            self.bias = None
        
        if device is not None:
            self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """5-stage forward pass."""
        x = x * self.scaling0.to(x.dtype)
        x = self.binary_multiplication1(x)
        x = x * self.scaling2.to(x.dtype)
        x = self.binary_multiplication3(x)
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

