"""
OneBit Layer implementation

Inference implementation for OneBit quantized Linear layers.
W ≈ a ⊙ sign(W) ⊙ b^T

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import logging
import traceback
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# ========================================
# Bit packing / unpacking (same as DBF)
# ========================================


def my_pack(x: torch.Tensor) -> torch.Tensor:
    """Convert ±1 to {0,1} and pack into uint8 at 8:1 ratio. Tail is padded with +1."""
    flat = (x.flatten() >= 0).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = F.pad(flat, (0, pad), value=1)
    out = torch.zeros((flat.numel() // 8,), device=flat.device, dtype=torch.uint8)
    for i in range(8):
        out += flat[i::8] << (7 - i)
    return out


def my_unpack(x: torch.Tensor) -> torch.Tensor:
    """Expand uint8 to int8 {-1,+1} at 8x expansion (slice to required size downstream)."""
    out = torch.zeros((x.shape[0], 8), device=x.device, dtype=torch.int8)
    for i in range(8):
        out[:, i] = (x >> (7 - i)) & 1
    return out.flatten() * 2 - 1


# ========================================
# OneBitLinear layer (with bit packing support)
# ========================================


class OneBitLinear(nn.Module):
    """
    OneBit quantized Linear layer (with bit packing support).

    Computation: out = (a ⊙ sign(W) ⊙ b^T) @ x

    Where:
    - a: Row-wise scaling (out_features,)
    - b: Column-wise scaling (in_features,)
    - sign: Sign matrix {-1, +1} (out_features, in_features)

    Memory efficiency:
    - Sign matrix is stored as a packed uint8 buffer at 8:1 ratio.
    - Inference unpacks from sign_packed on demand, mirroring DBF.
    - sign_matrix is used only as a temporary non-persistent override for
        optimisation flows such as blockwise / CBQ.
    """

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        sign: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Scaling vectors (normalize to FP16, detach to drop autograd graph)
        self.register_buffer("a", a.detach().to(torch.float16))
        self.register_buffer("b", b.detach().to(torch.float16))

        # Dimension information
        self.out_features = self.a.shape[0]
        self.in_features = self.b.shape[0]

        # Bit packing of the sign matrix
        if sign.dtype == torch.uint8:
            # Already packed
            self.register_buffer("sign_packed", sign.detach().clone())
            self._sign_numel = self.out_features * self.in_features
        else:
            # Pack ±1 matrix
            self._sign_numel = sign.numel()
            sign_packed = my_pack(sign.detach().flatten())
            self.register_buffer("sign_packed", sign_packed)

        # Temporary non-persistent override used only by optimisation flows.
        self.register_buffer("sign_matrix", None, persistent=False)

        # Bias (normalize to FP16, clone to avoid aliasing the source Linear)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().to(torch.float16))
        else:
            self.bias = None

    def _unpack_sign_matrix(self) -> torch.Tensor:
        """Unpack sign_packed to a dense int8 {-1,+1} matrix."""
        return (
            my_unpack(self.sign_packed)[: self._sign_numel]
            .reshape(self.out_features, self.in_features)
            .to(torch.int8)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if f"{prefix}sign_packed" in state_dict:
            self.sign_matrix = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: out = (a ⊙ sign(W) ⊙ b^T) @ x

        Efficient computation:
        1. x = x * b (column-wise scaling)
        2. out = (sign * a[:, None]) @ x (sign matrix with row-wise scaling)
        """
        sign = self.sign_matrix if self.sign_matrix is not None else self._unpack_sign_matrix()

        # Apply b (column-wise scaling) to input
        x_scaled = x * self.b.to(x.dtype)

        # Weight combining sign matrix with a (row-wise scaling)
        # sign: {-1, +1} → float
        weight_matrix = sign.to(x.dtype) * self.a.to(x.dtype).unsqueeze(1)

        # Matrix multiplication
        out = torch.matmul(x_scaled, weight_matrix.t())

        # Add bias
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)

        return out

    def extra_repr(self) -> str:
        mode = "packed+override" if self.sign_matrix is not None else "packed"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={mode}"
        )

    @classmethod
    def from_quantization_result(
        cls,
        result,
        bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Build OneBitLinear from a OnebitResult.

        This mirrors the from_quantization_result pattern used by
        GPTQLinear and DoubleBinaryLinear.

        Args:
            result: OnebitResult with a, b, sign attributes.
            bias: Optional bias tensor.
            device: Optional device to move the layer to.
        Returns:
            OneBitLinear instance.
        """
        layer = cls(a=result.a, b=result.b, sign=result.sign, bias=bias)
        if device is not None:
            layer = layer.to(device)
        return layer

    @classmethod
    def from_saved_state(
        cls,
        layer_state_dict: dict,
        in_features: int,
        out_features: int,
        empty: bool = False,
    ):
        """Build OneBitLinear from saved state_dict tensors.

        Saved keys: a, b, sign_packed, (optional) bias.
        Uses the same cls.__new__ pattern as DoubleBinaryLinear.

        Args:
            layer_state_dict: Sub-state_dict with keys a, b, sign_packed.
            in_features: Input feature size.
            out_features: Output feature size.
            empty: If True, create zero params of the same shape.

        Returns:
            OneBitLinear instance.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)

        self.out_features = out_features
        self.in_features = in_features
        self._sign_numel = out_features * in_features

        def _p(k):
            t = layer_state_dict[k]
            return torch.zeros_like(t) if empty else t

        self.register_buffer("a", _p("a"))
        self.register_buffer("b", _p("b"))
        self.register_buffer("sign_packed", _p("sign_packed"))
        self.register_buffer("sign_matrix", None, persistent=False)

        bias_tensor = layer_state_dict.get("bias")
        if bias_tensor is not None:
            self.register_buffer("bias", torch.zeros_like(bias_tensor) if empty else bias_tensor)
        else:
            self.bias = None

        return self
