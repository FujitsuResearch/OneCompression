"""
GPTQ Quantized Linear Layer for Fast Inference

Implements a Linear layer for GPTQ-quantized models.
Runs inference in quantized (INT) form for better memory and speed.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from logging import getLogger

logger = getLogger(__name__)

# Optional GemLite integration
try:
    from onecomp.quantizer.gemlite import create_gemlite_linear, is_gemlite_available
    HAS_GEMLITE_SUPPORT = True
except ImportError:
    HAS_GEMLITE_SUPPORT = False


# ========================================
# Bit packing / unpacking
# ========================================

def pack_int_weights(weights: torch.Tensor, wbits: int) -> torch.Tensor:
    """
    Pack INT quantized weights for memory efficiency.

    Args:
        weights: Quantized weights (INT32, values 0 to 2^wbits-1)
        wbits: Bit width (2, 3, 4, 8, etc.)

    Returns:
        Packed weights (uint8 or int32)
    """
    if wbits >= 8:
        # No packing for 8+ bits (8bit -> int8 for efficiency; higher unchanged)
        return weights.to(torch.int8) if wbits == 8 else weights

    # How many wbits values fit in 32 bits
    values_per_int32 = 32 // wbits
    flat = weights.flatten()

    # Padding
    pad_size = (-flat.numel()) % values_per_int32
    if pad_size > 0:
        flat = F.pad(flat, (0, pad_size), value=0)

    # Bit packing
    packed = torch.zeros((flat.numel() // values_per_int32,), 
                         device=flat.device, dtype=torch.int32)
    for i in range(values_per_int32):
        packed += (flat[i::values_per_int32].int() << (i * wbits))
    
    return packed


def unpack_int_weights(
    packed: torch.Tensor, wbits: int, original_shape: Union[torch.Size, Tuple[int, ...]]
) -> torch.Tensor:
    """
    Unpack packed INT weights.

    Args:
        packed: Packed weights
        wbits: Bit width
        original_shape: Original shape

    Returns:
        Unpacked weights (INT32)
    """
    if wbits >= 8:
        return packed.reshape(original_shape)

    values_per_int32 = 32 // wbits
    mask = (1 << wbits) - 1

    # Unpack
    n = packed.numel()
    flat = torch.zeros(n * values_per_int32, dtype=torch.int32, device=packed.device)
    for i in range(values_per_int32):
        flat[i::values_per_int32] = (packed >> (i * wbits)) & mask

    # Truncate to original size
    if isinstance(original_shape, torch.Size):
        numel = original_shape.numel()
    else:
        numel = 1
        for s in original_shape:
            numel *= s
    return flat[:numel].reshape(original_shape).int()


# ========================================
# GPTQ quantized Linear layer
# ========================================

class GPTQLinear(nn.Module):
    """
    GPTQ quantized Linear layer.

    Option: GemLite acceleration
    - use_gemlite=True: Use GemLite (3-5x faster when available)
    - use_gemlite=False: PyTorch implementation (default, no extra deps)
    - use_gemlite=None: Auto (use if available)

    GemLite requirements:
    - actorder=False (actorder not compatible with GemLite)
    - groupsize > 0 (group quantization required)
    - wbits in [2, 4, 8] (supported bit widths)

    Args:
        in_features: Input feature size
        out_features: Output feature size
        wbits: Quantization bit width
        groupsize: Group size (-1 = no grouping)
        actorder: Whether columns were reordered by activation order
        quantized_weight: Packed quantized weights (INT)
        scale: Scale (FP16)
        zero: Zero point (FP16)
        perm: Column permutation (when actorder=True)
        bias: Bias (optional)
        use_gemlite: GemLite flag (None=auto)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        wbits: int,
        groupsize: int,
        actorder: bool,
        quantized_weight: torch.Tensor,  # INT32, shape: (out_features, in_features)
        scale: torch.Tensor,             # FP16
        zero: torch.Tensor,              # FP16
        perm: Optional[torch.Tensor] = None,  # INT64
        bias: Optional[torch.Tensor] = None,
        device: str = "cuda",
        pack_weights: bool = True,  # Pack INT weights for memory efficiency
        use_gemlite: Optional[bool] = None,  # GemLite flag
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.wbits = wbits
        self.groupsize = groupsize
        self.actorder = actorder
        
        device = torch.device(device) if isinstance(device, str) else device
        
        # Decide whether to use GemLite
        if use_gemlite is None:
            use_gemlite = (
                HAS_GEMLITE_SUPPORT and
                is_gemlite_available() and
                not actorder and  # actorder not compatible with GemLite
                groupsize > 0 and  # group quantization required
                wbits in [2, 4, 8]  # supported bit widths
            )

        gemlite_layer = None
        if use_gemlite and HAS_GEMLITE_SUPPORT and not actorder:
            # Dequantize INT weights to FP16 for GemLite
            weight_dequant = quantized_weight.float()
            if groupsize == -1:
                # Per-channel: scale/zero shape is (out_features, 1)
                weight_dequant = scale * (weight_dequant - zero)
            else:
                # Grouped: scale/zero shape is (out_features, num_groups)
                num_groups = in_features // groupsize
                for i in range(num_groups):
                    start = i * groupsize
                    end = start + groupsize
                    weight_dequant[:, start:end] = (
                        scale[:, i:i+1] * (weight_dequant[:, start:end] - zero[:, i:i+1])
                    )
            weight_for_gemlite = weight_dequant.to(torch.float16)

            gemlite_layer = create_gemlite_linear(
                weight_for_gemlite,
                nbits=wbits,
                group_size=groupsize,
                device=device
            )

        if gemlite_layer is not None:
            # GemLite succeeded
            self.gemlite_layer = gemlite_layer
            self.using_gemlite = True
            self.packed_weight = None
            self.weight_shape = None
            self.quantized_weight = None
            # scale/zero managed inside GemLite
            self.scale = None
            self.zero = None
        else:
            # PyTorch implementation (fallback)
            self.gemlite_layer = None
            self.using_gemlite = False

            # Weight packing (memory efficiency)
            if pack_weights and wbits < 8:
                packed_weight = pack_int_weights(quantized_weight, wbits)
                self.register_buffer('packed_weight', packed_weight.to(device))
                self.register_buffer('weight_shape', torch.tensor(quantized_weight.shape))
                self.quantized_weight = None  # Save memory
            else:
                self.register_buffer('quantized_weight', quantized_weight.to(device))
                self.packed_weight = None
                self.weight_shape = None

            # Scale and zero point
            if scale.dim() == 1:
                scale = scale.unsqueeze(1)
            if zero.dim() == 1:
                zero = zero.unsqueeze(1)
            self.register_buffer('scale', scale.to(torch.float16).to(device))
            self.register_buffer('zero', zero.to(torch.float16).to(device))

        # Permutation order
        if perm is not None and actorder:
            self.register_buffer('perm', perm.to(device))
        else:
            self.perm = None

        # Bias
        if bias is not None:
            self.register_buffer('bias', bias.to(torch.float16).to(device))
        else:
            self.bias = None
        
        # Group index (when groupsize != -1)
        if groupsize != -1 and not self.using_gemlite:
            if actorder and perm is not None:
                # groups are defined in perm order.
                invperm = torch.argsort(perm)
                g_idx = (invperm // groupsize).to(torch.int32).to(device)
            else:
                g_idx = torch.arange(in_features, dtype=torch.int32, device=device) // groupsize
            self.register_buffer('g_idx', g_idx, persistent=False)
        else:
            self.g_idx = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Fast path when using GemLite
        if self.using_gemlite and self.gemlite_layer is not None:
            # GemLite handles all quantization internally
            output = self.gemlite_layer(x)
            if self.bias is not None:
                output = output + self.bias.to(output.dtype)
            return output

        # PyTorch implementation
        # Unpack weights if packed
        if self.packed_weight is not None:
            weight_int = unpack_int_weights(
                self.packed_weight, 
                self.wbits, 
                tuple(self.weight_shape.tolist())
            )
        else:
            weight_int = self.quantized_weight

        # Dequantize: weight = scale * (weight_int - zero)
        if self.groupsize == -1:
            # Per-channel: scale/zero shape is (out_features, 1) 
            weight = self.scale * (weight_int.float() - self.zero)
        else:
            scale_expanded = self.scale[:, self.g_idx]
            zero_expanded = self.zero[:, self.g_idx]
            weight = scale_expanded * (weight_int.float() - zero_expanded)

        # Linear op
        bias = self.bias.to(weight.dtype) if self.bias is not None else None
        output = F.linear(x, weight, bias)

        return output
        
    @classmethod
    def from_quantization_result(cls, result, bias=None, device="cuda", pack_weights=True, use_gemlite=None):
        """
        Build GPTQLinear from GPTQResult (quantizer.results).

        Convenience method using quantizer.results directly;
        makes from_linear_and_config() unnecessary.

        Args:
            result: GPTQResult from quantizer.results[name]
            bias: Optional bias tensor
            device: Device to place the layer on
            pack_weights: Pack INT weights for memory efficiency
            use_gemlite: Use GemLite acceleration (None=auto, True/False=force)

        Returns:
            GPTQLinear instance

        Example:
            >>> # Used inside save_quantized_model()
            >>> for name, module in model.named_modules():
            >>>     if name in quantizer.results:
            >>>         result = quantizer.results[name]
            >>>         quantized_layer = GPTQLinear.from_quantization_result(
            >>>             result, bias=module.bias, device=module.weight.device, use_gemlite=True
            >>>         )
        """
        return cls(
            in_features=result.quantized_weight.shape[1],
            out_features=result.quantized_weight.shape[0],
            wbits=result.wbits,
            groupsize=result.groupsize,
            actorder=result.actorder,
            quantized_weight=result.quantized_weight,
            scale=result.scale,
            zero=result.zero,
            perm=result.perm,
            bias=bias,
            device=device,
            pack_weights=pack_weights,
            use_gemlite=use_gemlite,
        )

