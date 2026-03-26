"""
GemLite acceleration layer.

Optional: When GemLite is installed, quantized layers are accelerated
via Triton kernels.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Optional GemLite/HQQ imports
try:
    from gemlite.core import GemLiteLinearTriton, DType
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    HAS_GEMLITE = True
except (ImportError, AttributeError):
    HAS_GEMLITE = False


# Constants
DEFAULT_GROUP_SIZE = 128


def pad_cols_to_multiple(t: torch.Tensor, multiple: int, value: int) -> torch.Tensor:
    """Pad columns (in_features) to a multiple of the given value."""
    if t.ndim != 2:
        return t
    pad_cols = (-t.shape[1]) % multiple
    if pad_cols:
        t = F.pad(t, (0, pad_cols, 0, 0), value=value)  # (left,right,top,bottom) = (0,pad,0,0)
    return t


def create_gemlite_linear(
    weights: torch.Tensor,
    nbits: int = 1,
    group_size: int = DEFAULT_GROUP_SIZE,
    device: torch.device = torch.device("cuda"),
) -> Optional[nn.Module]:
    """
    Create a GemLiteLinear layer.

    Args:
        weights: Quantized weight matrix (out_features, in_features)
        nbits: Bit width (1=Binary, 2-8=INT quantization)
        group_size: Group size
        device: Device

    Returns:
        GemLiteLinearTriton or None on failure.

    Note:
        - Returns None if GemLite is not available
        - Returns None if in_features is not a multiple of group_size
        - Returns None on error (automatic fallback)

    Usage:
        >>> # DBF (1-bit)
        >>> gemlite_layer = create_gemlite_linear(binary_weights, nbits=1)

        >>> # GPTQ (3-bit)
        >>> gemlite_layer = create_gemlite_linear(quantized_weights, nbits=3)
    """
    if not HAS_GEMLITE:
        return None

    try:
        orig_out, orig_in = weights.shape

        # GemLite requires in_features to be a multiple of group_size
        if orig_in % group_size != 0:
            return None

        # Pad columns (in_features) to a multiple of group_size
        weights = pad_cols_to_multiple(weights, group_size, 1)
        out_features, in_features = weights.shape

        # Create a temporary Linear layer
        linear = nn.Linear(in_features, out_features, bias=False, device="cpu")
        linear.weight = nn.Parameter(weights.to(torch.float16), requires_grad=False)

        # HQQ quantization config
        quant_config = BaseQuantizeConfig(
            nbits=nbits, group_size=group_size, quant_zero=False, quant_scale=False, axis=1
        )
        quant_config["weight_quant_params"]["optimize"] = False

        # Create HQQLinear
        hqq_layer = HQQLinear(
            linear,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device="cpu",
            del_orig=False,
        )

        # Create GemLiteLinear
        gemlite_linear = GemLiteLinearTriton(
            W_nbits=nbits,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            input_dtype=DType.FP16,
            output_dtype=DType.FP16,
        )

        # Get metadata
        scale = hqq_layer.meta["scale"].clone()
        zero = (
            hqq_layer.meta["zero"].clone() if "zero" in hqq_layer.meta else torch.zeros_like(scale)
        )

        # Pack
        gemlite_linear.pack(
            hqq_layer.unpack(dtype=torch.uint8).view((out_features, in_features)),
            scale,
            zero,
            bias=None,
        )

        return gemlite_linear.to(device)

    except Exception as e:
        import warnings

        warnings.warn(
            f"GemLite initialization failed: {e}. Falling back to standard implementation."
        )
        return None


def is_gemlite_available() -> bool:
    """Return whether GemLite is available."""
    return HAS_GEMLITE
