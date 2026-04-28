"""Copyright 2025-2026 Fujitsu Ltd."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemlite.core import GemLiteLinearTriton, DType
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
from vllm.logger import init_logger

HQQLinear.set_backend(HQQBackend.PYTORCH)
logger = init_logger(__name__)
GROUP_SIZE = 128


def pad_to_multiple(tensor, multiple=32, value=0):
    """Pad a tensor to the multiple of `multiple` (GemLite-compatible)."""
    pad = []
    for dim in reversed(range(tensor.ndim)):
        size = tensor.shape[dim]
        padding_needed = (multiple - size % multiple) % multiple
        pad.extend([0, padding_needed])
    return F.pad(tensor, pad, mode="constant", value=value)


def pad_cols_to_multiple(t: torch.Tensor, multiple: int, value: int) -> torch.Tensor:
    """Align only axis=1 (columns=in_feature) to group_size."""
    if t.ndim != 2:
        return t
    pad_cols = (-t.shape[1]) % multiple
    if pad_cols:
        # (left,right,top,bottom) = (0,pad,0,0)
        t = F.pad(t, (0, pad_cols, 0, 0), value=value)
    return t


def _pad_to_group(t: torch.Tensor, gs: int, value: int = 1) -> torch.Tensor:
    """Pad both rows and cols of a 2-D tensor to multiples of *gs*."""
    pad_rows = (-t.shape[0]) % gs
    pad_cols = (-t.shape[1]) % gs
    if pad_rows or pad_cols:
        t = F.pad(t, (0, pad_cols, 0, pad_rows), value=value)
    return t


def get_gemlite_linear(weights: torch.Tensor):
    """Build a GemLiteLinear layer (rows & cols padded to GROUP_SIZE)."""
    try:
        W_nbits = 1

        weights = _pad_to_group(weights, GROUP_SIZE, 1)
        out_features, in_features = weights.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        linear = nn.Linear(in_features, out_features, bias=False, device=device)
        linear.weight = nn.Parameter(weights.to(torch.bfloat16), requires_grad=False)

        quant_config = BaseQuantizeConfig(
            nbits=W_nbits, group_size=GROUP_SIZE, quant_zero=False, quant_scale=False, axis=1
        )
        quant_config["weight_quant_params"]["optimize"] = False

        hqq_layer = HQQLinear(
            linear,
            quant_config=quant_config,
            compute_dtype=torch.bfloat16,
            device=device,
            del_orig=False,
        )

        gemlite_linear = GemLiteLinearTriton(
            W_nbits=W_nbits,
            group_size=GROUP_SIZE,
            in_features=in_features,
            out_features=out_features,
            input_dtype=DType.BF16,
            output_dtype=DType.BF16,
        )

        # Fall back to a zero tensor when 'zero' is absent in meta (avoid passing None)
        scale = hqq_layer.meta["scale"].clone()
        zero = (
            hqq_layer.meta["zero"].clone() if "zero" in hqq_layer.meta else torch.zeros_like(scale)
        )

        gemlite_linear.pack(
            hqq_layer.unpack(dtype=torch.uint8).view((out_features, in_features)),
            scale,
            zero,
            bias=None,
        )
        return gemlite_linear
    except Exception as e:
        logger.warning(f"[DBF] Failed to create GemLite linear: {e}")
        return None


def unpack_sign_bits(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpack a uint8 tensor into a ±1 tensor.

    Args:
        packed: Packed uint8 tensor.
        original_shape: Shape of the original tensor.

    Returns:
        unpacked: Tensor unpacked ±1 values.
    """
    device = packed.device
    dtype = torch.float16

    # Unpack bits in the same order as my_unpack in the quantization script:
    # extract bits as (x >> 7, >> 6, >> 5, ..., >> 0)
    int8_tensor = packed

    # Bit-shift amounts: 7, 6, 5, ..., 0
    shifts = torch.arange(7, -1, -1, device=device).view(1, 8)
    expanded_int8 = int8_tensor.unsqueeze(-1)

    # Extract each bit and convert {0, 1} → {-1, +1}
    unpacked_bits = ((expanded_int8 >> shifts) & 1).to(dtype)
    unpacked_bits = unpacked_bits.view(int8_tensor.shape[0], -1)

    # {0,1} → {-1,+1}
    fp16_tensor = (unpacked_bits.to(torch.int8)) * 2 - 1

    # Reshape to original shape, truncating any excess elements
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    return fp16_tensor.flatten()[:total_elements].reshape(original_shape)


class DBFLinear_GEMLITE(nn.Module):
    def __init__(self, w_bit, in_features, out_features, bias, dev, training=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        mid_features1 = int(w_bit * (in_features * out_features) / (in_features + out_features))
        mid_features = min(min(in_features, out_features), mid_features1)
        self.mid_features = mid_features
        self.w_bit = w_bit
        self.training = training

        # logger.debug(f"in_features={in_features} out_features={out_features} mid_features={mid_features} w_bit={w_bit} ")
        # assert in_features % (8 // self.w_bit) == 0
        self.register_buffer(
            "scaling0",
            torch.zeros(
                (in_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "bp1",
            torch.zeros(
                ((mid_features * in_features + (8) - 1) // (8)),
                dtype=torch.uint8,
                device=dev,
            ),
        )
        self.register_buffer(
            "scaling2",
            torch.zeros(
                (mid_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "bp3",
            torch.zeros(
                ((out_features * mid_features + (8) - 1) // (8)),
                dtype=torch.uint8,
                device=dev,
            ),
        )
        self.register_buffer(
            "scaling4",
            torch.zeros(
                (out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear,
        w_bit,
        init_only=False,
    ):
        dbf_linear = cls(
            w_bit,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return dbf_linear

        return dbf_linear

    def post_init(self):
        bp1_int8 = unpack_sign_bits(self.bp1, (self.mid_features, self.in_features)).to(torch.int8)
        bp3_int8 = unpack_sign_bits(self.bp3, (self.out_features, self.mid_features)).to(
            torch.int8
        )
        bp1_gemlite = pad_cols_to_multiple(bp1_int8, GROUP_SIZE, 1)
        bp3_gemlite = pad_cols_to_multiple(bp3_int8, GROUP_SIZE, 1)
        self.binary1 = get_gemlite_linear(bp1_gemlite)
        self.binary3 = get_gemlite_linear(bp3_gemlite)

    def forward(self, x):
        assert hasattr(self, "binary1"), (
            "module.post_init() must be called before module.forward(). "
            "Use gemlitelinear_post_init() on the whole model."
        )
        assert hasattr(self, "binary3"), (
            "module.post_init() must be called before module.forward(). "
            "Use gemlitelinear_post_init() on the whole model."
        )

        input_dtype = x.dtype
        x = x * self.scaling0.to(x.dtype)
        if torch.bfloat16 != x.dtype:
            x = x.bfloat16()
        if x.shape[-1] != ((x.shape[-1] + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE:
            pad_size = ((x.shape[-1] + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE - x.shape[
                -1
            ]  # 24
            x = F.pad(x, (0, pad_size), mode="constant", value=0)
        x = self.binary1(x)
        x = x[..., : self.mid_features]
        x = x * self.scaling2.to(x.dtype)
        if torch.bfloat16 != x.dtype:
            x = x.bfloat16()
        if x.shape[-1] != ((x.shape[-1] + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE:
            pad_size = ((x.shape[-1] + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE - x.shape[-1]
            x = F.pad(x, (0, pad_size), mode="constant", value=0)
        x = self.binary3(x)
        x = x[..., : self.out_features]
        x = x * self.scaling4.to(x.dtype)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        if input_dtype != x.dtype:
            x = x.to(dtype=input_dtype)
        return x


def gemlitelinear_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, DBFLinear_GEMLITE):
            submodule.post_init()

    return model
