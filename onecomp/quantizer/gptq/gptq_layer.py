"""
GPTQ Quantized Linear Layer for Fast Inference

Implements a Linear layer for GPTQ-quantized models.
Runs inference in quantized (INT) form for better memory and speed.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from logging import getLogger
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _pack_rows(matrix: torch.Tensor, wbits: int) -> torch.Tensor:
    """Pack integer values along dim-0 into INT32 (AutoGPTQ continuous bit-stream).

    Args:
        matrix: (rows, cols), integer values in [0, 2^wbits - 1]
        wbits: Bit width (2, 3, 4, or 8)

    Returns:
        Packed INT32 tensor of shape (rows * wbits // 32, cols).
    """
    rows, cols = matrix.shape
    matrix = matrix.int()
    # Mask inputs to wbits so any out-of-range or negative value does not
    # corrupt neighboring slots of the packed INT32 word via sign-extended bits.
    # This also covers the GPTQ v1 qzero=0 case, where qzero - 1 is stored as -1.
    mask = (1 << wbits) - 1

    if wbits in (2, 4, 8):
        pack_factor = 32 // wbits
        assert (
            rows % pack_factor == 0
        ), f"rows ({rows}) must be divisible by pack_factor ({pack_factor})"
        reshaped = matrix.reshape(rows // pack_factor, pack_factor, cols)
        packed = torch.zeros(rows // pack_factor, cols, dtype=torch.int32, device=matrix.device)
        for i in range(pack_factor):
            packed |= (reshaped[:, i, :] & mask) << (i * wbits)
        return packed

    if wbits == 3:
        # 32 values → 96 bits → 3 INT32s (continuous bit-stream, no waste)
        assert rows % 32 == 0, f"rows ({rows}) must be divisible by 32 for 3-bit packing"
        num_blocks = rows // 32
        reshaped = matrix.reshape(num_blocks, 32, cols)
        packed = torch.zeros(num_blocks, 3, cols, dtype=torch.int32, device=matrix.device)
        for k in range(32):
            start_bit = k * 3
            word_idx = start_bit // 32
            bit_offset = start_bit % 32
            val = reshaped[:, k, :] & mask
            if bit_offset + 3 <= 32:
                packed[:, word_idx, :] |= val << bit_offset
            else:
                bits_first = 32 - bit_offset
                packed[:, word_idx, :] |= (val & ((1 << bits_first) - 1)) << bit_offset
                packed[:, word_idx + 1, :] |= val >> bits_first
        return packed.reshape(num_blocks * 3, cols)

    raise ValueError(f"Unsupported wbits for packing: {wbits}")


def _unpack_rows(packed: torch.Tensor, wbits: int, num_rows: int) -> torch.Tensor:
    """Unpack INT32 values along dim-0 back to integer values.

    Args:
        packed: (packed_rows, cols), INT32
        wbits: Bit width (2, 3, 4, or 8)
        num_rows: Number of original rows

    Returns:
        (num_rows, cols), INT32 values in [0, 2^wbits - 1]
    """
    packed_rows, cols = packed.shape
    mask = (1 << wbits) - 1

    if wbits in (2, 4, 8):
        pack_factor = 32 // wbits
        unpacked = torch.zeros(
            packed_rows, pack_factor, cols, dtype=torch.int32, device=packed.device
        )
        for i in range(pack_factor):
            unpacked[:, i, :] = (packed >> (i * wbits)) & mask
        return unpacked.reshape(packed_rows * pack_factor, cols)[:num_rows]

    if wbits == 3:
        num_blocks = packed_rows // 3
        packed_3d = packed.reshape(num_blocks, 3, cols)
        unpacked = torch.zeros(num_blocks, 32, cols, dtype=torch.int32, device=packed.device)
        for k in range(32):
            start_bit = k * 3
            word_idx = start_bit // 32
            bit_offset = start_bit % 32
            if bit_offset + 3 <= 32:
                unpacked[:, k, :] = (packed_3d[:, word_idx, :] >> bit_offset) & mask
            else:
                bits_first = 32 - bit_offset
                lower = (packed_3d[:, word_idx, :] >> bit_offset) & ((1 << bits_first) - 1)
                upper = packed_3d[:, word_idx + 1, :] & ((1 << (3 - bits_first)) - 1)
                unpacked[:, k, :] = lower | (upper << bits_first)
        return unpacked.reshape(num_blocks * 32, cols)[:num_rows]

    raise ValueError(f"Unsupported wbits for unpacking: {wbits}")


def pack_int_weights(weights: torch.Tensor, wbits: int) -> torch.Tensor:
    """Pack quantized weights in AutoGPTQ format.

    Args:
        weights: (out_features, in_features), integer values in [0, 2^wbits - 1]
        wbits: Bit width (2, 3, 4, 8)

    Returns:
        Packed INT32 tensor, shape (in_features * wbits // 32, out_features).
    """
    return _pack_rows(weights.t().contiguous(), wbits)


def unpack_int_weights(
    packed: torch.Tensor, wbits: int, original_shape: Union[torch.Size, Tuple[int, ...]]
) -> torch.Tensor:
    """Unpack AutoGPTQ-format packed weights.

    Args:
        packed: (in_features * wbits // 32, out_features) INT32
        wbits: Bit width (2, 3, 4, 8)
        original_shape: (out_features, in_features)

    Returns:
        (out_features, in_features), INT32
    """
    in_features = original_shape[1]
    unpacked = _unpack_rows(packed, wbits, in_features)  # (in_features, out_features)
    return unpacked.t().contiguous()


def pack_zeros(zeros: torch.Tensor, wbits: int) -> torch.Tensor:
    """Pack zero points in AutoGPTQ format (pack along out_features dim).

    Args:
        zeros: (num_groups, out_features), integer zero points
        wbits: Bit width

    Returns:
        Packed INT32 tensor, shape (num_groups, out_features * wbits // 32).
    """
    return _pack_rows(zeros.t().contiguous(), wbits).t().contiguous()


def unpack_zeros(packed_zeros: torch.Tensor, wbits: int, out_features: int) -> torch.Tensor:
    """Unpack zero points from AutoGPTQ format.

    Args:
        packed_zeros: (num_groups, packed_cols) INT32
        wbits: Bit width
        out_features: Original out_features

    Returns:
        (num_groups, out_features), INT32
    """
    return _unpack_rows(packed_zeros.t().contiguous(), wbits, out_features).t().contiguous()


def normalize_scale_zero(tensor: torch.Tensor, out_features: int) -> torch.Tensor:
    """Normalize scale/zero to (num_groups, out_features) for AutoGPTQ.

    Args:
        tensor: Scale or zero tensor in any of the layouts OneComp produces —
            ``(out_features,)``, ``(out_features, 1)``, or already
            ``(num_groups, out_features)``.
        out_features: Output feature count, used to disambiguate the
            ``(out_features, 1)`` per-channel layout.

    Returns:
        (num_groups, out_features) tensor. Any other shape is returned as-is,
        unvalidated.
    """
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)  # (out_features,) → (1, out_features)
    if tensor.dim() == 2 and tensor.shape == (out_features, 1):
        return tensor.t()  # (out_features, 1) → (1, out_features)
    return tensor  # (num_groups, out_features)


# Bit widths OneComp can store in the AutoGPTQ packed INT32 format
# (see ``_pack_rows`` / ``_unpack_rows``). Widths outside this set — notably
# JointQ's 1-bit weights — have no packing layout and are kept unpacked.
PACKABLE_WBITS = (2, 3, 4, 8)


def _normalize_wbits(wbits: int) -> int:
    """Return an integral ``wbits`` as a built-in int.

    Accepts integers and integer-valued floats. Rejects ``bool``, non-integral
    or non-finite floats, and all other types with ``ValueError``.
    """
    if isinstance(wbits, bool):
        raise ValueError(f"wbits must be an integer value, got {wbits!r}.")
    if isinstance(wbits, int):
        # Normalize int subclasses as well.
        return int(wbits)
    if isinstance(wbits, float) and wbits.is_integer():
        return int(wbits)
    raise ValueError(f"wbits must be an integer value, got {wbits!r}.")


def is_packable_wbits(wbits: int) -> bool:
    """Return whether GPTQ INT tensors of this bit width can be packed.

    Single source of truth for the packable-width check used by
    ``GPTQLinear.pack_in_place`` and (later) the save path's
    ``_packable_gptq_wbits`` helper, so the two cannot drift apart.

    Comparisons are exact: ``2.0`` is accepted, while ``2.5`` is not rounded.
    """
    return wbits in PACKABLE_WBITS


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
        quantized_weight: Quantized weights INT32, shape: (out_features, in_features)
        scale: Scale (FP16)
        zero: Zero point (FP16)
        perm: Column permutation (when actorder=True)
        bias: Bias (optional)
        pack_weights: Whether to store qweight/qzeros in packed AutoGPTQ format
        use_gemlite: GemLite flag (None=auto)
        input_weight_is_packed: Whether quantized_weight is already packed
        input_zero_is_packed: Whether zero is already packed
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        in_features: int,
        out_features: int,
        wbits: int,
        groupsize: int,
        actorder: bool,
        quantized_weight: torch.Tensor,  # INT32, shape: (out_features, in_features)
        scale: torch.Tensor,  # FP16
        zero: torch.Tensor,  # FP16
        perm: Optional[torch.Tensor] = None,  # INT64
        bias: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = "cuda",
        pack_weights: bool = True,  # Pack INT weights for memory efficiency
        use_gemlite: Optional[bool] = None,  # GemLite flag
        input_weight_is_packed: bool = False,
        input_zero_is_packed: bool = False,
    ):
        super().__init__()

        # Normalize before storing or using wbits in bit operations.
        wbits = _normalize_wbits(wbits)

        self.in_features = in_features
        self.out_features = out_features
        self.wbits = wbits
        self.groupsize = groupsize
        self.actorder = actorder

        device = torch.device(device) if isinstance(device, str) else device
        original_shape = (out_features, in_features)
        wbits_mask = (1 << wbits) - 1
        should_pack_weights = bool(pack_weights and is_packable_wbits(wbits))
        if pack_weights and not should_pack_weights:
            logger.warning(
                "pack_weights=True was requested, but GPTQ wbits=%d does not support "
                "packing; storing weights unpacked.",
                wbits,
            )

        quantized_weight_unpacked = None
        zero_int = None
        zero_raw = None

        def _get_quantized_weight_unpacked():
            nonlocal quantized_weight_unpacked
            if quantized_weight_unpacked is None:
                if input_weight_is_packed:
                    quantized_weight_unpacked = unpack_int_weights(
                        quantized_weight.to(torch.int32), wbits, original_shape
                    )
                else:
                    quantized_weight_unpacked = quantized_weight
            return quantized_weight_unpacked

        def _get_zero_raw():
            nonlocal zero_raw
            if zero_raw is None:
                if input_zero_is_packed:
                    zero_raw = (_get_zero_int() + 1) & wbits_mask
                else:
                    # Normalize to the saved/inference layout used by AutoGPTQ.
                    zero_raw = normalize_scale_zero(zero, out_features)
            return zero_raw

        def _get_zero_int():
            nonlocal zero_int
            if zero_int is None:
                if input_zero_is_packed:
                    zero_int = unpack_zeros(zero.to(torch.int32), wbits, out_features)
                else:
                    zero_int = _get_zero_raw().round().to(torch.int32) - 1
            return zero_int

        # Decide whether to use GemLite
        if use_gemlite is None:
            use_gemlite = (
                HAS_GEMLITE_SUPPORT
                and is_gemlite_available()
                and not actorder  # actorder not compatible with GemLite
                and groupsize > 0  # group quantization required
                and wbits in [2, 4, 8]  # supported bit widths
            )

        gemlite_layer = None
        if use_gemlite and HAS_GEMLITE_SUPPORT and not actorder:
            # Dequantize INT weights to FP16 for GemLite
            weight_dequant = _get_quantized_weight_unpacked().float()
            zero_raw_for_gemlite = _get_zero_raw()
            if groupsize == -1:
                zero_for_gemlite = zero_raw_for_gemlite
                if zero_for_gemlite.dim() == 2 and zero_for_gemlite.shape == (1, out_features):
                    zero_for_gemlite = zero_for_gemlite.t()
                weight_dequant = scale * (weight_dequant - zero_for_gemlite)
            else:
                # Grouped: scales/qzeros shape is (num_groups, out_features)
                num_groups = in_features // groupsize
                for i in range(num_groups):
                    start, end = i * groupsize, (i + 1) * groupsize
                    weight_dequant[:, start:end] = scale[i, :].unsqueeze(1) * (
                        weight_dequant[:, start:end] - zero_raw_for_gemlite[i, :].unsqueeze(1)
                    )
            weight_for_gemlite = weight_dequant.to(torch.float16)

            gemlite_layer = create_gemlite_linear(
                weight_for_gemlite, nbits=wbits, group_size=groupsize, device=device
            )

        self.using_gemlite = False

        # --- Weight: AutoGPTQ-compatible packed format ---
        self.checkpoint_format = "gptq"  # always v1: OneComp generates v1 tensors unconditionally
        if should_pack_weights:
            if input_weight_is_packed:
                packed = quantized_weight.to(torch.int32)
            else:
                packed = pack_int_weights(_get_quantized_weight_unpacked(), wbits)
            self.register_buffer("qweight", packed.to(device))
        else:
            self.register_buffer("qweight", _get_quantized_weight_unpacked().to(device))

        # --- Scales: normalize to (num_groups, out_features) ---
        scale = normalize_scale_zero(scale, out_features)
        self.register_buffer("scales", scale.to(torch.float16).to(device))

        # --- Zeros: normalize then pack (AutoGPTQ v1 convention) ---
        # v1: store (raw_zero - 1); vLLM exllama kernel restores via stored + 1
        if should_pack_weights:
            if input_zero_is_packed:
                self.register_buffer("qzeros", zero.to(torch.int32).to(device))
            else:
                self.register_buffer("qzeros", pack_zeros(_get_zero_int(), wbits).to(device))
        else:
            self.register_buffer("qzeros", _get_zero_int().to(device))

        self._gemlite_layer = gemlite_layer if gemlite_layer is not None else None
        if self._gemlite_layer is not None:
            self.using_gemlite = True

        # Permutation order
        if perm is not None and actorder:
            self.register_buffer("perm", perm.to(device))
        else:
            self.perm = None

        # Bias
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16).to(device))
        else:
            self.bias = None

        # Group index (when groupsize != -1)
        if groupsize != -1:
            if actorder and perm is not None:
                # groups are defined in perm order.
                invperm = torch.argsort(perm)
                g_idx = (invperm // groupsize).to(torch.int32).to(device)
            else:
                g_idx = torch.arange(in_features, dtype=torch.int32, device=device) // groupsize
        else:
            g_idx = torch.zeros(in_features, dtype=torch.int32, device=device)
        self.register_buffer("g_idx", g_idx)
        self._weight_is_packed = should_pack_weights

    def unpack_in_place(self) -> None:
        """Expand ``qweight``/``qzeros`` to dense INT form, in place.

        Re-registers the ``qweight`` and ``qzeros`` buffers as their unpacked
        equivalents and flips ``_weight_is_packed`` to ``False``.

        Intended for post-processes (e.g. LoRA SFT) that train against a base
        layer: keeping the weights unpacked avoids re-running the unpack on
        every forward, which is costly over many epochs.

        Idempotent — a no-op when the layer is already unpacked. The v1 qzeros
        ``-1`` offset is preserved verbatim because ``unpack_zeros`` does not
        reinterpret the stored integers (a stored ``-1`` simply re-reads as
        ``2^wbits - 1``, which ``forward`` wraps back to ``0`` modularly).
        """
        if not self._weight_is_packed:
            return  # already unpacked → idempotent no-op
        if self.using_gemlite:
            # GemLite holds the dequantized weights; the qweight buffer is not
            # the source of truth, so mutating it would be meaningless. Post-
            # process and save paths always pass use_gemlite=False, so this
            # guard is purely defensive.
            logger.warning(
                "unpack_in_place skipped: GPTQLinear is using GemLite "
                "(qweight buffer is not the source of truth)."
            )
            return

        device = self.qweight.device
        weight_int = unpack_int_weights(
            self.qweight, self.wbits, (self.out_features, self.in_features)
        )
        zeros_int = unpack_zeros(self.qzeros, self.wbits, self.out_features)
        self.register_buffer("qweight", weight_int.contiguous().to(device))
        self.register_buffer("qzeros", zeros_int.contiguous().to(device))
        self._weight_is_packed = False

    def pack_in_place(self) -> None:
        """Re-pack ``qweight``/``qzeros`` into the AutoGPTQ INT32 form, in place.

        Re-registers the ``qweight`` and ``qzeros`` buffers as their packed
        equivalents and flips ``_weight_is_packed`` to ``True``. Used to restore
        a layer to its packed state after a post-process unpacked it for
        training.

        Idempotent — a no-op when the layer is already packed. Guards:

        - Non-packable ``wbits`` (anything outside :data:`PACKABLE_WBITS`, e.g.
          JointQ's 1-bit) → no-op with a log message; the layer stays unpacked.
        - ``using_gemlite=True`` → no-op (qweight is not the source of truth).

        The v1 qzeros ``-1`` offset is preserved verbatim because ``pack_zeros``
        does not reinterpret the stored integers (``_pack_rows`` masks a stored
        ``-1`` to ``2^wbits - 1`` rather than corrupting neighboring slots).
        """
        if self._weight_is_packed:
            return  # already packed → idempotent no-op
        if self.using_gemlite:
            logger.warning(
                "pack_in_place skipped: GPTQLinear is using GemLite "
                "(qweight buffer is not the source of truth)."
            )
            return
        if not is_packable_wbits(self.wbits):
            # e.g. JointQ 1-bit: no AutoGPTQ packing layout exists, so leave the
            # weights unpacked (matches how such layers are constructed/saved).
            logger.info(
                "pack_in_place no-op: wbits=%d is not packable; " "leaving GPTQLinear unpacked.",
                self.wbits,
            )
            return

        device = self.qweight.device
        packed_weight = pack_int_weights(self.qweight.to(torch.int32), self.wbits)
        packed_zeros = pack_zeros(self.qzeros.to(torch.int32), self.wbits)
        self.register_buffer("qweight", packed_weight.contiguous().to(device))
        self.register_buffer("qzeros", packed_zeros.contiguous().to(device))
        self._weight_is_packed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Fast path when using GemLite
        if self.using_gemlite and self._gemlite_layer is not None:
            # GemLite is initialized with fp16 weights; cast input to fp16
            # to avoid Triton dtype mismatch (e.g. bf16 models like Gemma 4),
            # then restore the original dtype.
            orig_dtype = x.dtype
            output = self._gemlite_layer(x.to(torch.float16))
            output = output.to(orig_dtype)
            if self.bias is not None:
                output = output + self.bias.to(output.dtype)
            return output

        # Unpack weights: (out_features, in_features)
        if self._weight_is_packed:
            weight_int = unpack_int_weights(
                self.qweight, self.wbits, (self.out_features, self.in_features)
            )
        else:
            weight_int = self.qweight

        # Unpack zeros: (num_groups, out_features)
        # OneComp always writes v1 (qzeros stored with -1 offset), but from_saved_state
        # may load external checkpoints saved as gptq_v2 (qzeros stored as-is, no offset).
        # The v1 restoration is modular: stored 2^wbits-1 (from qzero=0) must wrap to 0,
        # matching the AutoGPTQ/vLLM exllama kernel's modular arithmetic.
        _v1 = getattr(self, "checkpoint_format", "gptq") != "gptq_v2"
        wbits_mask = (1 << self.wbits) - 1
        if self._weight_is_packed:
            zeros = unpack_zeros(self.qzeros, self.wbits, self.out_features)
            if _v1:
                zeros = (zeros + 1) & wbits_mask
        else:
            zeros = ((self.qzeros + 1) & wbits_mask) if _v1 else self.qzeros

        # Dequantize: weight = scale * (weight_int - zero)
        # scales: (num_groups, out_features), g_idx: (in_features,)
        scale_expanded = self.scales[self.g_idx, :].T  # (out_features, in_features)
        zero_expanded = zeros[self.g_idx, :].T  # (out_features, in_features)
        weight = scale_expanded * (weight_int.float() - zero_expanded)

        # Cast dequantized weight to input dtype (e.g. float32 -> float16)
        weight = weight.to(x.dtype)

        # MPS F.linear produces incorrect results on non-contiguous tensors.
        if weight.device.type == "mps":
            weight = weight.contiguous()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        output = F.linear(x, weight, bias)

        return output

    @classmethod
    def from_quantization_result(  # pylint: disable=too-many-positional-arguments
        cls, result, bias=None, device="cuda", pack_weights=True, use_gemlite=None
    ):
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
                Packed qweight inputs use result.qweight_original_shape for layer size.

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
        original_shape = getattr(result, "qweight_original_shape", None)
        if original_shape is None:
            out_features, in_features = result.qweight.shape
        else:
            out_features, in_features = original_shape

        return cls(
            in_features=in_features,
            out_features=out_features,
            wbits=result.wbits,
            groupsize=result.groupsize,
            actorder=result.actorder,
            quantized_weight=result.qweight,
            scale=result.scales,
            zero=result.qzeros,
            perm=result.perm,
            bias=bias,
            device=device,
            pack_weights=pack_weights,
            use_gemlite=use_gemlite,
            input_weight_is_packed=bool(getattr(result, "qweight_is_packed", False)),
            input_zero_is_packed=bool(getattr(result, "qzeros_is_packed", False)),
        )

    @classmethod
    def from_saved_state(  # pylint: disable=too-many-positional-arguments
        cls,
        layer_state_dict: dict,
        in_features: int,
        out_features: int,
        wbits: int,
        groupsize: int = -1,
        actorder: bool = False,
        empty: bool = False,
        checkpoint_format: str = "gptq",
    ):
        """Build GPTQLinear from saved state_dict tensors (AutoGPTQ format).

        Args:
            layer_state_dict: Sub-state_dict for this layer (keys: qweight, scales, qzeros, etc.)
            in_features: Input feature size.
            out_features: Output feature size.
            wbits, groupsize, actorder: Quantization config.
            empty: If True, create zero buffers of the same shape (for
                "replace then load_state_dict" flow). If False, use tensors
                from layer_state_dict directly.
            checkpoint_format: ``"gptq"`` (v1, qzeros stored with -1 offset) or
                ``"gptq_v2"`` (v2, qzeros stored as-is). OneComp always writes v1;
                this param exists to support external AutoGPTQ v2 checkpoints.

        Returns:
            GPTQLinear instance.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)

        # __init__ is bypassed, so normalize here as well.
        wbits = _normalize_wbits(wbits)

        self.in_features = in_features
        self.out_features = out_features
        self.wbits = wbits
        self.groupsize = groupsize
        self.actorder = actorder
        self.checkpoint_format = checkpoint_format
        # JointQ wbits=1 is saved with pack_weights=False
        # (GPTQLinear packing does not support 1-bit), so load it unpacked as well.
        self._weight_is_packed = wbits != 1

        def _t(k):
            t = layer_state_dict[k]
            return torch.zeros_like(t) if empty else t

        dev = layer_state_dict["qweight"].device
        self.register_buffer("qweight", _t("qweight"))
        self.register_buffer("scales", _t("scales"))
        self.register_buffer("qzeros", _t("qzeros"))

        g_idx = layer_state_dict.get("g_idx")
        if g_idx is not None:
            self.register_buffer("g_idx", torch.zeros_like(g_idx) if empty else g_idx)
        else:
            if groupsize != -1:
                self.register_buffer(
                    "g_idx",
                    torch.arange(in_features, dtype=torch.int32, device=dev) // groupsize,
                )
            else:
                self.register_buffer(
                    "g_idx",
                    torch.zeros(in_features, dtype=torch.int32, device=dev),
                )

        perm = layer_state_dict.get("perm")
        if perm is not None:
            self.register_buffer("perm", torch.zeros_like(perm) if empty else perm)
        else:
            self.perm = None

        bias = layer_state_dict.get("bias")
        if bias is not None:
            self.register_buffer("bias", torch.zeros_like(bias) if empty else bias)
        else:
            self.bias = None

        self.using_gemlite = False
        self._gemlite_layer = None

        return self
