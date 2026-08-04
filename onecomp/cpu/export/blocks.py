"""Lossless packing of GPTQ integer weights into GGUF legacy block formats.

The key observation behind OneComp's direct GPTQ -> GGUF export is that the
GGUF legacy quantization types use a fixed block size of 32 and a single
(scale[, min]) per block. OneComp's GPTQ weights are grouped (default group
size 128) with a per-group, per-output-channel scale and zero point. Because
the GPTQ group size is a multiple of 32, every GGUF block of 32 input features
falls entirely inside a single GPTQ group, so we can reuse the exact GPTQ
integer codes and scales without any re-quantization:

  * 4-bit symmetric  (zero == 2^(b-1) == 8)  -> Q4_0   (value = d * (q - 8))
  * 4-bit asymmetric                          -> Q4_1   (value = d * q + m)
  * 8-bit symmetric  (zero == 128)            -> Q8_0   (value = d * q8)

This means a QEP/GPTQ-optimized checkpoint keeps its accuracy when exported to
GGUF, unlike the naive "dequantize to fp16 then re-quantize with llama-quantize"
path which throws the GPTQ error correction away.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from gguf.constants import GGMLQuantizationType
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The 'gguf' package is required for GGUF export. "
        "Install it with: pip install 'onecomp[gguf]' (or [llamacpp])."
    ) from exc

QK = 32  # GGUF legacy block size (elements per block)


class UnsupportedGPTQLayout(ValueError):
    """Raised when a GPTQ layer cannot be losslessly mapped to a GGUF block type."""


def select_gguf_type(wbits: int, sym: bool) -> GGMLQuantizationType:
    """Pick the GGUF legacy block type that losslessly represents a GPTQ layer.

    Args:
        wbits: Quantization bit width of the GPTQ layer.
        sym: Whether the GPTQ layer is symmetric.

    Returns:
        The matching ``GGMLQuantizationType``.

    Raises:
        UnsupportedGPTQLayout: If no lossless mapping exists.
    """
    if wbits == 4:
        return GGMLQuantizationType.Q4_0 if sym else GGMLQuantizationType.Q4_1
    if wbits == 8 and sym:
        return GGMLQuantizationType.Q8_0
    raise UnsupportedGPTQLayout(
        f"No lossless GGUF block type for wbits={wbits}, sym={sym}. "
        "Supported: 4-bit (sym->Q4_0, asym->Q4_1), 8-bit symmetric (->Q8_0). "
        "Use the dequantize->convert->llama-quantize fallback path instead."
    )


def _block_group_indices(in_features: int, groupsize: int) -> np.ndarray:
    """Group index for each 32-element GGUF block along the input dimension."""
    if in_features % QK != 0:
        raise UnsupportedGPTQLayout(
            f"in_features ({in_features}) must be a multiple of {QK} for GGUF export."
        )
    n_blocks = in_features // QK
    if groupsize == -1:
        return np.zeros(n_blocks, dtype=np.int64)
    if groupsize % QK != 0:
        raise UnsupportedGPTQLayout(
            f"GPTQ groupsize ({groupsize}) must be a multiple of {QK} (or -1) so that "
            "each GGUF block stays inside one GPTQ group."
        )
    # block b covers input features [32b, 32b+32); all inside group (32b)//groupsize.
    return (np.arange(n_blocks, dtype=np.int64) * QK) // groupsize


def _gather_per_block_scale_zero(
    scales: np.ndarray,
    zeros: Optional[np.ndarray],
    in_features: int,
    out_features: int,
    groupsize: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Expand per-group (num_groups, out) tensors to per-block (out, n_blocks)."""
    g_idx = _block_group_indices(in_features, groupsize)  # (n_blocks,)
    # scales: (num_groups, out_features) -> (out_features, n_blocks)
    d = scales[g_idx, :].T.astype(np.float32, copy=False)
    z = None
    if zeros is not None:
        z = zeros[g_idx, :].T.astype(np.float32, copy=False)
    assert d.shape == (out_features, len(g_idx)), (d.shape, out_features, len(g_idx))
    return np.ascontiguousarray(d), (np.ascontiguousarray(z) if z is not None else None)


def _f16_bytes(values: np.ndarray) -> np.ndarray:
    """View float values as little-endian fp16 byte pairs along a new last axis."""
    le = np.ascontiguousarray(values.astype("<f2"))
    return le.view(np.uint8).reshape(*values.shape, 2)


def pack_q4_0(q_int: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Pack 4-bit symmetric codes into Q4_0 blocks.

    Args:
        q_int: (out, in) uint codes in [0, 15].
        d: (out, n_blocks) per-block scale.

    Returns:
        (out, n_blocks * 18) uint8 array.
    """
    out_f, in_f = q_int.shape
    n_blocks = in_f // QK
    q = q_int.reshape(out_f, n_blocks, QK).astype(np.uint8)
    low = q[:, :, :16]
    high = q[:, :, 16:]
    qs = (low | (high << 4)).astype(np.uint8)  # (out, n_blocks, 16)
    d_bytes = _f16_bytes(d)  # (out, n_blocks, 2)
    block = np.concatenate([d_bytes, qs], axis=-1)  # (out, n_blocks, 18)
    return block.reshape(out_f, n_blocks * 18)


def pack_q4_1(q_int: np.ndarray, d: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Pack 4-bit asymmetric codes into Q4_1 blocks (value = d*q + m)."""
    out_f, in_f = q_int.shape
    n_blocks = in_f // QK
    q = q_int.reshape(out_f, n_blocks, QK).astype(np.uint8)
    qs = (q[:, :, :16] | (q[:, :, 16:] << 4)).astype(np.uint8)
    d_bytes = _f16_bytes(d)
    m_bytes = _f16_bytes(m)
    block = np.concatenate([d_bytes, m_bytes, qs], axis=-1)  # (out, n_blocks, 20)
    return block.reshape(out_f, n_blocks * 20)


def pack_q8_0(q8: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Pack 8-bit symmetric codes into Q8_0 blocks (value = d*q8, q8 in int8)."""
    out_f, in_f = q8.shape
    n_blocks = in_f // QK
    qs = q8.reshape(out_f, n_blocks, QK).astype(np.int8).view(np.uint8)
    d_bytes = _f16_bytes(d)
    block = np.concatenate([d_bytes, qs], axis=-1)  # (out, n_blocks, 34)
    return block.reshape(out_f, n_blocks * 34)


def pack_gptq_linear(
    q_int: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    wbits: int,
    sym: bool,
    groupsize: int,
) -> Tuple[np.ndarray, GGMLQuantizationType]:
    """Convert one unpacked GPTQ linear into GGUF block bytes (no re-quantization).

    Args:
        q_int: (out_features, in_features) integer codes in [0, 2^wbits - 1].
        scales: (num_groups, out_features) float scales.
        zeros: (num_groups, out_features) restored integer zero points.
        wbits: Bit width (4 or 8).
        sym: Symmetric flag.
        groupsize: GPTQ group size (multiple of 32, or -1 for per-channel).

    Returns:
        (packed_uint8_byteshape, gguf_type).
    """
    qtype = select_gguf_type(wbits, sym)
    out_features, in_features = q_int.shape
    q_int = np.ascontiguousarray(q_int)
    scales = np.asarray(scales)
    zeros = np.asarray(zeros)

    d, z = _gather_per_block_scale_zero(scales, zeros, in_features, out_features, groupsize)

    if qtype == GGMLQuantizationType.Q4_0:
        if z is not None and not np.all(z == 8):
            raise UnsupportedGPTQLayout(
                "Q4_0 requires a symmetric zero point of 8; got non-8 zeros."
            )
        elif q_int.max(initial=0) > 15:
            raise UnsupportedGPTQLayout("Q4_0 expects codes in [0, 15].")
        packed = pack_q4_0(q_int, d)
    elif qtype == GGMLQuantizationType.Q4_1:
        if z is not None:
            m = (-d * z).astype(np.float32)
            packed = pack_q4_1(q_int, d, m)
        else:
            raise UnsupportedGPTQLayout("Q4_1 expects codes mapping into int4 range.")
    else:  # Q8_0
        # 8-bit symmetric: GGUF Q8_0 is value = d * q8 (q8 signed int8, no min),
        # so the GPTQ zero point must be exactly 128 for a lossless mapping.
        if z is not None and not np.all(z == 128):
            raise UnsupportedGPTQLayout(
                "Q8_0 requires a symmetric zero point of 128; got non-128 zeros."
            )
        q8 = q_int.astype(np.int32) - 128
        if q8.min(initial=0) < -128 or q8.max(initial=0) > 127:
            raise UnsupportedGPTQLayout("Q8_0 expects codes mapping into int8 range.")
        packed = pack_q8_0(q8.astype(np.int8), d)

    return np.ascontiguousarray(packed), qtype
