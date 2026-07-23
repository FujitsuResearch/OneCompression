"""Microscaling floating-point format codecs (NVFP4 / MXFP4 / FP8)

This module provides tensor-level codecs for microscaling floating-point
formats used in fake-quantization. All operations are implemented with
plain torch tensor arithmetic so that they run on CPU and GPU alike.

Supported formats:
    - NVFP4: FP4 (E2M1) elements with per-block (16) FP8 E4M3 scales and a
      single per-tensor FP32 scale (two-level scaling, NVIDIA definition).
    - MXFP4: FP4 (E2M1) elements with per-block (32) E8M0 power-of-two
      scales (OCP Microscaling specification).
    - FP8: FP8 E4M3 elements with per-channel (or per-tensor) FP32 scales.

Functions:
    e2m1_grid: Return the sorted FP4 E2M1 value grid.
    quantize_to_grid: Round values to the nearest grid point (returns indices).
    dequantize_from_grid: Look up grid values from grid indices.
    round_to_e4m3: Round values to the FP8 E4M3 grid (saturating).
    e8m0_block_scale: Compute E8M0 power-of-two block scales (OCP MX).
    nvfp4_quantize: Quantize a 2D weight to NVFP4 codes and scales.
    nvfp4_dequantize: Reconstruct a weight from NVFP4 codes and scales.
    mxfp4_quantize: Quantize a 2D weight to MXFP4 codes and scales.
    mxfp4_dequantize: Reconstruct a weight from MXFP4 codes and scales.
    fp8_quantize: Fake-quantize a 2D weight to FP8 E4M3 values and scales.
    fp8_dequantize: Reconstruct a weight from FP8 values and scales.
    sweep_nvfp4_block_scales: Select E4M3 block scales by local sweep.
    sweep_mxfp4_block_scales: Select E8M0 block scales by local sweep.
    grid_codes_to_fp4_bits: Convert E2M1 grid indices to 4-bit FP4 codes.
    fp4_bits_to_grid_codes: Convert 4-bit FP4 codes to E2M1 grid indices.
    pack_fp4_codes: Pack E2M1 grid indices into two-per-byte uint8 tensors.
    unpack_fp4_codes: Unpack two-per-byte uint8 tensors to grid indices.
    e8m0_scales_to_uint8: Encode power-of-two scales as biased E8M0 bytes.
    uint8_to_e8m0_scales: Decode biased E8M0 bytes to power-of-two scales.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import torch

# FP4 E2M1 representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Maximum representable magnitudes
E2M1_MAX = 6.0
E4M3_MAX = 448.0

# Smallest positive E4M3 value (subnormal): 2^-9
E4M3_MIN_SUBNORMAL = 2.0**-9

# Largest E2M1 exponent (6 = 1.5 * 2^2), used by the OCP MX scale rule
_E2M1_EMAX = 2.0

# E8M0 exponent range (unbiased)
E8M0_MIN_EXP = -127.0
E8M0_MAX_EXP = 127.0


def e2m1_grid(device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the sorted FP4 E2M1 value grid.

    The grid contains the 15 distinct values
    {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6} in ascending order.

    Args:
        device (torch.device, optional): Device to create the grid on.
        dtype (torch.dtype, optional): Data type of the grid.
            Default is torch.float32.

    Returns:
        torch.Tensor: 1D tensor of shape (15,) with grid values sorted
            in ascending order.
    """
    positive = torch.tensor(_E2M1_POSITIVE, device=device, dtype=dtype)
    negative = -positive[1:].flip(0)
    return torch.cat([negative, positive])


def quantize_to_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Round values to the nearest grid point and return grid indices.

    The grid must be symmetric around zero with an odd number of entries
    sorted in ascending order (as returned by :func:`e2m1_grid`).
    Rounding is sign-symmetric round-to-nearest; exact midpoint ties
    resolve to the magnitude with an even index in the positive half of
    the grid, which for E2M1 is the value with an even mantissa
    (round-half-to-even, matching IEEE-style casts). Values outside the
    grid range (including +-inf) are clamped to the outermost grid
    points.

    Args:
        x (torch.Tensor): Input tensor (any shape).
        grid (torch.Tensor): 1D symmetric tensor of grid values sorted
            in ascending order with an odd number of entries.

    Returns:
        torch.Tensor: Integer tensor of grid indices (torch.int8),
            same shape as ``x``.
    """
    zero_index = (grid.numel() - 1) // 2
    magnitudes = grid[zero_index:]
    boundaries = (magnitudes[:-1] + magnitudes[1:]) / 2

    magnitude = x.abs().contiguous()
    # bucketize resolves boundary hits to the lower bucket with
    # right=False and to the upper bucket with right=True; the two only
    # differ at exact midpoints, where the even index is selected.
    low = torch.bucketize(magnitude, boundaries)
    high = torch.bucketize(magnitude, boundaries, right=True)
    codes = torch.where(high % 2 == 0, high, low)

    sign = torch.where(x < 0, -1, 1).to(codes.dtype)
    return (zero_index + sign * codes).to(torch.int8)


def dequantize_from_grid(codes: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Look up grid values from grid indices.

    Args:
        codes (torch.Tensor): Integer tensor of grid indices.
        grid (torch.Tensor): 1D tensor of grid values sorted in
            ascending order.

    Returns:
        torch.Tensor: Tensor of grid values with the dtype of ``grid``,
            same shape as ``codes``.
    """
    return grid[codes.long()]


def round_to_e4m3(s: torch.Tensor) -> torch.Tensor:
    """Round values to the FP8 E4M3 grid.

    Implemented as a round-trip cast through ``torch.float8_e4m3fn``.
    Values with magnitude above 448 saturate to +-448, and tiny values
    round to E4M3 subnormals (minimum positive value 2^-9).

    Args:
        s (torch.Tensor): Input tensor (floating point).

    Returns:
        torch.Tensor: Tensor with values on the E4M3 grid, cast back to
            the input dtype.
    """
    clamped = torch.clamp(s, min=-E4M3_MAX, max=E4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).to(s.dtype)


def e8m0_block_scale(block_amax: torch.Tensor, mode: str = "ceil") -> torch.Tensor:
    """Compute E8M0 power-of-two block scales.

    Two exponent rules are supported:

    - ``"ceil"`` (default): the smallest power of two such that the block
      maximum maps to at most the E2M1 maximum 6, i.e.
      ``scale = 2^(ceil(log2(block_amax / 6)))``. No element ever saturates,
      which removes the up-to-25% clipping error the OCP rule incurs on
      block maxima in ``(6, 8) * scale``.
    - ``"ocp"``: the OCP Microscaling v1.0 rule
      ``scale = 2^(floor(log2(block_amax)) - emax_elem)`` with
      ``emax_elem = 2``, mapping the block maximum into [4, 8); elements
      above 6 saturate during element rounding. Use this for bit-exact
      parity with strictly OCP-compliant implementations.

    The exponent is clamped to the E8M0 range [-127, 127]. All-zero blocks
    get scale 1 (their codes map to zero regardless of the scale).

    Args:
        block_amax (torch.Tensor): Per-block maximum magnitudes (>= 0).
        mode (str): ``"ceil"`` or ``"ocp"``.

    Returns:
        torch.Tensor: Tensor of power-of-two scale values with the same
            dtype and shape as the input.
    """
    safe_amax = torch.clamp(block_amax, min=2.0**E8M0_MIN_EXP)
    if mode == "ceil":
        exponent = torch.ceil(torch.log2(safe_amax / E2M1_MAX))
    elif mode == "ocp":
        exponent = torch.floor(torch.log2(safe_amax)) - _E2M1_EMAX
    else:
        raise ValueError(f"unknown E8M0 scale mode: {mode!r} (expected 'ceil' or 'ocp')")
    exponent = torch.clamp(exponent, min=E8M0_MIN_EXP, max=E8M0_MAX_EXP)
    scale = torch.exp2(exponent).to(block_amax.dtype)
    return torch.where(block_amax == 0, torch.ones_like(scale), scale)


# E4M3 bit patterns for positive values: 0x00 encodes 0, 0x01 the smallest
# subnormal (2^-9), ..., 0x7E the maximum normal (448); 0x7F is NaN.
_E4M3_MIN_BITS = 1
_E4M3_MAX_BITS = 126


def _e4m3_neighbor_scales(base_scales: torch.Tensor, offset: int) -> torch.Tensor:
    """Shift positive E4M3 scales by ``offset`` steps in bit-pattern space.

    Because positive E4M3 bit patterns are monotonically ordered, adding
    ``offset`` to the byte representation moves the scale to the
    ``offset``-th next (or previous) representable E4M3 value. The result
    is clamped to the positive range [2^-9, 448].

    Args:
        base_scales (torch.Tensor): Positive scales already on the E4M3
            grid (any floating dtype).
        offset (int): Number of bit-pattern steps to shift by.

    Returns:
        torch.Tensor: FP32 tensor of shifted E4M3 scale values.
    """
    bits = base_scales.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    bits = torch.clamp(bits + offset, _E4M3_MIN_BITS, _E4M3_MAX_BITS)
    return bits.to(torch.uint8).contiguous().view(torch.float8_e4m3fn).float()


def _block_quantization_error(
    blocks: torch.Tensor,
    effective_scales: torch.Tensor,
    grid: torch.Tensor,
    importance: torch.Tensor = None,
) -> torch.Tensor:
    """Per-block (weighted) squared reconstruction error for FP4 rounding.

    Args:
        blocks (torch.Tensor): Blocked weights (..., num_blocks, block_size).
        effective_scales (torch.Tensor): Scales of shape (..., num_blocks).
        grid (torch.Tensor): E2M1 value grid.
        importance (torch.Tensor, optional): Non-negative per-element
            importance broadcastable to ``blocks`` (e.g. Hessian diagonal).

    Returns:
        torch.Tensor: Error tensor of shape (..., num_blocks).
    """
    eff = effective_scales.unsqueeze(-1)
    codes = quantize_to_grid(blocks / eff, grid)
    dequantized = dequantize_from_grid(codes, grid) * eff
    err = (blocks - dequantized).square()
    if importance is not None:
        err = err * importance
    return err.sum(dim=-1)


def sweep_nvfp4_block_scales(
    blocks: torch.Tensor,
    tensor_scale: torch.Tensor,
    importance: torch.Tensor = None,
    offset_down: int = None,
    offset_up: int = 7,
) -> torch.Tensor:
    """Select NVFP4 E4M3 block scales by local sweep in bit-pattern space.

    Starting from the AbsMax-based E4M3 scale, this evaluates the
    neighboring representable E4M3 scales and keeps, per block, the one
    minimizing the (optionally importance-weighted) squared
    reconstruction error. The default sweep window follows the
    theoretically justified bounds of ScaleSweep (arXiv:2606.07618):
    ``[-3, +7]`` bit-pattern steps for the MSE objective and ``[-8, +7]``
    when an importance weighting is supplied (WMSE).

    Args:
        blocks (torch.Tensor): Blocked FP32 weights of shape
            (out_features, num_blocks, block_size).
        tensor_scale (torch.Tensor): Per-tensor FP32 scale (0-dim).
        importance (torch.Tensor, optional): Non-negative importance
            broadcastable to ``blocks`` (e.g. the Hessian diagonal of the
            corresponding input columns, shaped (1, num_blocks,
            block_size)).
        offset_down (int, optional): Steps to sweep below the AbsMax
            scale. Defaults to 3 (MSE) or 8 (with importance).
        offset_up (int, optional): Steps to sweep above. Default 7.

    Returns:
        torch.Tensor: Selected E4M3-representable block scales (FP32,
            shape (out_features, num_blocks)), to be multiplied by
            ``tensor_scale`` for the effective scale.
    """
    if offset_down is None:
        offset_down = 8 if importance is not None else 3

    grid = e2m1_grid(device=blocks.device)
    block_amax = blocks.abs().amax(dim=-1)
    base = round_to_e4m3(block_amax / (E2M1_MAX * tensor_scale))
    base = torch.clamp(base, min=E4M3_MIN_SUBNORMAL)

    best_scales = base
    best_err = _block_quantization_error(blocks, base * tensor_scale, grid, importance)
    for offset in range(-offset_down, offset_up + 1):
        if offset == 0:
            continue
        candidate = _e4m3_neighbor_scales(base, offset)
        err = _block_quantization_error(blocks, candidate * tensor_scale, grid, importance)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scales = torch.where(better, candidate, best_scales)
    return best_scales


def sweep_mxfp4_block_scales(
    blocks: torch.Tensor,
    importance: torch.Tensor = None,
    offset_down: int = 2,
    offset_up: int = 1,
) -> torch.Tensor:
    """Select MXFP4 E8M0 block scales by local sweep over exponents.

    Starting from the non-clipping ``ceil`` scale, this evaluates
    neighboring power-of-two exponents and keeps, per block, the one
    minimizing the (optionally importance-weighted) squared
    reconstruction error. Lower exponents trade saturation of the block
    maximum for finer resolution of the remaining elements, which is
    frequently the better trade for E8M0's coarse power-of-two grid.

    Args:
        blocks (torch.Tensor): Blocked FP32 weights of shape
            (out_features, num_blocks, block_size).
        importance (torch.Tensor, optional): Non-negative importance
            broadcastable to ``blocks``.
        offset_down (int, optional): Exponent steps to sweep below the
            ceil scale. Default 2.
        offset_up (int, optional): Exponent steps to sweep above.
            Default 1.

    Returns:
        torch.Tensor: Selected power-of-two block scales (FP32, shape
            (out_features, num_blocks)).
    """
    grid = e2m1_grid(device=blocks.device)
    block_amax = blocks.abs().amax(dim=-1)
    base = e8m0_block_scale(block_amax)
    base_exp = torch.round(torch.log2(base))

    best_scales = base
    best_err = _block_quantization_error(blocks, base, grid, importance)
    for offset in range(-offset_down, offset_up + 1):
        if offset == 0:
            continue
        exponent = torch.clamp(base_exp + offset, min=E8M0_MIN_EXP, max=E8M0_MAX_EXP)
        candidate = torch.exp2(exponent)
        err = _block_quantization_error(blocks, candidate, grid, importance)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scales = torch.where(better, candidate, best_scales)
    return best_scales


def _reshape_to_blocks(w: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape a 2D weight into row-wise blocks.

    Args:
        w (torch.Tensor): Weight tensor of shape (out_features, in_features).
        block_size (int): Number of consecutive elements per block along
            the input dimension.

    Returns:
        torch.Tensor: Tensor of shape (out_features, num_blocks, block_size).

    Raises:
        ValueError: If ``in_features`` is not divisible by ``block_size``.
    """
    if w.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(w.shape)}.")
    out_features, in_features = w.shape
    if in_features % block_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by block_size={block_size}."
        )
    return w.reshape(out_features, in_features // block_size, block_size)


def nvfp4_quantize(
    w: torch.Tensor,
    block_size: int = 16,
    tensor_scale: torch.Tensor = None,
    scale_search: bool = False,
    importance: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to NVFP4 (two-level scaling, NVIDIA definition).

    Each row is split into blocks of ``block_size`` elements. Every block
    stores an FP8 E4M3 scale, and the whole tensor stores one FP32 scale:

        tensor_scale = global_amax / (448 * 6)
        block_scale  = round_to_e4m3(block_amax / (6 * tensor_scale))
        code         = round_to_e2m1(w / (block_scale * tensor_scale))

    Args:
        w (torch.Tensor): Weight tensor of shape (out_features, in_features).
        block_size (int, optional): Block size along the input dimension.
            Default is 16.
        tensor_scale (torch.Tensor, optional): Per-tensor FP32 scale
            (0-dim tensor) to use instead of deriving it from the weight.
            Used to share one global scale across layers that inference
            engines fuse into a single matrix (e.g. q/k/v projections).
        scale_search (bool, optional): If True, refine each E4M3 block
            scale by a local sweep in bit-pattern space instead of the
            AbsMax heuristic (see :func:`sweep_nvfp4_block_scales`).
            Default is False.
        importance (torch.Tensor, optional): Non-negative per-column
            importance of shape (in_features,) used as the WMSE weighting
            during the scale sweep (e.g. the Hessian diagonal). Only used
            when ``scale_search=True``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - codes: E2M1 grid indices (torch.int8), same shape as ``w``.
            - block_scales: Decoded E4M3 block scales (FP32), shape
              (out_features, in_features // block_size).
            - tensor_scale: Per-tensor FP32 scale (0-dim tensor).
    """
    w = w.float()
    blocks = _reshape_to_blocks(w, block_size)

    if tensor_scale is None:
        global_amax = w.abs().amax()
        tensor_scale = global_amax / (E4M3_MAX * E2M1_MAX)
        if tensor_scale == 0:
            tensor_scale = torch.ones_like(tensor_scale)
    else:
        tensor_scale = tensor_scale.to(device=w.device, dtype=torch.float32).reshape(())

    if scale_search:
        block_scales = sweep_nvfp4_block_scales(
            blocks, tensor_scale, importance=_blocked_importance(importance, block_size)
        )
    else:
        block_amax = blocks.abs().amax(dim=-1)
        block_scales = round_to_e4m3(block_amax / (E2M1_MAX * tensor_scale))
    # Blocks whose amax rounds to scale 0 (all-zero or magnitudes below half
    # the E4M3 minimum subnormal) are clamped to that minimum, 2^-9, instead
    # of 1.0: a scale of 1.0 would round every element of a tiny non-zero
    # block to code 0 and erase it.
    block_scales = torch.clamp(block_scales, min=E4M3_MIN_SUBNORMAL)

    grid = e2m1_grid(device=w.device)
    effective_scale = (block_scales * tensor_scale).unsqueeze(-1)
    codes = quantize_to_grid(blocks / effective_scale, grid).reshape(w.shape)

    return codes, block_scales, tensor_scale


def _blocked_importance(importance: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape a per-column importance vector for blocked broadcasting.

    Args:
        importance (torch.Tensor, optional): Vector of shape (in_features,).
        block_size (int): Block size along the input dimension.

    Returns:
        torch.Tensor: Tensor of shape (1, in_features // block_size,
            block_size), or None when ``importance`` is None.
    """
    if importance is None:
        return None
    return importance.float().reshape(1, -1, block_size)


def nvfp4_dequantize(
    codes: torch.Tensor,
    block_scales: torch.Tensor,
    tensor_scale: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Reconstruct a weight from NVFP4 codes and scales.

    Args:
        codes (torch.Tensor): E2M1 grid indices of shape
            (out_features, in_features).
        block_scales (torch.Tensor): Decoded E4M3 block scales of shape
            (out_features, in_features // block_size).
        tensor_scale (torch.Tensor): Per-tensor FP32 scale (0-dim tensor).
        block_size (int, optional): Block size along the input dimension.
            Default is 16.

    Returns:
        torch.Tensor: Reconstructed FP32 weight of shape
            (out_features, in_features).
    """
    grid = e2m1_grid(device=codes.device)
    values = dequantize_from_grid(codes, grid)
    blocks = _reshape_to_blocks(values, block_size)
    blocks = blocks * (block_scales * tensor_scale).unsqueeze(-1)
    return blocks.reshape(values.shape)


def mxfp4_quantize(
    w: torch.Tensor,
    block_size: int = 32,
    scale_search: bool = False,
    importance: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to MXFP4 (OCP Microscaling specification).

    Each row is split into blocks of ``block_size`` elements, and every
    block stores an E8M0 power-of-two scale (default ``ceil`` rule, see
    :func:`e8m0_block_scale`):

        block_scale = 2^(ceil(log2(block_amax / 6)))
        code        = round_to_e2m1(w / block_scale)

    Elements whose scaled magnitude exceeds the E2M1 maximum 6 saturate
    to +-6 during rounding.

    Args:
        w (torch.Tensor): Weight tensor of shape (out_features, in_features).
        block_size (int, optional): Block size along the input dimension.
            Default is 32.
        scale_search (bool, optional): If True, refine each E8M0 exponent
            by a local sweep instead of the ceil rule (see
            :func:`sweep_mxfp4_block_scales`). Default is False.
        importance (torch.Tensor, optional): Non-negative per-column
            importance of shape (in_features,) used as the WMSE weighting
            during the scale sweep. Only used when ``scale_search=True``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - codes: E2M1 grid indices (torch.int8), same shape as ``w``.
            - block_scales: E8M0 power-of-two block scales (FP32), shape
              (out_features, in_features // block_size).
    """
    w = w.float()
    blocks = _reshape_to_blocks(w, block_size)

    if scale_search:
        block_scales = sweep_mxfp4_block_scales(
            blocks, importance=_blocked_importance(importance, block_size)
        )
    else:
        block_amax = blocks.abs().amax(dim=-1)
        block_scales = e8m0_block_scale(block_amax)

    grid = e2m1_grid(device=w.device)
    codes = quantize_to_grid(blocks / block_scales.unsqueeze(-1), grid).reshape(w.shape)

    return codes, block_scales


def mxfp4_dequantize(
    codes: torch.Tensor,
    block_scales: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Reconstruct a weight from MXFP4 codes and scales.

    Args:
        codes (torch.Tensor): E2M1 grid indices of shape
            (out_features, in_features).
        block_scales (torch.Tensor): E8M0 block scales of shape
            (out_features, in_features // block_size).
        block_size (int, optional): Block size along the input dimension.
            Default is 32.

    Returns:
        torch.Tensor: Reconstructed FP32 weight of shape
            (out_features, in_features).
    """
    grid = e2m1_grid(device=codes.device)
    values = dequantize_from_grid(codes, grid)
    blocks = _reshape_to_blocks(values, block_size)
    blocks = blocks * block_scales.unsqueeze(-1)
    return blocks.reshape(values.shape)


def fp8_quantize(w: torch.Tensor, per_channel: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize a 2D weight to FP8 E4M3.

    The weight is scaled so that its (per-channel or per-tensor) maximum
    magnitude maps to the E4M3 maximum (448), then rounded to the E4M3
    grid via a round-trip cast:

        scale = amax / 448
        q     = round_to_e4m3(w / scale)

    Args:
        w (torch.Tensor): Weight tensor of shape (out_features, in_features).
        per_channel (bool, optional): If True, use one scale per output
            channel (row). If False, use a single per-tensor scale.
            Default is True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - values: E4M3-rounded values (FP32), same shape as ``w``.
            - scales: FP32 scales of shape (out_features, 1) when
              ``per_channel=True``, otherwise shape (1, 1).
    """
    w = w.float()
    if w.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(w.shape)}.")

    if per_channel:
        amax = w.abs().amax(dim=-1, keepdim=True)
    else:
        amax = w.abs().amax().reshape(1, 1)

    scales = amax / E4M3_MAX
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    values = round_to_e4m3(w / scales)
    return values, scales


def fp8_dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reconstruct a weight from FP8 values and scales.

    Args:
        values (torch.Tensor): E4M3-rounded values (FP32).
        scales (torch.Tensor): FP32 scales broadcastable to ``values``.

    Returns:
        torch.Tensor: Reconstructed FP32 weight.
    """
    return values * scales


# =====================================================================
# Bit-level packing (storage layout used by vLLM / compressed-tensors)
# =====================================================================

# The FP4 E2M1 bit encoding is ``sign(1) | exponent(2) | mantissa(1)``.
# Positive codes 0..7 decode to {0, 0.5, 1, 1.5, 2, 3, 4, 6}, which is
# exactly the positive half of :func:`e2m1_grid`; bit 3 is the sign.


def grid_codes_to_fp4_bits(codes: torch.Tensor) -> torch.Tensor:
    """Convert E2M1 grid indices to 4-bit FP4 (E2M1) codes.

    Grid indices are the symmetric indices produced by
    :func:`quantize_to_grid` (0..14 with 7 = zero); FP4 codes follow the
    IEEE-style ``sign | exponent | mantissa`` layout used by packed
    checkpoints (compressed-tensors, ModelOpt, ...).

    Args:
        codes (torch.Tensor): Integer tensor of grid indices (0..14).

    Returns:
        torch.Tensor: uint8 tensor of FP4 codes (0..15), same shape.
    """
    centered = codes.to(torch.int16) - 7
    magnitude = centered.abs()
    sign = (centered < 0).to(torch.int16)
    return ((sign << 3) | magnitude).to(torch.uint8)


def fp4_bits_to_grid_codes(bits: torch.Tensor) -> torch.Tensor:
    """Convert 4-bit FP4 (E2M1) codes to E2M1 grid indices.

    Inverse of :func:`grid_codes_to_fp4_bits`. The negative-zero code
    (0b1000) maps to the single zero grid index 7.

    Args:
        bits (torch.Tensor): uint8 tensor of FP4 codes (0..15).

    Returns:
        torch.Tensor: int8 tensor of grid indices (0..14), same shape.
    """
    bits = bits.to(torch.int16)
    magnitude = bits & 0x7
    sign = (bits >> 3) & 0x1
    return (7 + (1 - 2 * sign) * magnitude).to(torch.int8)


def pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 grid indices into two-per-byte uint8 tensors.

    Adjacent elements along the last dimension share one byte with the
    even-indexed element in the low nibble, matching the layout of
    ``compressed-tensors`` ``*-pack-quantized`` checkpoints (and the
    input expected by vLLM's FP4 kernels).

    Args:
        codes (torch.Tensor): Grid indices of shape (..., in_features)
            with ``in_features`` even.

    Returns:
        torch.Tensor: uint8 tensor of shape (..., in_features // 2).
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {codes.shape[-1]}.")
    bits = grid_codes_to_fp4_bits(codes)
    low = bits[..., 0::2]
    high = bits[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack two-per-byte uint8 tensors to E2M1 grid indices.

    Inverse of :func:`pack_fp4_codes`.

    Args:
        packed (torch.Tensor): uint8 tensor of shape (..., in_features // 2).

    Returns:
        torch.Tensor: int8 tensor of grid indices, shape (..., in_features).
    """
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    bits = torch.stack([low, high], dim=-1).flatten(start_dim=-2)
    return fp4_bits_to_grid_codes(bits)


def e8m0_scales_to_uint8(scales: torch.Tensor) -> torch.Tensor:
    """Encode power-of-two scales as biased E8M0 bytes (bias 127).

    Args:
        scales (torch.Tensor): Power-of-two scale values in
            [2^-127, 2^127] (as produced by :func:`e8m0_block_scale`).

    Returns:
        torch.Tensor: uint8 tensor of biased exponents, same shape.
    """
    exponents = torch.round(torch.log2(scales.float()))
    if ((exponents < E8M0_MIN_EXP) | (exponents > E8M0_MAX_EXP)).any():
        raise ValueError("E8M0 scales out of the representable exponent range.")
    return (exponents + 127).to(torch.uint8)


def uint8_to_e8m0_scales(encoded: torch.Tensor) -> torch.Tensor:
    """Decode biased E8M0 bytes (bias 127) to power-of-two FP32 scales.

    Args:
        encoded (torch.Tensor): uint8 tensor of biased exponents.

    Returns:
        torch.Tensor: FP32 tensor of power-of-two scales, same shape.
    """
    return torch.exp2(encoded.float() - 127.0)
