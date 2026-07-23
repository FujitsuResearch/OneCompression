"""Tests for the microscaling floating-point format codecs.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import math

import pytest
import torch

from onecomp.quantizer.floatquant.formats import (
    E2M1_MAX,
    E4M3_MAX,
    dequantize_from_grid,
    e2m1_grid,
    e8m0_block_scale,
    e8m0_scales_to_uint8,
    fp4_bits_to_grid_codes,
    fp8_dequantize,
    fp8_quantize,
    grid_codes_to_fp4_bits,
    mxfp4_dequantize,
    mxfp4_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
    pack_fp4_codes,
    quantize_to_grid,
    round_to_e4m3,
    uint8_to_e8m0_scales,
    unpack_fp4_codes,
)


class TestE2M1Grid:
    """Tests for the E2M1 grid and nearest-neighbour rounding."""

    def test_grid_values(self):
        """The grid contains exactly {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}."""
        grid = e2m1_grid()
        expected = torch.tensor(
            [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        )
        assert torch.equal(grid, expected)

    @pytest.mark.parametrize(
        "value,expected",
        [
            (5.9, 6.0),  # closer to 6 than 4
            (2.4, 2.0),  # closer to 2 than 3
            (2.6, 3.0),  # closer to 3 than 2
            (4.9, 4.0),  # below the (4, 6) midpoint 5
            (5.1, 6.0),  # above the (4, 6) midpoint 5
            (0.24, 0.0),  # below the (0, 0.5) midpoint 0.25
            (0.26, 0.5),  # above the (0, 0.5) midpoint 0.25
            (-3.4, -3.0),  # closer to -3 than -4
            (-5.9, -6.0),  # closer to -6 than -4
            (100.0, 6.0),  # clamps to the grid maximum
            (-100.0, -6.0),  # clamps to the grid minimum
            (0.0, 0.0),  # exact grid point
            (1.5, 1.5),  # exact grid point
        ],
    )
    def test_nearest_rounding(self, value, expected):
        """Values round to the nearest grid point."""
        grid = e2m1_grid()
        codes = quantize_to_grid(torch.tensor([value]), grid)
        assert codes.dtype == torch.int8
        rounded = dequantize_from_grid(codes, grid)
        assert rounded.item() == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.25, 0.0),  # tie 0 / 0.5 -> even mantissa 0
            (0.75, 1.0),  # tie 0.5 / 1 -> even mantissa 1
            (1.25, 1.0),  # tie 1 / 1.5 -> even mantissa 1
            (1.75, 2.0),  # tie 1.5 / 2 -> even mantissa 2
            (2.5, 2.0),  # tie 2 / 3 -> even mantissa 2
            (3.5, 4.0),  # tie 3 / 4 -> even mantissa 4
            (5.0, 4.0),  # tie 4 / 6 -> even mantissa 4
        ],
    )
    def test_midpoint_ties_round_half_to_even(self, value, expected):
        """Exact midpoints round to the grid value with an even mantissa."""
        grid = e2m1_grid()
        for signed, want in ((value, expected), (-value, -expected)):
            codes = quantize_to_grid(torch.tensor([signed]), grid)
            assert dequantize_from_grid(codes, grid).item() == pytest.approx(want)

    def test_sign_symmetry(self):
        """quantize(-x) == -quantize(x), including at midpoint ties."""
        grid = e2m1_grid()
        x = torch.cat(
            [
                torch.linspace(0.0, 8.0, 4001),
                torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]),
            ]
        )
        pos = dequantize_from_grid(quantize_to_grid(x, grid), grid)
        neg = dequantize_from_grid(quantize_to_grid(-x, grid), grid)
        assert torch.equal(pos, -neg)

    def test_infinite_inputs_clamp(self):
        """+-inf clamps to the outermost grid values."""
        grid = e2m1_grid()
        codes = quantize_to_grid(torch.tensor([float("inf"), float("-inf")]), grid)
        values = dequantize_from_grid(codes, grid)
        assert values.tolist() == [E2M1_MAX, -E2M1_MAX]

    def test_roundtrip_shape_and_range(self):
        """Codes cover valid indices and dequantization preserves shape."""
        grid = e2m1_grid()
        x = torch.randn(64, 32) * 3
        codes = quantize_to_grid(x, grid)
        assert codes.shape == x.shape
        assert codes.min() >= 0
        assert codes.max() <= len(grid) - 1
        values = dequantize_from_grid(codes, grid)
        assert values.shape == x.shape
        assert torch.isfinite(values).all()


class TestScaleRounding:
    """Tests for E4M3 and E8M0 scale rounding."""

    def test_e4m3_exact_values(self):
        """Representable values pass through unchanged."""
        exact = torch.tensor([0.0, 1.0, 448.0, -448.0, 2.0**-9])
        assert torch.equal(round_to_e4m3(exact), exact)

    def test_e4m3_saturates(self):
        """Values beyond the E4M3 range saturate to +-448."""
        out = round_to_e4m3(torch.tensor([1e6, -1e6, 449.0]))
        assert torch.equal(out, torch.tensor([448.0, -448.0, 448.0]))

    def test_e4m3_relative_error_bound(self):
        """Rounding error is within the E4M3 relative precision (2^-3)."""
        x = torch.exp(torch.randn(1000)) + 0.01
        rounded = round_to_e4m3(x)
        rel = ((rounded - x).abs() / x).max().item()
        assert rel <= 2.0**-3

    def test_e4m3_idempotent(self):
        """Rounding twice equals rounding once (values lie on the grid)."""
        x = torch.randn(256) * 10
        once = round_to_e4m3(x)
        assert torch.equal(round_to_e4m3(once), once)

    def test_e8m0_is_power_of_two(self):
        """E8M0 block scales are exact powers of two."""
        amax = torch.exp(torch.randn(1000)) + 1e-6
        scale = e8m0_block_scale(amax)
        exponents = torch.log2(scale)
        assert torch.equal(exponents, exponents.round())

    def test_e8m0_default_never_clips_block_max(self):
        """Default (ceil) scales map the block maximum to at most E2M1 max 6."""
        amax = torch.exp(torch.randn(1000) * 3) + 1e-6
        scale = e8m0_block_scale(amax)
        expected = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
        assert torch.equal(scale, expected)
        ratio = amax / scale
        assert (ratio <= 6.0).all()
        assert (ratio > 3.0).all()

    def test_e8m0_ocp_mode_follows_spec_rule(self):
        """mode='ocp' equals 2^(floor(log2(amax)) - 2) per the OCP MX spec."""
        amax = torch.exp(torch.randn(1000) * 3) + 1e-6
        scale = e8m0_block_scale(amax, mode="ocp")
        expected = torch.exp2(torch.floor(torch.log2(amax)) - 2)
        assert torch.equal(scale, expected)
        # The block maximum maps into [4, 8); values in (6, 8) saturate.
        ratio = amax / scale
        assert (ratio >= 4.0).all()
        assert (ratio < 8.0).all()

    def test_e8m0_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            e8m0_block_scale(torch.tensor([1.0]), mode="round")

    def test_e8m0_zero_amax_guard(self):
        """All-zero blocks get scale 1 (codes are zero anyway)."""
        scale = e8m0_block_scale(torch.tensor([0.0, 2.0]))
        assert scale[0].item() == 1.0
        assert scale[1].item() == pytest.approx(0.5)

    def test_e8m0_clamps_exponent(self):
        """Exponents are clamped to the E8M0 range [-127, 127]."""
        tiny = e8m0_block_scale(torch.tensor([1e-45]))
        huge = e8m0_block_scale(torch.tensor([2.0**129]))
        assert tiny.item() == pytest.approx(2.0**-127)
        assert huge.item() == pytest.approx(2.0**127)


class TestNVFP4:
    """Tests for the NVFP4 codec."""

    def test_shapes_and_dtypes(self):
        """Codes and scales have the documented shapes and dtypes."""
        torch.manual_seed(0)
        w = torch.randn(64, 64)
        codes, block_scales, tensor_scale = nvfp4_quantize(w, block_size=16)
        assert codes.shape == w.shape
        assert codes.dtype == torch.int8
        assert block_scales.shape == (64, 64 // 16)
        assert tensor_scale.dim() == 0

    def test_tensor_scale_definition(self):
        """Per-tensor scale follows the NVIDIA definition amax / (448 * 6)."""
        torch.manual_seed(0)
        w = torch.randn(32, 32)
        _, _, tensor_scale = nvfp4_quantize(w, block_size=16)
        expected = w.abs().max() / (E4M3_MAX * E2M1_MAX)
        assert tensor_scale.item() == pytest.approx(expected.item())

    def test_block_amax_maps_to_six(self):
        """Each block's maximum-magnitude element maps to the E2M1 maximum 6."""
        torch.manual_seed(0)
        w = torch.randn(64, 64)
        codes, _, _ = nvfp4_quantize(w, block_size=16)
        grid = e2m1_grid()
        values = dequantize_from_grid(codes, grid).reshape(64, 4, 16)
        amax_idx = w.reshape(64, 4, 16).abs().argmax(dim=-1, keepdim=True)
        amax_values = values.gather(-1, amax_idx)
        assert (amax_values.abs() == E2M1_MAX).all()

    def test_roundtrip_error(self):
        """Round-trip is finite with a modest relative Frobenius error."""
        torch.manual_seed(0)
        w = torch.randn(256, 256)
        codes, block_scales, tensor_scale = nvfp4_quantize(w, block_size=16)
        w_hat = nvfp4_dequantize(codes, block_scales, tensor_scale, block_size=16)
        assert torch.isfinite(w_hat).all()
        rel = (torch.norm(w - w_hat) / torch.norm(w)).item()
        assert rel < 0.15

    def test_invalid_block_size_raises(self):
        """A block size that does not divide in_features raises ValueError."""
        with pytest.raises(ValueError):
            nvfp4_quantize(torch.randn(4, 10), block_size=16)

    def test_zero_weight(self):
        """An all-zero weight reconstructs to exactly zero."""
        w = torch.zeros(8, 32)
        codes, block_scales, tensor_scale = nvfp4_quantize(w, block_size=16)
        w_hat = nvfp4_dequantize(codes, block_scales, tensor_scale, block_size=16)
        assert torch.equal(w_hat, w)


class TestMXFP4:
    """Tests for the MXFP4 codec."""

    def test_scales_are_powers_of_two(self):
        """MXFP4 block scales are exact powers of two (E8M0)."""
        torch.manual_seed(0)
        w = torch.randn(64, 64)
        _, block_scales = mxfp4_quantize(w, block_size=32)
        exponents = torch.log2(block_scales)
        assert torch.equal(exponents, exponents.round())

    def test_scales_follow_ceil_rule(self):
        """Block scales equal 2^(ceil(log2(block_amax / 6))) (no clipping)."""
        torch.manual_seed(0)
        w = torch.randn(64, 64)
        _, block_scales = mxfp4_quantize(w, block_size=32)
        block_amax = w.reshape(64, 2, 32).abs().amax(dim=-1)
        expected = torch.exp2(torch.ceil(torch.log2(block_amax / 6.0)))
        assert torch.equal(block_scales, expected)
        assert (block_amax / block_scales <= 6.0).all()

    def test_zero_block_reconstructs_to_zero(self):
        """An all-zero block reconstructs to exactly zero without NaN."""
        w = torch.randn(4, 64)
        w[:, 32:] = 0.0
        codes, block_scales = mxfp4_quantize(w, block_size=32)
        w_hat = mxfp4_dequantize(codes, block_scales, block_size=32)
        assert torch.isfinite(w_hat).all()
        assert (w_hat[:, 32:] == 0).all()

    def test_roundtrip_error(self):
        """Round-trip is finite with a bounded relative Frobenius error."""
        torch.manual_seed(0)
        w = torch.randn(256, 256)
        codes, block_scales = mxfp4_quantize(w, block_size=32)
        w_hat = mxfp4_dequantize(codes, block_scales, block_size=32)
        assert torch.isfinite(w_hat).all()
        rel = (torch.norm(w - w_hat) / torch.norm(w)).item()
        assert rel < 0.25

    def test_nvfp4_error_not_worse_than_mxfp4(self):
        """NVFP4 error <= MXFP4 error on the same Gaussian matrices (mean over seeds)."""
        nv_errors = []
        mx_errors = []
        for seed in range(10):
            torch.manual_seed(seed)
            w = torch.randn(128, 128)
            codes, bs, ts = nvfp4_quantize(w, block_size=16)
            nv_hat = nvfp4_dequantize(codes, bs, ts, block_size=16)
            nv_errors.append((torch.norm(w - nv_hat) / torch.norm(w)).item())
            codes, bs = mxfp4_quantize(w, block_size=32)
            mx_hat = mxfp4_dequantize(codes, bs, block_size=32)
            mx_errors.append((torch.norm(w - mx_hat) / torch.norm(w)).item())
        assert sum(nv_errors) / len(nv_errors) <= sum(mx_errors) / len(mx_errors)


class TestFP8:
    """Tests for the FP8 E4M3 codec."""

    def test_per_channel_shapes(self):
        """Per-channel scales have shape (out_features, 1)."""
        torch.manual_seed(0)
        w = torch.randn(16, 64)
        values, scales = fp8_quantize(w, per_channel=True)
        assert values.shape == w.shape
        assert scales.shape == (16, 1)

    def test_per_tensor_shapes(self):
        """Per-tensor scale has shape (1, 1)."""
        torch.manual_seed(0)
        w = torch.randn(16, 64)
        _, scales = fp8_quantize(w, per_channel=False)
        assert scales.shape == (1, 1)

    def test_roundtrip_error(self):
        """FP8 fake-quant has a small relative Frobenius error."""
        torch.manual_seed(0)
        w = torch.randn(256, 256)
        values, scales = fp8_quantize(w, per_channel=True)
        w_hat = fp8_dequantize(values, scales)
        assert torch.isfinite(w_hat).all()
        rel = (torch.norm(w - w_hat) / torch.norm(w)).item()
        assert rel < 0.05

    def test_values_on_e4m3_grid(self):
        """Quantized values lie exactly on the E4M3 grid."""
        torch.manual_seed(0)
        w = torch.randn(16, 64)
        values, _ = fp8_quantize(w, per_channel=True)
        assert torch.equal(round_to_e4m3(values), values)


class TestFp4BitPacking:
    """Tests for the FP4 bit encoding and two-per-byte packing."""

    def test_bit_encoding_matches_e2m1_values(self):
        """FP4 codes decode to the E2M1 values via the IEEE bit layout."""
        grid = e2m1_grid()
        codes = torch.arange(15, dtype=torch.int8)
        bits = grid_codes_to_fp4_bits(codes)
        # sign | exponent(2) | mantissa(1): positive magnitudes 0..7 are
        # exactly the positive half of the grid.
        expected_magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        for code, bit in zip(codes.tolist(), bits.tolist()):
            value = grid[code].item()
            sign = -1.0 if (bit >> 3) & 1 else 1.0
            magnitude = expected_magnitudes[bit & 0x7].item()
            assert sign * magnitude == pytest.approx(value)

    def test_bits_roundtrip_all_codes(self):
        """grid -> bits -> grid is the identity for all 15 grid indices."""
        codes = torch.arange(15, dtype=torch.int8)
        assert torch.equal(fp4_bits_to_grid_codes(grid_codes_to_fp4_bits(codes)), codes)

    def test_negative_zero_decodes_to_zero(self):
        """The FP4 negative-zero code (0b1000) maps to the zero grid index."""
        bits = torch.tensor([0b1000], dtype=torch.uint8)
        assert fp4_bits_to_grid_codes(bits).item() == 7

    def test_pack_unpack_roundtrip(self):
        """pack -> unpack is the identity on random code tensors."""
        torch.manual_seed(0)
        codes = torch.randint(0, 15, (32, 64), dtype=torch.int8)
        packed = pack_fp4_codes(codes)
        assert packed.dtype == torch.uint8
        assert packed.shape == (32, 32)
        assert torch.equal(unpack_fp4_codes(packed), codes)

    def test_low_nibble_is_even_element(self):
        """Even-indexed elements occupy the low nibble (vLLM convention)."""
        codes = torch.tensor([[8, 0]], dtype=torch.int8)  # [+0.5, -6.0]
        packed = pack_fp4_codes(codes)
        low, high = packed.item() & 0xF, packed.item() >> 4
        assert low == 0b0001  # +0.5 -> exponent 0, mantissa 1
        assert high == 0b1111  # -6.0 -> sign 1, exponent 3, mantissa 1

    def test_pack_rejects_odd_width(self):
        """Packing requires an even number of elements per row."""
        with pytest.raises(ValueError):
            pack_fp4_codes(torch.zeros(2, 3, dtype=torch.int8))

    def test_quantize_pack_dequantize_consistency(self):
        """NVFP4 dequantization is unchanged by a pack/unpack round-trip."""
        torch.manual_seed(0)
        w = torch.randn(16, 64)
        codes, bs, ts = nvfp4_quantize(w, block_size=16)
        recovered = unpack_fp4_codes(pack_fp4_codes(codes))
        assert torch.equal(
            nvfp4_dequantize(recovered, bs, ts, block_size=16),
            nvfp4_dequantize(codes, bs, ts, block_size=16),
        )


class TestE8M0Encoding:
    """Tests for the biased-exponent E8M0 byte encoding."""

    def test_roundtrip_random_exponents(self):
        """uint8 -> scale -> uint8 is the identity across the E8M0 range."""
        exponents = torch.arange(-127, 128, dtype=torch.float32)
        scales = torch.exp2(exponents)
        encoded = e8m0_scales_to_uint8(scales)
        assert encoded.dtype == torch.uint8
        assert torch.equal(encoded, (exponents + 127).to(torch.uint8))
        assert torch.equal(uint8_to_e8m0_scales(encoded), scales)

    def test_mxfp4_scales_encode_exactly(self):
        """Scales produced by mxfp4_quantize encode without error."""
        torch.manual_seed(0)
        w = torch.randn(8, 64)
        _, block_scales = mxfp4_quantize(w, block_size=32)
        encoded = e8m0_scales_to_uint8(block_scales)
        assert torch.equal(uint8_to_e8m0_scales(encoded), block_scales)

    def test_out_of_range_raises(self):
        """Values outside [2^-127, 2^127] are rejected."""
        with pytest.raises(ValueError):
            e8m0_scales_to_uint8(torch.tensor([2.0**200]))


class TestNvfp4TensorScaleOverride:
    """Tests for nvfp4_quantize with an externally supplied global scale."""

    def test_override_is_used_verbatim(self):
        """The returned tensor_scale equals the override."""
        torch.manual_seed(0)
        w = torch.randn(8, 32)
        override = torch.tensor(0.01)
        codes, bs, ts = nvfp4_quantize(w, block_size=16, tensor_scale=override)
        assert ts.item() == pytest.approx(0.01)
        w_hat = nvfp4_dequantize(codes, bs, ts, block_size=16)
        rel = (torch.norm(w - w_hat) / torch.norm(w)).item()
        assert rel < 0.25

    def test_default_matches_derived_scale(self):
        """Passing the derived scale explicitly reproduces the default output."""
        torch.manual_seed(1)
        w = torch.randn(8, 32)
        codes_a, bs_a, ts_a = nvfp4_quantize(w, block_size=16)
        codes_b, bs_b, ts_b = nvfp4_quantize(w, block_size=16, tensor_scale=ts_a)
        assert torch.equal(codes_a, codes_b)
        assert torch.equal(bs_a, bs_b)
        assert torch.equal(ts_a, ts_b)


def _reconstruction_mse(w, w_hat, importance=None):
    err = (w - w_hat).square()
    if importance is not None:
        err = err * importance
    return err.sum().item()


class TestScaleSearch:
    """Tests for the local block-scale sweep (MSE / WMSE objectives)."""

    def test_nvfp4_sweep_never_increases_mse(self):
        """Swept scales achieve at most the AbsMax reconstruction MSE."""
        torch.manual_seed(0)
        for _ in range(5):
            w = torch.randn(16, 64) * torch.rand(16, 1) * 3
            codes, bs, ts = nvfp4_quantize(w, block_size=16)
            base = _reconstruction_mse(w, nvfp4_dequantize(codes, bs, ts, 16))
            codes_s, bs_s, ts_s = nvfp4_quantize(w, block_size=16, scale_search=True)
            swept = _reconstruction_mse(w, nvfp4_dequantize(codes_s, bs_s, ts_s, 16))
            assert swept <= base + 1e-9

    def test_nvfp4_sweep_improves_heavy_tailed_blocks(self):
        """A single outlier per block is the classic case AbsMax loses."""
        torch.manual_seed(1)
        w = torch.randn(32, 64)
        w[:, ::16] *= 10.0  # one outlier per 16-block
        codes, bs, ts = nvfp4_quantize(w, block_size=16)
        base = _reconstruction_mse(w, nvfp4_dequantize(codes, bs, ts, 16))
        codes_s, bs_s, ts_s = nvfp4_quantize(w, block_size=16, scale_search=True)
        swept = _reconstruction_mse(w, nvfp4_dequantize(codes_s, bs_s, ts_s, 16))
        assert swept < base

    def test_nvfp4_swept_scales_stay_on_e4m3_grid(self):
        """Every swept block scale must remain E4M3-representable."""
        torch.manual_seed(2)
        w = torch.randn(8, 64) * 5
        _, bs, _ = nvfp4_quantize(w, block_size=16, scale_search=True)
        assert torch.equal(bs, round_to_e4m3(bs))

    def test_mxfp4_sweep_never_increases_mse(self):
        """Swept E8M0 exponents achieve at most the ceil-rule MSE."""
        torch.manual_seed(3)
        for _ in range(5):
            w = torch.randn(16, 64) * torch.rand(16, 1) * 3
            codes, bs = mxfp4_quantize(w, block_size=32)
            base = _reconstruction_mse(w, mxfp4_dequantize(codes, bs, 32))
            codes_s, bs_s = mxfp4_quantize(w, block_size=32, scale_search=True)
            swept = _reconstruction_mse(w, mxfp4_dequantize(codes_s, bs_s, 32))
            assert swept <= base + 1e-9

    def test_mxfp4_swept_scales_are_powers_of_two(self):
        """Every swept MXFP4 scale must remain a power of two."""
        torch.manual_seed(4)
        w = torch.randn(8, 64) * 5
        _, bs = mxfp4_quantize(w, block_size=32, scale_search=True)
        exponents = torch.log2(bs)
        assert torch.equal(exponents, torch.round(exponents))

    def test_wmse_importance_shifts_the_optimum(self):
        """The weighted sweep must never be worse than AbsMax in WMSE."""
        torch.manual_seed(5)
        w = torch.randn(16, 64)
        w[:, ::16] *= 8.0
        importance = torch.rand(64) * 10
        codes, bs, ts = nvfp4_quantize(w, block_size=16)
        base = _reconstruction_mse(
            w, nvfp4_dequantize(codes, bs, ts, 16), importance.reshape(1, -1)
        )
        codes_s, bs_s, ts_s = nvfp4_quantize(
            w, block_size=16, scale_search=True, importance=importance
        )
        swept = _reconstruction_mse(
            w, nvfp4_dequantize(codes_s, bs_s, ts_s, 16), importance.reshape(1, -1)
        )
        assert swept <= base + 1e-9
