"""Tests for ``pseudo_quantize_tensor`` group handling.

Copyright 2025-2026 Fujitsu Ltd.
"""

import math

import pytest
import torch

from onecomp.quantizer.rtn.quantizer import pseudo_quantize_tensor

ROWS = 16
COLS = 40
GROUPSIZE = 16  # two full groups and a last group of 8 columns


def _weight(rows: int = ROWS, cols: int = COLS) -> torch.Tensor:
    """Return a seeded random weight of shape ``(rows, cols)``."""
    torch.manual_seed(0)
    return torch.randn(rows, cols)


@pytest.mark.parametrize("zero_point", [True, False])
@pytest.mark.parametrize("mse", [False, True])
def test_ragged_last_group_matches_quantizing_it_separately(zero_point: bool, mse: bool) -> None:
    """The shorter last group is quantized from its own columns only."""
    w = _weight()
    full = COLS - COLS % GROUPSIZE
    kwargs = {"n_bit": 4, "zero_point": zero_point, "mse": mse}

    outputs = pseudo_quantize_tensor(w, q_group_size=GROUPSIZE, **kwargs)
    head = pseudo_quantize_tensor(w[:, :full], q_group_size=GROUPSIZE, **kwargs)
    tail = pseudo_quantize_tensor(w[:, full:], q_group_size=COLS - full, **kwargs)

    for output, head_part, tail_part in zip(outputs, head, tail):
        assert torch.equal(output, torch.cat([head_part, tail_part], dim=-1))


def test_ragged_outputs_keep_input_shape_and_are_contiguous() -> None:
    """Weights keep the input shape; scale and zero have one column per group."""
    w = _weight()

    w_quant, scale, zero, w_int = pseudo_quantize_tensor(w, n_bit=4, q_group_size=GROUPSIZE)

    assert w_quant.shape == w_int.shape == w.shape
    assert w_quant.is_contiguous() and w_int.is_contiguous()
    assert scale.shape == zero.shape == (ROWS, math.ceil(COLS / GROUPSIZE))


@pytest.mark.parametrize("groupsize", [COLS, 64])
def test_groupsize_not_smaller_than_in_features_matches_per_channel(groupsize: int) -> None:
    """A group covering the whole row is per-channel quantization."""
    w = _weight()

    grouped = pseudo_quantize_tensor(w, n_bit=4, q_group_size=groupsize)
    per_channel = pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1)

    for output, expected in zip(grouped, per_channel):
        assert torch.equal(output, expected)


@pytest.mark.parametrize("groupsize", [8, -1])
def test_divisible_groupsize_keeps_layout_of_transposed_input(groupsize: int) -> None:
    """Outputs keep the input strides, so ``run_rtn`` results for Conv1D keep their layout."""
    w = _weight(rows=32, cols=16).t()

    outputs = pseudo_quantize_tensor(w, n_bit=4, q_group_size=groupsize)
    expected = pseudo_quantize_tensor(w.contiguous(), n_bit=4, q_group_size=groupsize)

    w_quant, _, _, w_int = outputs
    assert w_quant.stride() == w_int.stride() == w.stride()
    for output, expected_output in zip(outputs, expected):
        assert torch.equal(output, expected_output)
