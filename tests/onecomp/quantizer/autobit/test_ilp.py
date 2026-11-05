"""Tests for the RTN error estimates of AutoBit's ILP assignment.

Copyright 2025-2026 Fujitsu Ltd.
"""

import logging
import math

import pytest
import torch

from onecomp.quantizer.autobit.ilp import _rtn_errors
from onecomp.quantizer.gptq import GPTQ


@pytest.mark.parametrize("activation_aware", [False, True])
def test_rtn_errors_accept_groupsize_that_does_not_divide_in_features(
    activation_aware: bool,
) -> None:
    """in_features 40 with groupsize 16 leaves a last group of 8 columns."""
    torch.manual_seed(0)
    out_features, in_features = 8, 40
    weight = torch.randn(out_features, in_features)
    quantizers = [GPTQ(wbits=4, groupsize=16), GPTQ(wbits=8, groupsize=16)]
    activation = {}
    if activation_aware:
        activation = {
            "a_diag": torch.rand(in_features) + 0.1,
            "b_diag": torch.rand(out_features) + 0.1,
        }

    error_4bit, error_8bit = _rtn_errors(
        weight, quantizers, logging.getLogger(__name__), **activation
    )

    assert all(math.isfinite(error) and error > 0 for error in (error_4bit, error_8bit))
    assert error_4bit > error_8bit
