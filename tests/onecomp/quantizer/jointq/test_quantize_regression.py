"""Regression test for jointq.quantize.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Regression test to verify that quantization quality (MSE) does not degrade
after refactoring of jointq.quantize.

Test Contents
-------------
test_quantize_regression:
    Re-run quantize with fixed data and parameters, then compare the MSE
    against the hardcoded expected value (EXPECTED_MSE in
    regression_quantize_helper.py) with rel_tol=1e-2.

    Only MSE is checked because exact tensor values (integers_z, zero_point,
    scales) can vary across CUDA/PyTorch versions due to floating-point
    rounding without affecting overall quantization quality.

    Skipped if CUDA is unavailable or input data is not present.
"""

import math

import pytest
import torch

from .regression_quantize_helper import DEFAULT_DATA_PATH, EXPECTED_MSE, run_quantize

_SKIP_NO_CUDA = not torch.cuda.is_available()
_SKIP_NO_DATA = not DEFAULT_DATA_PATH.exists()


@pytest.mark.skipif(_SKIP_NO_CUDA, reason="CUDA is not available")
@pytest.mark.skipif(_SKIP_NO_DATA, reason=f"Data file not found: {DEFAULT_DATA_PATH}")
def test_quantize_regression():
    """Test that quantize produces the same quality as the expected baseline."""

    actual = run_quantize(DEFAULT_DATA_PATH, device_id=0)

    assert math.isclose(
        actual["mse"], EXPECTED_MSE, rel_tol=1e-2
    ), f"MSE mismatch: actual={actual['mse']}, expected={EXPECTED_MSE}"
