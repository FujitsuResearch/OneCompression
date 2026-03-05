"""
QEP regression tests.

Verify that GPTQ+QEP quantization results closely match the reference data.
Run after refactoring to ensure the results have not changed.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

Usage:
    pytest tests/onecomp/test_qep_regression.py -v
"""

import json
import pytest
from pathlib import Path

from tests.onecomp.fixtures.generate_qep_gptq_reference import run_qep_gptq_quantization

# Path to reference data
FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_FILE = FIXTURES_DIR / "qep_gptq_reference.json"

# Relative tolerance (considered equal if within 10%)
RELATIVE_TOLERANCE = 0.1
# Absolute tolerance (for very small values)
ABSOLUTE_TOLERANCE = 1e-10


@pytest.fixture(scope="module")
def reference_data():
    """Load reference data."""
    if not REFERENCE_FILE.exists():
        pytest.skip(f"Reference file not found: {REFERENCE_FILE}")
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def qep_results(reference_data):  # pylint: disable=unused-argument
    """Run GPTQ+QEP quantization and obtain the results.

    Quantization is performed with the same settings as when the reference
    data was generated.

    Note:
        With scope="module", this runs only once within this module and
        the results are shared across all test methods (quantization runs once).
    """
    runner = run_qep_gptq_quantization()
    return runner.quantizer.results


def _is_close(actual, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE):
    """Check whether a value is within the tolerance range.

    Args:
        actual: Actual value.
        expected: Expected value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        bool: True if within the tolerance range.
    """
    if expected == 0:
        return abs(actual) <= atol
    relative_error = abs(actual - expected) / abs(expected)
    return relative_error <= rtol or abs(actual - expected) <= atol


def _get_relative_error(actual, expected):
    """Calculate the relative error."""
    if expected == 0:
        return float("inf") if actual != 0 else 0
    return abs(actual - expected) / abs(expected)


class TestQEPGPTQRegression:
    """Regression tests for GPTQ+QEP quantization."""

    def test_same_layers_quantized(self, qep_results, reference_data):
        """Verify that the same layers are quantized."""
        expected_layers = set(reference_data.keys())
        actual_layers = set(qep_results.keys())
        assert expected_layers == actual_layers, (
            f"Layer mismatch:\n"
            f"  Expected: {sorted(expected_layers)}\n"
            f"  Actual: {sorted(actual_layers)}"
        )

    def test_output_squared_error(self, qep_results, reference_data):
        """Verify that the output squared error closely matches the reference data."""
        for name, expected in reference_data.items():
            actual = qep_results[name].output_squared_error
            expected_val = expected["output_squared_error"]
            rel_err = _get_relative_error(actual, expected_val)
            assert _is_close(actual, expected_val), (
                f"{name}: output_squared_error mismatch\n"
                f"  Expected: {expected_val:.6e}\n"
                f"  Actual:   {actual:.6e}\n"
                f"  Relative error: {rel_err:.2%}"
            )

    def test_mean_output_squared_error(self, qep_results, reference_data):
        """Verify that the mean output squared error closely matches the reference data."""
        for name, expected in reference_data.items():
            actual = qep_results[name].mean_output_squared_error
            expected_val = expected["mean_output_squared_error"]
            rel_err = _get_relative_error(actual, expected_val)
            assert _is_close(actual, expected_val), (
                f"{name}: mean_output_squared_error mismatch\n"
                f"  Expected: {expected_val:.6e}\n"
                f"  Actual:   {actual:.6e}\n"
                f"  Relative error: {rel_err:.2%}"
            )

    def test_weight_squared_error(self, qep_results, reference_data):
        """Verify that the weight squared error closely matches the reference data."""
        for name, expected in reference_data.items():
            actual = qep_results[name].weight_squared_error
            expected_val = expected["weight_squared_error"]
            rel_err = _get_relative_error(actual, expected_val)
            assert _is_close(actual, expected_val), (
                f"{name}: weight_squared_error mismatch\n"
                f"  Expected: {expected_val:.6e}\n"
                f"  Actual:   {actual:.6e}\n"
                f"  Relative error: {rel_err:.2%}"
            )

    def test_mean_weight_squared_error(self, qep_results, reference_data):
        """Verify that the mean weight squared error closely matches the reference data."""
        for name, expected in reference_data.items():
            actual = qep_results[name].mean_weight_squared_error
            expected_val = expected["mean_weight_squared_error"]
            rel_err = _get_relative_error(actual, expected_val)
            assert _is_close(actual, expected_val), (
                f"{name}: mean_weight_squared_error mismatch\n"
                f"  Expected: {expected_val:.6e}\n"
                f"  Actual:   {actual:.6e}\n"
                f"  Relative error: {rel_err:.2%}"
            )
