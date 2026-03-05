"""
Regression tests for cumulative error analysis.

Verify that analyze_cumulative_error results closely match the reference data.
Run after refactoring to ensure the results have not changed.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

Usage:
    pytest tests/onecomp/test_cumulative_error_regression.py -v
"""

import json
import pytest
from pathlib import Path

from tests.onecomp.fixtures.generate_qep_gptq_reference import run_qep_gptq_quantization

# Path to reference data
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CUMULATIVE_ERROR_REFERENCE_FILE = FIXTURES_DIR / "qep_gptq_cumulative_error_reference.json"

# Keywords used for cumulative error analysis (same as when generating reference data)
LAYER_KEYWORDS = ["mlp.down_proj", "self_attn.o_proj"]

# Relative tolerance (considered equal if within 10%)
RELATIVE_TOLERANCE = 0.1
# Absolute tolerance (for very small values)
ABSOLUTE_TOLERANCE = 1e-10


@pytest.fixture(scope="module")
def cumulative_error_reference():
    """Load cumulative error reference data."""
    if not CUMULATIVE_ERROR_REFERENCE_FILE.exists():
        pytest.skip(
            f"Cumulative error reference file not found: {CUMULATIVE_ERROR_REFERENCE_FILE}"
        )
    with open(CUMULATIVE_ERROR_REFERENCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cumulative_error_results(cumulative_error_reference):  # pylint: disable=unused-argument
    """Run quantization and obtain cumulative error analysis results.

    Note:
        With scope="module", this runs only once within this module and
        the results are shared across all test methods.
    """
    runner = run_qep_gptq_quantization()
    results = runner.analyze_cumulative_error(layer_keywords=LAYER_KEYWORDS)
    return results


def _is_close(actual, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE):
    """Check whether a value is within the tolerance range."""
    if expected == 0:
        return abs(actual) <= atol
    relative_error = abs(actual - expected) / abs(expected)
    return relative_error <= rtol or abs(actual - expected) <= atol


def _get_relative_error(actual, expected):
    """Calculate the relative error."""
    if expected == 0:
        return float("inf") if actual != 0 else 0
    return abs(actual - expected) / abs(expected)


class TestCumulativeErrorRegression:
    """Regression tests for cumulative error analysis (default: per-keyword mode)."""

    def test_same_keywords(self, cumulative_error_results, cumulative_error_reference):
        """Verify that the same keywords are present in the results."""
        expected_keywords = set(cumulative_error_reference.keys())
        actual_keywords = set(cumulative_error_results.keys())
        assert expected_keywords == actual_keywords, (
            f"Keyword mismatch:\n"
            f"  Expected: {sorted(expected_keywords)}\n"
            f"  Actual: {sorted(actual_keywords)}"
        )

    def test_same_layers_per_keyword(
        self, cumulative_error_results, cumulative_error_reference
    ):
        """Verify that the same layers are analyzed for each keyword."""
        for keyword in cumulative_error_reference:
            expected_layers = set(cumulative_error_reference[keyword].keys())
            actual_layers = set(cumulative_error_results[keyword].keys())
            assert expected_layers == actual_layers, (
                f"Keyword '{keyword}' layer mismatch:\n"
                f"  Expected: {sorted(expected_layers)}\n"
                f"  Actual: {sorted(actual_layers)}"
            )

    def test_squared_error(self, cumulative_error_results, cumulative_error_reference):
        """Verify that the cumulative squared error closely matches the reference data."""
        for keyword, layers in cumulative_error_reference.items():
            for name, expected in layers.items():
                actual = cumulative_error_results[keyword][name]["squared_error"]
                expected_val = expected["squared_error"]
                rel_err = _get_relative_error(actual, expected_val)
                assert _is_close(actual, expected_val), (
                    f"[{keyword}] {name}: squared_error mismatch\n"
                    f"  Expected: {expected_val:.6e}\n"
                    f"  Actual:   {actual:.6e}\n"
                    f"  Relative error: {rel_err:.2%}"
                )

    def test_mean_squared_error(self, cumulative_error_results, cumulative_error_reference):
        """Verify that the cumulative mean squared error closely matches the reference data."""
        for keyword, layers in cumulative_error_reference.items():
            for name, expected in layers.items():
                actual = cumulative_error_results[keyword][name]["mean_squared_error"]
                expected_val = expected["mean_squared_error"]
                rel_err = _get_relative_error(actual, expected_val)
                assert _is_close(actual, expected_val), (
                    f"[{keyword}] {name}: mean_squared_error mismatch\n"
                    f"  Expected: {expected_val:.6e}\n"
                    f"  Actual:   {actual:.6e}\n"
                    f"  Relative error: {rel_err:.2%}"
                )


# --- Tests for batch_keywords=True ---


@pytest.fixture(scope="module")
def cumulative_error_results_batch(cumulative_error_reference):  # pylint: disable=unused-argument
    """Run cumulative error analysis with batch_keywords=True.

    This mode processes all keywords at once.
    Verify that it produces the same results as the per-keyword mode.
    """
    runner = run_qep_gptq_quantization()
    results = runner.analyze_cumulative_error(
        layer_keywords=LAYER_KEYWORDS, batch_keywords=True
    )
    return results


class TestCumulativeErrorRegressionBatch:
    """Regression tests for cumulative error analysis (batch_keywords=True mode).

    Verify that batch_keywords=True produces the same results as per-keyword (default).
    """

    def test_same_keywords(self, cumulative_error_results_batch, cumulative_error_reference):
        """Verify that the same keywords are present in the results."""
        expected_keywords = set(cumulative_error_reference.keys())
        actual_keywords = set(cumulative_error_results_batch.keys())
        assert expected_keywords == actual_keywords, (
            f"Keyword mismatch:\n"
            f"  Expected: {sorted(expected_keywords)}\n"
            f"  Actual: {sorted(actual_keywords)}"
        )

    def test_same_layers_per_keyword(
        self, cumulative_error_results_batch, cumulative_error_reference
    ):
        """Verify that the same layers are analyzed for each keyword."""
        for keyword in cumulative_error_reference:
            expected_layers = set(cumulative_error_reference[keyword].keys())
            actual_layers = set(cumulative_error_results_batch[keyword].keys())
            assert expected_layers == actual_layers, (
                f"Keyword '{keyword}' layer mismatch:\n"
                f"  Expected: {sorted(expected_layers)}\n"
                f"  Actual: {sorted(actual_layers)}"
            )

    def test_squared_error(self, cumulative_error_results_batch, cumulative_error_reference):
        """Verify that the cumulative squared error closely matches the reference data."""
        for keyword, layers in cumulative_error_reference.items():
            for name, expected in layers.items():
                actual = cumulative_error_results_batch[keyword][name]["squared_error"]
                expected_val = expected["squared_error"]
                rel_err = _get_relative_error(actual, expected_val)
                assert _is_close(actual, expected_val), (
                    f"[batch] [{keyword}] {name}: squared_error mismatch\n"
                    f"  Expected: {expected_val:.6e}\n"
                    f"  Actual:   {actual:.6e}\n"
                    f"  Relative error: {rel_err:.2%}"
                )

    def test_mean_squared_error(
        self, cumulative_error_results_batch, cumulative_error_reference
    ):
        """Verify that the cumulative mean squared error closely matches the reference data."""
        for keyword, layers in cumulative_error_reference.items():
            for name, expected in layers.items():
                actual = cumulative_error_results_batch[keyword][name]["mean_squared_error"]
                expected_val = expected["mean_squared_error"]
                rel_err = _get_relative_error(actual, expected_val)
                assert _is_close(actual, expected_val), (
                    f"[batch] [{keyword}] {name}: mean_squared_error mismatch\n"
                    f"  Expected: {expected_val:.6e}\n"
                    f"  Actual:   {actual:.6e}\n"
                    f"  Relative error: {rel_err:.2%}"
                )
