"""
Consistency test between generic and architecture-aware QEP implementations.

Verify that general=True and general=False produce similar quantization
results (dequantized weights and weight errors per layer).

Note:
    The architecture-aware implementation (general=False) does not capture
    per-layer input activations (quant_input_activation=None), so output
    quantization errors (output_squared_error, etc.) are not available.
    This test compares dequantized weights and weight-based errors only.

    Layers whose input does not pass through attention (q_proj, k_proj,
    v_proj) produce bit-identical results between the two implementations.
    Layers whose input depends on attention output (o_proj, gate_proj,
    up_proj, down_proj) show ~14% relative weight difference due to
    different forward-pass batch sizes: the generic implementation
    captures activations via a single full-model forward (batch=N),
    while the architecture-aware implementation uses block-level
    forwards in smaller batches.  Tiny floating-point differences in
    attention matmul/softmax are amplified by GPTQ's discrete
    quantization and error-feedback mechanism.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    pytest tests/onecomp/test_qep_general_consistency.py -v
"""

import gc

import pytest
import torch

from onecomp import ModelConfig, Runner, setup_logger
from onecomp.qep import QEPConfig
from onecomp.quantizer.gptq import GPTQ


# Relative tolerance for dequantized weight comparison (Frobenius norm).
# Layers after attention (o_proj, MLP) show ~14% difference due to
# forward-pass batch-size mismatch between the two implementations.
WEIGHT_RELATIVE_TOLERANCE = 0.20

# Relative tolerance for scalar error metrics
RELATIVE_TOLERANCE = 0.20
# Absolute tolerance (for very small values)
ABSOLUTE_TOLERANCE = 1e-10

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
NUM_LAYERS = 7  # 1 Transformer block (q,k,v,o,gate,up,down)


def _run_qep(general: bool) -> dict:
    """Run GPTQ+QEP quantization with the given general flag.

    Args:
        general: If True, use the generic implementation;
                 if False, use the architecture-aware implementation.

    Returns:
        dict: Quantization results (quantizer.results).
    """
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = GPTQ(
        wbits=4,
        groupsize=128,
        sym=False,
        num_layers=NUM_LAYERS,
        calc_quant_error=True,
    )
    qep_config = QEPConfig(general=general)

    runner = Runner(
        model_config=model_config,
        max_length=512,
        num_calibration_samples=128,
        calibration_strategy="drop_rand",
        calibration_seed=0,
        quantizer=quantizer,
        qep=True,
        qep_config=qep_config,
    )
    runner.run()

    results = quantizer.results

    del runner, quantizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


@pytest.fixture(scope="module")
def results_general_true():
    """Run QEP with general=True."""
    return _run_qep(general=True)


@pytest.fixture(scope="module")
def results_general_false():
    """Run QEP with general=False."""
    return _run_qep(general=False)


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


class TestQEPGeneralConsistency:
    """Verify that general=True and general=False produce consistent results."""

    def test_same_layers_quantized(self, results_general_true, results_general_false):
        """Both implementations should quantize the same layers."""
        layers_true = set(results_general_true.keys())
        layers_false = set(results_general_false.keys())
        assert layers_true == layers_false, (
            f"Layer mismatch:\n"
            f"  general=True:  {sorted(layers_true)}\n"
            f"  general=False: {sorted(layers_false)}"
        )

    def test_dequantized_weight_close(self, results_general_true, results_general_false):
        """Dequantized weights should be close between implementations.

        Compares ||W_true - W_false||_F / ||W_true||_F for each layer.
        """
        for name in results_general_true:
            w_true = results_general_true[name].dequantized_weight.float()
            w_false = results_general_false[name].dequantized_weight.float()

            diff_norm = torch.norm(w_true - w_false).item()
            base_norm = torch.norm(w_true).item()
            rel_err = diff_norm / base_norm if base_norm > 0 else float("inf")

            assert rel_err <= WEIGHT_RELATIVE_TOLERANCE, (
                f"{name}: dequantized_weight mismatch\n"
                f"  ||W_true - W_false||_F: {diff_norm:.6e}\n"
                f"  ||W_true||_F:           {base_norm:.6e}\n"
                f"  Relative error:         {rel_err:.2%}"
            )

    def test_weight_squared_error(self, results_general_true, results_general_false):
        """Weight squared errors should be close between implementations."""
        for name in results_general_true:
            actual = results_general_false[name].weight_squared_error
            expected = results_general_true[name].weight_squared_error
            rel_err = _get_relative_error(actual, expected)
            assert _is_close(actual, expected), (
                f"{name}: weight_squared_error mismatch\n"
                f"  general=True:  {expected:.6e}\n"
                f"  general=False: {actual:.6e}\n"
                f"  Relative error: {rel_err:.2%}"
            )

    def test_mean_weight_squared_error(self, results_general_true, results_general_false):
        """Mean weight squared errors should be close between implementations."""
        for name in results_general_true:
            actual = results_general_false[name].mean_weight_squared_error
            expected = results_general_true[name].mean_weight_squared_error
            rel_err = _get_relative_error(actual, expected)
            assert _is_close(actual, expected), (
                f"{name}: mean_weight_squared_error mismatch\n"
                f"  general=True:  {expected:.6e}\n"
                f"  general=False: {actual:.6e}\n"
                f"  Relative error: {rel_err:.2%}"
            )

    def test_output_error_none_for_arch(self, results_general_false):
        """Architecture-aware impl should have None for output errors.

        general=False passes quant_input_activation=None, so output
        quantization errors cannot be computed.
        """
        for name in results_general_false:
            result = results_general_false[name]
            assert (
                result.output_squared_error is None
            ), f"{name}: output_squared_error should be None for general=False"
            assert (
                result.mean_output_squared_error is None
            ), f"{name}: mean_output_squared_error should be None for general=False"
