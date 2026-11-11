"""End-to-end test: JointQ + group_swap on full TinyLlama, then PPL.

Quantize *all* linear layers of TinyLlama with JointQ
(``enable_group_swap=True``), then evaluate the dequantized model's
WikiText-2 perplexity. The test asserts that:

1. The dequantized PPL is finite (not NaN / Inf).
2. The dequantized PPL stays below ``PPL_UPPER_BOUND``.

The bound is a coarse sanity check: TinyLlama's WikiText-2 PPL is in
the high single digits, so anything below ~10 indicates the
quantization pipeline ran successfully end-to-end without breaking
the model. Original (unquantized) PPL is not computed; we only care
that the dequantized model is not catastrophically broken.

This test runs the full pipeline end-to-end and is slow. It is marked
``@pytest.mark.slow`` and is skipped without CUDA.

Run with:

    pytest tests/onecomp/quantizer/jointq/test_jointq_groupswap_ppl_tinyllama.py \
        -v -s -m slow

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import gc
import math

import pytest
import torch

from onecomp import CalibrationConfig, JointQ, ModelConfig, Runner, setup_logger

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Upper bound for the dequantized PPL on WikiText-2 (test split).
# TinyLlama's reference PPL is in the high single digits; anything
# below this threshold confirms the full quantization pipeline ran
# end-to-end without breaking the model.
PPL_UPPER_BOUND = 10.5


def _quantize_and_eval_dequantized_ppl() -> float:
    """Quantize the full TinyLlama model and return the dequantized PPL."""
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = JointQ(
        bits=3,
        symmetric=True,
        actorder=True,
        group_size=128,
        log_level=1,
        enable_group_swap=True,
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=128,
            strategy="drop_rand",
            seed=0,
        ),
        qep=False,
    )
    runner.run()

    _, dequant_ppl, _ = runner.calculate_perplexity(
        original_model=False,
        dequantized_model=True,
        quantized_model=False,
    )
    print("dequant_ppl: ", dequant_ppl)

    del runner, quantizer
    gc.collect()
    torch.cuda.empty_cache()

    return float(dequant_ppl)


@pytest.fixture(scope="module")
def dequant_ppl():
    """End-to-end JointQ + group_swap on TinyLlama; dequantized PPL."""
    return _quantize_and_eval_dequantized_ppl()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestJointQGroupSwapPplTinyLlama:
    """Full-model end-to-end PPL check for JointQ with group swap."""

    def test_dequantized_ppl_finite(self, dequant_ppl):
        """Dequantized PPL must not be NaN / Inf."""
        assert math.isfinite(dequant_ppl), f"Dequantized PPL is not finite: {dequant_ppl}"

    def test_dequantized_ppl_within_bound(self, dequant_ppl):
        """Dequantized PPL should stay below the regression upper bound."""
        assert dequant_ppl < PPL_UPPER_BOUND, (
            f"Dequantized PPL regression: " f"dequant={dequant_ppl:.4f} (bound={PPL_UPPER_BOUND})"
        )
