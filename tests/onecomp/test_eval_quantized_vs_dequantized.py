"""
Test that quantized and dequantized models produce similar PPL values.

Quantize the model with multiple GPTQ configurations in a single Runner
invocation (shared calibration), then use ``benchmark_perplexity`` to
evaluate both the quantized and dequantized models for each configuration.
Assert that the PPL difference is within tolerance.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    pytest tests/onecomp/test_eval_quantized_vs_dequantized.py -v
"""

import gc

import pytest
import torch

from onecomp import GPTQ, ModelConfig, Runner, setup_logger

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Relative tolerance for PPL comparison between quantized and dequantized.
# Both representations decode the same quantized weights, so PPL should
# be very close; small differences arise from packing/unpacking numerics.
PPL_RELATIVE_TOLERANCE = 0.02

# GPTQ quantization variants to test
QUANTIZER_CONFIGS = [
    {
        "name": "gptq_g-1",
        "wbits": 4,
        "groupsize": -1,
        "actorder": False,
        "sym": False,
        "num_layers": 28,
    },
    {
        "name": "gptq_g128",
        "wbits": 4,
        "groupsize": 128,
        "actorder": False,
        "sym": False,
        "num_layers": 28,
    },
    {
        "name": "gptq_g128_act",
        "wbits": 4,
        "groupsize": 128,
        "actorder": True,
        "sym": False,
        "num_layers": 28,
    },
    {
        "name": "gptq_g128_sym",
        "wbits": 4,
        "groupsize": 128,
        "actorder": False,
        "sym": True,
        "num_layers": 28,
    },
]


def _build_quantizers():
    """Create GPTQ quantizer instances from QUANTIZER_CONFIGS."""
    quantizers = []
    for cfg in QUANTIZER_CONFIGS:
        q = GPTQ(
            wbits=cfg["wbits"],
            groupsize=cfg["groupsize"],
            actorder=cfg["actorder"],
            sym=cfg["sym"],
            num_layers=cfg["num_layers"],
        )
        q.name = cfg["name"]
        quantizers.append(q)
    return quantizers


@pytest.fixture(scope="module")
def ppl_dict():
    """Run quantization once with all configs and benchmark PPL."""
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizers = _build_quantizers()

    runner = Runner(
        model_config=model_config,
        quantizers=quantizers,
        qep=False,
        calibration_batch_size=128,
    )
    runner.run()

    result = runner.benchmark_perplexity(
        original_model=False,
        dequantized_model=True,
        quantized_model=True,
    )

    del runner, quantizers
    gc.collect()
    torch.cuda.empty_cache()

    return result


@pytest.fixture(params=[c["name"] for c in QUANTIZER_CONFIGS])
def config_name(request):
    return request.param


class TestQuantizedVsDequantizedPPL:
    """Verify that quantized and dequantized models produce similar PPL."""

    def test_original_ppl_not_computed(self, ppl_dict):
        assert (
            "original" not in ppl_dict
        ), "original PPL should not be computed (original_model=False)"

    def test_dequantized_ppl_is_not_none(self, ppl_dict, config_name):
        key = config_name + "_dequantized"
        assert (
            key in ppl_dict and ppl_dict[key] is not None
        ), f"{config_name}: dequantized PPL should not be None"

    def test_quantized_ppl_is_not_none(self, ppl_dict, config_name):
        assert (
            config_name in ppl_dict and ppl_dict[config_name] is not None
        ), f"{config_name}: quantized PPL should not be None"

    def test_ppl_difference_within_tolerance(self, ppl_dict, config_name):
        quant_ppl = ppl_dict.get(config_name)
        dequant_ppl = ppl_dict.get(config_name + "_dequantized")

        if dequant_ppl is None or quant_ppl is None:
            pytest.skip(f"{config_name}: one of the PPL values is None")

        rel_err = abs(quant_ppl - dequant_ppl) / dequant_ppl
        assert rel_err <= PPL_RELATIVE_TOLERANCE, (
            f"{config_name}: PPL mismatch between quantized and dequantized\n"
            f"  dequantized PPL: {dequant_ppl:.4f}\n"
            f"  quantized PPL:   {quant_ppl:.4f}\n"
            f"  relative error:  {rel_err:.4%}\n"
            f"  tolerance:       {PPL_RELATIVE_TOLERANCE:.4%}"
        )
