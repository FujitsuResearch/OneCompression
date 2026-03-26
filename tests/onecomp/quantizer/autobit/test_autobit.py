"""Tests for the AutoBit quantizer implementation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

import os

import pytest

from onecomp.quantizer.autobit._autobit import AutoBitQuantizer
from onecomp.quantizer.gptq import GPTQ
from onecomp.utils import estimate_wbits_from_vram
from onecomp import Runner, ModelConfig

pytestmark = pytest.mark.slow

# Large-model tests (e.g. 70B) are extremely slow and memory-intensive,
# so they are skipped by default.  To run them:
#   RUN_LARGE_MODEL_TESTS=1 pytest tests/onecomp/quantizer/autobit/
_skip_large = pytest.mark.skipif(
    not os.environ.get("RUN_LARGE_MODEL_TESTS"),
    reason="Large-model test skipped by default. Set RUN_LARGE_MODEL_TESTS=1 to run.",
)

SMALL_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
LARGE_MODEL_ID = "meta-llama/Meta-Llama-3-70B"

CANDIDATES = [GPTQ(wbits=b) for b in (2, 3, 4, 5)]


def _estimate(model_id, total_vram_gb):
    result = estimate_wbits_from_vram(model_id, total_vram_gb=total_vram_gb)
    return result.target_bitwidth


def test_autobit_small_model_ilp():
    target_bit = _estimate(SMALL_MODEL_ID, total_vram_gb=0.8)
    quantizer = AutoBitQuantizer(
        assignment_strategy="activation_aware",
        target_bit=target_bit,
        quantizers=CANDIDATES,
    )

    runner = Runner(
        model_config=ModelConfig(model_id=SMALL_MODEL_ID, device="cpu"),
        quantizer=quantizer,
        qep=True,
    )
    runner.run()

    # TODO: add assertions on get_quant_config() to verify assignment results


def test_autobit_small_model_dbf():
    target_bit = _estimate(SMALL_MODEL_ID, total_vram_gb=0.4)
    quantizer = AutoBitQuantizer(
        assignment_strategy="activation_aware",
        target_bit=target_bit,
        dbf_iters=10,
        quantizers=CANDIDATES,
    )

    runner = Runner(
        model_config=ModelConfig(model_id=SMALL_MODEL_ID, device="cpu"),
        quantizer=quantizer,
        qep=True,
    )
    runner.run()

    config = quantizer.get_quant_config()
    active = [q for q in config["quantizers"] if q["layers"]]
    assert len(active) == 1 and "dbf" == active[0]["quant_method"].lower(), (
        f"Expected all layers assigned to DBF, got: "
        f"{[(q['quant_method'], len(q['layers'])) for q in active]}"
    )


def test_autobit_small_model_error():
    target_bit = _estimate(SMALL_MODEL_ID, total_vram_gb=0.2)
    quantizer = AutoBitQuantizer(
        assignment_strategy="activation_aware",
        target_bit=target_bit,
        quantizers=CANDIDATES,
    )

    runner = Runner(
        model_config=ModelConfig(model_id=SMALL_MODEL_ID, device="cpu"),
        quantizer=quantizer,
        qep=True,
    )
    with pytest.raises(ValueError, match="target_bit=.* is below 1.0 bpw"):
        runner.run()


@_skip_large
def test_autobit_large_model():
    target_bit = _estimate(LARGE_MODEL_ID, total_vram_gb=40)
    quantizer = AutoBitQuantizer(
        assignment_strategy="activation_aware",
        target_bit=target_bit,
        quantizers=CANDIDATES,
    )

    runner = Runner(
        model_config=ModelConfig(model_id=LARGE_MODEL_ID, device="cpu"),
        quantizer=quantizer,
        qep=True,
    )
    runner.run()
