"""End-to-end test: DBF quantization -> vLLM inference.

Validates:
  1. DBF quantization produces a valid saved model
  2. The saved config.json contains DBF method entries in quantization_bits
  3. vLLM loads the model and generates output without errors

Quantization runs once per module (shared via fixture) with qep=False and
minimal calibration samples to keep runtime short.

Requirements: CUDA GPU.  vLLM tests additionally require vLLM.

Copyright 2025-2026 Fujitsu Ltd.

"""

import gc
import json
import os

import pytest
import torch

# Use the DBF naive vLLM path for this E2E.
os.environ.setdefault("ONECOMP_DBF_NAIVE_LINEAR", "1")

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

E2E_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


@pytest.fixture(scope="module")
def dbf_quantized_model_dir(tmp_path_factory):
    """Quantize once with DBF and save to a temp directory.

    Shared across all tests in this module.
    Uses qep=False and num_calibration_samples=8 for speed.
    """
    from onecomp import DBF, CalibrationConfig, ModelConfig, Runner

    quantizer = DBF(target_bits=2.0, iters=10, balance_iters=5)
    runner = Runner(
        model_config=ModelConfig(model_id=E2E_MODEL_ID, device="cuda:0"),
        quantizer=quantizer,
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
        qep=False,
    )
    runner.run()

    save_dir = str(tmp_path_factory.mktemp("dbf_model"))
    runner.save_quantized_model(save_dir)

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return save_dir


# ---------------------------------------------------------------------------
# Config verification (no vLLM needed)
# ---------------------------------------------------------------------------


class TestDbfQuantizeSaveConfig:
    """Verify the saved model contains DBF method entries in its config."""

    def test_config_json_exists(self, dbf_quantized_model_dir):
        assert os.path.exists(os.path.join(dbf_quantized_model_dir, "config.json"))

    def test_quant_method_is_dbf(self, dbf_quantized_model_dir):
        with open(os.path.join(dbf_quantized_model_dir, "config.json")) as f:
            qcfg = json.load(f).get("quantization_config", {})
        assert qcfg.get("quant_method") == "dbf"

    def test_quantization_bits_not_empty(self, dbf_quantized_model_dir):
        with open(os.path.join(dbf_quantized_model_dir, "config.json")) as f:
            qcfg = json.load(f).get("quantization_config", {})
        qbits = qcfg.get("quantization_bits", [])
        assert len(qbits) > 0, "quantization_bits is empty"

    def test_quantization_bits_carry_dbf_method(self, dbf_quantized_model_dir):
        with open(os.path.join(dbf_quantized_model_dir, "config.json")) as f:
            qcfg = json.load(f).get("quantization_config", {})
        qbits = qcfg.get("quantization_bits", [])

        methods = set()
        for layer_cfg in qbits:
            for mod_cfg in layer_cfg.values():
                methods.add(mod_cfg.get("method"))

        assert methods == {
            "dbf"
        }, f"Expected every module to be quantized with method 'dbf', found {methods}."


# ---------------------------------------------------------------------------
# vLLM inference (requires vLLM)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestDbfVllmInference:
    """Load the quantized model with vLLM and verify generation works."""

    def test_generate_produces_non_empty_output(self, dbf_quantized_model_dir):
        llm = LLM(
            model=dbf_quantized_model_dir,
            max_model_len=512,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.78,
        )

        outputs = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=16, temperature=0.0),
        )

        assert len(outputs) == 1
        text = outputs[0].outputs[0].text
        assert len(text) > 0, "vLLM generated empty output"

        del llm
        gc.collect()
        torch.cuda.empty_cache()
