"""End-to-end test: rotation preprocessing + DBF save + vLLM inference.

Validates:
  1. A rotation-preprocessed model can be quantized and saved for vLLM
  2. The saved config.json carries rotated metadata and keeps quant_method="dbf"
     (unlike GPTQ, the DBF plugin natively handles rotation via
     maybe_wrap_rotation_method and does not rename the quant_method)
  3. vLLM loads the saved directory and generates output without errors

To keep runtime manageable, rotation training is disabled and the test uses
compact calibration settings for DBF.

Copyright 2025-2026 Fujitsu Ltd.
"""

import gc
import json
import os

import pytest
import torch

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

from onecomp import CalibrationConfig, DBF, ModelConfig, Runner
from onecomp.pre_process.prepare_rotated_model import prepare_rotated_model

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

SMALL_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def _build_vllm_llm(model_path: str) -> LLM:
    return LLM(
        model=model_path,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.78,
    )


def _assert_non_empty_outputs(outputs, expected_count: int) -> None:
    assert len(outputs) == expected_count
    assert all(len(output.outputs[0].text) > 0 for output in outputs)


def _load_quantization_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        return json.load(f).get("quantization_config", {})


@pytest.fixture(scope="module")
def rotated_quantized_model_dir(tmp_path_factory):
    """Build one rotated + DBF-quantized checkpoint for all tests in this module."""
    model_config = ModelConfig(model_id=SMALL_MODEL_ID, device="cuda:0")

    rotated_dir = str(tmp_path_factory.mktemp("rotated_model"))
    rotated_config = prepare_rotated_model(
        model_config=model_config,
        save_directory=rotated_dir,
        rotation=True,
        scaling=False,
        enable_training=False,
        fp32_had=True,
    )

    runner = Runner(
        model_config=rotated_config,
        quantizer=DBF(target_bits=1.5),
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
        qep=False,
    )
    runner.run()

    save_dir = str(tmp_path_factory.mktemp("rotated_dbf_vllm"))
    runner.save_quantized_model(save_dir)

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return save_dir


@pytest.fixture(scope="module")
def plain_quantized_model_dir(tmp_path_factory):
    """Build one plain DBF-quantized checkpoint for comparison tests."""
    runner = Runner(
        model_config=ModelConfig(model_id=SMALL_MODEL_ID, device="cuda:0"),
        quantizer=DBF(target_bits=1.5),
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
        qep=False,
    )
    runner.run()

    save_dir = str(tmp_path_factory.mktemp("plain_dbf_vllm"))
    runner.save_quantized_model(save_dir)

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return save_dir


class TestRotatedDBFQuantizeSave:
    """Verify the saved config is routed through the vLLM plugin path."""

    def test_config_json_exists(self, rotated_quantized_model_dir):
        config_path = os.path.join(rotated_quantized_model_dir, "config.json")
        assert os.path.exists(config_path)

    def test_quant_method_is_dbf(self, rotated_quantized_model_dir):
        qcfg = _load_quantization_config(rotated_quantized_model_dir)
        # DBF plugin natively handles rotation; quant_method stays "dbf"
        # (unlike GPTQ which is renamed to "mixed_gptq" upon save)
        assert qcfg.get("quant_method") == "dbf"

    def test_rotation_metadata_is_saved(self, rotated_quantized_model_dir):
        qcfg = _load_quantization_config(rotated_quantized_model_dir)
        assert qcfg.get("rotated") is True
        assert qcfg.get("fp32_had") is True


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestRotatedDBFVllmInference:
    """Load the rotated DBF checkpoint with vLLM and verify generation works."""

    def test_generate_produces_non_empty_output(self, rotated_quantized_model_dir):
        llm = _build_vllm_llm(rotated_quantized_model_dir)

        outputs = llm.generate(
            ["Fujitsu is"],
            SamplingParams(max_tokens=16, temperature=0.0),
        )

        _assert_non_empty_outputs(outputs, expected_count=1)

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    def test_batched_generate_produces_outputs(self, rotated_quantized_model_dir):
        llm = _build_vllm_llm(rotated_quantized_model_dir)

        outputs = llm.generate(
            ["Fujitsu is", "Tokyo is"],
            SamplingParams(max_tokens=12, temperature=0.0),
        )

        _assert_non_empty_outputs(outputs, expected_count=2)

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    def test_repeated_generate_produces_outputs(self, rotated_quantized_model_dir):
        llm = _build_vllm_llm(rotated_quantized_model_dir)

        # Reusing the same LLM instance is intended to catch regressions where
        # the Hadamard pre-hook is registered twice, the cached Hadamard state
        # becomes stale, or the internal pre-hook-installed flag is corrupted.
        first_outputs = llm.generate(
            ["Fujitsu is"],
            SamplingParams(max_tokens=12, temperature=0.0),
        )
        second_outputs = llm.generate(
            ["Osaka is"],
            SamplingParams(max_tokens=12, temperature=0.0),
        )

        _assert_non_empty_outputs(first_outputs, expected_count=1)
        _assert_non_empty_outputs(second_outputs, expected_count=1)

        del llm
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestPlainDBFVllmInference:
    """Plain (non-rotated) DBF checkpoint loads and generates in vLLM.

    Kept (unlike the GPTQ side) because plain DBF still routes through the
    in-house ``dbf`` plugin, whereas plain GPTQ ("gptq") is handled by vLLM's
    built-in path and exercises no first-party code.
    """

    def test_plain_dbf_generate_produces_non_empty_output(self, plain_quantized_model_dir):
        llm = _build_vllm_llm(plain_quantized_model_dir)

        outputs = llm.generate(
            ["Fujitsu is"],
            SamplingParams(max_tokens=16, temperature=0.0),
        )

        _assert_non_empty_outputs(outputs, expected_count=1)

        del llm
        gc.collect()
        torch.cuda.empty_cache()
