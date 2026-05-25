"""Tensor-parallel smoke test for rotated GPTQ vLLM inference.

Copyright 2025-2026 Fujitsu Ltd.
"""

import gc

import pytest
import torch

from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner
from onecomp.pre_process.prepare_rotated_model import prepare_rotated_model

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices"),
]

SMALL_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


@pytest.fixture(scope="module")
def rotated_quantized_model_dir(tmp_path_factory):
    """Build one rotated + GPTQ-quantized checkpoint for TP=2 smoke tests."""
    model_config = ModelConfig(model_id=SMALL_MODEL_ID, device="cuda:0")

    rotated_dir = str(tmp_path_factory.mktemp("rotated_model_tp2"))
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
        quantizer=GPTQ(wbits=4, groupsize=128),
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
        qep=False,
    )
    runner.run()

    save_dir = str(tmp_path_factory.mktemp("rotated_gptq_vllm_tp2"))
    runner.save_quantized_model(save_dir)

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    return save_dir


class TestRotatedGPTQVllmTensorParallelSmoke:
    """Smoke-test rotated GPTQ vLLM inference with tensor parallelism enabled."""

    def test_generate_produces_non_empty_output_with_tp2(self, rotated_quantized_model_dir):
        llm = LLM(
            model=rotated_quantized_model_dir,
            tensor_parallel_size=2,
            max_model_len=512,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.78,
        )

        outputs = llm.generate(
            ["Fujitsu is"],
            SamplingParams(max_tokens=16, temperature=0.0),
        )

        assert len(outputs) == 1
        text = outputs[0].outputs[0].text
        assert len(text) > 0, "vLLM TP=2 generated empty output for rotated GPTQ model"

        del llm
        gc.collect()
        torch.cuda.empty_cache()