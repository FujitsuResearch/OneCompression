"""Tensor-parallel smoke test for rotated GPTQ vLLM inference.

Copyright 2025-2026 Fujitsu Ltd.
"""

import gc

import pytest
import torch

from tests.vllm_plugins.gptq.test_rotated_gptq_e2e import _HAS_VLLM, rotated_quantized_model_dir

if _HAS_VLLM:
    from vllm import LLM, SamplingParams


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices"),
]


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