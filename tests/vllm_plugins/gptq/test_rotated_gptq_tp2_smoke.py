"""Tensor-parallel smoke test for rotated GPTQ vLLM inference.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

try:
    from vllm import SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

from ..conftest import build_vllm_llm, release_gpu

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices"),
]


class TestRotatedGPTQVllmTensorParallelSmoke:
    """Smoke-test rotated GPTQ vLLM inference with tensor parallelism enabled."""

    def test_generate_produces_non_empty_output_with_tp2(self, rotated_gptq_quantized_dir):
        llm = build_vllm_llm(rotated_gptq_quantized_dir, tensor_parallel_size=2)
        try:
            outputs = llm.generate(
                ["Fujitsu is"],
                SamplingParams(max_tokens=16, temperature=0.0),
            )
            assert len(outputs) == 1
            text = outputs[0].outputs[0].text
            assert len(text) > 0, "vLLM TP=2 generated empty output for rotated GPTQ model"
        finally:
            del llm
            release_gpu()
