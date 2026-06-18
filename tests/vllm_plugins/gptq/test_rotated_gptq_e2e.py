"""End-to-end test: rotation preprocessing + GPTQ save + vLLM inference.

Validates:
  1. A rotation-preprocessed model can be quantized and saved for vLLM
  2. The saved config.json carries rotated metadata and uses mixed_gptq
  3. vLLM loads the saved directory and generates output without errors

To keep runtime manageable, rotation training is disabled and the test uses
compact calibration settings for GPTQ.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os

import pytest
import torch

try:
    from vllm import SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

from ..conftest import (
    assert_non_empty_outputs,
    build_vllm_llm,
    load_quantization_config,
    release_gpu,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


class TestRotatedGPTQQuantizeSave:
    """Verify the saved config is routed through the vLLM plugin path."""

    def test_config_json_exists(self, rotated_gptq_quantized_dir):
        config_path = os.path.join(rotated_gptq_quantized_dir, "config.json")
        assert os.path.exists(config_path)

    def test_quant_method_is_mixed_gptq(self, rotated_gptq_quantized_dir):
        qcfg = load_quantization_config(rotated_gptq_quantized_dir)
        assert qcfg.get("quant_method") == "mixed_gptq"

    def test_rotation_metadata_is_saved(self, rotated_gptq_quantized_dir):
        qcfg = load_quantization_config(rotated_gptq_quantized_dir)
        assert qcfg.get("rotated") is True
        assert qcfg.get("fp32_had") is True


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestRotatedGPTQVllmInference:
    """Load the rotated GPTQ checkpoint with vLLM and verify generation works."""

    def test_generate_produces_non_empty_output(self, rotated_gptq_quantized_dir):
        llm = build_vllm_llm(rotated_gptq_quantized_dir)
        try:
            outputs = llm.generate(
                ["Fujitsu is"],
                SamplingParams(max_tokens=16, temperature=0.0),
            )
            assert_non_empty_outputs(outputs, expected_count=1)
        finally:
            del llm
            release_gpu()

    def test_batched_generate_produces_outputs(self, rotated_gptq_quantized_dir):
        llm = build_vllm_llm(rotated_gptq_quantized_dir)
        try:
            outputs = llm.generate(
                ["Fujitsu is", "Tokyo is"],
                SamplingParams(max_tokens=12, temperature=0.0),
            )
            assert_non_empty_outputs(outputs, expected_count=2)
        finally:
            del llm
            release_gpu()

    def test_repeated_generate_produces_outputs(self, rotated_gptq_quantized_dir):
        llm = build_vllm_llm(rotated_gptq_quantized_dir)
        try:
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
            assert_non_empty_outputs(first_outputs, expected_count=1)
            assert_non_empty_outputs(second_outputs, expected_count=1)
        finally:
            del llm
            release_gpu()
