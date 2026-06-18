"""Shared constants, helpers, and fixtures for vllm_plugins tests.

Copyright 2025-2026 Fujitsu Ltd.
"""

import gc
import json
import os
from unittest.mock import MagicMock

import pytest
import torch

try:
    from vllm import LLM
except ImportError:
    LLM = None  # type: ignore[assignment]

from onecomp import CalibrationConfig, GPTQ, ModelConfig, Runner
from onecomp.pre_process.prepare_rotated_model import prepare_rotated_model

SMALL_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


class _DummyLayer:
    """Minimal fake linear layer for rotation hook tests."""

    def __init__(
        self,
        *,
        input_is_parallel: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        self.input_is_parallel = input_is_parallel
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.registered_hooks = []

    def register_forward_pre_hook(self, hook):
        self.registered_hooks.append(hook)
        return MagicMock(name="hook_handle")


def build_vllm_llm(model_path: str, **kwargs) -> "LLM":
    # gpu_memory_utilization is lowered from the vLLM default (0.92) to
    # accommodate DGX Spark's 128 GB Unified Memory and the test job's
    # SLURM cgroup limit (--mem=115G in run_test_vllm.sh):
    #   - 0.92 (~112 GiB) trips vLLM's own startup OOM check (only ~106
    #     GiB of UMA is free after AutoBit quantization runs in the same
    #     process).
    #   - 0.85 (~103 GiB) clears vLLM's check but the resulting Python
    #     residual (~16 GiB for vllm/transformers/torch imports + pytest
    #     state) plus 103 GiB allocation overflows the 115 GiB cgroup
    #     and the kernel OOM-kills the process.
    #   - 0.78 (~95 GiB) leaves ~4 GiB cgroup headroom and is the
    #     largest value we can use without cgroup OOM.
    if LLM is None:
        pytest.skip("vLLM is not installed; skipping test that requires it")
    return LLM(
        model=model_path,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.78,
        **kwargs,
    )


def assert_non_empty_outputs(outputs, expected_count: int) -> None:
    assert len(outputs) == expected_count
    assert all(len(output.outputs[0].text) > 0 for output in outputs)


def load_quantization_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        return json.load(f).get("quantization_config", {})


def release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def rotated_tinyllama_config(tmp_path_factory):
    """Rotation-preprocessed TinyLlama shared across all vllm_plugins test modules."""
    model_config = ModelConfig(model_id=SMALL_MODEL_ID, device="cuda:0")
    rotated_dir = str(tmp_path_factory.mktemp("rotated_tinyllama"))
    return prepare_rotated_model(
        model_config=model_config,
        save_directory=rotated_dir,
        rotation=True,
        scaling=False,
        enable_training=False,
        fp32_had=True,
    )


@pytest.fixture(scope="session")
def rotated_gptq_quantized_dir(rotated_tinyllama_config, tmp_path_factory):
    """Rotated + GPTQ-quantized checkpoint shared across all gptq test modules.

    Quantization runs once per session; both e2e and TP2 smoke tests load from
    the same directory since tensor_parallel_size is an inference-time setting
    and does not affect the saved weights.
    """
    runner = Runner(
        model_config=rotated_tinyllama_config,
        quantizer=GPTQ(wbits=4, groupsize=128),
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
        qep=False,
    )
    try:
        runner.run()
        save_dir = str(tmp_path_factory.mktemp("rotated_gptq_vllm"))
        runner.save_quantized_model(save_dir)
    finally:
        del runner
        release_gpu()
    return save_dir
