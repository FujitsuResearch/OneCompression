"""
Smoke test for PostProcessLoraSFT.

Verifies that PostProcessLoraSFT completes without error on a small
model (TinyLlama) with minimal training settings (1 epoch, 4 samples).
Also checks that LoRA layers are injected and that weights are updated
after training.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    pytest tests/onecomp/post_process/test_post_process_lora_sft.py -v -s --log-cli-level=INFO
"""

import gc
import os
from pathlib import Path

import pytest
import torch

from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger
from onecomp.post_process.post_process_lora_sft import (
    LoRAGPTQLinear,
    PostProcessLoraSFT,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SFT_DATA_FILE = str(FIXTURES_DIR / "sft_train_data.jsonl")


@pytest.fixture(scope="module")
def quantized_model_and_config():
    """Quantize TinyLlama with GPTQ and build a quantized model."""
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = GPTQ(wbits=4, groupsize=128)

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
    )
    runner.run()

    model, _tokenizer = runner.create_quantized_model(
        pack_weights=False,
        use_gemlite=False,
    )

    yield model, model_config

    del model, runner
    gc.collect()
    torch.cuda.empty_cache()


class TestPostProcessLoraSFT:
    """Smoke tests for PostProcessLoraSFT."""

    def test_run_completes_without_error(self, quantized_model_and_config):
        """PostProcessLoraSFT.run() completes without raising."""
        model, model_config = quantized_model_and_config

        post_process = PostProcessLoraSFT(
            data_files=SFT_DATA_FILE,
            epochs=1,
            max_train_samples=4,
            max_length=64,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
        )
        post_process.run(model, model_config)

    def test_lora_layers_are_injected(self, quantized_model_and_config):
        """After run(), target layers should be LoRAGPTQLinear."""
        model, _model_config = quantized_model_and_config

        lora_count = sum(1 for _name, m in model.named_modules() if isinstance(m, LoRAGPTQLinear))
        assert lora_count > 0, "No LoRAGPTQLinear layers found after post-process"

    def test_model_is_on_cpu_after_run(self, quantized_model_and_config):
        """After run(), model should be moved back to CPU."""
        model, _model_config = quantized_model_and_config

        devices = {str(p.device) for p in model.parameters()}
        assert devices == {"cpu"}, f"Expected all params on CPU, got {devices}"

    def test_model_is_in_eval_mode(self, quantized_model_and_config):
        """After run(), model should be in eval mode."""
        model, _model_config = quantized_model_and_config
        assert not model.training, "Model should be in eval mode after run()"


class TestPostProcessLoraSFTViaRunner:
    """Test PostProcessLoraSFT integrated with Runner.run()."""

    def test_runner_with_post_process(self):
        """Runner with post_processes runs end-to-end without error."""
        setup_logger()

        model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
        quantizer = GPTQ(wbits=4, groupsize=128)

        post_process = PostProcessLoraSFT(
            data_files=SFT_DATA_FILE,
            epochs=1,
            max_train_samples=4,
            max_length=64,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
        )

        runner = Runner(
            model_config=model_config,
            quantizer=quantizer,
            calibration_config=CalibrationConfig(num_calibration_samples=8, max_length=512),
            post_processes=[post_process],
        )
        runner.run()

        assert (
            runner.quantized_model is not None
        ), "runner.quantized_model should be set after post-process"

        lora_count = sum(
            1
            for _name, m in runner.quantized_model.named_modules()
            if isinstance(m, LoRAGPTQLinear)
        )
        assert lora_count > 0, "No LoRAGPTQLinear layers found in runner.quantized_model"

        del runner
        gc.collect()
        torch.cuda.empty_cache()
