"""
Integration tests for GlobalPTQ with DBF quantization.

This file is skipped by default because DBF quantization is heavy.
Set RUN_DBF_INTEGRATION_TESTS=1 to enable.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

Usage:
    RUN_DBF_INTEGRATION_TESTS=1 pytest tests/onecomp/post_process/test_global_ptq_integration_dbf.py -v -s --log-cli-level=INFO
"""

import gc
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_DBF_INTEGRATION_TESTS"),
    reason=(
        "DBF integration test is heavy (DBF quantize + GlobalPTQ on TinyLlama). "
        "Skipped by default; set RUN_DBF_INTEGRATION_TESTS=1 to enable. "
        "For end-to-end DBF + GlobalPTQ verification, use "
        "example/post_process/example_global_ptq_dbf.py."
    ),
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


@_requires_cuda
class TestGlobalPTQDbfIntegration:
    """Integration tests: GlobalPTQ on DBF quantized TinyLlama."""

    @pytest.fixture(scope="class")
    def dbf_quantized_tiny_llama(self):
        """Quantize TinyLlama with DBF and return the quantized model + config."""
        from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger
        from onecomp.quantizer.dbf import DBF

        setup_logger()

        model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
        quantizer = DBF()

        runner = Runner(
            model_config=model_config,
            quantizer=quantizer,
            calibration_config=CalibrationConfig(
                max_length=128,
                num_calibration_samples=4,
            ),
        )
        runner.run()

        model, _tokenizer = runner.create_quantized_model(use_gemlite=False)

        yield model, model_config

        del model, runner
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.slow
    def test_dbf_run_completes_without_error(self, dbf_quantized_tiny_llama):
        """GlobalPTQ.run() on DBF TinyLlama completes without raising."""
        model, model_config = dbf_quantized_tiny_llama
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq import GlobalPTQ

        gptq = GlobalPTQ(
            epochs=1,
            dbf_lr=5e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            eval_interval=1,
        )
        gptq.run(model, model_config)

    @pytest.mark.slow
    def test_dbf_model_on_cpu_after_run(self, dbf_quantized_tiny_llama):
        """After run(), model should be on CPU."""
        model, _config = dbf_quantized_tiny_llama
        devices = {str(p.device) for p in model.parameters()}
        assert devices == {"cpu"}, f"Expected CPU, got {devices}"

    @pytest.mark.slow
    def test_dbf_model_in_eval_mode(self, dbf_quantized_tiny_llama):
        """After run(), model should be in eval mode."""
        model, _config = dbf_quantized_tiny_llama
        assert not model.training
