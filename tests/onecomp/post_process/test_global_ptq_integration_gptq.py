"""
Integration tests for GlobalPTQ with GPTQ quantization.

Requires a CUDA device and downloads TinyLlama from Hugging Face Hub.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

Usage:
    pytest tests/onecomp/post_process/test_global_ptq_integration_gptq.py -v -s --log-cli-level=INFO
"""

import gc

import pytest
import torch

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


@pytest.fixture(scope="module")
def quantized_tiny_llama():
    """Quantize TinyLlama with GPTQ and return the quantized model + config."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger

    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = GPTQ(wbits=4, groupsize=128)

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=CalibrationConfig(
            max_length=512,
            num_calibration_samples=8,
        ),
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


@_requires_cuda
class TestGlobalPTQIntegration:
    """Integration tests: GlobalPTQ on quantized TinyLlama."""

    @pytest.mark.slow
    def test_run_completes_and_improves_kl(self, quantized_tiny_llama):
        """GlobalPTQ.run() on TinyLlama completes and improves KL."""
        model, model_config = quantized_tiny_llama
        from onecomp import CalibrationConfig
        from onecomp.post_process._global_ptq.core import run_kl_distillation

        results = run_kl_distillation(
            model,
            model_config,
            epochs=2,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            eval_interval=1,
        )
        assert results["global_executed"] is True
        assert results["final_kl"] <= results["initial_kl"], (
            f"KL should not increase: {results['initial_kl']:.6f} " f"-> {results['final_kl']:.6f}"
        )

    @pytest.mark.slow
    def test_model_on_cpu_after_run(self, quantized_tiny_llama):
        """After run(), model should be on CPU."""
        model, _config = quantized_tiny_llama
        devices = {str(p.device) for p in model.parameters()}
        assert devices == {"cpu"}, f"Expected CPU, got {devices}"

    @pytest.mark.slow
    def test_model_in_eval_mode(self, quantized_tiny_llama):
        """After run(), model should be in eval mode."""
        model, _config = quantized_tiny_llama
        assert not model.training

    @pytest.mark.slow
    def test_use_cache_restored_after_run(self, quantized_tiny_llama):
        """model.config.use_cache is restored after GlobalPTQ.run().

        Before fix, gradient checkpointing set use_cache=False and never
        restored it.  After fix, original value is saved and restored.
        """
        model, _config = quantized_tiny_llama
        assert (
            getattr(model.config, "use_cache", None) is True
        ), "use_cache should be restored to True after run() completes"


@_requires_cuda
class TestGlobalPTQViaRunner:
    """Test GlobalPTQ integrated with Runner.run()."""

    @pytest.mark.slow
    def test_runner_with_global_ptq(self):
        """Runner with GlobalPTQ runs end-to-end without error."""
        from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger
        from onecomp.post_process.global_ptq import GlobalPTQ

        setup_logger()

        model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
        quantizer = GPTQ(wbits=4, groupsize=128)

        post = GlobalPTQ(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            eval_interval=1,
        )

        runner = Runner(
            model_config=model_config,
            quantizer=quantizer,
            calibration_config=CalibrationConfig(
                max_length=512,
                num_calibration_samples=8,
            ),
            post_processes=[post],
        )
        runner.run()

        assert runner.quantized_model is not None

        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

        gptq_count = sum(
            1 for _n, m in runner.quantized_model.named_modules() if isinstance(m, GPTQLinear)
        )
        assert gptq_count > 0, "GPTQ layers should still be present after global PTQ"

        del runner
        gc.collect()
        torch.cuda.empty_cache()
