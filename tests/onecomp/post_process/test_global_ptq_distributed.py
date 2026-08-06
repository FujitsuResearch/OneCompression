"""
Tests for GlobalPTQDistributed (Trainer-based global PTQ).

Unit tests run on synthetic tensors (no GPU required).
Integration tests require a CUDA device and download TinyLlama from HF.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii

Usage:
    # Unit tests only (fast, CPU):
    pytest tests/onecomp/post_process/test_global_ptq_distributed.py -v -k "not slow"

    # Full suite (needs CUDA + HF access):
    pytest tests/onecomp/post_process/test_global_ptq_distributed.py -v -s --log-cli-level=INFO
"""

import gc
import os
from types import SimpleNamespace

# Restrict to single GPU to prevent Trainer from using DataParallel,
# which is incompatible with custom nn.Parameter attributes on
# quantization modules (GPTQLinear._opt_scales, etc.).
# Multi-GPU tests are in test_global_ptq_distributed_multigpu.py.
_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _visible.split(",")[0]

import pytest
import torch
import torch.nn as nn

from tests.onecomp.fixtures.quant_config import valid_quant_config

# ---------------------------------------------------------------------------
# Unit tests — compute_ntp_loss
# ---------------------------------------------------------------------------


class TestComputeNtpLoss:
    """Tests for compute_ntp_loss."""

    def test_gradient_flows(self):
        from onecomp.post_process._global_ptq.losses import compute_ntp_loss

        logits = torch.randn(2, 10, 100, requires_grad=True)
        input_ids = torch.randint(0, 100, (2, 10))
        loss = compute_ntp_loss(logits, input_ids)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_shift_is_correct(self):
        """Position t predicts token at t+1."""
        from onecomp.post_process._global_ptq.losses import compute_ntp_loss

        vocab = 10
        logits = torch.zeros(1, 3, vocab)
        input_ids = torch.tensor([[5, 7, 2]])
        logits[0, 0, 7] = 100.0
        logits[0, 1, 2] = 100.0
        loss = compute_ntp_loss(logits, input_ids)
        assert loss.item() < 0.01, "Perfect predictions should give near-zero loss"

    def test_no_nan_with_float16_inputs(self):
        from onecomp.post_process._global_ptq.losses import compute_ntp_loss

        logits = torch.randn(1, 8, 50, dtype=torch.float16)
        input_ids = torch.randint(0, 50, (1, 8))
        loss = compute_ntp_loss(logits, input_ids)
        assert not torch.isnan(loss), "Loss should not be NaN with fp16 input"
        assert not torch.isinf(loss), "Loss should not be Inf with fp16 input"

    def test_positive_loss(self):
        from onecomp.post_process._global_ptq.losses import compute_ntp_loss

        logits = torch.randn(2, 16, 200)
        input_ids = torch.randint(0, 200, (2, 16))
        loss = compute_ntp_loss(logits, input_ids)
        assert loss.item() > 0, "Random predictions should give positive loss"


# ---------------------------------------------------------------------------
# Unit tests — _KDDataset
# ---------------------------------------------------------------------------


class TestKDDataset:
    """Tests for _KDDataset."""

    def test_len(self):
        from onecomp.post_process._global_ptq.trainer import _KDDataset

        ds = _KDDataset(torch.randint(0, 100, (8, 32)))
        assert len(ds) == 8

    def test_getitem_returns_dict(self):
        from onecomp.post_process._global_ptq.trainer import _KDDataset

        ds = _KDDataset(torch.randint(0, 100, (4, 16)))
        item = ds[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert len(item["input_ids"]) == 16

    def test_values_match_input(self):
        from onecomp.post_process._global_ptq.trainer import _KDDataset

        input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
        ds = _KDDataset(input_ids)
        assert ds[0]["input_ids"] == [10, 20, 30]
        assert ds[1]["input_ids"] == [40, 50, 60]


class _PlainPostProcessModel(nn.Module):
    """Schema-valid model with no quantized inference layers."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.config = SimpleNamespace(quantization_config=valid_quant_config())


class TestGlobalPTQDistributedSkipMetadata:
    """Regression tests for early-return audit metadata."""

    def test_not_quantized_skip_records_reason(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        model = _PlainPostProcessModel()
        GlobalPTQDistributed().run(model, object())

        entry = model.config.quantization_config["onecomp_post_processes"][-1]
        assert entry["class"] == "GlobalPTQDistributed"
        assert entry["executed"] is False
        assert entry["reason"] == "not_quantized"

    def test_unsupported_method_skip_records_reason(self, monkeypatch):
        from onecomp.post_process._global_ptq import helpers
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        monkeypatch.setattr(
            helpers,
            "detect_quantization_method",
            lambda _model: ("onebit", []),
        )

        model = _PlainPostProcessModel()
        GlobalPTQDistributed().run(model, object())

        entry = model.config.quantization_config["onecomp_post_processes"][-1]
        assert entry["executed"] is False
        assert entry["reason"] == "unsupported_method_onebit"

    def test_no_params_skip_records_reason(self, monkeypatch):
        """A supported method with zero trainable params records ``no_params``."""
        from onecomp import calibration
        from onecomp.post_process._global_ptq import gptq_adapter, helpers
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        # Reach the trainable-parameter check with a supported method but no
        # differentiable parameters, without loading real calibration data.
        monkeypatch.setattr(
            helpers,
            "detect_quantization_method",
            lambda _model: ("gptq", []),
        )
        monkeypatch.setattr(
            calibration,
            "prepare_calibration_dataset",
            lambda **_k: {"input_ids": torch.zeros((1, 4), dtype=torch.long)},
        )
        monkeypatch.setattr(
            gptq_adapter,
            "setup_gptq_differentiable",
            lambda _modules, _dev: ({}, []),
        )
        # This build of torch is CUDA-enabled but the box may have no driver;
        # the skip path's cache cleanup must not force CUDA initialization.
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        fake_model_config = SimpleNamespace(load_tokenizer=lambda: object())

        model = _PlainPostProcessModel()
        GlobalPTQDistributed().run(model, fake_model_config)

        entry = model.config.quantization_config["onecomp_post_processes"][-1]
        assert entry["class"] == "GlobalPTQDistributed"
        assert entry["executed"] is False
        assert entry["reason"] == "no_params"


# ---------------------------------------------------------------------------
# Unit tests — GlobalPTQDistributed dataclass
# ---------------------------------------------------------------------------


class TestGlobalPTQDistributedDataclass:
    """Verify dataclass defaults and construction."""

    def test_defaults(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        g = GlobalPTQDistributed()
        assert g.w_distill == 1.0
        assert g.w_ntp == 0.0
        assert g.temperature == 1.0
        assert g.epochs == 5
        assert g.gptq_lr == 1e-5
        assert g.dbf_lr == 5e-5
        assert g.deepspeed_config is None
        assert g.use_gradient_checkpointing is True
        assert g.bf16 is True
        assert g.per_device_train_batch_size == 1
        assert g.gradient_accumulation_steps == 1
        assert g.calibration_config is not None
        assert g.calibration_config.num_calibration_samples == 128
        assert g.calibration_config.strategy == "drop_rand"
        assert g.output_dir is None
        assert g.lr_scheduler_type == "cosine"
        assert g.logging_steps == 1
        assert g.report_to == "none"

    def test_name_auto_set(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        g = GlobalPTQDistributed()
        assert g.name == "GlobalPTQDistributed"

    def test_custom_params(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        g = GlobalPTQDistributed(
            w_distill=0.0,
            w_ntp=1.0,
            epochs=3,
            gptq_lr=2e-5,
        )
        assert g.w_distill == 0.0
        assert g.w_ntp == 1.0
        assert g.epochs == 3
        assert g.gptq_lr == 2e-5

    def test_importable_from_top_level(self):
        from onecomp import GlobalPTQDistributed  # noqa: F401


# ---------------------------------------------------------------------------
# Unit tests — validation
# ---------------------------------------------------------------------------


class TestGlobalPTQDistributedValidation:
    """Verify early validation catches bad parameters."""

    def test_both_loss_weights_zero_raises(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        with pytest.raises(ValueError, match="Both w_distill and w_ntp are 0"):
            GlobalPTQDistributed(w_distill=0.0, w_ntp=0.0)

    def test_custom_calibration_config_stored(self):
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        cc = CalibrationConfig(
            calibration_dataset="wikitext2",
            num_calibration_samples=64,
            strategy="drop_head",
        )
        g = GlobalPTQDistributed(calibration_config=cc)
        assert g.calibration_config.calibration_dataset == "wikitext2"
        assert g.calibration_config.num_calibration_samples == 64
        assert g.calibration_config.strategy == "drop_head"

    def test_custom_output_dir_stored(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        g = GlobalPTQDistributed(output_dir="/tmp/my_output")
        assert g.output_dir == "/tmp/my_output"

    def test_custom_lr_scheduler_stored(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        g = GlobalPTQDistributed(lr_scheduler_type="linear")
        assert g.lr_scheduler_type == "linear"

    def test_custom_report_to_stored(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        g = GlobalPTQDistributed(report_to="wandb")
        assert g.report_to == "wandb"

    def test_custom_logging_steps_stored(self):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        g = GlobalPTQDistributed(logging_steps=10)
        assert g.logging_steps == 10


# ---------------------------------------------------------------------------
# Unit tests — GlobalPTQ calibration_dataset parameter
# ---------------------------------------------------------------------------


class TestGlobalPTQCalibrationConfig:
    """Verify GlobalPTQ accepts CalibrationConfig."""

    def test_default_calibration_config(self):
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq import GlobalPTQ

        g = GlobalPTQ()
        assert isinstance(g.calibration_config, CalibrationConfig)

    def test_custom_calibration_config_stored(self):
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq import GlobalPTQ

        cc = CalibrationConfig(calibration_dataset="wikitext2", num_calibration_samples=64)
        g = GlobalPTQ(calibration_config=cc)
        assert g.calibration_config.calibration_dataset == "wikitext2"
        assert g.calibration_config.num_calibration_samples == 64


# ---------------------------------------------------------------------------
# Integration tests — require CUDA + TinyLlama
# ---------------------------------------------------------------------------

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
class TestGlobalPTQDistributedIntegration:
    """Integration tests: GlobalPTQDistributed on quantized TinyLlama."""

    @pytest.mark.slow
    def test_run_completes_without_error(self, quantized_tiny_llama):
        """GlobalPTQDistributed.run() on TinyLlama completes without raising."""
        model, model_config = quantized_tiny_llama
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        gptq = GlobalPTQDistributed(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            eval_interval=1,
        )
        gptq.run(model, model_config)

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
    def test_gptq_layers_preserved(self, quantized_tiny_llama):
        """GPTQ layers should still exist after training."""
        model, _config = quantized_tiny_llama
        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

        gptq_count = sum(1 for _n, m in model.named_modules() if isinstance(m, GPTQLinear))
        assert gptq_count > 0

    @pytest.mark.slow
    def test_use_cache_restored_after_run(self, quantized_tiny_llama):
        """model.config.use_cache is restored after run() completes.

        Before fix, use_cache was set to False for gradient checkpointing
        but never restored.  After fix, original value is saved and restored.
        """
        model, _config = quantized_tiny_llama
        assert (
            getattr(model.config, "use_cache", None) is True
        ), "use_cache should be restored to True after run() completes"


@_requires_cuda
class TestGlobalPTQDistributedGptqRollback:
    """Regression: GPTQ rollback must not be overwritten by write_back.

    With an excessively high learning rate, training degrades eval_loss,
    triggering a rollback to initial state.  Before the fix,
    write_back_gptq_params() was called unconditionally after rollback,
    copying _opt_scales/_opt_zeros (trained, bad values) back into the
    GPTQLinear buffers, effectively nullifying the rollback.
    """

    @pytest.fixture(scope="class")
    def rollback_result(self):
        """Run GlobalPTQDistributed with extreme LR to force rollback."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed
        from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

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
        model, _tok = runner.create_quantized_model(
            pack_weights=False,
            use_gemlite=False,
        )

        initial_scales = {
            name: mod.scales.clone()
            for name, mod in model.named_modules()
            if isinstance(mod, GPTQLinear)
        }

        gptq = GlobalPTQDistributed(
            epochs=2,
            gptq_lr=100.0,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            eval_interval=1,
        )
        gptq.run(model, model_config)

        final_scales = {
            name: mod.scales.clone()
            for name, mod in model.named_modules()
            if isinstance(mod, GPTQLinear)
        }

        yield initial_scales, final_scales, gptq

        del model, runner
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.slow
    def test_rollback_preserves_initial_gptq_params(self, rollback_result):
        """After rollback, GPTQ scales must match the initial snapshot."""
        initial_scales, final_scales, _gptq = rollback_result
        for name in initial_scales:
            assert torch.equal(initial_scales[name].cpu(), final_scales[name].cpu()), (
                f"GPTQ scales for {name} differ after rollback — "
                "write_back_gptq_params may have overwritten the rollback"
            )


@_requires_cuda
class TestGlobalPTQDistributedNTP:
    """Integration tests: GlobalPTQDistributed with NTP loss."""

    @pytest.mark.slow
    def test_ntp_only_mode(self, quantized_tiny_llama):
        """QAT mode (w_distill=0, w_ntp=1) completes without error."""
        model, model_config = quantized_tiny_llama
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        gptq = GlobalPTQDistributed(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            eval_interval=1,
            w_distill=0.0,
            w_ntp=1.0,
        )
        gptq.run(model, model_config)

    @pytest.mark.slow
    def test_ntp_only_skips_teacher(self, quantized_tiny_llama, monkeypatch):
        """QAT mode should never call load_model for the teacher."""
        model, model_config = quantized_tiny_llama
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        load_calls = []
        original_load = model_config.load_model

        def tracking_load(*args, **kwargs):
            load_calls.append(1)
            return original_load(*args, **kwargs)

        monkeypatch.setattr(model_config, "load_model", tracking_load)

        from onecomp import CalibrationConfig

        gptq = GlobalPTQDistributed(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            w_distill=0.0,
            w_ntp=1.0,
        )
        gptq.run(model, model_config)
        assert len(load_calls) == 0, "Teacher model should not be loaded when w_distill=0"

    @pytest.mark.slow
    def test_combined_loss(self, quantized_tiny_llama):
        """Combined KL + NTP loss completes without error."""
        model, model_config = quantized_tiny_llama
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        gptq = GlobalPTQDistributed(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            eval_interval=1,
            w_distill=1.0,
            w_ntp=0.5,
        )
        gptq.run(model, model_config)


@_requires_cuda
class TestGlobalPTQDistributedCustomData:
    """Integration tests: custom calibration data and output settings."""

    @pytest.mark.slow
    def test_custom_calibration_dataset(self, quantized_tiny_llama, tmp_path):
        """Run with user-provided text data instead of C4."""
        model, model_config = quantized_tiny_llama
        import json

        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        custom_texts = [long_text] * 8

        # Save to a temporary jsonl file as per new API requirements (str only for dataset)
        temp_file = tmp_path / "custom_data.jsonl"
        with open(temp_file, "w") as f:
            for text in custom_texts:
                f.write(json.dumps({"text": text}) + "\n")

        from onecomp import CalibrationConfig

        gptq = GlobalPTQDistributed(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                calibration_dataset=str(temp_file),
                num_calibration_samples=4,
                max_length=128,
            ),
        )
        gptq.run(model, model_config)

        devices = {str(p.device) for p in model.parameters()}
        assert devices == {"cpu"}

    @pytest.mark.slow
    def test_custom_output_dir(self, quantized_tiny_llama, tmp_path):
        """Trainer outputs go to the user-specified directory."""
        model, model_config = quantized_tiny_llama
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        out = str(tmp_path / "ptq_output")

        from onecomp import CalibrationConfig

        gptq = GlobalPTQDistributed(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            output_dir=out,
        )
        gptq.run(model, model_config)

        import os

        assert os.path.isdir(out), "output_dir should be created by Trainer"

    @pytest.mark.slow
    def test_linear_lr_scheduler(self, quantized_tiny_llama):
        """Non-default LR scheduler should work."""
        model, model_config = quantized_tiny_llama
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        gptq = GlobalPTQDistributed(
            epochs=1,
            gptq_lr=1e-4,
            calibration_config=CalibrationConfig(
                num_calibration_samples=4,
                max_length=128,
            ),
            lr_scheduler_type="linear",
        )
        gptq.run(model, model_config)


@_requires_cuda
class TestGlobalPTQDistributedViaRunner:
    """Test GlobalPTQDistributed integrated with Runner.run()."""

    @pytest.mark.slow
    def test_runner_with_distributed_ptq(self):
        """Runner with GlobalPTQDistributed runs end-to-end."""
        from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner, setup_logger
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        setup_logger()

        model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
        quantizer = GPTQ(wbits=4, groupsize=128)

        post = GlobalPTQDistributed(
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
        assert gptq_count > 0

        del runner
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Unit tests — ds_zero2.json file location
# ---------------------------------------------------------------------------


class TestDsConfigFileLocation:
    """Verify ds_zero2.json exists at the correct test directory path."""

    def test_ds_zero2_json_exists(self):
        """The DeepSpeed config file exists in the test directory."""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ds_zero2.json",
        )
        assert os.path.isfile(config_path), f"ds_zero2.json not found at {config_path}"

    def test_ds_zero2_json_is_valid(self):
        """The config is valid JSON with ZeRO stage 2."""
        import json

        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ds_zero2.json",
        )
        with open(config_path) as f:
            config = json.load(f)
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 2
