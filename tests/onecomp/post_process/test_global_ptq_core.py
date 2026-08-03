"""
Tests for GlobalPTQ core helpers (LR schedule, gradient checkpointing, dataclass validation).

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

Usage:
    pytest tests/onecomp/post_process/test_global_ptq_core.py -v
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.onecomp.fixtures.quant_config import valid_quant_config

# ---------------------------------------------------------------------------
# cosine_warmup_lr_lambda
# ---------------------------------------------------------------------------


class TestCosineWarmupLrLambda:
    """Tests for cosine_warmup_lr_lambda."""

    def test_starts_at_near_zero(self):
        from onecomp.post_process._global_ptq.core import cosine_warmup_lr_lambda

        lr = cosine_warmup_lr_lambda(0, total_steps=100, warmup_steps=10)
        assert lr < 0.01

    def test_reaches_one_at_end_of_warmup(self):
        from onecomp.post_process._global_ptq.core import cosine_warmup_lr_lambda

        lr = cosine_warmup_lr_lambda(10, total_steps=100, warmup_steps=10)
        assert lr == pytest.approx(1.0, abs=0.01)

    def test_decays_to_min_at_end(self):
        from onecomp.post_process._global_ptq.core import cosine_warmup_lr_lambda

        lr = cosine_warmup_lr_lambda(100, total_steps=100, warmup_steps=10, min_lr_ratio=0.01)
        assert lr == pytest.approx(0.01, abs=0.01)


# ---------------------------------------------------------------------------
# LR schedule + gradient accumulation
# ---------------------------------------------------------------------------


class TestSchedulerGradAccum:
    """Verify LR scheduler correctly accounts for gradient accumulation.

    With ``grad_accum_steps > 1`` the scheduler must use
    ``effective_total_steps = total_batches // grad_accum_steps`` so
    that cosine decay completes by the last *optimiser* step.
    """

    def test_effective_steps_complete_decay(self):
        """LR must reach near min_lr_ratio at the last effective step."""
        from onecomp.post_process._global_ptq.core import cosine_warmup_lr_lambda

        epochs, num_batches, accum = 2, 8, 4
        total_batches = epochs * num_batches
        effective = max(1, total_batches // accum)
        warmup = int(effective * 0.1)

        lr = cosine_warmup_lr_lambda(
            effective - 1,
            effective,
            warmup,
            min_lr_ratio=0.01,
        )
        assert lr < 0.2, f"LR at last effective step should be near min_lr_ratio, got {lr:.4f}"

    def test_unadjusted_total_would_not_decay(self):
        """Without the fix, LR barely decays with grad_accum_steps > 1."""
        from onecomp.post_process._global_ptq.core import cosine_warmup_lr_lambda

        epochs, num_batches, accum = 2, 8, 4
        total_batches = epochs * num_batches
        effective = max(1, total_batches // accum)
        buggy_warmup = int(total_batches * 0.1)

        lr_buggy = cosine_warmup_lr_lambda(
            effective - 1,
            total_batches,
            buggy_warmup,
            min_lr_ratio=0.01,
        )
        assert (
            lr_buggy > 0.8
        ), f"With unadjusted total_steps, LR should still be high, got {lr_buggy:.4f}"

    def test_scheduler_driven_by_effective_steps(self):
        """Simulate the actual scheduler calling pattern with grad_accum."""
        from onecomp.post_process._global_ptq.core import cosine_warmup_lr_lambda

        epochs, num_batches, accum = 1, 12, 4
        total_batches = epochs * num_batches
        effective = max(1, total_batches // accum)
        warmup = int(effective * 0.1)

        base_lr = 1e-5
        param = nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=base_lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda s: cosine_warmup_lr_lambda(
                s,
                effective,
                warmup,
                0.01,
            ),
        )

        lrs = []
        for batch_idx in range(total_batches):
            is_boundary = (batch_idx + 1) % accum == 0 or batch_idx == total_batches - 1
            if is_boundary:
                sched.step()
                lrs.append(sched.get_last_lr()[0])

        assert len(lrs) == effective
        assert (
            lrs[-1] < base_lr * 0.15
        ), f"Final LR {lrs[-1]:.2e} should be near min_lr_ratio * base_lr"


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------


class TestGradientCheckpointing:
    """Tests for enable/disable gradient checkpointing helpers."""

    def test_enable_on_model_with_method(self):
        from onecomp.post_process._global_ptq.helpers import (
            disable_gradient_checkpointing,
            enable_gradient_checkpointing,
        )

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._ckpt = False

            def gradient_checkpointing_enable(self, **kwargs):
                self._ckpt = True

            def gradient_checkpointing_disable(self):
                self._ckpt = False

        model = _FakeModel()
        assert enable_gradient_checkpointing(model) is True
        assert model._ckpt is True
        disable_gradient_checkpointing(model)
        assert model._ckpt is False

    def test_returns_false_on_unsupported_model(self):
        from onecomp.post_process._global_ptq.helpers import enable_gradient_checkpointing

        model = nn.Linear(4, 4)
        assert enable_gradient_checkpointing(model) is False


# ---------------------------------------------------------------------------
# Early stopping / dataclass validation
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Verify early stopping counter logic in results."""

    def test_patience_zero_means_disabled(self):
        from onecomp.post_process.global_ptq import GlobalPTQ

        g = GlobalPTQ(early_stopping_patience=0)
        assert g.early_stopping_patience == 0

    def test_dataclass_defaults(self):
        from onecomp.post_process.global_ptq import GlobalPTQ

        g = GlobalPTQ()
        assert g.use_gradient_checkpointing is True
        assert g.early_stopping_patience == 0
        assert g.use_mixed_precision is False
        assert g.grad_accum_steps == 1

    def test_epochs_zero_raises(self):
        from onecomp.post_process.global_ptq import GlobalPTQ

        with pytest.raises(ValueError, match="epochs must be >= 1"):
            GlobalPTQ(epochs=0)

    def test_num_calibration_samples_zero_raises(self):
        from onecomp import CalibrationConfig
        from onecomp.post_process.global_ptq import GlobalPTQ

        with pytest.raises(ValueError, match="num_calibration_samples"):
            GlobalPTQ(calibration_config=CalibrationConfig(num_calibration_samples=0))


# ---------------------------------------------------------------------------
# Redundant lr parameter removed
# ---------------------------------------------------------------------------


class TestLrParameterRemoved:
    """The redundant ``lr`` field was removed from GlobalPTQ.

    Only ``gptq_lr`` and ``dbf_lr`` remain as the correct per-method
    learning rate parameters.
    """

    def test_lr_keyword_raises_type_error(self):
        """Passing lr= to GlobalPTQ must raise TypeError (removed field)."""
        from onecomp.post_process.global_ptq import GlobalPTQ

        with pytest.raises(TypeError):
            GlobalPTQ(lr=1e-3)


class TestRemovedDiscreteFields:
    """Discrete parameter and advanced optimization fields have been removed.
    Passing any of them must raise TypeError (unexpected keyword argument).
    """

    @pytest.mark.parametrize(
        "field,value",
        [
            ("gptq_optimize_intweight", True),
            ("gptq_intweight_lr", 1e-4),
            ("optimize_binary", True),
            ("ste_k", 100.0),
        ],
    )
    def test_discrete_fields_rejected_global_ptq(self, field, value):
        from onecomp.post_process.global_ptq import GlobalPTQ

        with pytest.raises(TypeError, match=field):
            GlobalPTQ(**{field: value})

    @pytest.mark.parametrize(
        "field,value",
        [
            ("gptq_optimize_intweight", True),
            ("gptq_intweight_lr", 1e-4),
            ("optimize_binary", True),
            ("ste_k", 100.0),
        ],
    )
    def test_discrete_fields_rejected_distributed(self, field, value):
        from onecomp.post_process.global_ptq_distributed import GlobalPTQDistributed

        with pytest.raises(TypeError, match=field):
            GlobalPTQDistributed(**{field: value})

    @pytest.mark.parametrize(
        "field,value",
        [
            ("use_sam", True),
            ("sam_rho", 0.02),
            ("use_ema", True),
            ("ema_decay", 0.99),
            ("use_lookahead", True),
            ("lookahead_k", 5),
            ("lookahead_alpha", 0.5),
            ("use_fisher_lr", True),
            ("fisher_n_samples", 4),
            ("use_entropy_reg", True),
            ("entropy_lambda", 0.1),
            ("use_inter_loss", True),
            ("lambda_inter", 10.0),
            ("use_progressive_unfreeze", True),
        ],
    )
    def test_advanced_fields_rejected_global_ptq(self, field, value):
        from onecomp.post_process.global_ptq import GlobalPTQ

        with pytest.raises(TypeError, match=field):
            GlobalPTQ(**{field: value})

    @pytest.mark.parametrize(
        "symbol",
        [
            "smooth_ste_round",
            "smooth_sign_ste",
            "SAMOptimizer",
            "EMATracker",
            "LookaheadOptimizer",
            "compute_entropy_loss",
            "setup_intermediate_hooks",
            "compute_intermediate_loss",
            "build_param_to_layer_map",
            "set_layer_grad",
        ],
    )
    def test_removed_symbols_not_importable(self, symbol):
        import importlib

        mod_map = {
            "smooth_ste_round": "onecomp.post_process._global_ptq.helpers",
            "smooth_sign_ste": "onecomp.post_process._global_ptq.helpers",
            "SAMOptimizer": "onecomp.post_process._global_ptq.core",
            "EMATracker": "onecomp.post_process._global_ptq.core",
            "LookaheadOptimizer": "onecomp.post_process._global_ptq.core",
            "compute_entropy_loss": "onecomp.post_process._global_ptq.losses",
            "setup_intermediate_hooks": "onecomp.post_process._global_ptq.losses",
            "compute_intermediate_loss": "onecomp.post_process._global_ptq.losses",
            "build_param_to_layer_map": "onecomp.post_process._global_ptq.core",
            "set_layer_grad": "onecomp.post_process._global_ptq.core",
        }
        mod = importlib.import_module(mod_map[symbol])
        assert not hasattr(mod, symbol), f"{symbol} should not be defined (removed)"


# ---------------------------------------------------------------------------
# GlobalPTQ — skip-path audit metadata
# ---------------------------------------------------------------------------


class _PlainPostProcessModel(nn.Module):
    """Schema-valid model with no quantized inference layers."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.config = SimpleNamespace(quantization_config=valid_quant_config())


class TestGlobalPTQSkipMetadata:
    """Regression tests for early-return audit metadata (cf. GlobalPTQDistributed)."""

    def test_not_quantized_skip_records_reason(self):
        from onecomp.post_process.global_ptq import GlobalPTQ

        model = _PlainPostProcessModel()
        GlobalPTQ().run(model, object())

        entry = model.config.quantization_config["onecomp_post_processes"][-1]
        assert entry["class"] == "GlobalPTQ"
        assert entry["executed"] is False
        assert entry["reason"] == "not_quantized"

    def test_unsupported_method_skip_records_reason(self, monkeypatch):
        from onecomp.post_process._global_ptq import core
        from onecomp.post_process.global_ptq import GlobalPTQ

        monkeypatch.setattr(
            core,
            "detect_quantization_method",
            lambda _model: ("onebit", []),
        )

        model = _PlainPostProcessModel()
        GlobalPTQ().run(model, object())

        entry = model.config.quantization_config["onecomp_post_processes"][-1]
        assert entry["executed"] is False
        assert entry["reason"] == "unsupported_method_onebit"

    def test_no_params_skip_records_reason(self, monkeypatch):
        """A supported method with zero trainable params records ``no_params``."""
        from onecomp.post_process._global_ptq import core
        from onecomp.post_process.global_ptq import GlobalPTQ

        # Reach the trainable-parameter check with a supported method but no
        # differentiable parameters, without loading real calibration data or
        # a teacher model.
        monkeypatch.setattr(
            core,
            "detect_quantization_method",
            lambda _model: ("gptq", []),
        )
        monkeypatch.setattr(core, "_prepare_dataloader", lambda *a, **k: [])
        monkeypatch.setattr(
            core,
            "setup_gptq_differentiable",
            lambda _modules, _dev: ({}, []),
        )
        # This build of torch is CUDA-enabled but the box may have no driver;
        # the skip path's cache cleanup must not force CUDA initialization.
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        fake_model_config = SimpleNamespace(load_model=lambda **_k: nn.Linear(2, 2))

        model = _PlainPostProcessModel()
        GlobalPTQ().run(model, fake_model_config)

        entry = model.config.quantization_config["onecomp_post_processes"][-1]
        assert entry["class"] == "GlobalPTQ"
        assert entry["executed"] is False
        assert entry["reason"] == "no_params"
