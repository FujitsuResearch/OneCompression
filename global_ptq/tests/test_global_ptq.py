"""
Tests for onecomp_globalptq GlobalPTQ and GlobalPTQDistributed.

Unit tests run on synthetic tensors / tiny models (no GPU required).
Integration tests require a CUDA device and download TinyLlama from HF.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

Usage:
    # Unit tests only (fast, CPU):
    uv run pytest tests/test_global_ptq.py -v -m "not slow"

    # Full suite (needs CUDA + HF access):
    uv run pytest tests/test_global_ptq.py -v -s --log-cli-level=INFO
"""

import gc
import math
import os

# Restrict to single GPU to prevent Trainer from using DataParallel,
# which is incompatible with custom nn.Parameter attributes on
# quantization modules (GPTQLinear._opt_scales, etc.).
# Multi-GPU tests are in test_global_ptq_distributed_multigpu.py.
_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _visible.split(",")[0]

import pytest
import torch
import torch.nn as nn

from onecomp_globalptq import GlobalPTQ, GlobalPTQDistributed
from onecomp import PostQuantizationProcess


# ===========================================================================
# Dataclass / import tests
# ===========================================================================


class TestGlobalPTQDataclass:
    """Verify dataclass construction and defaults."""

    def test_is_post_quantization_process(self):
        ptq = GlobalPTQ()
        assert isinstance(ptq, PostQuantizationProcess)

    def test_default_name(self):
        ptq = GlobalPTQ()
        assert ptq.name == "GlobalPTQ"

    def test_custom_name(self):
        ptq = GlobalPTQ(name="my-global-ptq")
        assert ptq.name == "my-global-ptq"

    def test_defaults(self):
        g = GlobalPTQ()
        assert g.epochs == 5
        assert g.gptq_lr == 1e-5
        assert g.dbf_lr == 5e-5
        assert g.temperature == 1.0
        assert g.grad_clip == 1.0
        assert g.gptq_optimize_intweight is False
        assert g.optimize_binary is False
        assert g.calibration_dataset is None
        assert g.calibration_strategy == "drop_rand"
        assert g.use_sam is False
        assert g.use_ema is False
        assert g.use_lookahead is False
        assert g.use_fisher_lr is False
        assert g.use_entropy_reg is False
        assert g.use_inter_loss is False
        assert g.use_progressive_unfreeze is False
        assert g.use_gradient_checkpointing is True
        assert g.early_stopping_patience == 0
        assert g.use_mixed_precision is False
        assert g.grad_accum_steps == 1

    def test_invalid_calibration_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration_strategy"):
            GlobalPTQ(calibration_strategy="nonexistent")

    def test_lr_keyword_raises_type_error(self):
        """Old ``lr`` field was removed; only gptq_lr / dbf_lr exist."""
        with pytest.raises(TypeError):
            GlobalPTQ(lr=1e-3)

    def test_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            GlobalPTQ(epochs=0)

    def test_num_calibration_samples_zero_raises(self):
        with pytest.raises(ValueError, match="num_calibration_samples must be >= 1"):
            GlobalPTQ(num_calibration_samples=0)

    def test_importable_from_top_level(self):
        from onecomp_globalptq import GlobalPTQ as G  # noqa: F401


class TestGlobalPTQDistributedDataclass:
    """Verify GlobalPTQDistributed construction and defaults."""

    def test_is_post_quantization_process(self):
        ptq = GlobalPTQDistributed()
        assert isinstance(ptq, PostQuantizationProcess)

    def test_default_name(self):
        ptq = GlobalPTQDistributed()
        assert ptq.name == "GlobalPTQDistributed"

    def test_defaults(self):
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
        assert g.lr_scheduler_type == "cosine"
        assert g.report_to == "none"

    def test_both_loss_weights_zero_raises(self):
        with pytest.raises(ValueError, match="Both w_distill and w_ntp are 0"):
            GlobalPTQDistributed(w_distill=0.0, w_ntp=0.0)

    def test_invalid_calibration_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration_strategy"):
            GlobalPTQDistributed(calibration_strategy="nonexistent")

    def test_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            GlobalPTQDistributed(epochs=0)

    def test_num_calibration_samples_zero_raises(self):
        with pytest.raises(ValueError, match="num_calibration_samples must be >= 1"):
            GlobalPTQDistributed(num_calibration_samples=0)

    def test_importable_from_top_level(self):
        from onecomp_globalptq import GlobalPTQDistributed as G  # noqa: F401


# ===========================================================================
# Unit tests — STE helpers
# ===========================================================================


class TestSmoothSteRound:
    def test_forward_equals_round_clamp(self):
        from onecomp_globalptq.global_ptq._core.helpers import smooth_ste_round
        x = torch.tensor([0.3, 1.7, -0.5, 3.2, 7.9])
        result = smooth_ste_round(x, min_val=0, max_val=7)
        expected = x.clamp(0, 7).round()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_gradient_flows(self):
        from onecomp_globalptq.global_ptq._core.helpers import smooth_ste_round
        x = torch.tensor([1.3, 2.7, 0.1], requires_grad=True)
        y = smooth_ste_round(x, min_val=0, max_val=7, k=10.0)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestSmoothSignSte:
    def test_forward_equals_sign(self):
        from onecomp_globalptq.global_ptq._core.helpers import smooth_sign_ste
        x = torch.tensor([-2.0, -0.5, 0.3, 1.5])
        result = smooth_sign_ste(x)
        expected = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_zero_maps_to_plus_one(self):
        from onecomp_globalptq.global_ptq._core.helpers import smooth_sign_ste
        result = smooth_sign_ste(torch.tensor([0.0]))
        assert result.item() == 1.0

    def test_gradient_flows(self):
        from onecomp_globalptq.global_ptq._core.helpers import smooth_sign_ste
        x = torch.tensor([-0.5, 0.3, 1.0], requires_grad=True)
        y = smooth_sign_ste(x, k=10.0)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_output_is_binary(self):
        from onecomp_globalptq.global_ptq._core.helpers import smooth_sign_ste
        x = torch.randn(100)
        result = smooth_sign_ste(x)
        assert set(result.unique().tolist()).issubset({-1.0, 1.0})


# ===========================================================================
# Unit tests — losses
# ===========================================================================


class TestComputeKlLoss:
    def test_identical_logits_give_zero_loss(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_kl_loss
        logits = torch.randn(2, 10, 100)
        loss = compute_kl_loss(logits, logits.clone(), temperature=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_logits_give_positive_loss(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_kl_loss
        loss = compute_kl_loss(torch.randn(2, 10, 100), torch.randn(2, 10, 100))
        assert loss.item() > 0

    def test_gradient_flows_through_student(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_kl_loss
        student = torch.randn(2, 5, 30, requires_grad=True)
        loss = compute_kl_loss(torch.randn(2, 5, 30), student)
        loss.backward()
        assert student.grad is not None

    def test_no_nan_with_float16_inputs(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_kl_loss
        torch.manual_seed(0)
        loss = compute_kl_loss(torch.randn(1, 4, 32000).half(), torch.randn(1, 4, 32000).half())
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestComputeNtpLoss:
    def test_gradient_flows(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_ntp_loss
        logits = torch.randn(2, 10, 100, requires_grad=True)
        loss = compute_ntp_loss(logits, torch.randint(0, 100, (2, 10)))
        loss.backward()
        assert logits.grad is not None

    def test_shift_is_correct(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_ntp_loss
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 7] = 100.0
        logits[0, 1, 2] = 100.0
        loss = compute_ntp_loss(logits, torch.tensor([[5, 7, 2]]))
        assert loss.item() < 0.01

    def test_no_nan_with_float16_inputs(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_ntp_loss
        loss = compute_ntp_loss(torch.randn(1, 8, 50, dtype=torch.float16), torch.randint(0, 50, (1, 8)))
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestComputeEntropyLoss:
    def test_uniform_gives_lower_loss(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_entropy_loss
        uniform = torch.zeros(2, 10, 50)
        peaked = torch.zeros(2, 10, 50)
        peaked[..., 0] = 100.0
        assert compute_entropy_loss(uniform).item() < compute_entropy_loss(peaked).item()


# ===========================================================================
# Unit tests — GPTQ adapter
# ===========================================================================


def _make_synthetic_gptq_linear(in_f=32, out_f=16, wbits=4, groupsize=-1, device="cpu"):
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    num_groups = 1 if groupsize == -1 else in_f // groupsize
    weight = torch.randint(0, (1 << wbits), (out_f, in_f), dtype=torch.int32)
    scale = torch.randn(num_groups, out_f).abs().to(torch.float16) + 0.01
    zero = torch.randint(0, (1 << wbits), (num_groups, out_f)).float()
    return GPTQLinear(
        in_features=in_f, out_features=out_f, wbits=wbits, groupsize=groupsize,
        actorder=False, quantized_weight=weight, scale=scale, zero=zero,
        device=device, pack_weights=False, use_gemlite=False,
    )


class _TinyGPTQModel(nn.Module):
    def __init__(self, hidden=32, wbits=4, device="cpu"):
        super().__init__()
        self.layer1 = _make_synthetic_gptq_linear(hidden, hidden, wbits, device=device)
        self.layer2 = _make_synthetic_gptq_linear(hidden, hidden, wbits, device=device)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class TestGptqAdapter:
    def test_find_modules(self):
        from onecomp_globalptq.global_ptq._core.gptq_adapter import find_gptq_modules
        assert len(find_gptq_modules(_TinyGPTQModel())) == 2
        assert find_gptq_modules(nn.Linear(10, 10)) == []

    def test_setup_creates_opt_parameters(self):
        from onecomp_globalptq.global_ptq._core.gptq_adapter import find_gptq_modules, setup_gptq_differentiable
        model = _TinyGPTQModel()
        modules = find_gptq_modules(model)
        _fwd, scaling, intweight = setup_gptq_differentiable(modules, torch.device("cpu"))
        assert len(scaling) == 4
        assert len(intweight) == 0

    def test_setup_with_intweight(self):
        from onecomp_globalptq.global_ptq._core.gptq_adapter import find_gptq_modules, setup_gptq_differentiable
        model = _TinyGPTQModel()
        modules = find_gptq_modules(model)
        _, _, intweight = setup_gptq_differentiable(modules, torch.device("cpu"), optimize_intweight=True)
        assert len(intweight) == 2

    def test_differentiable_forward_gradient_flows(self):
        from onecomp_globalptq.global_ptq._core.gptq_adapter import find_gptq_modules, setup_gptq_differentiable
        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        _, scaling, _ = setup_gptq_differentiable(modules, torch.device("cpu"))
        out = model(torch.randn(2, 32))
        out.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in scaling)

    def test_restore_original_forward(self):
        from onecomp_globalptq.global_ptq._core.gptq_adapter import find_gptq_modules, setup_gptq_differentiable, restore_gptq_original
        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        x = torch.randn(2, 32)
        out_before = model(x).detach()
        orig_fwd, _, _ = setup_gptq_differentiable(modules, torch.device("cpu"))
        restore_gptq_original(modules, orig_fwd)
        out_after = model(x).detach()
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_write_back_changes_buffers(self):
        from onecomp_globalptq.global_ptq._core.gptq_adapter import find_gptq_modules, setup_gptq_differentiable, write_back_gptq_params
        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        _, scaling, _ = setup_gptq_differentiable(modules, torch.device("cpu"))
        scales_before = modules[0][1].scales.clone()
        with torch.no_grad():
            for p in scaling:
                p.add_(0.1)
        write_back_gptq_params(modules)
        assert not torch.allclose(scales_before, modules[0][1].scales)

    def test_state_save_load_roundtrip(self):
        from onecomp_globalptq.global_ptq._core.gptq_adapter import find_gptq_modules, save_gptq_state, load_gptq_state
        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        state = save_gptq_state(modules)
        modules[0][1].scales.fill_(0.0)
        load_gptq_state(modules, state)
        assert not torch.all(modules[0][1].scales == 0.0)


# ===========================================================================
# Unit tests — DBF adapter
# ===========================================================================


def _make_synthetic_dbf_linear(in_dim=16, out_dim=16, mid_dim=8, device="cpu"):
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear
    Da = torch.randn(out_dim).abs() + 0.01
    A = torch.sign(torch.randn(out_dim, mid_dim)); A[A == 0] = 1
    mid = torch.randn(mid_dim).abs() + 0.01
    B = torch.sign(torch.randn(mid_dim, in_dim)); B[B == 0] = 1
    Db = torch.randn(in_dim).abs() + 0.01
    return DoubleBinaryLinear(dbf_Da=Da, dbf_A=A, dbf_mid=mid, dbf_B=B, dbf_Db=Db, device=device, use_gemlite=False)


class _TinyDBFModel(nn.Module):
    def __init__(self, in_dim=16, out_dim=16, mid_dim=8, device="cpu"):
        super().__init__()
        self.layer1 = _make_synthetic_dbf_linear(in_dim, out_dim, mid_dim, device=device)
        self.layer2 = _make_synthetic_dbf_linear(in_dim, out_dim, mid_dim, device=device)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class TestDbfAdapter:
    def test_find_modules(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules
        assert len(find_dbf_modules(_TinyDBFModel())) == 2
        assert find_dbf_modules(nn.Linear(10, 10)) == []

    def test_scaling_params_require_grad(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules, setup_dbf_differentiable
        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        _, scaling, binary = setup_dbf_differentiable(modules, optimize_binary=False)
        assert len(scaling) == 6
        assert len(binary) == 0
        assert all(p.requires_grad for p in scaling)

    def test_binary_params_with_optimize_binary(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules, setup_dbf_differentiable
        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        _, scaling, binary = setup_dbf_differentiable(modules, optimize_binary=True)
        assert len(scaling) == 6
        assert len(binary) == 4
        assert all(bp.requires_grad for bp in binary)

    def test_scaling_params_are_float32(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules, setup_dbf_differentiable
        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        _, scaling, _ = setup_dbf_differentiable(modules)
        assert all(p.dtype == torch.float32 for p in scaling)

    def test_scaling_gradient_flows(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules, setup_dbf_differentiable
        model = _TinyDBFModel(in_dim=16, out_dim=16, mid_dim=8)
        modules = find_dbf_modules(model)
        _, scaling, _ = setup_dbf_differentiable(modules)
        model(torch.randn(2, 16)).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in scaling)

    def test_binary_weight_receives_gradient(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules, setup_dbf_differentiable, restore_dbf_original
        model = _TinyDBFModel(in_dim=16, out_dim=16, mid_dim=8)
        modules = find_dbf_modules(model)
        orig, _, binary = setup_dbf_differentiable(modules, optimize_binary=True)
        model(torch.randn(2, 16)).sum().backward()
        restore_dbf_original(modules, orig)
        assert any(bp.grad is not None and bp.grad.abs().sum() > 0 for bp in binary)

    def test_write_back_scaling_converts_to_float16(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules, setup_dbf_differentiable, write_back_dbf_scaling
        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        setup_dbf_differentiable(modules)
        write_back_dbf_scaling(modules)
        for _, mod in modules:
            for attr in ("scaling0", "scaling2", "scaling4"):
                assert getattr(mod, attr).dtype == torch.float16

    def test_state_save_load_roundtrip(self):
        from onecomp_globalptq.global_ptq._core.dbf_adapter import find_dbf_modules, save_dbf_state, load_dbf_state
        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        state = save_dbf_state(modules)
        original_s0 = modules[0][1].scaling0.data.clone()
        modules[0][1].scaling0.data.fill_(0.0)
        load_dbf_state(modules, state)
        assert torch.allclose(modules[0][1].scaling0.data, original_s0)


# ===========================================================================
# Unit tests — helpers (detect_quantization_method)
# ===========================================================================


class TestDetectQuantizationMethod:
    def test_detects_gptq(self):
        from onecomp_globalptq.global_ptq._core.helpers import detect_quantization_method
        method, modules = detect_quantization_method(_TinyGPTQModel())
        assert method == "gptq"
        assert len(modules) == 2

    def test_detects_dbf(self):
        from onecomp_globalptq.global_ptq._core.helpers import detect_quantization_method
        method, modules = detect_quantization_method(_TinyDBFModel())
        assert method == "dbf"
        assert len(modules) == 2

    def test_plain_model_returns_none(self):
        from onecomp_globalptq.global_ptq._core.helpers import detect_quantization_method
        method, modules = detect_quantization_method(nn.Linear(10, 10))
        assert method is None
        assert modules == []


# ===========================================================================
# Unit tests — optimiser wrappers (SAM, EMA, Lookahead)
# ===========================================================================


class TestSAMOptimizer:
    def test_two_step_changes_params(self):
        from onecomp_globalptq.global_ptq._core.core import SAMOptimizer
        w = nn.Parameter(torch.randn(4, 4))
        sam = SAMOptimizer(torch.optim.SGD([w], lr=0.1), rho=0.05)
        w_before = w.data.clone()
        (w ** 2).sum().backward()
        sam.first_step()
        (w ** 2).sum().backward()
        sam.second_step()
        assert not torch.allclose(w.data, w_before)

    def test_undo_first_step_restores(self):
        from onecomp_globalptq.global_ptq._core.core import SAMOptimizer
        w = nn.Parameter(torch.randn(4, 4))
        sam = SAMOptimizer(torch.optim.SGD([w], lr=0.1), rho=0.05)
        w_before = w.data.clone()
        (w ** 2).sum().backward()
        sam.first_step()
        sam.undo_first_step()
        assert torch.allclose(w.data, w_before, atol=1e-6)


class TestEMATracker:
    def test_update_moves_shadow(self):
        from onecomp_globalptq.global_ptq._core.core import EMATracker
        w = nn.Parameter(torch.ones(4))
        ema = EMATracker([w], decay=0.9)
        with torch.no_grad():
            w.fill_(2.0)
        ema.update([w])
        expected = 0.9 * 1.0 + 0.1 * 2.0
        assert ema.shadow[id(w)].allclose(torch.full((4,), expected))

    def test_apply_and_restore(self):
        from onecomp_globalptq.global_ptq._core.core import EMATracker
        w = nn.Parameter(torch.ones(4) * 5.0)
        ema = EMATracker([w], decay=0.9)
        ema.apply([w])
        ema.restore([w])
        assert w.data.allclose(torch.full((4,), 5.0))


class TestLookaheadOptimizer:
    def test_slow_weight_update_after_k_steps(self):
        from onecomp_globalptq.global_ptq._core.core import LookaheadOptimizer
        w = nn.Parameter(torch.zeros(4))
        la = LookaheadOptimizer(torch.optim.SGD([w], lr=0.1), k=3, alpha=0.5)
        slow_before = la._slow_weights[id(w)].clone()
        for _ in range(3):
            (w - 1.0).pow(2).sum().backward()
            la.step()
            la.zero_grad()
        assert not torch.allclose(slow_before, la._slow_weights[id(w)])


# ===========================================================================
# Unit tests — Fisher LR
# ===========================================================================


class TestFisherLR:
    def test_build_param_to_layer_map(self):
        from onecomp_globalptq.global_ptq._core.core import build_param_to_layer_map
        layer0, layer1 = nn.Linear(4, 4), nn.Linear(4, 4)
        layer0.weight.requires_grad = True
        layer1.weight.requires_grad = True
        ptl = build_param_to_layer_map([("model.layers.0.proj", layer0), ("model.layers.1.proj", layer1)])
        assert ptl[id(layer0.weight)] == 0
        assert ptl[id(layer1.weight)] == 1

    def test_fisher_lr_multipliers_range(self):
        from onecomp_globalptq.global_ptq._core.core import build_fisher_lr_multipliers
        mult = build_fisher_lr_multipliers({0: 10.0, 1: 0.1, 2: 1.0}, min_mult=0.1, max_mult=10.0)
        assert all(0.1 <= v <= 10.0 for v in mult.values())

    def test_uniform_fisher_gives_unit(self):
        from onecomp_globalptq.global_ptq._core.core import build_fisher_lr_multipliers
        mult = build_fisher_lr_multipliers({0: 5.0, 1: 5.0})
        assert all(v == pytest.approx(1.0) for v in mult.values())


# ===========================================================================
# Unit tests — progressive layer unfreezing
# ===========================================================================


class TestSetLayerGrad:
    def test_enables_and_disables_grad(self):
        from onecomp_globalptq.global_ptq._core.core import set_layer_grad
        layer0, layer1 = nn.Linear(4, 4), nn.Linear(4, 4)
        modules = [("model.layers.0.proj", layer0), ("model.layers.1.proj", layer1)]
        set_layer_grad(modules, {0}, False)
        assert not layer0.weight.requires_grad
        assert layer1.weight.requires_grad
        set_layer_grad(modules, {0}, True)
        assert layer0.weight.requires_grad


# ===========================================================================
# Unit tests — LR schedule
# ===========================================================================


class TestCosineWarmupLrLambda:
    def test_starts_near_zero(self):
        from onecomp_globalptq.global_ptq._core.core import cosine_warmup_lr_lambda
        assert cosine_warmup_lr_lambda(0, 100, 10) < 0.01

    def test_reaches_one_at_warmup_end(self):
        from onecomp_globalptq.global_ptq._core.core import cosine_warmup_lr_lambda
        assert cosine_warmup_lr_lambda(10, 100, 10) == pytest.approx(1.0, abs=0.01)

    def test_decays_to_min_at_end(self):
        from onecomp_globalptq.global_ptq._core.core import cosine_warmup_lr_lambda
        assert cosine_warmup_lr_lambda(100, 100, 10, min_lr_ratio=0.01) == pytest.approx(0.01, abs=0.01)


class TestSchedulerGradAccum:
    """Verify LR scheduler correctly accounts for gradient accumulation."""

    def test_effective_steps_complete_decay(self):
        from onecomp_globalptq.global_ptq._core.core import cosine_warmup_lr_lambda

        epochs, num_batches, accum = 2, 8, 4
        total_batches = epochs * num_batches
        effective = max(1, total_batches // accum)
        warmup = int(effective * 0.1)

        lr = cosine_warmup_lr_lambda(
            effective - 1, effective, warmup, min_lr_ratio=0.01,
        )
        assert lr < 0.2, (
            f"LR at last effective step should be near min_lr_ratio, got {lr:.4f}"
        )


class TestGradientCheckpointing:
    def test_enable_on_model_with_method(self):
        from onecomp_globalptq.global_ptq._core.helpers import (
            enable_gradient_checkpointing,
            disable_gradient_checkpointing,
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
        from onecomp_globalptq.global_ptq._core.helpers import enable_gradient_checkpointing
        assert enable_gradient_checkpointing(nn.Linear(4, 4)) is False


class TestKlLossFloat16Accuracy:
    """KL loss from float16 logits should closely match float32 reference."""

    def test_float16_matches_float32_reference(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_kl_loss

        torch.manual_seed(42)
        teacher = torch.randn(1, 8, 32000)
        student = teacher + 0.5 * torch.randn_like(teacher)

        ref = compute_kl_loss(teacher, student, temperature=1.0).item()
        fp16 = compute_kl_loss(teacher.half(), student.half(), temperature=1.0).float().item()

        rel_err = abs(ref - fp16) / max(abs(ref), 1e-10)
        assert rel_err < 0.01, (
            f"float16 KL ({fp16:.6f}) deviates {rel_err*100:.1f}% from "
            f"float32 reference ({ref:.6f})"
        )


# ===========================================================================
# Unit tests — _KDDataset
# ===========================================================================


class TestKDDataset:
    def test_len(self):
        from onecomp_globalptq.global_ptq._core.trainer import _KDDataset
        data = {"input_ids": torch.randint(0, 100, (8, 32))}
        assert len(_KDDataset(data)) == 8

    def test_getitem(self):
        from onecomp_globalptq.global_ptq._core.trainer import _KDDataset
        data = {"input_ids": torch.tensor([[10, 20, 30], [40, 50, 60]])}
        ds = _KDDataset(data)
        assert ds[0]["input_ids"] == [10, 20, 30]
        assert ds[1]["input_ids"] == [40, 50, 60]


# ===========================================================================
# Unit tests — intermediate hooks
# ===========================================================================


class TestIntermediateHooks:
    def _make_small_transformer(self):
        layer1 = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
        layer2 = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
        model = nn.Module()
        model.layers = nn.ModuleList([layer1, layer2])
        return model

    def test_hooks_capture_outputs(self):
        from onecomp_globalptq.global_ptq._core.losses import setup_intermediate_hooks, clear_hooks, remove_hooks
        model = self._make_small_transformer()
        hooks = setup_intermediate_hooks(model)
        x = torch.randn(1, 4, 16)
        for layer in model.layers:
            x = layer(x)
        assert len(hooks["outputs"]) == 2
        clear_hooks(hooks)
        assert len(hooks["outputs"]) == 0
        remove_hooks(hooks)
        assert len(hooks["handles"]) == 0

    def test_identical_outputs_zero_loss(self):
        from onecomp_globalptq.global_ptq._core.losses import compute_intermediate_loss
        h = torch.randn(1, 4, 16)
        loss = compute_intermediate_loss({"outputs": [h]}, {"outputs": [h.clone()]})
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# Integration tests — require CUDA + TinyLlama
# ===========================================================================

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available",
)


@pytest.fixture(scope="module")
def quantized_tiny_llama():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from onecomp import GPTQ, ModelConfig, Runner, CalibrationConfig, setup_logger
    setup_logger()
    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    runner = Runner(
        model_config=model_config,
        quantizer=GPTQ(wbits=4, groupsize=128),
        calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=8),
    )
    runner.run()
    model, _ = runner.create_quantized_model(pack_weights=False, use_gemlite=False)
    yield model, model_config
    del model, runner
    gc.collect()
    torch.cuda.empty_cache()


@_requires_cuda
class TestGlobalPTQIntegration:
    @pytest.mark.slow
    def test_run_with_progressive_unfreeze(self, quantized_tiny_llama):
        model, model_config = quantized_tiny_llama
        from onecomp_globalptq.global_ptq._core.core import run_kl_distillation

        results = run_kl_distillation(
            model, model_config,
            epochs=2, gptq_lr=1e-4,
            num_calibration_samples=4, max_length=128,
            use_progressive_unfreeze=True,
        )
        assert results["global_executed"] is True
        assert results["features"]["progressive_unfreeze"] is True

    @pytest.mark.slow
    def test_run_with_fisher_lr(self, quantized_tiny_llama):
        model, model_config = quantized_tiny_llama
        from onecomp_globalptq.global_ptq._core.core import run_kl_distillation

        results = run_kl_distillation(
            model, model_config,
            epochs=1, gptq_lr=1e-4,
            num_calibration_samples=4, max_length=128,
            use_fisher_lr=True, fisher_n_samples=2,
        )
        assert results["global_executed"] is True
        assert results["features"]["fisher_lr"] is True

    @pytest.mark.slow
    def test_run_with_discrete_params(self, quantized_tiny_llama):
        model, model_config = quantized_tiny_llama
        from onecomp_globalptq.global_ptq._core.core import run_kl_distillation

        results = run_kl_distillation(
            model, model_config,
            epochs=1, gptq_lr=1e-4,
            num_calibration_samples=4, max_length=128,
            gptq_optimize_intweight=True, gptq_intweight_lr=1e-4,
        )
        assert results["global_executed"] is True

    @pytest.mark.slow
    def test_run_completes_and_improves_kl(self, quantized_tiny_llama):
        model, model_config = quantized_tiny_llama
        from onecomp_globalptq.global_ptq._core.core import run_kl_distillation

        results = run_kl_distillation(
            model, model_config,
            epochs=2, gptq_lr=1e-4,
            num_calibration_samples=4, max_length=128,
        )
        assert results["global_executed"] is True
        assert results["final_kl"] <= results["initial_kl"], (
            f"KL should not increase: {results['initial_kl']:.6f} "
            f"-> {results['final_kl']:.6f}"
        )

    @pytest.mark.slow
    def test_model_on_cpu_after_run(self, quantized_tiny_llama):
        model, _ = quantized_tiny_llama
        assert {str(p.device) for p in model.parameters()} == {"cpu"}

    @pytest.mark.slow
    def test_model_in_eval_mode(self, quantized_tiny_llama):
        model, _ = quantized_tiny_llama
        assert not model.training

    @pytest.mark.slow
    def test_use_cache_restored(self, quantized_tiny_llama):
        model, _ = quantized_tiny_llama
        assert getattr(model.config, "use_cache", None) is True


@_requires_cuda
class TestGlobalPTQAdvancedIntegration:
    @pytest.mark.slow
    def test_sam_and_ema(self, quantized_tiny_llama):
        model, mc = quantized_tiny_llama
        GlobalPTQ(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                  use_sam=True, use_ema=True).run(model, mc)

    @pytest.mark.slow
    def test_lookahead(self, quantized_tiny_llama):
        model, mc = quantized_tiny_llama
        GlobalPTQ(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                  use_lookahead=True, lookahead_k=2).run(model, mc)

    @pytest.mark.slow
    def test_entropy_reg(self, quantized_tiny_llama):
        model, mc = quantized_tiny_llama
        GlobalPTQ(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                  use_entropy_reg=True).run(model, mc)

    @pytest.mark.slow
    def test_inter_loss(self, quantized_tiny_llama):
        model, mc = quantized_tiny_llama
        GlobalPTQ(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                  use_inter_loss=True, lambda_inter=1.0).run(model, mc)

    @pytest.mark.slow
    def test_intweight_optimization(self, quantized_tiny_llama):
        """GlobalPTQ-exclusive: integer weight optimization via Smooth STE."""
        model, mc = quantized_tiny_llama
        GlobalPTQ(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                  gptq_optimize_intweight=True).run(model, mc)


@_requires_cuda
class TestGlobalPTQDistributedIntegration:
    @pytest.mark.slow
    def test_run_completes(self, quantized_tiny_llama):
        model, mc = quantized_tiny_llama
        GlobalPTQDistributed(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128).run(model, mc)

    @pytest.mark.slow
    def test_model_on_cpu_after_run(self, quantized_tiny_llama):
        model, _ = quantized_tiny_llama
        assert {str(p.device) for p in model.parameters()} == {"cpu"}

    @pytest.mark.slow
    def test_ntp_only_mode(self, quantized_tiny_llama):
        model, mc = quantized_tiny_llama
        GlobalPTQDistributed(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                             w_distill=0.0, w_ntp=1.0).run(model, mc)

    @pytest.mark.slow
    def test_combined_kl_ntp(self, quantized_tiny_llama):
        model, mc = quantized_tiny_llama
        GlobalPTQDistributed(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                             w_distill=1.0, w_ntp=0.5).run(model, mc)

    @pytest.mark.slow
    def test_use_cache_restored(self, quantized_tiny_llama):
        model, _ = quantized_tiny_llama
        assert getattr(model.config, "use_cache", None) is True

    @pytest.mark.slow
    def test_intweight_optimization_distributed(self, quantized_tiny_llama):
        """GlobalPTQ-exclusive: integer weight optimization in distributed mode."""
        model, mc = quantized_tiny_llama
        GlobalPTQDistributed(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128,
                             gptq_optimize_intweight=True).run(model, mc)


@_requires_cuda
class TestGlobalPTQViaRunner:
    @pytest.mark.slow
    def test_runner_with_global_ptq(self):
        from onecomp import GPTQ, ModelConfig, Runner, CalibrationConfig, setup_logger
        setup_logger()
        runner = Runner(
            model_config=ModelConfig(model_id=MODEL_ID, device="cuda:0"),
            quantizer=GPTQ(wbits=4, groupsize=128),
            calibration_config=CalibrationConfig(max_length=512, num_calibration_samples=8),
            post_processes=[GlobalPTQ(epochs=1, gptq_lr=1e-4, num_calibration_samples=4, max_length=128)],
        )
        runner.run()
        assert runner.quantized_model is not None
        del runner
        gc.collect()
        torch.cuda.empty_cache()


@_requires_cuda
class TestGlobalPTQDbfIntegration:
    @pytest.fixture(scope="class")
    def dbf_quantized_tiny_llama(self):
        from onecomp import ModelConfig, Runner, CalibrationConfig, setup_logger
        from onecomp.quantizer.dbf import DBF
        setup_logger()
        runner = Runner(
            model_config=ModelConfig(model_id=MODEL_ID, device="cuda:0"),
            quantizer=DBF(),
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=4),
        )
        runner.run()
        model, _ = runner.create_quantized_model(use_gemlite=False)
        yield model, runner.model_config
        del model, runner
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.slow
    def test_dbf_run_completes(self, dbf_quantized_tiny_llama):
        model, mc = dbf_quantized_tiny_llama
        GlobalPTQ(epochs=1, dbf_lr=5e-4, num_calibration_samples=4, max_length=128).run(model, mc)

    @pytest.mark.slow
    def test_dbf_with_binary_optimization(self, dbf_quantized_tiny_llama):
        """GlobalPTQ-exclusive: binary matrix optimization via Sign STE."""
        model, mc = dbf_quantized_tiny_llama
        GlobalPTQ(epochs=1, dbf_lr=5e-4, optimize_binary=True,
                  num_calibration_samples=4, max_length=128).run(model, mc)
