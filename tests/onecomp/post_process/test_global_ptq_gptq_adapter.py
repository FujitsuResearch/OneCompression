"""
Tests for GPTQ adapter and quantization method detection used by GlobalPTQ.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

Usage:
    pytest tests/onecomp/post_process/test_global_ptq_gptq_adapter.py -v
"""

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Synthetic model helpers
# ---------------------------------------------------------------------------


def _make_synthetic_gptq_linear(in_f=32, out_f=16, wbits=4, groupsize=-1, device="cpu"):
    """Build a GPTQLinear with random quantisation parameters."""
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear

    num_groups = 1 if groupsize == -1 else in_f // groupsize
    weight = torch.randint(0, (1 << wbits), (out_f, in_f), dtype=torch.int32)
    scale = torch.randn(num_groups, out_f).abs().to(torch.float16) + 0.01
    zero = torch.randint(0, (1 << wbits), (num_groups, out_f)).float()

    return GPTQLinear(
        in_features=in_f,
        out_features=out_f,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        quantized_weight=weight,
        scale=scale,
        zero=zero,
        device=device,
        pack_weights=False,
        use_gemlite=False,
    )


class _TinyGPTQModel(nn.Module):
    """Minimal model wrapping two GPTQLinear layers for adapter tests."""

    def __init__(self, hidden=32, wbits=4, device="cpu"):
        super().__init__()
        self.layer1 = _make_synthetic_gptq_linear(hidden, hidden, wbits, device=device)
        self.layer2 = _make_synthetic_gptq_linear(hidden, hidden, wbits, device=device)

    def forward(self, x):
        return self.layer2(self.layer1(x))


def _make_synthetic_dbf_linear(in_dim=16, out_dim=16, mid_dim=8, device="cpu"):
    """Build a DoubleBinaryLinear with random parameters."""
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear

    Da = torch.randn(out_dim).abs() + 0.01
    A = torch.sign(torch.randn(out_dim, mid_dim))
    A[A == 0] = 1
    mid = torch.randn(mid_dim).abs() + 0.01
    B = torch.sign(torch.randn(mid_dim, in_dim))
    B[B == 0] = 1
    Db = torch.randn(in_dim).abs() + 0.01

    return DoubleBinaryLinear(
        dbf_Da=Da,
        dbf_A=A,
        dbf_mid=mid,
        dbf_B=B,
        dbf_Db=Db,
        device=device,
        use_gemlite=False,
    )


class _TinyDBFModel(nn.Module):
    """Minimal model wrapping two DoubleBinaryLinear layers for adapter tests."""

    def __init__(self, in_dim=16, out_dim=16, mid_dim=8, device="cpu"):
        super().__init__()
        self.layer1 = _make_synthetic_dbf_linear(in_dim, out_dim, mid_dim, device=device)
        self.layer2 = _make_synthetic_dbf_linear(in_dim, out_dim, mid_dim, device=device)

    def forward(self, x):
        return self.layer2(self.layer1(x))


# ---------------------------------------------------------------------------
# GPTQ adapter tests
# ---------------------------------------------------------------------------


class TestGptqAdapterFindModules:
    """Tests for find_gptq_modules."""

    def test_finds_all_gptq_layers(self):
        from onecomp.post_process._global_ptq.gptq_adapter import find_gptq_modules

        model = _TinyGPTQModel()
        modules = find_gptq_modules(model)
        assert len(modules) == 2

    def test_empty_model_returns_empty(self):
        from onecomp.post_process._global_ptq.gptq_adapter import find_gptq_modules

        model = nn.Linear(10, 10)
        assert find_gptq_modules(model) == []


class TestGptqDifferentiableSetup:
    """Tests for setup/teardown of differentiable GPTQ forward."""

    def test_setup_creates_opt_parameters(self):
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            setup_gptq_differentiable,
        )

        model = _TinyGPTQModel()
        modules = find_gptq_modules(model)
        _fwd, scaling = setup_gptq_differentiable(
            modules,
            torch.device("cpu"),
        )
        assert len(scaling) == 4  # 2 layers * (scales + zeros)
        for _name, mod in modules:
            assert hasattr(mod, "_opt_scales")
            assert hasattr(mod, "_opt_zeros")
            assert isinstance(mod._opt_scales, nn.Parameter)

    def test_differentiable_forward_output_shape(self):
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            setup_gptq_differentiable,
        )

        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        setup_gptq_differentiable(modules, torch.device("cpu"))

        x = torch.randn(2, 32)
        out = model(x)
        assert out.shape == (2, 32)

    def test_differentiable_forward_gradient_flows(self):
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            setup_gptq_differentiable,
        )

        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        _fwd, scaling = setup_gptq_differentiable(modules, torch.device("cpu"))

        x = torch.randn(2, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in scaling)
        assert has_grad, "Gradient should flow to scaling parameters"

    def test_restore_original_forward(self):
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            restore_gptq_original,
            setup_gptq_differentiable,
        )

        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)

        x = torch.randn(2, 32)
        out_before = model(x).detach()

        orig_fwd, _ = setup_gptq_differentiable(modules, torch.device("cpu"))
        restore_gptq_original(modules, orig_fwd)

        out_after = model(x).detach()
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_restore_removes_opt_params_with_cleanup(self):
        """_opt_scales/_opt_zeros must be removed when cleanup=True."""
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            restore_gptq_original,
            setup_gptq_differentiable,
        )

        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        orig_fwd, _ = setup_gptq_differentiable(modules, torch.device("cpu"))
        restore_gptq_original(modules, orig_fwd, cleanup=True)

        for _name, mod in modules:
            assert not hasattr(
                mod, "_opt_scales"
            ), "_opt_scales should be removed after restore with cleanup=True"
            assert not hasattr(
                mod, "_opt_zeros"
            ), "_opt_zeros should be removed after restore with cleanup=True"

    def test_restore_keeps_opt_params_without_cleanup(self):
        """_opt_scales/_opt_zeros must be kept when cleanup=False (default)."""
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            restore_gptq_original,
            setup_gptq_differentiable,
        )

        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        orig_fwd, _ = setup_gptq_differentiable(modules, torch.device("cpu"))
        restore_gptq_original(modules, orig_fwd)

        for _name, mod in modules:
            assert hasattr(
                mod, "_opt_scales"
            ), "_opt_scales should be kept after restore without cleanup"
            assert hasattr(
                mod, "_opt_zeros"
            ), "_opt_zeros should be kept after restore without cleanup"


class TestGptqWriteBack:
    """Tests for write_back_gptq_params."""

    def test_write_back_changes_buffers(self):
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            setup_gptq_differentiable,
            write_back_gptq_params,
        )

        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        _, scaling = setup_gptq_differentiable(modules, torch.device("cpu"))

        scales_before = modules[0][1].scales.clone()

        with torch.no_grad():
            for p in scaling:
                p.add_(0.1)

        write_back_gptq_params(modules)
        scales_after = modules[0][1].scales
        assert not torch.allclose(scales_before, scales_after)


class TestGptqStateSaveLoad:
    """Tests for save/load GPTQ state."""

    def test_roundtrip(self):
        from onecomp.post_process._global_ptq.gptq_adapter import (
            find_gptq_modules,
            load_gptq_state,
            save_gptq_state,
        )

        model = _TinyGPTQModel(hidden=32)
        modules = find_gptq_modules(model)
        state = save_gptq_state(modules)

        modules[0][1].scales.fill_(0.0)
        load_gptq_state(modules, state)

        assert not torch.all(modules[0][1].scales == 0.0)


# ---------------------------------------------------------------------------
# detect_quantization_method tests
# ---------------------------------------------------------------------------


class TestDetectQuantizationMethod:
    """Tests for detect_quantization_method."""

    def test_detects_gptq(self):
        from onecomp.post_process._global_ptq.helpers import detect_quantization_method

        model = _TinyGPTQModel(hidden=32)
        method, modules = detect_quantization_method(model)
        assert method == "gptq"
        assert len(modules) == 2

    def test_plain_model_returns_none(self):
        from onecomp.post_process._global_ptq.helpers import detect_quantization_method

        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        method, modules = detect_quantization_method(model)
        assert method is None
        assert modules == []

    def test_detects_dbf(self):
        from onecomp.post_process._global_ptq.helpers import detect_quantization_method

        model = _TinyDBFModel(in_dim=16, out_dim=16, mid_dim=8)
        method, modules = detect_quantization_method(model)
        assert method == "dbf"
        assert len(modules) == 2

    def test_mixed_gptq_dbf_returns_gptq_and_warns(self, caplog):
        """Mixed GPTQ+DBF model should return gptq and emit a warning."""
        import logging

        from onecomp.post_process._global_ptq.helpers import detect_quantization_method

        class _MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gptq = _make_synthetic_gptq_linear(32, 32, 4)
                self.dbf = _make_synthetic_dbf_linear(16, 16, 8)

        model = _MixedModel()
        with caplog.at_level(logging.WARNING, logger="onecomp.post_process._global_ptq.helpers"):
            method, modules = detect_quantization_method(model)

        assert method == "gptq"
        assert len(modules) == 1
        assert any(
            "Mixed GPTQ + DBF" in msg for msg in caplog.messages
        ), "Expected 'Mixed GPTQ + DBF' warning in log output"
