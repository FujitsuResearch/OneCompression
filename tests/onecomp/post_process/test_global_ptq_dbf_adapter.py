"""
Tests for DBF adapter used by GlobalPTQ.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

Usage:
    pytest tests/onecomp/post_process/test_global_ptq_dbf_adapter.py -v
"""

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Synthetic DBF model helpers
# ---------------------------------------------------------------------------


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
# DBF adapter tests
# ---------------------------------------------------------------------------


class TestDbfAdapterFindModules:
    """Tests for find_dbf_modules."""

    def test_finds_all_dbf_layers(self):
        from onecomp.post_process._global_ptq.dbf_adapter import find_dbf_modules

        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        assert len(modules) == 2

    def test_empty_model_returns_empty(self):
        from onecomp.post_process._global_ptq.dbf_adapter import find_dbf_modules

        model = nn.Linear(10, 10)
        assert find_dbf_modules(model) == []


class TestDbfDifferentiableSetup:
    """Tests for setup_dbf_differentiable."""

    def test_scaling_params_require_grad(self):
        from onecomp.post_process._global_ptq.dbf_adapter import (
            find_dbf_modules,
            setup_dbf_differentiable,
        )

        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        _fwd, scaling = setup_dbf_differentiable(modules)

        assert len(scaling) == 6  # 2 layers * 3 scaling (scaling0, scaling2, scaling4)
        for p in scaling:
            assert p.requires_grad

    def test_scaling_params_are_float32(self):
        """Scaling params must be promoted to float32 for stable Adam updates.

        DoubleBinaryLinear stores scalings in float16.  If the adapter
        does not upcast to float32, Adam's epsilon underflows in fp16,
        causing NaN parameter values.
        """
        from onecomp.post_process._global_ptq.dbf_adapter import (
            find_dbf_modules,
            setup_dbf_differentiable,
        )

        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        _fwd, scaling = setup_dbf_differentiable(modules)

        for p in scaling:
            assert (
                p.dtype == torch.float32
            ), f"Scaling param should be float32 for stable optimisation, got {p.dtype}"

    def test_scaling_gradient_flows_through_forward(self):
        """Scaling params must receive non-zero gradients via the forward pass."""
        from onecomp.post_process._global_ptq.dbf_adapter import (
            find_dbf_modules,
            setup_dbf_differentiable,
        )

        model = _TinyDBFModel(in_dim=16, out_dim=16, mid_dim=8)
        modules = find_dbf_modules(model)
        _fwd, scaling = setup_dbf_differentiable(modules)

        x = torch.randn(2, 16)
        out = model(x)
        out.sum().backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in scaling)
        assert has_grad, "Scaling params should receive non-zero gradients"

    def test_forward_still_works(self):
        from onecomp.post_process._global_ptq.dbf_adapter import (
            find_dbf_modules,
            setup_dbf_differentiable,
        )

        model = _TinyDBFModel(in_dim=16, out_dim=16, mid_dim=8)
        modules = find_dbf_modules(model)
        _fwd, _s = setup_dbf_differentiable(modules)

        x = torch.randn(2, 16)
        out = model(x)
        assert out.shape == (2, 16)


class TestDbfWriteBack:
    """Tests for write_back_dbf_scaling."""

    def test_converts_to_float16(self):
        """After write-back, scaling params should be float16 for inference."""
        from onecomp.post_process._global_ptq.dbf_adapter import (
            find_dbf_modules,
            setup_dbf_differentiable,
            write_back_dbf_scaling,
        )

        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        _fwd, _s = setup_dbf_differentiable(modules)

        write_back_dbf_scaling(modules)

        for _name, mod in modules:
            for attr in ("scaling0", "scaling2", "scaling4"):
                assert (
                    getattr(mod, attr).dtype == torch.float16
                ), f"{attr} should be float16 after write-back"

    def test_values_preserved_through_roundtrip(self):
        """float16 -> float32 (setup) -> float16 (write-back) should be lossless."""
        from onecomp.post_process._global_ptq.dbf_adapter import (
            find_dbf_modules,
            setup_dbf_differentiable,
            write_back_dbf_scaling,
        )

        model = _TinyDBFModel()
        modules = find_dbf_modules(model)

        originals = {
            name: {a: getattr(mod, a).data.clone() for a in ("scaling0", "scaling2", "scaling4")}
            for name, mod in modules
        }

        _fwd, _s = setup_dbf_differentiable(modules)
        write_back_dbf_scaling(modules)

        for name, mod in modules:
            for attr in ("scaling0", "scaling2", "scaling4"):
                assert torch.equal(
                    getattr(mod, attr).data, originals[name][attr]
                ), f"{attr} changed after float16 -> float32 -> float16 roundtrip"


class TestDbfStateSaveLoad:
    """Tests for save/load DBF state."""

    def test_roundtrip(self):
        from onecomp.post_process._global_ptq.dbf_adapter import (
            find_dbf_modules,
            load_dbf_state,
            save_dbf_state,
        )

        model = _TinyDBFModel()
        modules = find_dbf_modules(model)
        state = save_dbf_state(modules)

        original_s0 = modules[0][1].scaling0.data.clone()
        modules[0][1].scaling0.data.fill_(0.0)
        load_dbf_state(modules, state)

        assert torch.allclose(modules[0][1].scaling0.data, original_s0)
