"""Tests for the MDBF adapter used by GlobalPTQ."""

import torch
import torch.nn as nn


def _make_model():
    from onecomp.quantizer.mdbf.initialize import MDBFParams
    from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear

    def make_layer():
        params = MDBFParams(
            A_sign=torch.ones(8, 4),
            B_sign=torch.ones(4, 8),
            A_amp=torch.rand(8, 2) + 0.1,
            B_amp=torch.rand(8, 2) + 0.1,
            Q_U_amp=torch.rand(4, 2) + 0.1,
            Q_V_amp=torch.rand(4, 2) + 0.1,
        )
        return MultipathMDBFLinear([params], use_gemlite=False)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = make_layer()

        def forward(self, inputs):
            return self.layer(inputs)

    return TinyModel()


def test_mdbf_setup_exposes_trainable_amplitudes_and_gradients():
    from onecomp.post_process._global_ptq.mdbf_adapter import (
        find_mdbf_modules,
        setup_mdbf_differentiable,
    )

    model = _make_model()
    modules = find_mdbf_modules(model)
    _original, amplitudes, binary = setup_mdbf_differentiable(modules)

    assert len(modules) == 1
    assert len(amplitudes) == 4
    assert binary == []
    model(torch.randn(2, 8)).sum().backward()
    assert all(parameter.grad is not None for parameter in amplitudes)


def test_mdbf_binary_setup_and_write_back():
    from onecomp.post_process._global_ptq.mdbf_adapter import (
        find_mdbf_modules,
        setup_mdbf_differentiable,
        write_back_mdbf_amp,
        write_back_mdbf_binary,
    )

    model = _make_model()
    modules = find_mdbf_modules(model)
    _original, amplitudes, binary = setup_mdbf_differentiable(modules, optimize_binary=True)

    assert len(amplitudes) == 4
    assert len(binary) == 2
    with torch.no_grad():
        amplitudes[0].add_(1.0)
        binary[0].mul_(-1.0)
    write_back_mdbf_amp(modules)
    write_back_mdbf_binary(modules)
    path = modules[0][1].paths[0]
    assert path.A_amp.dtype == torch.float16


def test_mdbf_state_round_trip():
    from onecomp.post_process._global_ptq.mdbf_adapter import (
        find_mdbf_modules,
        load_mdbf_state,
        save_mdbf_state,
    )

    model = _make_model()
    modules = find_mdbf_modules(model)
    state = save_mdbf_state(modules)
    original = modules[0][1].paths[0].A_amp.detach().clone()
    modules[0][1].paths[0].A_amp.zero_()
    load_mdbf_state(modules, state)
    assert torch.equal(modules[0][1].paths[0].A_amp, original)