"""Tests for the MDBF quantizer implementation.

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import logging
import os
import sys

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from test_module import BaseQuantizeSpec

from onecomp.quantizer.gemlite import is_gemlite_available
from onecomp.quantizer.mdbf import admm, mdbf_layer
from onecomp.quantizer.mdbf._mdbf import MAX_SEED, MDBF, MDBFResult
from onecomp.quantizer.mdbf.initialize import MDBFParams, initialize_MDBF, lowrank_osvd
from onecomp.quantizer.mdbf.mdbf_layer import MDBFLinear, MultipathMDBFLinear
from onecomp.quantizer.mdbf.utils import (
    DEFAULT_L,
    DEFAULT_P,
    bpw_from_rank,
    rank_from_bpw,
    reconstruct_weight,
)


class TestMDBF(BaseQuantizeSpec):
    """Test cases for MDBF quantization."""

    __test__ = True
    quantizer_cls = MDBF
    result_cls = MDBFResult
    default_parameter_for_test = {
        "target_bits": 1.0,
        "l": 1,
        "P": 1,
        "svd_mode": "svd",
        "use_admm": False,
        "use_gradient_refine": False,
    }

    boundary_parameters = [
        # target_bits: float > 0
        {"target_bits": 1e-10},  # lower boundary (near zero, positive)
        {"target_bits": 100.0},  # large value
        # l: int >= 1
        {"l": 1},  # lower boundary
        {"l": 5},  # large value
        # P: in {1, 2}
        {"P": 1},  # minimum valid value
        {"P": 2},  # maximum valid value
        # svd_mode: in {"svd", "svd_llm"}
        {"svd_mode": "svd"},  # default value (explicit check)
        {"svd_mode": "svd_llm"},  # alternate valid value
        # act_init: in {"none", "osvd", "svd_llm"}
        {"act_init": "none"},  # first valid value
        {"act_init": "osvd"},  # second valid value
        {"act_init": "svd_llm"},  # third valid value
        # scale_bits: int >= 0
        {"scale_bits": 0},  # lower boundary (binary-only accounting)
        {"scale_bits": 16},  # default value (FP16)
        {"scale_bits": 32},  # large value
        # admm_reg: float >= 0 (always validated)
        {"admm_reg": 0.0},  # lower boundary (0.0 is valid)
        {"admm_reg": 100.0},  # large value
        # admm_seed: int in [0, MAX_SEED] or None
        {"admm_seed": None},  # default value (global RNG)
        {"admm_seed": 0},  # lower boundary
        {"admm_seed": MAX_SEED},  # upper boundary (largest seed torch accepts)
        # use_admm: bool
        {
            "use_admm": True,
            "admm_outer_iters": 1,
            "admm_inner_iters": 1,
        },  # lower boundary combo when enabled
        {"use_admm": False},  # disabled individually
        # admm_outer_iters: int >= 1 (validated when use_admm=True)
        {"use_admm": True, "admm_outer_iters": 1},  # lower boundary
        {"use_admm": True, "admm_outer_iters": 100},  # large value
        # admm_inner_iters: int >= 1 (validated when use_admm=True)
        {"use_admm": True, "admm_inner_iters": 1},  # lower boundary
        {"use_admm": True, "admm_inner_iters": 10},  # large value
        # admm_* not validated when use_admm=False
        {"use_admm": False, "admm_outer_iters": 0},  # admm_outer_iters=0 allowed when admm off
        {"use_admm": False, "admm_inner_iters": 0},  # admm_inner_iters=0 allowed when admm off
        # use_gradient_refine: bool
        {
            "use_gradient_refine": True,
            "gradient_iters": 1,
            "gradient_lr": 1e-3,
        },  # lower boundary combo when enabled
        {"use_gradient_refine": False},  # disabled individually
        # gradient_iters: int >= 1 (validated when use_gradient_refine=True)
        {"use_gradient_refine": True, "gradient_iters": 1},  # lower boundary
        {"use_gradient_refine": True, "gradient_iters": 100},  # large value
        # gradient_lr: float > 0, strict (validated when use_gradient_refine=True)
        {
            "use_gradient_refine": True,
            "gradient_lr": 1e-10,
        },  # lower boundary (near zero, positive)
        {"use_gradient_refine": True, "gradient_lr": 100.0},  # large value
        # gradient_* not validated when use_gradient_refine=False
        {
            "use_gradient_refine": False,
            "gradient_iters": 0,
        },  # gradient_iters=0 allowed when refine off
        {
            "use_gradient_refine": False,
            "gradient_lr": 0.0,
        },  # gradient_lr=0.0 allowed when refine off
        # activation_aware: bool
        {
            "activation_aware": True,
            "P": 1,
        },  # P=1 required to exercise the activation_aware code path
        {"activation_aware": True, "P": 2},  # P!=1 fallback (warning only, no error)
        {"activation_aware": False},  # disabled individually
        # mlp_target_bits: float > 0 or None
        {"mlp_target_bits": 1e-10},  # lower boundary (near zero, positive)
        {"mlp_target_bits": 100.0},  # large value
        # module_target_bits: dict[str, float>0] or None
        {"module_target_bits": {"model.layers.0.self_attn.q_proj": 1.5}},  # single-layer override
        {
            "module_target_bits": {  # multi-layer override
                "model.layers.0.self_attn.q_proj": 1.0,
                "model.layers.0.mlp.gate_proj": 2.0,
            }
        },
        # combo: all bools False
        {"use_admm": False, "use_gradient_refine": False, "activation_aware": False},
        # combo: all numerics at lower bounds
        # use_admm not specified -> class default True applies,
        # so admm_outer_iters/admm_inner_iters are validated
        {
            "target_bits": 1.0,
            "l": 1,
            "P": 1,
            "scale_bits": 0,
            "admm_reg": 0.0,
            "admm_outer_iters": 1,
            "admm_inner_iters": 1,
        },
        # all class defaults spelled out (the defaults themselves are pinned by
        # test_class_defaults_are_multi_envelope)
        {
            "target_bits": 1.0,
            "l": 2,
            "P": 1,
            "svd_mode": "svd",
            "use_admm": True,
            "admm_outer_iters": 260,
            "admm_inner_iters": 3,
            "admm_reg": 0.03,
            "admm_seed": None,
            "use_gradient_refine": False,
            "gradient_iters": 1000,
            "gradient_lr": 0.01,
            "activation_aware": False,
            "act_init": "osvd",
            "scale_bits": 16,
        },
        # all minimum (use_admm=False, use_gradient_refine=False skips condition-guarded validation)
        {
            "target_bits": 1e-10,
            "l": 1,
            "P": 1,
            "svd_mode": "svd",
            "use_admm": False,
            "admm_outer_iters": 0,
            "admm_inner_iters": 0,
            "admm_reg": 0.0,
            "use_gradient_refine": False,
            "gradient_iters": 0,
            "gradient_lr": 0.0,
            "activation_aware": False,
            "act_init": "none",
            "scale_bits": 0,
        },
        # all maximum
        {
            "target_bits": 100.0,
            "l": 5,
            "P": 2,
            "svd_mode": "svd_llm",
            "use_admm": True,
            "admm_outer_iters": 100,
            "admm_inner_iters": 10,
            "admm_reg": 100.0,
            "admm_seed": 12345,
            "use_gradient_refine": True,
            "gradient_iters": 100,
            "gradient_lr": 100.0,
            "activation_aware": False,
            "act_init": "svd_llm",
            "mlp_target_bits": 100.0,
            "module_target_bits": {"model.layers.0.self_attn.q_proj": 100.0},
            "scale_bits": 32,
        },
    ]

    abnormal_parameters = [
        # target_bits: float > 0, strict
        {"target_bits": 0.0},  # boundary (target_bits > 0, strict)
        {"target_bits": -1.0},  # negative value
        # l: int >= 1
        {"l": 0},  # below lower boundary (l >= 1)
        {"l": -1},  # negative value
        # P: in {1, 2}
        {"P": 0},  # not in {1, 2} (below range)
        {"P": 3},  # not in {1, 2} (above range)
        # scale_bits: int >= 0
        {"scale_bits": -1},  # below lower boundary (scale_bits >= 0)
        # admm_reg: float >= 0
        {"admm_reg": -0.01},  # below lower boundary (admm_reg >= 0)
        # admm_seed: int in [0, MAX_SEED] or None
        {"admm_seed": -1},  # below lower boundary (admm_seed >= 0)
        {"admm_seed": MAX_SEED + 1},  # above upper boundary (torch overflows)
        {"admm_seed": 1.5},  # wrong type (not an int)
        {"admm_seed": True},  # wrong type (bool; torch.manual_seed rejects it)
        # admm_outer_iters: int >= 1 (validated when use_admm=True)
        {
            "use_admm": True,
            "admm_outer_iters": 0,
        },  # below lower boundary (admm_outer_iters >= 1 when use_admm=True)
        {"use_admm": True, "admm_outer_iters": -1},  # negative value
        # admm_inner_iters: int >= 1 (validated when use_admm=True)
        {
            "use_admm": True,
            "admm_inner_iters": 0,
        },  # below lower boundary (admm_inner_iters >= 1 when use_admm=True)
        {"use_admm": True, "admm_inner_iters": -1},  # negative value
        # gradient_iters: int >= 1 (validated when use_gradient_refine=True)
        {"use_gradient_refine": True, "gradient_iters": 0},  # below lower boundary
        # gradient_lr: float > 0, strict (validated when use_gradient_refine=True)
        {"use_gradient_refine": True, "gradient_lr": 0.0},  # boundary (gradient_lr > 0, strict)
        # svd_mode
        {"svd_mode": "invalid"},  # not in {"svd", "svd_llm"}
        # act_init
        {"act_init": "invalid"},  # not in {"none", "osvd", "svd_llm"}
        # mlp_target_bits: float > 0, strict
        {"mlp_target_bits": 0.0},  # boundary (mlp_target_bits > 0, strict)
        {"mlp_target_bits": -1.0},  # negative value
        # module_target_bits
        {"module_target_bits": "not a dict"},  # wrong type
        {"module_target_bits": {"model.layers.0.self_attn.q_proj": 0.0}},  # value not > 0
        {"module_target_bits": {"model.layers.0.self_attn.q_proj": -1.0}},  # negative value
    ]

    logger = logging.getLogger(__name__)

    def check_quantize_layer(self, result: MDBFResult, layer: torch.nn.Module):
        assert isinstance(result, self.result_cls)

        for attr in [
            "mdbf_A_sign",
            "mdbf_B_sign",
            "mdbf_A_amp",
            "mdbf_B_amp",
            "mdbf_Q_U_amp",
            "mdbf_Q_V_amp",
        ]:
            assert hasattr(result, attr)

        assert isinstance(result.is_mdbf_quantized, bool)

        n, m = layer.weight.shape
        r = result.r
        configured_l = result.l

        for p in range(result.P):
            A_sign = result.mdbf_A_sign[p]
            B_sign = result.mdbf_B_sign[p]
            A_amp = result.mdbf_A_amp[p]
            B_amp = result.mdbf_B_amp[p]
            Q_U_amp = result.mdbf_Q_U_amp[p]
            Q_V_amp = result.mdbf_Q_V_amp[p]

            for tensor in [A_sign, B_sign, A_amp, B_amp, Q_U_amp, Q_V_amp]:
                assert isinstance(tensor, torch.Tensor)
                assert tensor.device == torch.device("cpu")
                assert tensor.dtype == layer.weight.dtype

            effective_l = A_amp.shape[1]
            assert A_sign.shape == (n, r)
            assert B_sign.shape == (r, m)
            assert 1 <= effective_l <= configured_l
            assert A_amp.shape == (n, effective_l)
            assert B_amp.shape == (m, effective_l)
            assert Q_U_amp.shape == (r, effective_l)
            assert Q_V_amp.shape == (r, effective_l)

            # sign matrices should contain only -1 and +1
            A_vals = set(A_sign.flatten().tolist())
            B_vals = set(B_sign.flatten().tolist())
            assert all(v in (-1.0, 1.0) for v in A_vals)
            assert all(v in (-1.0, 1.0) for v in B_vals)

        W_recon = result.compute_dequantized_weight()
        assert W_recon.shape == layer.weight.shape
        assert W_recon.dtype == torch.float16
        assert W_recon.device == torch.device("cpu")

        # optional: validate per-path reconstruction sums to W_recon
        W_sum = None
        for p in range(result.P):
            Wp = reconstruct_weight(
                result.mdbf_A_sign[p].float(),
                result.mdbf_B_sign[p].float(),
                result.mdbf_A_amp[p].float(),
                result.mdbf_B_amp[p].float(),
                result.mdbf_Q_U_amp[p].float(),
                result.mdbf_Q_V_amp[p].float(),
            )
            W_sum = Wp if W_sum is None else W_sum + Wp

        assert torch.allclose(W_sum.to(torch.float16).cpu(), W_recon, rtol=1, atol=1)

    def check_equal_results(self, r1, r2):
        assert torch.equal(r1.compute_dequantized_weight(), r2.compute_dequantized_weight())
        assert r1.is_mdbf_quantized == r2.is_mdbf_quantized

        for attr in [
            "mdbf_A_sign",
            "mdbf_B_sign",
            "mdbf_A_amp",
            "mdbf_B_amp",
            "mdbf_Q_U_amp",
            "mdbf_Q_V_amp",
        ]:
            for p in range(r1.P):
                assert torch.equal(getattr(r1, attr)[p], getattr(r2, attr)[p])

    def check_quantize_error(self, error, max_error):
        assert error < 0.4
        assert max_error < 1.71

    def check_forward_error(
        self,
        error_original_vs_dequantized,
        error_dequantized_vs_applied,
        max_error_dequantized_vs_applied,
    ):
        self.logger.info(
            "[MDBF forward error] "
            f"original_vs_mdbf(rel={error_original_vs_dequantized:.8f}), "
            f"mdbf_vs_mdbfl(max={max_error_dequantized_vs_applied:.8f}), "
            f"mdbf_vs_mdbfl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert (
            max_error_dequantized_vs_applied < 1e-2
        ), f"MDBF dequantized vs applied max error too large: {max_error_dequantized_vs_applied}"

    def apply_quantized_weights(self, module, result, device):
        dtype = module.weight.data.dtype
        module.weight.data = result.compute_dequantized_weight().to(device).to(dtype)

        # attach MDBF_params so layer-replacement utilities can use them
        params_list = [
            MDBFParams(
                A_sign=result.mdbf_A_sign[p],
                B_sign=result.mdbf_B_sign[p],
                A_amp=result.mdbf_A_amp[p],
                B_amp=result.mdbf_B_amp[p],
                Q_U_amp=result.mdbf_Q_U_amp[p],
                Q_V_amp=result.mdbf_Q_V_amp[p],
            )
            for p in range(result.P)
        ]
        module.MDBF_params = params_list
        module.is_quantized = True


def _random_sign(shape, device, dtype=torch.float16):
    return (torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1).to(dtype)


def _make_mdbf_params(n, m, r, l, device, dtype=torch.float16):
    return MDBFParams(
        A_sign=_random_sign((n, r), device, dtype),
        B_sign=_random_sign((r, m), device, dtype),
        A_amp=torch.randn(n, l, device=device, dtype=dtype),
        B_amp=torch.randn(m, l, device=device, dtype=dtype),
        Q_U_amp=torch.randn(r, l, device=device, dtype=dtype),
        Q_V_amp=torch.randn(r, l, device=device, dtype=dtype),
    )


def _assert_gemlite_output_matches_dense(y_dense, y_gemlite):
    diff = (y_dense.float() - y_gemlite.float()).abs()
    rel = (torch.norm(y_dense.float() - y_gemlite.float()) / torch.norm(y_dense.float())).item()
    assert rel < 1e-3, f"GemLite relative output error too large: {rel}"
    assert diff.max().item() < 4.0, f"GemLite max abs output error too large: {diff.max().item()}"


def _quantize_linear_for_inference_test(in_features, out_features, device, p=1):
    layer = torch.nn.Linear(
        in_features, out_features, bias=False, device=device, dtype=torch.float32
    )
    inp = torch.randn(3, 4, in_features, device=device, dtype=torch.float32)
    quantizer = MDBF(
        target_bits=1.0,
        l=1,
        P=p,
        svd_mode="svd",
        use_admm=False,
        use_gradient_refine=False,
    )
    hessian, nsamples = quantizer.calculate_hessian(layer, inp)
    result = quantizer.quantize_layer(layer, inp, hessian=hessian, nsamples=nsamples)
    return layer, inp, quantizer, result


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gemlite_available(),
    reason="GemLite unavailable or CUDA not available",
)
def test_mdbflinear_gemlite_matches_dense_forward():
    torch.manual_seed(0)
    device = torch.device("cuda")
    params = _make_mdbf_params(n=96, m=256, r=128, l=1, device=device)

    dense_layer = MDBFLinear(params, device=device, use_gemlite=False)
    gemlite_layer = MDBFLinear(params, device=device, use_gemlite=True)

    assert gemlite_layer.use_gemlite
    assert set(gemlite_layer._gemlite_layers) == {"A", "B"}

    x = torch.randn(7, 256, device=device, dtype=torch.float16)
    with torch.no_grad():
        y_dense = dense_layer(x)
        y_gemlite = gemlite_layer(x)

    assert y_dense.shape == y_gemlite.shape
    _assert_gemlite_output_matches_dense(y_dense, y_gemlite)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gemlite_available(),
    reason="GemLite unavailable or CUDA not available",
)
def test_multipath_mdbflinear_gemlite_matches_dense_forward():
    torch.manual_seed(1)
    device = torch.device("cuda")
    params_list = [
        _make_mdbf_params(n=80, m=256, r=128, l=2, device=device),
        _make_mdbf_params(n=80, m=256, r=128, l=2, device=device),
    ]
    bias = torch.randn(80, device=device, dtype=torch.float16)

    dense_layer = MultipathMDBFLinear(
        params_list=params_list,
        bias=bias,
        device=device,
        use_gemlite=False,
    )
    gemlite_layer = MultipathMDBFLinear(
        params_list=params_list,
        bias=bias,
        device=device,
        use_gemlite=True,
    )

    assert all(path.use_gemlite for path in gemlite_layer.paths)

    x = torch.randn(5, 256, device=device, dtype=torch.float16)
    with torch.no_grad():
        y_dense = dense_layer(x)
        y_gemlite = gemlite_layer(x)

    assert y_dense.shape == y_gemlite.shape
    _assert_gemlite_output_matches_dense(y_dense, y_gemlite)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gemlite_available(),
    reason="GemLite unavailable or CUDA not available",
)
@pytest.mark.parametrize("p", [1, 2])
def test_mdbf_create_inference_layer_gemlite_matches_dequantized_forward(p):
    torch.manual_seed(7 + p)
    device = torch.device("cuda")
    layer, inp, quantizer, result = _quantize_linear_for_inference_test(
        in_features=256,
        out_features=128,
        device=device,
        p=p,
    )

    dequantized_layer = torch.nn.Linear(256, 128, bias=False, device=device, dtype=torch.float32)
    dequantized_layer.weight.data.copy_(
        result.compute_dequantized_weight().to(device=device, dtype=torch.float32)
    )

    dense_layer = quantizer.create_inference_layer(
        result=result,
        linear_module=layer,
        use_gemlite=False,
    )
    gemlite_layer = quantizer.create_inference_layer(
        result=result,
        linear_module=layer,
        use_gemlite=True,
    )

    assert isinstance(dense_layer, MultipathMDBFLinear)
    assert isinstance(gemlite_layer, MultipathMDBFLinear)
    assert all(path.use_gemlite for path in gemlite_layer.paths)

    with torch.no_grad():
        y_dequantized = dequantized_layer(inp).float()
        y_dense = dense_layer(inp.to(torch.float16)).float()
        y_gemlite = gemlite_layer(inp.to(torch.float16)).float()

    _assert_gemlite_output_matches_dense(y_dense, y_gemlite)
    _assert_gemlite_output_matches_dense(y_dequantized, y_gemlite)


def test_mdbflinear_falls_back_to_dense_when_gemlite_unavailable(monkeypatch):
    """When GemLite is not installed, forcing use_gemlite=True must fall back to
    the dense path instead of raising (Test plan: dense fallback)."""
    # Simulate an environment without GemLite support.
    monkeypatch.setattr(mdbf_layer, "HAS_GEMLITE_SUPPORT", False)

    device = torch.device("cpu")
    torch.manual_seed(0)
    params = _make_mdbf_params(n=16, m=32, r=8, l=1, device=device, dtype=torch.float32)

    forced_layer = MDBFLinear(params, device=device, use_gemlite=True)
    dense_layer = MDBFLinear(params, device=device, use_gemlite=False)

    # GemLite is unavailable, so even use_gemlite=True must not enable a kernel.
    assert forced_layer.use_gemlite is False
    assert forced_layer._gemlite_layers == {}

    x = torch.randn(4, 32, device=device, dtype=torch.float32)
    with torch.no_grad():
        y_forced = forced_layer(x)
        y_dense = dense_layer(x)

    # The fallback must take the exact same dense forward path.
    assert y_forced.shape == (4, 16)
    assert torch.equal(y_forced, y_dense)


def test_class_defaults_are_multi_envelope():
    """The shipped defaults must be (l, P) = (2, 1), not a baseline format.

    (l, P) = (1, 1) reproduces DBF and (1, 2) reproduces LittleBit, so a regression in
    these defaults would silently quantize with a format MDBF is only measured against.
    """
    q = MDBF(target_bits=1.0)
    assert (q.l, q.P) == (DEFAULT_L, DEFAULT_P) == (2, 1)
    config = q.get_quant_config()
    assert (config["l"], config["P"]) == (2, 1)


def test_rank_from_bpw_matches_paper_formula():
    """rank_from_bpw() with scale_bits=16 must be consistent with the paper's BPW
    formula b = P * [r(n+m) + 16*l*(n+m+2r)] / (nm) (Test plan: rank_from_bpw)."""
    n, m, l, P = 512, 2048, 8, 1

    for r_true in (16, 64, 128, 256):
        # bpw_from_rank uses scale_bits=16, the paper default.
        b = bpw_from_rank(n, m, r_true, l=l, P=P)
        # floor rounding must never exceed the requested budget.
        r_floor = rank_from_bpw(n, m, b, l=l, P=P, scale_bits=16, rounding="floor")
        assert r_floor <= r_true
        assert bpw_from_rank(n, m, r_floor, l=l, P=P) <= b + 1e-9
        # round-trip: rounding to nearest recovers the exact rank.
        r_round = rank_from_bpw(n, m, b, l=l, P=P, scale_bits=16, rounding="round")
        assert r_round == r_true

    # scale_bits=0 drops the envelope cost, so more rank fits in the same budget.
    b = bpw_from_rank(n, m, 64, l=l, P=P)
    r_with_scale = rank_from_bpw(n, m, b, l=l, P=P, scale_bits=16, rounding="round")
    r_without_scale = rank_from_bpw(n, m, b, l=l, P=P, scale_bits=0, rounding="round")
    assert r_without_scale > r_with_scale


def test_lowrank_osvd_beats_plain_svd_in_hessian_error():
    """OSVD (H^{1/2}=Q diag(sqrt(λ)) Q^T whitening) must achieve a Hessian-weighted
    output error no larger than plain rank-r SVD for a non-diagonal H
    (Test plan: OSVD in activation-aware mode)."""
    torch.manual_seed(0)
    n, m, r = 32, 24, 4
    W = torch.randn(n, m, dtype=torch.float64)

    # Non-diagonal SPD Hessian (the H-weighting only matters when Q != I).
    A = torch.randn(m, m, dtype=torch.float64)
    H = A @ A.T + 0.1 * torch.eye(m, dtype=torch.float64)

    def hessian_error(W_hat):
        E = W - W_hat
        return torch.trace(E @ H @ E.T).item()

    # OSVD reconstruction: W_hat = U' @ V'^T.
    U_prime, V_prime = lowrank_osvd(W, H, r, ridge=0.0)
    err_osvd = hessian_error(U_prime @ V_prime.T)

    # Plain rank-r SVD (ignores H).
    U_s, S_s, Vh_s = torch.linalg.svd(W, full_matrices=False)
    W_svd = (U_s[:, :r] * S_s[:r]) @ Vh_s[:r, :]
    err_svd = hessian_error(W_svd)

    # OSVD is tailored to the H-weighted objective, so it must not be worse.
    assert err_osvd <= err_svd + 1e-6 * abs(
        err_svd
    ), f"OSVD H-weighted error ({err_osvd:.6e}) should be <= plain SVD ({err_svd:.6e})"
    # For a genuinely non-diagonal H the two solutions must differ.
    assert abs(err_osvd - err_svd) > 1e-8 * abs(
        err_svd
    ), "OSVD and plain SVD errors are identical: the H-weighting is not being exercised"


def _spy_on_projection_seeds(monkeypatch):
    """Record the seed handed to every svd_abs_rank_l() call and return the log."""
    seen = []
    original = admm.svd_abs_rank_l

    def spy(W, l, seed=None):
        seen.append(seed)
        return original(W, l, seed=seed)

    monkeypatch.setattr(admm, "svd_abs_rank_l", spy)
    return seen


def _make_admm_inputs(n=32, m=24, r=8, l=2):
    """Build a weight matrix and its Phase 1 MDBF parameters for ADMM tests."""
    torch.manual_seed(0)
    W = torch.randn(n, m, dtype=torch.float32)
    params_list, _ = initialize_MDBF(W, r, l, P=1)
    return W, params_list


def _make_seeded_quantizer(seed, activation_aware=False):
    """Quantizer for the seed-propagation tests.

    target_bits=2.0 on the layer size below yields r=6, which must stay above l: at
    r <= l the projection takes its exact (non-randomized) branch and would never touch
    the seed, making a propagation test pass without exercising anything.
    """
    return MDBF(
        target_bits=2.0,
        l=2,
        P=1,
        use_admm=True,
        admm_outer_iters=2,
        admm_inner_iters=2,
        use_gradient_refine=False,
        activation_aware=activation_aware,
        admm_seed=seed,
    )


def test_admm_seed_reaches_projection_from_quantizer(monkeypatch):
    """MDBF(admm_seed=...) must reach svd_abs_rank_l(): the projection takes a seed,
    so the whole quantizer -> run_mdbf -> ADMM chain has to forward it, otherwise the
    randomized SVD silently falls back to the global RNG (Test plan: admm_seed)."""
    seen = _spy_on_projection_seeds(monkeypatch)

    torch.manual_seed(0)
    layer = torch.nn.Linear(64, 32, bias=False, dtype=torch.float32)
    inp = torch.randn(2, 4, 64, dtype=torch.float32)
    quantizer = _make_seeded_quantizer(seed=1234)
    result = quantizer.quantize_layer(layer, inp)

    assert result.r > quantizer.l, "rank too low: the projection skips its random branch"
    assert seen, "ADMM never reached the MDBF projection"
    assert set(seen) == {1234}


def test_admm_seed_reaches_projection_in_hessian_mode(monkeypatch):
    """The Hessian-based (activation-aware) ADMM path must forward the seed too, again
    all the way from the quantizer."""
    seen = _spy_on_projection_seeds(monkeypatch)

    torch.manual_seed(0)
    layer = torch.nn.Linear(64, 32, bias=False, dtype=torch.float32)
    inp = torch.randn(2, 4, 64, dtype=torch.float32)
    quantizer = _make_seeded_quantizer(seed=99, activation_aware=True)
    hessian, nsamples = quantizer.calculate_hessian(layer, inp)
    result = quantizer.quantize_layer(layer, inp, hessian=hessian, nsamples=nsamples)

    # Without this the run would have fallen back to the plain ADMM path, which the
    # test above already covers.
    assert result.actual_activation_aware is True

    assert result.r > quantizer.l, "rank too low: the projection skips its random branch"
    assert seen, "Hessian-based ADMM never reached the MDBF projection"
    assert set(seen) == {99}


@pytest.mark.parametrize("l", [1, 2])
def test_admm_seed_makes_result_independent_of_global_rng(l):
    """A fixed seed must pin the ADMM result regardless of the ambient RNG state,
    which is the point of being able to fix it at all."""
    W, params_list = _make_admm_inputs(l=l)
    kwargs = dict(iters=3, inner_iters=2)

    torch.manual_seed(11)
    _, recon_seeded_a = admm.optimize_MDBF_admm(W, params_list, l=l, seed=7, **kwargs)
    torch.manual_seed(123456)
    _, recon_seeded_b = admm.optimize_MDBF_admm(W, params_list, l=l, seed=7, **kwargs)

    assert torch.equal(recon_seeded_a, recon_seeded_b)
