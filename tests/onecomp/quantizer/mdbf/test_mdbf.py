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

from onecomp.quantizer.mdbf._mdbf import MDBF, MDBFResult
from onecomp.quantizer.mdbf.initialize import MDBFParams
from onecomp.quantizer.mdbf.utils import reconstruct_weight

from test_module import BaseQuantizeSpec


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
        {"target_bits": 1e-10},      # lower boundary (near zero, positive)
        {"target_bits": 100.0},      # large value
        # l: int >= 1
        {"l": 1},                    # lower boundary
        {"l": 5},                    # large value
        # P: in {1, 2}
        {"P": 1},                    # minimum valid value
        {"P": 2},                    # maximum valid value
        # svd_mode: in {"svd", "svd_llm"}
        {"svd_mode": "svd"},         # default value (explicit check)
        {"svd_mode": "svd_llm"},     # alternate valid value
        # act_init: in {"none", "osvd", "svd_llm"}
        {"act_init": "none"},        # first valid value
        {"act_init": "osvd"},        # second valid value
        {"act_init": "svd_llm"},     # third valid value
        # scale_bits: int >= 0
        {"scale_bits": 0},           # lower boundary (binary-only accounting)
        {"scale_bits": 16},          # default value (FP16)
        {"scale_bits": 32},          # large value
        # admm_reg: float >= 0 (always validated)
        {"admm_reg": 0.0},           # lower boundary (0.0 is valid)
        {"admm_reg": 100.0},         # large value
        # use_admm: bool
        {"use_admm": True, "admm_iters": 1, "admm_inner_iters": 1},  # lower boundary combo when enabled
        {"use_admm": False},         # disabled individually
        # admm_iters: int >= 1 (validated when use_admm=True)
        {"use_admm": True, "admm_iters": 1},    # lower boundary
        {"use_admm": True, "admm_iters": 100},  # large value
        # admm_inner_iters: int >= 1 (validated when use_admm=True)
        {"use_admm": True, "admm_inner_iters": 1},   # lower boundary
        {"use_admm": True, "admm_inner_iters": 10},  # large value
        # admm_* not validated when use_admm=False
        {"use_admm": False, "admm_iters": 0},        # admm_iters=0 allowed when admm off
        {"use_admm": False, "admm_inner_iters": 0},  # admm_inner_iters=0 allowed when admm off
        # use_gradient_refine: bool
        {"use_gradient_refine": True, "gradient_iters": 1, "gradient_lr": 1e-3},  # lower boundary combo when enabled
        {"use_gradient_refine": False},  # disabled individually
        # gradient_iters: int >= 1 (validated when use_gradient_refine=True)
        {"use_gradient_refine": True, "gradient_iters": 1},    # lower boundary
        {"use_gradient_refine": True, "gradient_iters": 100},  # large value
        # gradient_lr: float > 0, strict (validated when use_gradient_refine=True)
        {"use_gradient_refine": True, "gradient_lr": 1e-10},  # lower boundary (near zero, positive)
        {"use_gradient_refine": True, "gradient_lr": 100.0},  # large value
        # gradient_* not validated when use_gradient_refine=False
        {"use_gradient_refine": False, "gradient_iters": 0},   # gradient_iters=0 allowed when refine off
        {"use_gradient_refine": False, "gradient_lr": 0.0},    # gradient_lr=0.0 allowed when refine off
        # activation_aware: bool
        {"activation_aware": True, "P": 1},   # P=1 required to exercise the activation_aware code path
        {"activation_aware": True, "P": 2},   # P!=1 fallback (warning only, no error)
        {"activation_aware": False},           # disabled individually
        # mlp_target_bits: float > 0 or None
        {"mlp_target_bits": 1e-10},   # lower boundary (near zero, positive)
        {"mlp_target_bits": 100.0},   # large value
        # module_target_bits: dict[str, float>0] or None
        {"module_target_bits": {"model.layers.0.self_attn.q_proj": 1.5}},       # single-layer override
        {"module_target_bits": {                                                  # multi-layer override
            "model.layers.0.self_attn.q_proj": 1.0,
            "model.layers.0.mlp.gate_proj": 2.0,
        }},
        # combo: all bools False
        {"use_admm": False, "use_gradient_refine": False, "activation_aware": False},
        # combo: all numerics at lower bounds
        # use_admm not specified -> class default True applies, so admm_iters/admm_inner_iters are validated
        {
            "target_bits": 1.0,
            "l": 1,
            "P": 1,
            "scale_bits": 0,
            "admm_reg": 0.0,
            "admm_iters": 1,
            "admm_inner_iters": 1,
        },
        # all class defaults
        {
            "target_bits": 1.0,
            "l": 1,
            "P": 2,
            "svd_mode": "svd",
            "use_admm": True,
            "admm_iters": 260,
            "admm_inner_iters": 3,
            "admm_reg": 0.03,
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
            "admm_iters": 0,
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
            "admm_iters": 100,
            "admm_inner_iters": 10,
            "admm_reg": 100.0,
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
        {"target_bits": 0.0},    # boundary (target_bits > 0, strict)
        {"target_bits": -1.0},   # negative value
        # l: int >= 1
        {"l": 0},                # below lower boundary (l >= 1)
        {"l": -1},               # negative value
        # P: in {1, 2}
        {"P": 0},                # not in {1, 2} (below range)
        {"P": 3},                # not in {1, 2} (above range)
        # scale_bits: int >= 0
        {"scale_bits": -1},      # below lower boundary (scale_bits >= 0)
        # admm_reg: float >= 0
        {"admm_reg": -0.01},     # below lower boundary (admm_reg >= 0)
        # admm_iters: int >= 1 (validated when use_admm=True)
        {"use_admm": True, "admm_iters": 0},   # below lower boundary (admm_iters >= 1 when use_admm=True)
        {"use_admm": True, "admm_iters": -1},  # negative value
        # admm_inner_iters: int >= 1 (validated when use_admm=True)
        {"use_admm": True, "admm_inner_iters": 0},   # below lower boundary (admm_inner_iters >= 1 when use_admm=True)
        {"use_admm": True, "admm_inner_iters": -1},  # negative value
        # gradient_iters: int >= 1 (validated when use_gradient_refine=True)
        {"use_gradient_refine": True, "gradient_iters": 0},  # below lower boundary
        # gradient_lr: float > 0, strict (validated when use_gradient_refine=True)
        {"use_gradient_refine": True, "gradient_lr": 0.0},   # boundary (gradient_lr > 0, strict)
        # svd_mode
        {"svd_mode": "invalid"},  # not in {"svd", "svd_llm"}
        # act_init
        {"act_init": "invalid"},  # not in {"none", "osvd", "svd_llm"}
        # mlp_target_bits: float > 0, strict
        {"mlp_target_bits": 0.0},   # boundary (mlp_target_bits > 0, strict)
        {"mlp_target_bits": -1.0},  # negative value
        # module_target_bits
        {"module_target_bits": "not a dict"},                                          # wrong type
        {"module_target_bits": {"model.layers.0.self_attn.q_proj": 0.0}},   # value not > 0
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

    def check_forward_error(self, error_original_vs_dequantized, error_dequantized_vs_applied, max_error_dequantized_vs_applied):
        self.logger.info(
            "[MDBF forward error] "
            f"original_vs_mdbf(rel={error_original_vs_dequantized:.8f}), "
            f"mdbf_vs_mdbfl(max={max_error_dequantized_vs_applied:.8f}), "
            f"mdbf_vs_mdbfl(rel={error_dequantized_vs_applied:.8f})"
        )

        assert max_error_dequantized_vs_applied < 1e-2, (
            f"MDBF dequantized vs applied max error too large: {max_error_dequantized_vs_applied}"
        )

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
