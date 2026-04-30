"""Tests for the MDBF quantizer implementation.

Copyright 2026 Fujitsu Ltd.
"""

import logging
import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from onecomp.quantizer.mdbf._mdbf import MDBF, MDBFResult
from onecomp.quantizer.mdbf.initialize import MSVIDParams
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
        {"target_bits": 1e-10},
        {"target_bits": 100.0},
        {"l": 1},
        {"l": 5},
        {"P": 1},
        {"P": 2},
        {"svd_mode": "svd_llm"},
        {"act_init": "osvd"},
        {"use_admm": True, "admm_iters": 1, "admm_inner_iters": 1},
        {"use_gradient_refine": True, "gradient_iters": 1, "gradient_lr": 1e-3},
    ]

    abnormal_parameters = [
        {"target_bits": 0.0},
        {"l": 0},
        {"P": 3},
        {"admm_reg": -0.01},
        {"svd_mode": "invalid"},
        {"act_init": "invalid"},
        {"module_target_bits": "not a dict"},
    ]

    logger = logging.getLogger(__name__)

    def make_quantizer(self, **params):
        """Return a quantizer instance wrapped to accept (hessian, nsamples) tuples."""
        q = self.quantizer_cls(**params)
        orig_quantize_layer = q.quantize_layer

        def _wrapped_quantize_layer(module, input=None, hessian=None, nsamples=None, *args, **kwargs):
            if isinstance(hessian, (list, tuple)):
                try:
                    h, ns = hessian
                except Exception:
                    h = hessian[0]
                    ns = None
                hessian_local = h
                if nsamples is None:
                    nsamples = ns
            else:
                hessian_local = hessian

            if getattr(q, "flag_nsamples", False) and nsamples is not None:
                return orig_quantize_layer(module, input, hessian=hessian_local, nsamples=nsamples, *args, **kwargs)
            else:
                return orig_quantize_layer(module, input, hessian=hessian_local, *args, **kwargs)

        q.quantize_layer = _wrapped_quantize_layer
        return q

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
        l = result.l

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

            assert A_sign.shape == (n, r)
            assert B_sign.shape == (r, m)
            assert A_amp.shape == (n, l)
            assert B_amp.shape == (m, l)
            assert Q_U_amp.shape == (r, l)
            assert Q_V_amp.shape == (r, l)

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

        # attach msvid_params so layer-replacement utilities can use them
        params_list = [
            MSVIDParams(
                A_sign=result.mdbf_A_sign[p],
                B_sign=result.mdbf_B_sign[p],
                A_amp=result.mdbf_A_amp[p],
                B_amp=result.mdbf_B_amp[p],
                Q_U_amp=result.mdbf_Q_U_amp[p],
                Q_V_amp=result.mdbf_Q_V_amp[p],
            )
            for p in range(result.P)
        ]
        module.msvid_params = params_list
        module.is_quantized = True
