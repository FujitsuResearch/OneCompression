"""Unit tests for ``_rtn_fallback_result``.

Covers the RTN fallback path used when an MoE expert layer receives no
tokens during calibration: it must still be quantized (via RTN) instead
of skipped, and packaged as a ``GPTQResult`` so it flows through the same
``create_inference_layer`` / export path as calibrated GPTQ experts.

Copyright 2025-2026 Fujitsu Ltd.
"""

import torch
import torch.nn as nn

from onecomp.qep._quantize_with_qep_arch import (
    _resolve_gptq_for_rtn_fallback,
    _rtn_fallback_result,
)
from onecomp.quantizer.autobit import AutoBitQuantizer
from onecomp.quantizer.gptq import GPTQ
from onecomp.quantizer.gptq._gptq import GPTQResult
from onecomp.quantizer.rtn import RTN


def _linear(in_features=32, out_features=16, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features, bias=False)


class TestRtnFallbackResult:
    def test_returns_gptq_result(self):
        module = _linear()
        quantizer = GPTQ(wbits=4, groupsize=-1, sym=True)
        result = _rtn_fallback_result(module, quantizer, "model.layers.0.mlp.experts.0.down_proj")
        assert isinstance(result, GPTQResult)

    def test_actorder_is_false_and_perm_is_none(self):
        """RTN has no notion of activation order, unlike calibrated GPTQ."""
        module = _linear()
        quantizer = GPTQ(wbits=4, groupsize=-1, sym=True, actorder=True)
        result = _rtn_fallback_result(module, quantizer, "model.layers.0.mlp.experts.0.down_proj")
        assert result.actorder is False
        assert result.perm is None

    def test_resolves_wbits_and_groupsize_from_quantizer(self):
        module = _linear()
        quantizer = GPTQ(wbits=4, groupsize=128, sym=True, mlp_wbits=2, mlp_groupsize=32)
        # name contains "mlp" -> mlp_wbits/mlp_groupsize override should apply
        result = _rtn_fallback_result(module, quantizer, "model.layers.0.mlp.experts.0.down_proj")
        assert result.wbits == 2
        assert result.groupsize == 32

    def test_module_wbits_override_takes_priority(self):
        module = _linear()
        name = "model.layers.0.mlp.experts.0.down_proj"
        quantizer = GPTQ(wbits=4, groupsize=-1, sym=True, mlp_wbits=2, module_wbits={name: 8})
        result = _rtn_fallback_result(module, quantizer, name)
        assert result.wbits == 8

    def test_sym_propagated(self):
        module = _linear()
        for sym in (True, False):
            quantizer = GPTQ(wbits=4, groupsize=-1, sym=sym)
            result = _rtn_fallback_result(module, quantizer, "mlp.experts.0.down_proj")
            assert result.sym is sym

    def test_scales_and_qzeros_are_transposed_to_group_major(self):
        """RTN's raw scale/zero are (out_features, num_groups); GPTQResult

        expects (num_groups, out_features).
        """
        out_features, in_features, groupsize = 16, 32, 8
        module = _linear(in_features=in_features, out_features=out_features)
        quantizer = GPTQ(wbits=4, groupsize=groupsize, sym=True)
        result = _rtn_fallback_result(module, quantizer, "mlp.experts.0.down_proj")

        num_groups = in_features // groupsize
        assert result.scales.shape == (num_groups, out_features)
        assert result.qzeros.shape == (num_groups, out_features)
        assert result.qweight.shape == (out_features, in_features)

    def test_dequantized_weight_matches_shape(self):
        module = _linear(in_features=32, out_features=16)
        quantizer = GPTQ(wbits=4, groupsize=-1, sym=True)
        result = _rtn_fallback_result(module, quantizer, "mlp.experts.0.down_proj")
        assert result.dequantized_weight.shape == module.weight.data.shape

    def test_compute_dequantized_weight_roundtrip(self):
        """The packaged qweight/scales/qzeros must reconstruct a weight

        consistent with the shapes GPTQResult.compute_dequantized_weight expects
        (this is what create_inference_layer / export relies on downstream).
        """
        module = _linear(in_features=32, out_features=16)
        quantizer = GPTQ(wbits=4, groupsize=16, sym=True)
        result = _rtn_fallback_result(module, quantizer, "mlp.experts.0.down_proj")

        reconstructed = result.compute_dequantized_weight()
        assert reconstructed.shape == module.weight.data.shape


class TestResolveGptqForRtnFallback:
    """Covers which quantizers the no-tokens RTN fallback supports.

    Plain GPTQ always supports it. AutoBitQuantizer supports it only when
    every child it assigned is GPTQ (so a per-module bits/groupsize/sym can
    be resolved); any non-GPTQ quantizer (plain or as an AutoBitQuantizer
    child) is unsupported.
    """

    def test_plain_gptq_resolves_to_itself(self):
        module = _linear()
        quantizer = GPTQ(wbits=4, groupsize=-1, sym=True)
        assert _resolve_gptq_for_rtn_fallback(quantizer, module) is quantizer

    def test_non_gptq_quantizer_is_unsupported(self):
        module = _linear()
        quantizer = RTN(wbits=4, groupsize=-1, sym=True)
        assert _resolve_gptq_for_rtn_fallback(quantizer, module) is None

    def test_autobit_with_all_gptq_children_resolves_module_specific_child(self):
        module = _linear()
        assigned_child = GPTQ(wbits=2, groupsize=32, sym=True)
        other_child = GPTQ(wbits=4, groupsize=128, sym=True)
        autobit = AutoBitQuantizer(quantizers=[assigned_child, other_child])
        autobit._module_to_quantizer = {module: assigned_child}
        autobit._name_to_quantizer = {"mlp.experts.0.down_proj": assigned_child}

        resolved = _resolve_gptq_for_rtn_fallback(autobit, module)

        assert resolved is assigned_child

    def test_autobit_with_a_non_gptq_child_is_unsupported(self):
        module = _linear()
        gptq_child = GPTQ(wbits=4, groupsize=-1, sym=True)
        rtn_child = RTN(wbits=4, groupsize=-1, sym=True)
        autobit = AutoBitQuantizer(quantizers=[gptq_child, rtn_child])
        autobit._module_to_quantizer = {module: gptq_child}
        autobit._name_to_quantizer = {
            "mlp.experts.0.down_proj": gptq_child,
            "mlp.experts.1.down_proj": rtn_child,
        }

        assert _resolve_gptq_for_rtn_fallback(autobit, module) is None
