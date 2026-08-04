"""Tests for ``Runner._strip_moe_expert_g_idx_for_vllm``.

vLLM's GPTQ MoE kernel (MoeWNA16) has no ``g_idx`` parameter, so a
full-wrapper export must drop per-expert ``g_idx`` buffers before they can
crash vLLM weight loading. Dropping is only safe when desc_act/actorder is
disabled -- otherwise g_idx carries real activation-order information that
cannot be silently discarded, so we must raise instead.

These tests use a plain ``Runner.__new__`` stub (only ``logger`` is
needed), matching the pattern in ``test_save_format_full_wrapper.py``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from logging import getLogger

import pytest
import torch

from onecomp.runner import Runner


def _make_runner_stub():
    runner = Runner.__new__(Runner)
    runner.logger = getLogger("test_strip_moe_expert_g_idx")
    return runner


def _state_dict(*keys):
    return {k: torch.zeros(1) for k in keys}


class TestNoMoeGIdxKeys:
    def test_returns_same_object_when_no_moe_g_idx_keys(self):
        runner = _make_runner_stub()
        state_dict = _state_dict(
            "model.layers.0.self_attn.q_proj.g_idx",
            "model.layers.0.mlp.down_proj.qweight",
        )
        result = runner._strip_moe_expert_g_idx_for_vllm(state_dict, {})
        assert result is state_dict

    def test_non_expert_g_idx_is_untouched(self):
        """Only '.mlp.experts.' g_idx keys are in scope; regular layer

        g_idx (e.g. self_attn) must survive untouched.
        """
        runner = _make_runner_stub()
        state_dict = _state_dict("model.layers.0.self_attn.q_proj.g_idx")
        result = runner._strip_moe_expert_g_idx_for_vllm(state_dict, {"desc_act": True})
        assert result == state_dict


class TestDropsTrivialGIdx:
    def test_drops_moe_expert_g_idx_when_desc_act_false(self):
        runner = _make_runner_stub()
        state_dict = _state_dict(
            "model.layers.0.mlp.experts.0.down_proj.g_idx",
            "model.layers.0.mlp.experts.0.down_proj.qweight",
            "model.layers.0.mlp.experts.1.up_proj.g_idx",
        )
        result = runner._strip_moe_expert_g_idx_for_vllm(state_dict, {"desc_act": False})
        assert "model.layers.0.mlp.experts.0.down_proj.g_idx" not in result
        assert "model.layers.0.mlp.experts.1.up_proj.g_idx" not in result
        assert "model.layers.0.mlp.experts.0.down_proj.qweight" in result

    def test_defaults_to_dropping_when_desc_act_absent(self):
        runner = _make_runner_stub()
        state_dict = _state_dict("model.layers.0.mlp.experts.0.down_proj.g_idx")
        result = runner._strip_moe_expert_g_idx_for_vllm(state_dict, {})
        assert result == {}

    def test_actorder_alias_also_honoured(self):
        """quant_config may spell this ``actorder`` instead of ``desc_act``."""
        runner = _make_runner_stub()
        state_dict = _state_dict("model.layers.0.mlp.experts.0.down_proj.g_idx")
        with pytest.raises(RuntimeError):
            runner._strip_moe_expert_g_idx_for_vllm(state_dict, {"actorder": True})

    def test_does_not_mutate_input_dict(self):
        runner = _make_runner_stub()
        state_dict = _state_dict("model.layers.0.mlp.experts.0.down_proj.g_idx")
        original = dict(state_dict)
        runner._strip_moe_expert_g_idx_for_vllm(state_dict, {"desc_act": False})
        assert state_dict == original


class TestRaisesOnDescAct:
    def test_raises_when_desc_act_true(self):
        runner = _make_runner_stub()
        state_dict = _state_dict("model.layers.0.mlp.experts.0.down_proj.g_idx")
        with pytest.raises(RuntimeError, match="actorder/desc_act"):
            runner._strip_moe_expert_g_idx_for_vllm(state_dict, {"desc_act": True})
