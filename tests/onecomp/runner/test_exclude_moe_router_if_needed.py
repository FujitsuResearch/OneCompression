"""Unit tests for ``Runner._exclude_moe_router_if_needed``.

Pins that MoE models get both ``"router"`` and ``"shared_expert_gate"``
added to ``exclude_layer_keywords`` (vLLM's GateLinear can't be
quantized), that each keyword is tracked independently so one being
present doesn't block the other, and that non-MoE models are untouched.

Copyright 2025-2026 Fujitsu Ltd.
"""

from types import SimpleNamespace

from onecomp.runner import Runner


class _FakeModelConfig:
    """Stand-in for ``ModelConfig`` exposing only ``load_config``."""

    def __init__(self, config):
        self._config = config

    def load_config(self):
        return self._config


class _FakeQuantizer:
    """Stand-in for ``Quantizer`` exposing only ``exclude_layer_keywords``."""

    def __init__(self, exclude_layer_keywords=None):
        self.exclude_layer_keywords = exclude_layer_keywords


def _make_runner(config, **kwargs):
    return Runner(model_config=_FakeModelConfig(config), **kwargs)


def test_dense_model_without_num_experts_is_untouched():
    """No ``num_experts`` anywhere on the config => no-op."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=None)
    runner = _make_runner(SimpleNamespace(), quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords is None


def test_num_experts_zero_is_untouched():
    """``num_experts=0`` is treated the same as "absent"."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=["some_other_keyword"])
    runner = _make_runner(SimpleNamespace(num_experts=0), quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords == ["some_other_keyword"]


def test_moe_model_with_top_level_num_experts_sets_keywords_from_none():
    """MoE model detected via ``config.num_experts``, starting from ``None``."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=None)
    runner = _make_runner(SimpleNamespace(num_experts=8), quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords == ["router", "shared_expert_gate"]


def test_moe_model_with_nested_text_config_num_experts():
    """MoE model detected via ``config.text_config.num_experts`` (VLM-style)."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=None)
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=8))
    runner = _make_runner(config, quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords == ["router", "shared_expert_gate"]


def test_moe_model_appends_to_existing_exclude_list():
    """Existing keywords are preserved; router/shared_expert_gate are appended."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=["lm_head"])
    runner = _make_runner(SimpleNamespace(num_experts=8), quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords == ["lm_head", "router", "shared_expert_gate"]


def test_moe_model_adds_shared_expert_gate_when_only_router_present():
    """``"router"`` already present must not block adding ``"shared_expert_gate"``."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=["router"])
    runner = _make_runner(SimpleNamespace(num_experts=8), quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords == ["router", "shared_expert_gate"]


def test_moe_model_adds_router_when_only_shared_expert_gate_present():
    """``"shared_expert_gate"`` already present must not block adding ``"router"``."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=["shared_expert_gate"])
    runner = _make_runner(SimpleNamespace(num_experts=8), quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords == ["shared_expert_gate", "router"]


def test_moe_model_is_noop_when_both_keywords_already_present():
    """No duplicates are introduced when both keywords are already excluded."""
    quantizer = _FakeQuantizer(exclude_layer_keywords=["router", "shared_expert_gate"])
    runner = _make_runner(SimpleNamespace(num_experts=8), quantizer=quantizer)

    runner._exclude_moe_router_if_needed()

    assert quantizer.exclude_layer_keywords == ["router", "shared_expert_gate"]


def test_moe_model_applies_to_all_quantizers_in_list():
    """With multiple ``quantizers``, every one gets the exclusion applied."""
    q1 = _FakeQuantizer(exclude_layer_keywords=None)
    q2 = _FakeQuantizer(exclude_layer_keywords=["lm_head"])
    runner = _make_runner(SimpleNamespace(num_experts=8), quantizers=[q1, q2])

    runner._exclude_moe_router_if_needed()

    assert q1.exclude_layer_keywords == ["router", "shared_expert_gate"]
    assert q2.exclude_layer_keywords == ["lm_head", "router", "shared_expert_gate"]


def test_moe_model_does_not_alias_keyword_list_across_quantizers():
    """Two quantizers starting from ``None`` must not share the same list object."""
    q1 = _FakeQuantizer(exclude_layer_keywords=None)
    q2 = _FakeQuantizer(exclude_layer_keywords=None)
    runner = _make_runner(SimpleNamespace(num_experts=8), quantizers=[q1, q2])

    runner._exclude_moe_router_if_needed()

    assert q1.exclude_layer_keywords is not q2.exclude_layer_keywords
    q1.exclude_layer_keywords.append("extra")
    assert q2.exclude_layer_keywords == ["router", "shared_expert_gate"]
