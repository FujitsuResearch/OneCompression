"""Regression tests for Hadamard hook target-type collection on this branch.

Three bugs are pinned:
  Bug 1 (runner.py): old ``next(...)`` sampled only the first ``down_proj``;
    now collect all distinct non-``nn.Linear`` types.
  Bug 2 (runner.py): a leading ``nn.Linear`` down_proj caused duplicate hooks on
    every ``nn.Linear`` and no hooks on ``GPTQLinear``; the filter prevents this.
  Bug 3 (quantized_model_loader.py): an unknown ``quant_method`` string yielded
    ``layers_cls=None`` (hooks silently disabled); now types come from the model.

Test layering:
  - The collection logic lives in the pure helpers
    ``onecomp.pre_process.rotation_utils.collect_quantized_down_proj_types`` /
    ``collect_down_proj_types`` (co-located with their sole consumer
    ``register_online_hadamard_hooks``); the helper tests import and call them
    directly for sharp, fast branch coverage (Bug 1/2 and the loader's no-filter
    behaviour).
  - The integration tests drive the real ``load_quantized_model`` /
    ``create_quantized_model`` to pin that the call sites actually use the
    helpers — i.e. Bug 3 (the loader no longer derives types from the
    ``quant_method`` string) and that the runner passes the collected types to
    ``register_online_hadamard_hooks``.  These cannot be covered by the helper
    tests alone.

CPU-only: helper tests use stub modules; the runner integration test mocks
model/quantizer; the loader integration test builds a tiny in-memory Llama.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
import logging
import types
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from safetensors.torch import save_file

from onecomp.pre_process.rotation_utils import (
    collect_down_proj_types,
    collect_quantized_down_proj_types,
)

# ── stub quantized layer types ─────────────────────────────────────


class _FakeQuantLinear(nn.Module):
    """Stub non-nn.Linear quantized layer (e.g. GPTQLinear stand-in)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))

    def forward(self, x):
        return x


class _AnotherQuantLinear(nn.Module):
    """Second distinct stub quantized layer (e.g. DoubleBinaryLinear stand-in)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))

    def forward(self, x):
        return x


# ── helpers ────────────────────────────────────────────────────────


def _build_model_with_down_proj(*layers):
    """Return a tiny nn.Module with realistic ``...mlp.down_proj`` module paths.

    Each layer is nested as ``block.layers_<i>.mlp.down_proj`` so that
    ``named_modules()`` yields paths ending in ``.down_proj`` (mirroring real
    transformer naming like ``model.layers.0.mlp.down_proj``), exercising the
    same substring match used in production.
    """

    class _Mlp(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.add_module("down_proj", layer)

        def forward(self, x):
            return x

    class _Block(nn.Module):
        def __init__(self, children):
            super().__init__()
            for i, layer in enumerate(children):
                self.add_module(f"layers_{i}", _Mlp(layer))

        def forward(self, x):
            return x

    root = nn.Module()
    root.add_module("block", _Block(layers))
    return root


# ── runner.py: collect_quantized_down_proj_types unit tests ────────


class TestCollectQuantizedDownProjTypes:
    """Pins Bug 1 and Bug 2 by calling the runner helper directly."""

    def test_includes_quantized_excludes_linear(self):
        """Bug 1: quantized types are collected; a co-located nn.Linear is excluded."""
        model = _build_model_with_down_proj(_FakeQuantLinear(), nn.Linear(4, 4))
        result = collect_quantized_down_proj_types(model)
        assert _FakeQuantLinear in result
        assert nn.Linear not in result

    def test_collects_all_distinct_types(self):
        """Bug 2: all distinct quantized types are collected (not just the first).

        The old ``next()``-based code returned only the first down_proj type;
        a second distinct type would have been silently lost.
        """
        model = _build_model_with_down_proj(
            _FakeQuantLinear(),
            _AnotherQuantLinear(),
            _FakeQuantLinear(),  # duplicate → deduped by set
        )
        result = collect_quantized_down_proj_types(model)
        assert set(result) == {_FakeQuantLinear, _AnotherQuantLinear}

    def test_first_down_proj_linear_does_not_suppress_quantized(self):
        """Bug 2 (direct reproduction): a leading nn.Linear does not suppress quantized types.

        The old next()-based code would have taken the first (nn.Linear) type as
        the sole target, causing duplicate hooks on every nn.Linear and no hooks
        on the quantized down_proj.
        """
        model = _build_model_with_down_proj(nn.Linear(4, 4), _FakeQuantLinear())
        result = collect_quantized_down_proj_types(model)
        assert _FakeQuantLinear in result
        assert nn.Linear not in result

    def test_no_quantized_down_proj_returns_empty(self):
        """All-nn.Linear down_proj layers yield an empty list (hooks then skipped)."""
        model = _build_model_with_down_proj(nn.Linear(4, 4), nn.Linear(4, 4))
        assert collect_quantized_down_proj_types(model) == []

    def test_non_down_proj_names_excluded(self):
        """Name-filter isolation: modules without 'down_proj' in the name are excluded."""
        model = nn.Module()
        model.add_module("up_proj", _FakeQuantLinear())
        model.add_module("gate_proj", _FakeQuantLinear())
        assert collect_quantized_down_proj_types(model) == []


# ── quantized_model_loader.py: collect_down_proj_types unit tests ──


class TestCollectDownProjTypes:
    """Pins the loader helper, including its intentional difference from runner."""

    def test_collects_down_proj_types(self):
        """down_proj types are derived from the module tree."""
        model = _build_model_with_down_proj(_FakeQuantLinear(), _FakeQuantLinear())
        assert _FakeQuantLinear in collect_down_proj_types(model)

    def test_does_not_filter_nn_linear(self):
        """Unlike the runner helper, nn.Linear is NOT filtered out.

        The loader receives an already-quantized model, so this difference is
        intentional; the test pins it so it is not accidentally removed.
        """
        model = _build_model_with_down_proj(nn.Linear(4, 4))
        assert nn.Linear in collect_down_proj_types(model)


# ── runner.py: create_quantized_model wiring (integration) ─────────


def _run_create_quantized_model(stub_model, *, has_additional_data=True):
    """Drive the real ``Runner.create_quantized_model`` over a stub model.

    Everything except the hook-collection block is mocked, so the real call site
    runs on ``stub_model``.  Returns ``{"called": bool, "layers_cls": list|None}``
    capturing what was passed to ``register_online_hadamard_hooks``.
    """
    from onecomp.runner import Runner

    runner = Runner.__new__(Runner)  # bypass heavy __init__
    runner.logger = logging.getLogger("test")

    if not hasattr(stub_model, "config"):
        stub_model.config = types.SimpleNamespace(num_hidden_layers=1)

    quantizer = MagicMock()
    quantizer.get_quant_config.return_value = {}
    quantizer.results = {}
    quantizer.finalize_quant_config_for_save.return_value = {}
    runner.quantizer = quantizer

    model_config = MagicMock()
    model_config.load_model.return_value = stub_model
    model_config.load_tokenizer.return_value = object()
    model_config.has_additional_data.return_value = has_additional_data
    model_config.fp32_had = False
    runner.model_config = model_config

    captured = {"called": False, "layers_cls": None}

    def _fake_register(model, layers_cls, fp32_had):
        captured["called"] = True
        captured["layers_cls"] = layers_cls
        return []

    with (
        patch("onecomp.utils.unfuse_moe.unfuse_moe_experts", return_value=False),
        patch(
            "onecomp.pre_process.rotation_utils.register_online_hadamard_hooks",
            side_effect=_fake_register,
        ),
    ):
        runner.create_quantized_model()

    return captured


class TestRunnerCreateQuantizedModelWiring:
    """Pins that ``create_quantized_model`` uses the helper and feeds the hooks.

    The helper tests above guarantee the collection logic; these guarantee the
    call site still calls it and passes the result to
    ``register_online_hadamard_hooks`` (a regression to inline ``next()`` would
    fail here).
    """

    def test_collected_types_reach_register_hooks(self):
        """The non-nn.Linear collected types are passed to register_online_hadamard_hooks."""
        model = _build_model_with_down_proj(nn.Linear(4, 4), _FakeQuantLinear())
        captured = _run_create_quantized_model(model)
        assert captured["called"]
        assert _FakeQuantLinear in captured["layers_cls"]
        assert nn.Linear not in captured["layers_cls"]

    def test_no_quantized_down_proj_skips_registration(self):
        """Empty collected types → register_online_hadamard_hooks is not called."""
        model = _build_model_with_down_proj(nn.Linear(4, 4))
        captured = _run_create_quantized_model(model)
        assert captured["called"] is False

    def test_not_rotated_skips_registration(self):
        """Guard: no re-registration for non-rotation-preprocessed models."""
        model = _build_model_with_down_proj(_FakeQuantLinear())
        captured = _run_create_quantized_model(model, has_additional_data=False)
        assert captured["called"] is False


# ── quantized_model_loader.py: load_quantized_model wiring (integration) ─


class TestLoaderLoadQuantizedModelWiring:
    """Pins Bug 3 by driving the real ``load_quantized_model``.

    The old code derived the hook target from the ``quant_method`` string, so an
    unknown method produced ``layers_cls=None`` (hooks silently disabled).  The
    helper unit tests cannot detect that regression — only driving the real
    loader can confirm the call site now derives types from the model.
    """

    def test_passes_model_derived_types_for_unknown_quant_method(self, tmp_path):
        """Unknown quant_method still passes a non-None, model-derived type list.

        ``modules_in_block_to_quantize`` is empty, so no layer is replaced and
        the down_proj stays ``nn.Linear``; since the loader does not filter
        nn.Linear, it must appear in the captured types.  This pins both the
        unknown-method → non-None behaviour and the absence of the nn.Linear
        filter at the real call site.
        """
        from transformers import LlamaConfig, LlamaForCausalLM

        from onecomp.quantized_model_loader import QuantizedModelLoader

        config = LlamaConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_hidden_layers=1,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=32,
            vocab_size=64,
        )
        model = LlamaForCausalLM(config).to(torch.float16).eval()
        state_dict = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}

        save_dir = tmp_path / "rotated_model"
        save_dir.mkdir()
        cfg_dict = config.to_dict()
        cfg_dict["quantization_config"] = {
            "quant_method": "unknown_future_method",  # ← would produce layers_cls=None in old code
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "modules_in_block_to_quantize": [],
            "rotated": True,
        }
        (save_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
        save_file(state_dict, str(save_dir / "model.safetensors"))

        captured = {}

        def _fake_register(model, layers_cls, fp32_had):
            captured["layers_cls"] = layers_cls
            return []

        with (
            patch(
                "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
                return_value=object(),
            ),
            patch(
                "onecomp.pre_process.rotation_utils.register_online_hadamard_hooks",
                side_effect=_fake_register,
            ),
        ):
            QuantizedModelLoader.load_quantized_model(
                str(save_dir),
                device_map="",
                local_files_only=True,
            )

        assert "layers_cls" in captured, "register_online_hadamard_hooks was not called"
        assert captured["layers_cls"] is not None, (
            "layers_cls must not be None for unknown quant_method "
            "(old code regression: unknown method → layers_cls=None)"
        )
        assert isinstance(captured["layers_cls"], list)
        assert (
            nn.Linear in captured["layers_cls"]
        ), "loader must not filter nn.Linear (down_proj stays nn.Linear when nothing is replaced)"
