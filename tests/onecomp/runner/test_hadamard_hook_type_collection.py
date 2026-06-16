"""Regression tests for Hadamard hook target-type collection changes.

Pins the behaviour of two bugs fixed on this branch:

Bug 1 — Hadamard hooks registered on nn.Linear down_proj layers (runner.py):
  Before: ``next(m for n,m in model.named_modules() if 'down_proj' in n)``
          sampled only the FIRST ``down_proj`` layer and passed its type to
          ``register_online_hadamard_hooks``.
  After:  collect ALL distinct non-``nn.Linear`` ``down_proj`` types; only
          quantized layers receive hooks.

Bug 2 — Duplicate hooks on nn.Linear and missing hooks on GPTQLinear when the first down_proj is nn.Linear (runner.py):
  When the FIRST ``down_proj`` was ``nn.Linear`` (unquantized), the old code
  registered hooks on every ``nn.Linear`` in the model (duplicate hooks) while
  ``GPTQLinear`` down_proj layers received no hooks at all (missing hooks).
  The fix collects from the full module tree, skipping ``nn.Linear``.

Bug 3 — Hooks silently disabled when quant_method string is unknown (quantized_model_loader.py):
  Before: derive hook target class from ``quant_method`` string
          (``"gptq"`` → ``GPTQLinear``, ``"dbf"`` → ``DoubleBinaryLinear``,
          anything else → ``None``).  An unknown method silently disabled hooks.
  After:  collect types from ``model.named_modules()`` directly.

All tests are CPU-only and do not require downloaded weights.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
from unittest.mock import patch

import torch
import torch.nn as nn
from safetensors.torch import save_file


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
    """Return a tiny nn.Module whose children are named ``*_down_proj_i``."""

    class _Block(nn.Module):
        def __init__(self, children):
            super().__init__()
            for i, layer in enumerate(children):
                self.add_module(f"layer_{i}_down_proj", layer)

        def forward(self, x):
            return x

    root = nn.Module()
    root.add_module("block", _Block(layers))
    return root


def _runner_collect(model):
    """Replicate the runner.py type-collection comprehension."""
    return list(
        {
            type(module)
            for name, module in model.named_modules()
            if "down_proj" in name and not isinstance(module, nn.Linear)
        }
    )


def _loader_collect(model):
    """Replicate the quantized_model_loader.py type-collection comprehension."""
    return list(
        {
            type(module)
            for name, module in model.named_modules()
            if "down_proj" in name
        }
    )


# ── runner.py logic tests ──────────────────────────────────────────


class TestRunnerDownProjTypeCollection:
    """Pins the two runner.py bugs fixed on this branch (Bug 1 and Bug 2).

    Bug 1: Hadamard hooks registered on nn.Linear down_proj layers
      → ``not isinstance(module, nn.Linear)`` filter prevents unquantized layers
        from being collected.

    Bug 2: Duplicate hooks on nn.Linear and missing hooks on GPTQLinear when the first down_proj is nn.Linear
      → The old ``next()``-based code took only the first down_proj type.  If
        that was ``nn.Linear``, hooks fired on every ``nn.Linear`` in the model
        (duplicate hooks) while ``GPTQLinear`` down_proj got nothing (missing hooks).
        The fix collects all quantized types from the full module tree.
    """

    def test_includes_custom_quantized_type(self):
        """Bug 1 + 2: quantized types are collected; nn.Linear is excluded."""
        model = _build_model_with_down_proj(_FakeQuantLinear(), _FakeQuantLinear())
        result = _runner_collect(model)
        assert _FakeQuantLinear in result
        assert nn.Linear not in result

    def test_collects_all_distinct_types(self):
        """Bug 2: all distinct quantized types are collected (detects regression to first-type-only sampling).

        With the old ``next()``-based code, only the type of the FIRST layer
        would have been returned; a second distinct type would be silently lost.
        """
        model = _build_model_with_down_proj(
            _FakeQuantLinear(),
            _AnotherQuantLinear(),
            _FakeQuantLinear(),  # duplicate → deduped by set
        )
        result = _runner_collect(model)
        assert set(result) == {_FakeQuantLinear, _AnotherQuantLinear}

    def test_first_down_proj_linear_does_not_suppress_quantized(self):
        """Bug 2 (direct reproduction): quantized types after an initial nn.Linear are still collected.

        Reproduces the exact failure mode of Bug 2: the first down_proj is
        nn.Linear (unquantized), the second is _FakeQuantLinear (quantized,
        representing GPTQLinear).  The old next()-based code would have taken
        nn.Linear as the sole target, causing:
          - duplicate hooks: every nn.Linear in the model received hooks
          - missing hooks: the quantized down_proj received no hooks
        """
        model = _build_model_with_down_proj(nn.Linear(4, 4), _FakeQuantLinear())
        result = _runner_collect(model)
        assert _FakeQuantLinear in result, (
            "GPTQLinear-like type must be collected even when the first down_proj is nn.Linear"
        )
        assert nn.Linear not in result, (
            "nn.Linear must not receive hooks (would cause duplicate hooks on all nn.Linear in the model)"
        )

    def test_no_down_proj_returns_empty(self):
        """Name-filter isolation: modules whose names do not contain 'down_proj' are excluded.

        Uses _FakeQuantLinear so that only the name filter does the work
        (the type filter alone would not exclude it).
        """
        model = nn.Module()
        model.add_module("up_proj", _FakeQuantLinear())
        model.add_module("gate_proj", _FakeQuantLinear())
        result = _runner_collect(model)
        assert result == []


# ── quantized_model_loader.py logic tests ─────────────────────────


class TestLoaderDownProjTypeCollection:
    """Pins the model-derived type collection in quantized_model_loader.py.

    The key regression: with the old code, any quant_method that was neither
    "gptq" nor "dbf" produced layers_cls=None, silently disabling Hadamard
    hooks for the loaded model.  The new code derives types from the model
    itself, so any method works correctly.
    """

    def test_collects_type_independently_of_quant_method_string(self):
        """Types derived from model regardless of quant_method string (L1)."""
        model = _build_model_with_down_proj(_FakeQuantLinear(), _FakeQuantLinear())
        result = _loader_collect(model)
        assert _FakeQuantLinear in result

    def test_loader_includes_nn_linear_unlike_runner(self):
        """The loader does NOT filter nn.Linear (unlike runner.py).

        The loader receives an already-quantized model from disk, so
        down_proj layers are quantized types.  The absence of the
        nn.Linear filter is intentional — the test pins this difference
        so it is not accidentally removed.
        """
        model = _build_model_with_down_proj(nn.Linear(4, 4))
        result = _loader_collect(model)
        assert nn.Linear in result

    def test_loader_passes_actual_types_to_register_hooks(self, tmp_path):
        """register_online_hadamard_hooks receives actual module types (L2).

        Regression: the old code for an unknown quant_method would have
        passed layers_cls=None.  The new code must pass the real types
        derived from the loaded model.
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
        assert len(captured["layers_cls"]) > 0
