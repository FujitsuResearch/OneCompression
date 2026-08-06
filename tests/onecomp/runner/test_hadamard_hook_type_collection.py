"""Regression tests for Hadamard hook target-type collection on this branch.

Four bugs are pinned:
  Bug 1 (runner.py): old ``next(...)`` sampled only the first ``down_proj``;
    now collect all distinct non-``nn.Linear`` types.
  Bug 2 (runner.py): a leading ``nn.Linear`` down_proj caused duplicate hooks on
    every ``nn.Linear`` and no hooks on ``GPTQLinear``; the filter prevents this.
  Bug 3 (quantized_model_loader.py): an unknown ``quant_method`` string yielded
    ``layers_cls=None`` (hooks silently disabled); now types come from the model.
  Bug 4 (rotation_utils.py): the ``"down_proj" in name`` filter also matched a
    quantized layer's *descendants*, so MDBF's ``nn.ModuleList`` of paths landed
    in ``layers_cls`` and aborted the search -- zero hooks, no error.

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
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from safetensors.torch import save_file

from onecomp.pre_process.rotation_utils import (
    collect_down_proj_types,
    collect_quantized_down_proj_types,
    register_online_hadamard_hooks,
)

# ── stub quantized layer types ─────────────────────────────────────


class _FakeQuantLinear(nn.Module):
    """Stub non-nn.Linear quantized layer (e.g. GPTQLinear stand-in).

    ``in_features`` is exposed like every real inference layer, since the
    Hadamard hook sizes ``get_hadK`` from it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_features = 4
        self.weight = nn.Parameter(torch.zeros(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _AnotherQuantLinear(nn.Module):
    """Second distinct stub quantized layer (e.g. DoubleBinaryLinear stand-in)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NestedQuantLinear(nn.Module):
    """Stub quantized layer assembled from submodules (MultipathMDBFLinear stand-in).

    The children are held in an ``nn.ModuleList`` and are themselves quantized
    stubs, so ``named_modules()`` exposes two extra types underneath the
    ``down_proj`` name — exactly the shape that used to poison ``layers_cls``.
    """

    def __init__(self, in_features: int = 4, paths: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.paths = nn.ModuleList(_FakeQuantLinear() for _ in range(paths))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LinearSubclassQuantLinear(nn.Linear):
    """Stub quantized layer that subclasses ``nn.Linear`` (e.g. ``QuantLinear``).

    ``find_linear_layers`` matches on exact type, so such a layer is invisible
    to the default ``[nn.Linear]`` search — it never receives a hook from
    ``RotatedModelConfig.load_model()`` and must not be filtered out of the
    re-registration pass either.
    """


# ── helpers ────────────────────────────────────────────────────────


def _build_model_with_down_proj(*layers: nn.Module) -> nn.Module:
    """Return a tiny nn.Module mirroring an HF decoder's module paths.

    Each layer becomes ``model.layers.<i>.mlp.down_proj``, matching real
    transformer naming and therefore the exact predicate production hooks on
    (``is_online_hadamard_target``).

    ``layers`` is a genuine ``nn.ModuleList``, which the collection logic
    depends on: ``find_linear_layers`` stops descending at the first type
    match, so an ``nn.ModuleList`` leaking into ``layers_cls`` matches this
    container itself and aborts the search before any ``down_proj`` is
    reached.  A plain ``nn.Module`` holder would hide that failure mode.
    """

    class _Mlp(nn.Module):
        def __init__(self, layer: nn.Module) -> None:
            super().__init__()
            self.add_module("down_proj", layer)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _Layer(nn.Module):
        def __init__(self, layer: nn.Module) -> None:
            super().__init__()
            self.mlp = _Mlp(layer)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _Inner(nn.Module):
        def __init__(self, children: tuple[nn.Module, ...]) -> None:
            super().__init__()
            self.layers = nn.ModuleList(_Layer(layer) for layer in children)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    root = nn.Module()
    root.add_module("model", _Inner(layers))
    return root


# ── runner.py: collect_quantized_down_proj_types unit tests ────────


class TestCollectQuantizedDownProjTypes:
    """Pins Bug 1 and Bug 2 by calling the runner helper directly."""

    def test_includes_quantized_excludes_linear(self) -> None:
        """Bug 1: quantized types are collected; a co-located nn.Linear is excluded."""
        model = _build_model_with_down_proj(_FakeQuantLinear(), nn.Linear(4, 4))
        result = collect_quantized_down_proj_types(model)
        assert _FakeQuantLinear in result
        assert nn.Linear not in result

    def test_collects_all_distinct_types(self) -> None:
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

    def test_first_down_proj_linear_does_not_suppress_quantized(self) -> None:
        """Bug 2 (direct reproduction): a leading nn.Linear does not suppress quantized types.

        The old next()-based code would have taken the first (nn.Linear) type as
        the sole target, causing duplicate hooks on every nn.Linear and no hooks
        on the quantized down_proj.
        """
        model = _build_model_with_down_proj(nn.Linear(4, 4), _FakeQuantLinear())
        result = collect_quantized_down_proj_types(model)
        assert _FakeQuantLinear in result
        assert nn.Linear not in result

    def test_no_quantized_down_proj_returns_empty(self) -> None:
        """All-nn.Linear down_proj layers yield an empty list (hooks then skipped)."""
        model = _build_model_with_down_proj(nn.Linear(4, 4), nn.Linear(4, 4))
        assert collect_quantized_down_proj_types(model) == []

    def test_non_down_proj_names_excluded(self) -> None:
        """Name-filter isolation: modules without 'down_proj' in the name are excluded."""
        model = nn.Module()
        model.add_module("up_proj", _FakeQuantLinear())
        model.add_module("gate_proj", _FakeQuantLinear())
        assert collect_quantized_down_proj_types(model) == []

    def test_reregistration_leaves_exactly_one_hook_per_down_proj(self) -> None:
        """Pins *why* this helper drops nn.Linear while the loader helper keeps it.

        On the runner path the model came from ``RotatedModelConfig.load_model()``,
        which already hooked every ``nn.Linear`` ``down_proj``;
        ``apply_results_to_model`` only drops the hooks of the layers it
        replaced.  Including ``nn.Linear`` here would add a *second* hook to the
        untouched ones and Hadamard-transform their input twice.
        """
        # layers.0 stays nn.Linear (unquantized); layers.1 stands for a layer
        # already replaced by apply_results_to_model (so it carries no hook).
        model = _build_model_with_down_proj(nn.Linear(4, 4), _FakeQuantLinear())

        register_online_hadamard_hooks(model)  # RotatedModelConfig.load_model()
        register_online_hadamard_hooks(  # runner re-registration
            model, layers_cls=collect_quantized_down_proj_types(model)
        )

        for layer in model.model.layers:
            assert len(layer.mlp.down_proj._forward_pre_hooks) == 1

    def test_nn_linear_subclass_is_not_treated_as_already_hooked(self) -> None:
        """Only *plain* nn.Linear is excluded; subclasses still need their hook.

        The exclusion exists to skip layers ``load_model()`` already hooked, and
        that pass matches on exact type — so an ``nn.Linear`` subclass was never
        hooked.  An ``isinstance`` filter would drop it here too and leave it
        with no hook at all, the very failure this helper was written to prevent.
        """
        model = _build_model_with_down_proj(_LinearSubclassQuantLinear(4, 4))
        down_proj = model.model.layers[0].mlp.down_proj

        register_online_hadamard_hooks(model)  # RotatedModelConfig.load_model()
        assert not down_proj._forward_pre_hooks, "exact-type search must skip the subclass"

        layers_cls = collect_quantized_down_proj_types(model)
        assert _LinearSubclassQuantLinear in layers_cls
        register_online_hadamard_hooks(model, layers_cls=layers_cls)
        assert len(down_proj._forward_pre_hooks) == 1


# ── quantized_model_loader.py: collect_down_proj_types unit tests ──


class TestCollectDownProjTypes:
    """Pins the loader helper, including its intentional difference from runner."""

    def test_collects_down_proj_types(self) -> None:
        """down_proj types are derived from the module tree."""
        model = _build_model_with_down_proj(_FakeQuantLinear(), _FakeQuantLinear())
        assert _FakeQuantLinear in collect_down_proj_types(model)

    def test_does_not_filter_nn_linear(self) -> None:
        """Unlike the runner helper, nn.Linear is NOT filtered out.

        The loader receives an already-quantized model, so this difference is
        intentional; the test pins it so it is not accidentally removed.
        """
        model = _build_model_with_down_proj(nn.Linear(4, 4))
        assert nn.Linear in collect_down_proj_types(model)


# ── nested quantized layers (MDBF) ─────────────────────────────────


class TestNestedQuantizedDownProj:
    """Pins Bug 4: quantized ``down_proj`` layers built from submodules.

    ``MultipathMDBFLinear`` wraps an ``nn.ModuleList`` of per-pass
    ``MDBFLinear``, so the old filter put ``nn.ModuleList`` into ``layers_cls``.
    ``find_linear_layers`` then matched ``model.layers`` (itself an
    ``nn.ModuleList``) and returned before reaching any ``down_proj`` — zero
    hooks for *every* layer, silently.  The real-MDBF end-to-end case lives in
    ``test_mdbf_save_load_roundtrip.py``.
    """

    def test_quantized_collector_excludes_container_and_child_types(self) -> None:
        """Only the wrapper type is collected; its ModuleList and paths are not."""
        model = _build_model_with_down_proj(_NestedQuantLinear())
        result = collect_quantized_down_proj_types(model)
        assert result == [_NestedQuantLinear]

    def test_loader_collector_excludes_container_and_child_types(self) -> None:
        """The loader helper (no nn.Linear filter) must exclude descendants too."""
        model = _build_model_with_down_proj(_NestedQuantLinear())
        result = collect_down_proj_types(model)
        assert result == [_NestedQuantLinear]

    def test_hooks_registered_on_every_nested_down_proj(self) -> None:
        """Direct reproduction: a real ``nn.ModuleList`` of layers still gets hooks.

        This is the assertion the old code failed with ``0 == 2``.
        """
        model = _build_model_with_down_proj(_NestedQuantLinear(), _NestedQuantLinear())
        layers_cls = collect_quantized_down_proj_types(model)
        hooks = register_online_hadamard_hooks(model, layers_cls=layers_cls)
        assert len(hooks) == 2

    def test_nested_layer_does_not_suppress_hooks_on_flat_siblings(self) -> None:
        """A nested layer must not disable hooks on unrelated flat down_proj layers.

        The container leak was model-wide: one MDBF ``down_proj`` was enough to
        strip the hooks off every other layer as well.
        """
        model = _build_model_with_down_proj(_NestedQuantLinear(), _FakeQuantLinear())
        layers_cls = collect_quantized_down_proj_types(model)
        hooks = register_online_hadamard_hooks(model, layers_cls=layers_cls)
        assert len(hooks) == 2


# ── runner.py: create_quantized_model wiring (integration) ─────────


def _run_create_quantized_model(
    stub_model: nn.Module, *, has_additional_data: bool = True
) -> dict:
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

    def _fake_register(model: nn.Module, layers_cls: list, fp32_had: bool) -> list:
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

    def test_collected_types_reach_register_hooks(self) -> None:
        """The non-nn.Linear collected types are passed to register_online_hadamard_hooks."""
        model = _build_model_with_down_proj(nn.Linear(4, 4), _FakeQuantLinear())
        captured = _run_create_quantized_model(model)
        assert captured["called"]
        assert _FakeQuantLinear in captured["layers_cls"]
        assert nn.Linear not in captured["layers_cls"]

    def test_no_quantized_down_proj_skips_registration(self) -> None:
        """Empty collected types → register_online_hadamard_hooks is not called."""
        model = _build_model_with_down_proj(nn.Linear(4, 4))
        captured = _run_create_quantized_model(model)
        assert captured["called"] is False

    def test_not_rotated_skips_registration(self) -> None:
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

    def test_passes_model_derived_types_for_unknown_quant_method(self, tmp_path: Path) -> None:
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

        def _fake_register(model: nn.Module, layers_cls: list, fp32_had: bool) -> list:
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
