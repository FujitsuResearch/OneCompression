"""Copyright 2025-2026 Fujitsu Ltd.

Tests for the rotation vLLM plugin integration.

Covers items from the test spec not present in test_rotation.py:
  - is_online_hadamard_target predicate (5 parametrized cases)
  - register_weight_loader_v2_supported_method dispatch registration
  - RotatedLinearMethod.process_weights_after_loading delegation
  - DbfConfig rotation metadata parsing and quant_method wrapping
"""

from unittest.mock import MagicMock

import pytest
import torch

from onecomp.pre_process.rotation_utils import is_online_hadamard_target
from vllm_plugins.utils import rotation
from vllm_plugins.utils.rotation import RotatedLinearMethod, RotationMetadata

try:
    from vllm.model_executor.layers.linear import LinearBase

    from vllm_plugins.dbf.vllm_plugin import DbfConfig, DBFLinearMethod

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

from .conftest import _DummyLayer

# ---------------------------------------------------------------------------
# is_online_hadamard_target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mlp.down_proj", True),
        ("model.layers.0.mlp.down_proj", True),
        ("model.layers.0.mlp.gate_proj", False),
        ("model.layers.0.block_sparse_moe.experts.0.down_proj", False),
        ("down_proj", False),
    ],
)
def test_is_online_hadamard_target(name, expected):
    assert is_online_hadamard_target(name) is expected


# ---------------------------------------------------------------------------
# register_weight_loader_v2_supported_method
# ---------------------------------------------------------------------------


def test_rotated_method_preserves_vllm_weight_loader_v2_dispatch(monkeypatch):
    supported = []
    monkeypatch.setattr(rotation, "WEIGHT_LOADER_V2_SUPPORTED", supported)
    monkeypatch.setattr(rotation, "_vllm_register_weight_loader_v2_supported_method", None)

    class _FakeLinearMethod:
        pass

    result = rotation.register_weight_loader_v2_supported_method(_FakeLinearMethod)

    assert "_FakeLinearMethod" in supported
    assert result is _FakeLinearMethod

    # Idempotent: second call must not duplicate the entry.
    rotation.register_weight_loader_v2_supported_method(_FakeLinearMethod)
    assert supported.count("_FakeLinearMethod") == 1


# ---------------------------------------------------------------------------
# RotatedLinearMethod.process_weights_after_loading
# ---------------------------------------------------------------------------


def test_process_weights_after_loading_delegates_to_base_method():
    base_method = MagicMock()
    method = RotatedLinearMethod(base_method, fp32_had=False)
    layer = _DummyLayer(input_is_parallel=False, tp_size=1)

    method.process_weights_after_loading(layer)

    base_method.process_weights_after_loading.assert_called_once_with(layer)


def test_process_weights_after_loading_is_noop_when_base_has_no_process():
    base_method = MagicMock(spec=[])
    method = RotatedLinearMethod(base_method, fp32_had=False)
    layer = _DummyLayer(input_is_parallel=False, tp_size=1)

    method.process_weights_after_loading(layer)  # must not raise


# ---------------------------------------------------------------------------
# DbfConfig rotation metadata parsing and quant_method wrapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestDbfConfigRotation:
    """Verify DbfConfig correctly parses rotation metadata and wraps down_proj layers."""

    def test_from_config_parses_rotation_metadata(self):
        config = DbfConfig.from_config(
            {
                "quantization_bits": [],
                "rotated": True,
                "fp32_had": True,
            }
        )

        assert config.rotation_metadata.rotated is True
        assert config.rotation_metadata.fp32_had is True

    def test_from_config_defaults_rotation_metadata_when_absent(self):
        config = DbfConfig.from_config({"quantization_bits": []})

        assert config.rotation_metadata.rotated is False
        assert config.rotation_metadata.fp32_had is False

    def test_get_quant_method_wraps_down_proj_with_rotated_linear_method(self):
        config = DbfConfig(
            quantization_bits=[],
            rotation_metadata=RotationMetadata(rotated=True, fp32_had=True),
        )
        layer = MagicMock(spec=LinearBase)

        method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        assert isinstance(method, RotatedLinearMethod)

    def test_get_quant_method_does_not_wrap_gate_proj(self):
        config = DbfConfig(
            quantization_bits=[],
            rotation_metadata=RotationMetadata(rotated=True, fp32_had=True),
        )
        layer = MagicMock(spec=LinearBase)

        method = config.get_quant_method(layer, "model.layers.0.mlp.gate_proj")

        assert not isinstance(method, RotatedLinearMethod)

    def test_get_quant_method_keeps_plain_path_when_rotation_disabled(self):
        config = DbfConfig(
            quantization_bits=[],
            rotation_metadata=RotationMetadata(rotated=False, fp32_had=True),
        )
        layer = MagicMock(spec=LinearBase)

        method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        assert not isinstance(method, RotatedLinearMethod)

    def test_get_quant_method_returns_none_for_non_linear_layer(self):
        config = DbfConfig(
            quantization_bits=[],
            rotation_metadata=RotationMetadata(rotated=True, fp32_had=True),
        )
        layer = MagicMock()  # Not a LinearBase instance

        method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        assert method is None

    def test_get_quant_method_wraps_dbf_linear_method_for_quantized_down_proj(self):
        # quantization_bits=[] exercises the Unquantized path; this test uses a
        # real entry so that DBFLinearMethod (not UnquantizedLinearMethod) is wrapped.
        quantization_bits = [{"mlp.down_proj": {"bits": 1.5, "method": "dbf"}}]
        config = DbfConfig(
            quantization_bits=quantization_bits,
            rotation_metadata=RotationMetadata(rotated=True, fp32_had=True),
        )
        layer = MagicMock(spec=LinearBase)

        method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        assert isinstance(method, RotatedLinearMethod)
        assert isinstance(method.base_method, DBFLinearMethod)
        assert layer._dbf_mod_cfg == {"bits": 1.5, "method": "dbf"}
        assert layer._dbf_prefix == "model.layers.0.mlp.down_proj"

    def test_prehook_is_installed_on_down_proj_after_process_weights(self):
        # Verify that _onecomp_hadamard_prehook_installed is set after
        # process_weights_after_loading is called through the DBF quantized path.
        quantization_bits = [{"mlp.down_proj": {"bits": 1.5, "method": "dbf"}}]
        config = DbfConfig(
            quantization_bits=quantization_bits,
            rotation_metadata=RotationMetadata(rotated=True, fp32_had=True),
        )
        spec_layer = MagicMock(spec=LinearBase)
        method = config.get_quant_method(spec_layer, "model.layers.0.mlp.down_proj")
        assert isinstance(method, RotatedLinearMethod)

        dummy_layer = _DummyLayer(input_is_parallel=False, tp_size=1)
        method.process_weights_after_loading(dummy_layer)

        assert getattr(dummy_layer, "_onecomp_hadamard_prehook_installed", False) is True
        assert len(dummy_layer.registered_hooks) == 1

    def test_create_weights_raises_for_tensor_parallel_size_greater_than_one(self, monkeypatch):
        # TP > 1 is unsupported; DBFLinearMethod.create_weights must raise immediately.
        import vllm_plugins.dbf.vllm_plugin as plugin_mod

        monkeypatch.setattr(plugin_mod, "get_tensor_model_parallel_world_size", lambda: 2)

        config = DbfConfig(quantization_bits=[], rotation_metadata=RotationMetadata())
        method = DBFLinearMethod(config)

        with pytest.raises(ValueError, match="tensor_parallel_size=1"):
            method.create_weights(
                layer=MagicMock(),
                input_size_per_partition=64,
                output_partition_sizes=[32],
                input_size=64,
                output_size=32,
                params_dtype=torch.float16,
            )
