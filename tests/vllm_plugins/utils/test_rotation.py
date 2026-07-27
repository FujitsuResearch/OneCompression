"""Unit tests for rotation helpers used by the vLLM plugin.

Copyright 2025-2026 Fujitsu Ltd.
"""

from unittest.mock import MagicMock

import pytest
import torch

from onecomp.pre_process.rotation_utils import make_online_hadamard_hook
from vllm_plugins.utils import rotation

from ..conftest import _DummyLayer


def test_rotation_metadata_from_quant_config_parses_normal():
    meta = rotation.RotationMetadata.from_quant_config({"rotated": True, "fp32_had": True})

    assert meta.rotated is True and meta.fp32_had is True


def test_rotation_metadata_from_quant_config_falls_back_on_none():
    meta = rotation.RotationMetadata.from_quant_config(None)

    assert meta.rotated is False and meta.fp32_had is False


def test_rotation_metadata_from_quant_config_falls_back_on_non_dict():
    meta = rotation.RotationMetadata.from_quant_config("invalid")

    assert meta.rotated is False and meta.fp32_had is False


def test_rotation_metadata_from_quant_config_falls_back_on_missing_keys():
    meta = rotation.RotationMetadata.from_quant_config({})

    assert meta.rotated is False and meta.fp32_had is False


@pytest.mark.parametrize(
    ("rotated", "prefix", "expected_wrapped"),
    [
        (True, "model.layers.0.mlp.down_proj", True),
        (True, "mlp.down_proj", True),
        (True, "model.layers.0.mlp.gate_proj", False),
        (True, "model.layers.0.block_sparse_moe.experts.0.down_proj", False),
    ],
)
def test_maybe_wrap_rotation_method_dispatch_table(rotated, prefix, expected_wrapped):
    metadata = rotation.RotationMetadata(rotated=rotated, fp32_had=True)
    base_method = MagicMock()

    result = rotation.maybe_wrap_rotation_method(
        base_method,
        prefix=prefix,
        metadata=metadata,
    )

    assert metadata.requires_hadamard(prefix) is expected_wrapped
    if expected_wrapped:
        assert isinstance(result, rotation.RotatedLinearMethod)
    else:
        assert result is base_method


def test_v2_variant_is_registered_as_weight_loader_v2_supported():
    """The v2 wrapper's class name must be in vLLM's registry, otherwise vLLM
    would downgrade a wrapped v2-native base to the v1 weight loader."""
    if rotation.WEIGHT_LOADER_V2_SUPPORTED is None:
        pytest.skip("vLLM weight_loader_v2 registry not available in this build")
    assert "RotatedLinearMethodV2" in rotation.WEIGHT_LOADER_V2_SUPPORTED
    # The plain variant must stay unregistered so v1-only bases keep the v1 loader.
    assert "RotatedLinearMethod" not in rotation.WEIGHT_LOADER_V2_SUPPORTED


def test_maybe_wrap_uses_v2_variant_for_v2_native_base(monkeypatch):
    """A v2-native base must be wrapped by the v2-registered variant so vLLM
    keeps selecting weight_loader_v2 for the layer."""
    monkeypatch.setattr(
        rotation, "WEIGHT_LOADER_V2_SUPPORTED", ["GPTQMarlinLinearMethod"], raising=False
    )

    class GPTQMarlinLinearMethod:
        pass

    metadata = rotation.RotationMetadata(rotated=True, fp32_had=True)
    result = rotation.maybe_wrap_rotation_method(
        GPTQMarlinLinearMethod(),
        prefix="model.layers.0.mlp.down_proj",
        metadata=metadata,
    )

    assert isinstance(result, rotation.RotatedLinearMethodV2)


def test_maybe_wrap_uses_v1_variant_for_v1_only_base(monkeypatch):
    """A v1-only base must stay on the v1 loader: it is wrapped by the plain
    (unregistered) variant, whose class name is absent from the v2 registry."""
    monkeypatch.setattr(
        rotation, "WEIGHT_LOADER_V2_SUPPORTED", ["GPTQMarlinLinearMethod"], raising=False
    )

    class LegacyV1LinearMethod:
        pass

    metadata = rotation.RotationMetadata(rotated=True, fp32_had=True)
    result = rotation.maybe_wrap_rotation_method(
        LegacyV1LinearMethod(),
        prefix="model.layers.0.mlp.down_proj",
        metadata=metadata,
    )

    assert isinstance(result, rotation.RotatedLinearMethod)
    assert not isinstance(result, rotation.RotatedLinearMethodV2)
    assert "RotatedLinearMethod" not in rotation.WEIGHT_LOADER_V2_SUPPORTED


def test_maybe_wrap_rotation_method_keeps_plain_path_when_rotation_disabled():
    metadata = rotation.RotationMetadata(rotated=False, fp32_had=True)
    base_method = MagicMock()

    result = rotation.maybe_wrap_rotation_method(
        base_method,
        prefix="model.layers.0.mlp.down_proj",
        metadata=metadata,
    )

    assert metadata.requires_hadamard("model.layers.0.mlp.down_proj") is False
    assert result is base_method


def test_make_hadamard_forward_pre_hook_transforms_first_input_only(monkeypatch):
    calls = []

    def fake_apply_online_hadamard(x, *, fp32_had, cache_owner):
        calls.append((x.clone(), fp32_had, cache_owner))
        return x + 7

    hook = rotation.make_hadamard_forward_pre_hook(fp32_had=True)
    layer = MagicMock()
    x = torch.ones(2, 4)
    aux = torch.tensor([3.0])

    monkeypatch.setattr(rotation, "apply_online_hadamard", fake_apply_online_hadamard)

    result = hook(layer, (x, aux))

    assert torch.equal(result[0], x + 7)
    assert result[1] is aux
    assert len(calls) == 1
    assert calls[0][1] is True
    assert calls[0][2] is layer


def test_make_hadamard_forward_pre_hook_passes_through_empty_inputs():
    hook = rotation.make_hadamard_forward_pre_hook(fp32_had=False)

    result = hook(MagicMock(), ())

    assert result == ()


def test_create_weights_installs_prehook_once_for_tp1_path():
    base_method = MagicMock()
    method = rotation.RotatedLinearMethod(base_method, fp32_had=False)
    layer = _DummyLayer(input_is_parallel=False, tp_size=1)

    method.create_weights(layer, "w1")
    method.create_weights(layer, "w2")

    assert len(layer.registered_hooks) == 1
    assert getattr(layer, "_onecomp_hadamard_prehook_installed") is True
    assert base_method.create_weights.call_count == 2


def test_create_weights_and_process_weights_after_loading_share_idempotent_hook_installation():
    base_method = MagicMock()
    method = rotation.RotatedLinearMethod(base_method, fp32_had=False)
    layer = _DummyLayer(input_is_parallel=False, tp_size=1)

    method.create_weights(layer, "w")
    method.process_weights_after_loading(layer)

    assert len(layer.registered_hooks) == 1
    assert getattr(layer, "_onecomp_hadamard_prehook_installed") is True
    base_method.process_weights_after_loading.assert_called_once_with(layer)


def test_create_weights_skips_prehook_for_tp_parallel_path():
    base_method = MagicMock()
    method = rotation.RotatedLinearMethod(base_method, fp32_had=False)
    layer = _DummyLayer(input_is_parallel=True, tp_size=2, tp_rank=0)

    method.create_weights(layer, "w")

    assert len(layer.registered_hooks) == 0
    assert not hasattr(layer, "_onecomp_hadamard_prehook_installed")
    base_method.create_weights.assert_called_once_with(layer, "w")


def test_apply_uses_base_method_directly_when_tp_gather_not_required():
    base_method = MagicMock(return_value=torch.tensor([[5.0]]))
    base_method.apply.return_value = torch.tensor([[5.0]])
    method = rotation.RotatedLinearMethod(base_method, fp32_had=False)
    layer = _DummyLayer(input_is_parallel=False, tp_size=1)
    x = torch.tensor([[1.0, 2.0]])
    bias = torch.tensor([0.5])

    out = method.apply(layer, x, bias)

    assert torch.equal(out, torch.tensor([[5.0]]))
    base_method.apply.assert_called_once_with(layer, x, bias)


def test_apply_tp_path_uses_fake_collectives_and_local_shard(monkeypatch):
    base_method = MagicMock()
    base_method.apply.return_value = torch.tensor([[99.0]])
    method = rotation.RotatedLinearMethod(base_method, fp32_had=True)
    layer = _DummyLayer(input_is_parallel=True, tp_size=2, tp_rank=1)
    local_x = torch.tensor([[1.0, 2.0]])
    bias = torch.tensor([3.0])

    calls = {}

    def fake_all_gather(x, dim=-1):
        calls["all_gather"] = (x.clone(), dim)
        return torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    def fake_apply_online_hadamard(x, *, fp32_had, cache_owner):
        calls["hadamard"] = (x.clone(), fp32_had, cache_owner)
        return x + 10.0

    def fake_split(tensor, num_partitions, contiguous_split_chunks=False):
        calls["split"] = (tensor.clone(), num_partitions, contiguous_split_chunks)
        return (
            torch.tensor([[11.0, 12.0]]),
            torch.tensor([[13.0, 14.0]]),
        )

    monkeypatch.setattr(rotation, "tensor_model_parallel_all_gather", fake_all_gather)
    monkeypatch.setattr(rotation, "apply_online_hadamard", fake_apply_online_hadamard)
    monkeypatch.setattr(rotation, "split_tensor_along_last_dim", fake_split)

    out = method.apply(layer, local_x, bias)

    assert torch.equal(out, torch.tensor([[99.0]]))
    assert calls["all_gather"][1] == -1
    assert calls["hadamard"][1] is True
    assert calls["hadamard"][2] is layer
    assert calls["split"][1:] == (2, True)
    base_method.apply.assert_called_once()
    called_layer, called_x, called_bias = base_method.apply.call_args.args
    assert called_layer is layer
    assert torch.equal(called_x, torch.tensor([[13.0, 14.0]]))
    assert torch.equal(called_bias, bias)


@pytest.mark.parametrize(
    ("layer", "message"),
    [
        (
            _DummyLayer(input_is_parallel=True, tp_size=2, tp_rank=None),
            "requires tp_rank and tp_size",
        ),
        (
            _DummyLayer(input_is_parallel=True, tp_size=None, tp_rank=0),
            "requires tp_rank and tp_size",
        ),
        (_DummyLayer(input_is_parallel=True, tp_size=1, tp_rank=0), "requires tp_size > 1"),
        (
            _DummyLayer(input_is_parallel=True, tp_size=2, tp_rank=-1),
            "Invalid tensor-parallel metadata",
        ),
        (
            _DummyLayer(input_is_parallel=True, tp_size=2, tp_rank=2),
            "Invalid tensor-parallel metadata",
        ),
    ],
)
def test_get_tp_metadata_error_paths(layer, message):
    method = rotation.RotatedLinearMethod(MagicMock(), fp32_had=False)

    with pytest.raises(RuntimeError, match=message):
        method._get_tp_metadata(layer)


def test_apply_online_hadamard_caches_had_params_by_input_width(monkeypatch):
    cache_owner = MagicMock()
    get_calls = []
    mm_calls = []

    def fake_get_hadK(dim):
        get_calls.append(dim)
        return (f"had-{dim}", dim // 2)

    def fake_matmul(x, had_k, block_size):
        mm_calls.append((x.shape, had_k, block_size, x.dtype))
        return x + 1

    monkeypatch.setattr(rotation, "get_hadK", fake_get_hadK)
    monkeypatch.setattr(rotation, "matmul_hadU_cuda", fake_matmul)

    out1 = rotation.apply_online_hadamard(
        torch.ones(2, 4),
        fp32_had=False,
        cache_owner=cache_owner,
    )
    out2 = rotation.apply_online_hadamard(
        torch.full((1, 4), 2.0),
        fp32_had=False,
        cache_owner=cache_owner,
    )
    out3 = rotation.apply_online_hadamard(
        torch.ones(1, 8),
        fp32_had=False,
        cache_owner=cache_owner,
    )

    assert torch.equal(out1, torch.full((2, 4), 2.0))
    assert torch.equal(out2, torch.full((1, 4), 3.0))
    assert torch.equal(out3, torch.full((1, 8), 2.0))
    assert get_calls == [4, 8]
    assert mm_calls[0][1:] == ("had-4", 2, torch.float32)
    assert mm_calls[1][1:] == ("had-4", 2, torch.float32)
    assert mm_calls[2][1:] == ("had-8", 4, torch.float32)


def test_apply_online_hadamard_fp32_had_uses_fp32_compute_and_restores_input_dtype(monkeypatch):
    dtypes = []

    def fake_get_hadK(dim):
        return ("had", dim)

    def fake_matmul(x, had_k, block_size):
        dtypes.append(x.dtype)
        return x + 0.5

    monkeypatch.setattr(rotation, "get_hadK", fake_get_hadK)
    monkeypatch.setattr(rotation, "matmul_hadU_cuda", fake_matmul)

    x = torch.ones(1, 4, dtype=torch.float16)
    out = rotation.apply_online_hadamard(x, fp32_had=True, cache_owner=MagicMock())

    assert dtypes == [torch.float32]
    assert out.dtype == torch.float16
    assert torch.equal(out, torch.full((1, 4), 1.5, dtype=torch.float16))


def test_plugin_hadamard_matches_onecomp_hook_numerically(monkeypatch):
    def fake_get_hadK(dim):
        return ("had", dim // 2)

    def fake_matmul(x, had_k, block_size):
        return x * 2.0 + block_size

    monkeypatch.setattr(rotation, "get_hadK", fake_get_hadK)
    monkeypatch.setattr(rotation, "matmul_hadU_cuda", fake_matmul)
    monkeypatch.setattr("onecomp.pre_process.rotation_utils.get_hadK", fake_get_hadK)
    monkeypatch.setattr("onecomp.pre_process.rotation_utils.matmul_hadU_cuda", fake_matmul)

    x = torch.arange(8, dtype=torch.float16).reshape(2, 4)
    layer = MagicMock()
    hook = make_online_hadamard_hook("had", 2, fp32_had=False)

    plugin_out = rotation.apply_online_hadamard(x, fp32_had=False, cache_owner=MagicMock())
    hook_out = hook(layer, (x.clone(),))[0]

    assert torch.equal(plugin_out, hook_out)
