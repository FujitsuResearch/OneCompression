"""
Unit tests for onecomp.utils.device.

Copyright 2025-2026 Fujitsu Ltd.

"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "device",
    _REPO_ROOT / "onecomp" / "utils" / "device.py",
)
_device_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_device_mod)

cleanup_memory = _device_mod.cleanup_memory
empty_cache = _device_mod.empty_cache
get_default_device = _device_mod.get_default_device
is_mps_device = _device_mod.is_mps_device


@pytest.mark.parametrize(
    "device,expected",
    [
        (None, False),
        ("cuda", False),
        ("cuda:0", False),
        ("cpu", False),
        ("mps", True),
        ("mps:0", True),
        (torch.device("mps"), True),
        (torch.device("cpu"), False),
    ],
)
def test_is_mps_device(device, expected):
    assert is_mps_device(device) is expected


@patch.object(_device_mod.torch.cuda, "is_available", return_value=True)
@patch.object(_device_mod, "_has_mps_backend", return_value=True)
@patch.object(_device_mod.torch.backends.mps, "is_available", return_value=True)
def test_get_default_device_prefers_cuda(*_):
    assert get_default_device() == torch.device("cuda")


@patch.object(_device_mod.torch.cuda, "is_available", return_value=False)
@patch.object(_device_mod, "_has_mps_backend", return_value=True)
@patch.object(_device_mod.torch.backends.mps, "is_available", return_value=True)
def test_get_default_device_selects_mps(*_):
    assert get_default_device() == torch.device("mps")


@patch.object(_device_mod.torch.cuda, "is_available", return_value=False)
@patch.object(_device_mod, "_has_mps_backend", return_value=False)
def test_get_default_device_falls_back_to_cpu_no_mps_backend(*_):
    assert get_default_device() == torch.device("cpu")


@patch.object(_device_mod.torch.cuda, "is_available", return_value=False)
@patch.object(_device_mod, "_has_mps_backend", return_value=True)
@patch.object(_device_mod.torch.backends.mps, "is_available", return_value=False)
def test_get_default_device_falls_back_to_cpu_mps_unavailable(*_):
    assert get_default_device() == torch.device("cpu")


@patch.object(_device_mod.torch.cuda, "empty_cache")
def test_empty_cache_cuda(mock_cuda_empty):
    empty_cache("cuda")
    mock_cuda_empty.assert_called_once()


@patch.object(_device_mod.torch.cuda, "empty_cache")
def test_empty_cache_cuda_device_object(mock_cuda_empty):
    empty_cache(torch.device("cuda:0"))
    mock_cuda_empty.assert_called_once()


@patch.object(_device_mod, "_has_mps_backend", return_value=True)
def test_empty_cache_mps(_has_mps):
    mock_mps_empty = MagicMock()
    with patch.object(_device_mod.torch, "mps") as mock_mps:
        mock_mps.empty_cache = mock_mps_empty
        empty_cache("mps")
    mock_mps_empty.assert_called_once()


@patch.object(_device_mod.torch.cuda, "empty_cache")
@patch.object(_device_mod, "_has_mps_backend", return_value=True)
def test_empty_cache_cpu_is_no_op(_has_mps, mock_cuda_empty):
    mock_mps_empty = MagicMock()
    with patch.object(_device_mod.torch, "mps") as mock_mps:
        mock_mps.empty_cache = mock_mps_empty
        empty_cache("cpu")
    mock_cuda_empty.assert_not_called()
    mock_mps_empty.assert_not_called()


@patch.object(_device_mod.torch.cuda, "empty_cache")
@patch.object(_device_mod, "_has_mps_backend", return_value=False)
def test_empty_cache_mps_without_backend_is_no_op(_has_mps, mock_cuda_empty):
    empty_cache("mps")
    mock_cuda_empty.assert_not_called()


@patch.object(_device_mod, "_has_mps_backend", return_value=True)
def test_empty_cache_mps_without_empty_cache_fn_is_no_op(_has_mps):
    with patch.object(_device_mod.torch, "mps", None):
        empty_cache("mps")


@patch.object(_device_mod, "get_default_device", return_value=torch.device("cuda"))
@patch.object(_device_mod.torch.cuda, "empty_cache")
def test_empty_cache_none_uses_default_device(mock_cuda_empty, _default):
    empty_cache(None)
    mock_cuda_empty.assert_called_once()


@patch.object(_device_mod.gc, "collect")
@patch.object(_device_mod.torch.cuda, "empty_cache")
def test_cleanup_memory_collects_then_empties_cache(mock_cuda_empty, mock_collect):
    cleanup_memory("cuda")
    mock_collect.assert_called_once()
    mock_cuda_empty.assert_called_once()


@patch.object(_device_mod.gc, "collect")
@patch.object(_device_mod.torch.cuda, "empty_cache")
@patch.object(_device_mod, "_has_mps_backend", return_value=True)
def test_cleanup_memory_collects_on_cpu(_has_mps, mock_cuda_empty, mock_collect):
    cleanup_memory("cpu")
    mock_collect.assert_called_once()
    mock_cuda_empty.assert_not_called()


@patch.object(_device_mod, "get_default_device", return_value=torch.device("cuda"))
@patch.object(_device_mod.gc, "collect")
@patch.object(_device_mod.torch.cuda, "empty_cache")
def test_cleanup_memory_none_uses_default_device(mock_cuda_empty, mock_collect, _default):
    cleanup_memory(None)
    mock_collect.assert_called_once()
    mock_cuda_empty.assert_called_once()


def test_empty_cache_does_not_collect():
    """empty_cache stays GC-free: hot loops call it per layer/batch."""
    with patch.object(_device_mod.gc, "collect") as mock_collect:
        empty_cache("cpu")
    mock_collect.assert_not_called()
