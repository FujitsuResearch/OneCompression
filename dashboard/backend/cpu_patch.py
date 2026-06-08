"""Monkey-patch torch.cuda for environments without NVIDIA GPU.

gemlite unconditionally calls torch.cuda.get_device_properties() at import time,
which crashes without an NVIDIA driver. These patches return dummy objects on failure
so that onecomp can be imported on CPU/MPS machines. On CUDA machines the original
functions succeed and the patches are transparent no-ops.

Copyright 2025-2026 Fujitsu Ltd.
"""

from unittest.mock import MagicMock

import torch

_original_get_device_properties = torch.cuda.get_device_properties


def _patched_get_device_properties(device):
    try:
        return _original_get_device_properties(device)
    except (RuntimeError, AssertionError):
        mock = MagicMock()
        mock.multi_processor_count = 0
        mock.major = 0
        mock.minor = 0
        return mock


torch.cuda.get_device_properties = _patched_get_device_properties

_original_lazy_init = torch.cuda._lazy_init


def _patched_lazy_init():
    try:
        _original_lazy_init()
    except (RuntimeError, AssertionError):
        pass


torch.cuda._lazy_init = _patched_lazy_init
