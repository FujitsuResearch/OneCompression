"""Shared constants and fixtures for pre_process tests.

Copyright 2025-2026 Fujitsu Ltd.
"""

import gc

import pytest
import torch

from onecomp.calibration import CalibrationConfig
from onecomp.pre_process.prepare_rotated_model import (
    _VALID_ROTATION_MODES,
    _VALID_SCALING_MODES,
    prepare_rotated_model,
)

TINYLLAMA_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
QWEN3_ID = "Qwen/Qwen3-0.6B"

E2E_CALIB = CalibrationConfig(num_calibration_samples=4, max_length=64)

E2E_FAST = dict(
    calibration_config=E2E_CALIB,
    wbits=4,
    sym=False,
    groupsize=-1,
)

PROMPT = "Hello, world!"


@pytest.fixture(autouse=True)
def cleanup():
    """Release GPU memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
