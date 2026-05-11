"""E2E smoke tests for prepare_rotated_model with TinyLlama.

Requires CUDA and model download.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami

Usage::

    pytest tests/onecomp/pre_process/test_prepare_rotated_model_e2e_tinyllama.py -v -s --log-cli-level=INFO
"""

import os

import pytest
import torch

from onecomp.pre_process.prepare_rotated_model import (
    _VALID_ROTATION_MODES,
    _VALID_SCALING_MODES,
    prepare_rotated_model,
)

from .conftest import E2E_FAST, TINYLLAMA_ID


def _e2e_cases():
    cases = []
    tag, mid = "tinyllama", TINYLLAMA_ID
    for mode in _VALID_ROTATION_MODES:
        cases.append(
            pytest.param(
                mid,
                dict(rotation=True, scaling=False, rotation_mode=mode),
                id=f"{tag}-rot_{mode}",
            )
        )
    for mode in _VALID_SCALING_MODES:
        cases.append(
            pytest.param(
                mid,
                dict(rotation=True, scaling=True, scaling_mode=mode),
                id=f"{tag}-rot_scale_{mode}",
            )
        )
    cases.append(
        pytest.param(
            mid,
            dict(rotation=False, scaling=False),
            id=f"{tag}-no_rot_no_scale",
        )
    )
    cases.append(
        pytest.param(
            mid,
            dict(rotation=False, scaling=True),
            id=f"{tag}-no_rot_scale",
        )
    )
    return cases


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPrepareRotatedModelE2ETinyLlama:
    """E2E smoke tests for prepare_rotated_model with TinyLlama-1.1B."""

    @pytest.mark.parametrize("model_id, kwargs", _e2e_cases())
    def test_prepare_completes_and_saves(self, model_id, kwargs, tmp_path):
        from onecomp import ModelConfig
        from onecomp.rotated_model_config import RotatedModelConfig

        model_config = ModelConfig(model_id=model_id, device="cuda:0")
        save_dir = str(tmp_path / "rotated")

        merged = {**E2E_FAST, **kwargs}
        need_train = merged.get("rotation", True) or merged.get("scaling", False)
        if need_train:
            merged["training_args_override"] = dict(
                output_dir=str(tmp_path / "train_output"),
                max_steps=2,
                per_device_train_batch_size=1,
            )

        result = prepare_rotated_model(
            model_config=model_config,
            save_directory=save_dir,
            **merged,
        )

        assert isinstance(result, RotatedModelConfig)
        assert result.path == save_dir
        assert os.path.isfile(os.path.join(save_dir, "config.json"))
