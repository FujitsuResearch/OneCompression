"""Threshold tests: TinyLlama, rotation+scaling, fp32_had=True.

12 cases: 4 rotation_modes x 3 scaling_modes.

Copyright 2025-2026 Fujitsu Ltd.

Usage::

    pytest tests/onecomp/pre_process/test_threshold_tinyllama_rotation_scaling_fp32.py -v
"""

import pytest
import torch

from onecomp.pre_process.prepare_rotated_model import (
    _VALID_ROTATION_MODES,
    _VALID_SCALING_MODES,
    prepare_rotated_model,
)

from .conftest import E2E_CALIB, PROMPT, TINYLLAMA_ID


def _cases():
    cases = []
    mid = TINYLLAMA_ID
    for rot_mode in _VALID_ROTATION_MODES:
        for scale_mode in _VALID_SCALING_MODES:
            cases.append(
                pytest.param(
                    mid,
                    True,
                    dict(
                        rotation=True,
                        scaling=True,
                        rotation_mode=rot_mode,
                        scaling_mode=scale_mode,
                    ),
                    id=f"tinyllama-rot_{rot_mode}-scale_{scale_mode}-fp32had",
                )
            )
    return cases


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestThresholdTinyLlamaRotScaleFP32:
    """Verify rotation+scaling preserves TinyLlama output (fp32_had=True)."""

    @pytest.mark.parametrize("model_id, fp32_had, kwargs", _cases())
    def test_output_within_threshold(self, model_id, fp32_had, kwargs, tmp_path):
        from onecomp import ModelConfig

        device = "cuda:0"
        model_config = ModelConfig(model_id=model_id, device=device)
        tokenizer = model_config.load_tokenizer()
        inputs = tokenizer(PROMPT, return_tensors="pt")

        original_model = model_config.load_model()
        original_model.eval()
        with torch.no_grad():
            logits_before = (
                original_model(**{k: v.to(device) for k, v in inputs.items()}).logits.float().cpu()
            )
        del original_model

        rot_dir = str(tmp_path / "rotated")
        rot_kwargs = dict(
            fp32_had=fp32_had,
            enable_training=True,
            calibration_config=E2E_CALIB,
            wbits=4,
            sym=False,
            groupsize=-1,
            **kwargs,
        )
        rot_kwargs["training_args_override"] = dict(
            output_dir=str(tmp_path / "train_output"),
            max_steps=2,
            per_device_train_batch_size=1,
        )
        rotated_config = prepare_rotated_model(
            model_config=model_config,
            save_directory=rot_dir,
            **rot_kwargs,
        )

        rotated_model = rotated_config.load_model()
        rotated_model.eval()
        with torch.no_grad():
            logits_after = (
                rotated_model(**{k: v.to(device) for k, v in inputs.items()}).logits.float().cpu()
            )
        del rotated_model

        max_diff = (logits_before - logits_after).abs().max().item()
        threshold = 0.5
        assert (
            max_diff < threshold
        ), f"max logits diff {max_diff:.6f} exceeds threshold {threshold}"
