"""Tests for _validate_prepare_rotated_model_params.

Lightweight parameter validation tests.  No GPU or model download required.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami

Usage::

    pytest tests/onecomp/pre_process/test_prepare_rotated_model_validation.py -v
"""

import pytest

from onecomp.calibration import CalibrationConfig
from onecomp.pre_process.prepare_rotated_model import (
    _validate_prepare_rotated_model_params,
)


class TestPrepareRotatedModelValidation:
    """Tests for _validate_prepare_rotated_model_params."""

    _VALID_DEFAULTS = dict(
        rotation=True,
        scaling=False,
        rotation_mode="random_hadamard",
        scaling_mode="identity",
        seed=0,
        enable_training=True,
        calibration_config=CalibrationConfig(),
        wbits=4,
        sym=False,
        groupsize=-1,
        mse=False,
        norm=2.4,
        grid=100,
        fp32_had=False,
        use_sdpa=False,
        training_args_override=None,
    )

    def _call(self, **overrides):
        calib_overrides = {}
        for key in (
            "calibration_dataset",
            "max_length",
            "num_calibration_samples",
            "calibration_strategy",
        ):
            if key in overrides:
                calib_key = "strategy" if key == "calibration_strategy" else key
                calib_overrides[calib_key] = overrides.pop(key)
        if calib_overrides:
            base = self._VALID_DEFAULTS.get("calibration_config", CalibrationConfig())
            overrides["calibration_config"] = CalibrationConfig(
                **{**base.__dict__, **calib_overrides}
            )
        params = {**self._VALID_DEFAULTS, **overrides}
        _validate_prepare_rotated_model_params(**params)

    # --- Valid parameters (should not raise) ---

    def test_valid_defaults(self):
        self._call()

    @pytest.mark.parametrize("mode", ["random_hadamard", "hadamard", "random", "identity"])
    def test_valid_rotation_modes(self, mode):
        self._call(rotation_mode=mode)

    @pytest.mark.parametrize("mode", ["identity", "random_ones", "random"])
    def test_valid_scaling_modes(self, mode):
        self._call(scaling_mode=mode)

    @pytest.mark.parametrize(
        "strategy",
        ["concat_chunk", "concat_chunk_align", "drop_head", "drop_rand"],
    )
    def test_valid_calibration_strategies(self, strategy):
        self._call(calibration_strategy=strategy)

    def test_valid_wbits_boundaries(self):
        self._call(wbits=1)
        self._call(wbits=64)

    def test_valid_groupsize_boundaries(self):
        self._call(groupsize=-1)
        self._call(groupsize=1)
        self._call(groupsize=128)

    def test_valid_mse_params(self):
        self._call(mse=True, grid=1, norm=1e-5)
        self._call(mse=True, grid=10000, norm=100.0)

    def test_mse_params_not_validated_when_false(self):
        self._call(mse=False, grid=0, norm=0.0)

    # --- Invalid parameters (should raise ValueError) ---

    def test_invalid_rotation_mode(self):
        with pytest.raises(ValueError, match="rotation_mode"):
            self._call(rotation_mode="invalid")

    def test_invalid_scaling_mode(self):
        with pytest.raises(ValueError, match="scaling_mode"):
            self._call(scaling_mode="invalid")

    def test_invalid_calibration_strategy(self):
        with pytest.raises(ValueError, match="calibration_strategy"):
            self._call(calibration_strategy="invalid")

    def test_invalid_wbits_zero(self):
        with pytest.raises(ValueError, match="wbits"):
            self._call(wbits=0)

    def test_invalid_wbits_too_large(self):
        with pytest.raises(ValueError, match="wbits"):
            self._call(wbits=65)

    def test_invalid_groupsize_zero(self):
        with pytest.raises(ValueError, match="groupsize"):
            self._call(groupsize=0)

    def test_invalid_groupsize_negative(self):
        with pytest.raises(ValueError, match="groupsize"):
            self._call(groupsize=-2)

    def test_invalid_num_calibration_samples_zero(self):
        with pytest.raises(ValueError, match="num_calibration_samples"):
            self._call(num_calibration_samples=0)

    def test_invalid_max_length_zero(self):
        with pytest.raises(ValueError, match="max_length"):
            self._call(max_length=0)

    def test_invalid_seed_negative(self):
        with pytest.raises(ValueError, match="seed"):
            self._call(seed=-1)

    def test_invalid_grid_zero_when_mse(self):
        with pytest.raises(ValueError, match="grid"):
            self._call(mse=True, grid=0)

    def test_invalid_norm_zero_when_mse(self):
        with pytest.raises(ValueError, match="norm"):
            self._call(mse=True, norm=0.0)

    def test_invalid_norm_negative_when_mse(self):
        with pytest.raises(ValueError, match="norm"):
            self._call(mse=True, norm=-1.0)

    def test_multiple_invalid_params(self):
        with pytest.raises(ValueError) as exc_info:
            self._call(wbits=0, groupsize=-2, rotation_mode="bad")
        msg = str(exc_info.value)
        assert "wbits" in msg
        assert "groupsize" in msg
        assert "rotation_mode" in msg
