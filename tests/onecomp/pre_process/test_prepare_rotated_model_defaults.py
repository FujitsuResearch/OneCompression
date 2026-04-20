"""Tests for prepare_rotated_model default parameter values.

Lightweight tests that verify function signatures and default values
without loading any model.  No GPU required.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami

Usage::

    pytest tests/onecomp/pre_process/test_prepare_rotated_model_defaults.py -v
"""

import inspect

import pytest

from onecomp.pre_process.prepare_rotated_model import (
    _validate_prepare_rotated_model_params,
    prepare_rotated_model,
)


class TestPrepareRotatedModelDefaults:
    """Verify default parameter values declared in prepare_rotated_model.

    Tests follow the argument order of ``prepare_rotated_model``.
    """

    @pytest.fixture(autouse=True)
    def _sig(self):
        self.sig = inspect.signature(prepare_rotated_model)
        self.defaults = {
            name: p.default
            for name, p in self.sig.parameters.items()
            if p.default is not inspect.Parameter.empty
        }

    # --- defaults in prepare_rotated_model argument order ---

    def test_rotation_default(self):
        assert self.defaults["rotation"] is True

    def test_scaling_default(self):
        assert self.defaults["scaling"] is False

    def test_rotation_mode_default(self):
        assert self.defaults["rotation_mode"] == "random_hadamard"

    def test_scaling_mode_default(self):
        assert self.defaults["scaling_mode"] == "identity"

    def test_seed_default(self):
        assert self.defaults["seed"] == 0

    def test_enable_training_default(self):
        assert self.defaults["enable_training"] is True

    def test_calibration_config_default(self):
        assert self.defaults["calibration_config"] is None

    def test_wbits_default(self):
        assert self.defaults["wbits"] == 4

    def test_sym_default(self):
        assert self.defaults["sym"] is False

    def test_groupsize_default(self):
        assert self.defaults["groupsize"] == -1

    def test_mse_default(self):
        assert self.defaults["mse"] is False

    def test_norm_default(self):
        assert self.defaults["norm"] == 2.4

    def test_grid_default(self):
        assert self.defaults["grid"] == 100

    def test_fp32_had_default(self):
        assert self.defaults["fp32_had"] is False

    def test_use_sdpa_default(self):
        assert self.defaults["use_sdpa"] is False

    def test_training_args_override_default(self):
        assert self.defaults["training_args_override"] is None

    def test_maxshrink_not_exposed(self):
        assert "maxshrink" not in self.defaults

    # --- validation coverage ---

    def test_validate_covers_all_keyword_params(self):
        """_validate_prepare_rotated_model_params accepts all keyword params."""
        keyword_params = set(self.defaults.keys())

        validate_sig = inspect.signature(_validate_prepare_rotated_model_params)
        validated_params = set(validate_sig.parameters.keys())

        assert keyword_params == validated_params
