"""Unit tests for ``LPCDConfig``.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Usage:
    pytest tests/onecomp/lpcd/test_lpcd_config.py -v
"""

from dataclasses import fields

import pytest

from onecomp import LPCDConfig


class TestLPCDConfigDefaults:
    """Verify default values of ``LPCDConfig``."""

    def test_default_enable_flags(self):
        """By default only ``enable_residual`` is True."""
        cfg = LPCDConfig()
        assert cfg.enable_qk is False
        assert cfg.enable_vo is False
        assert cfg.enable_ud is False
        assert cfg.enable_residual is True

    def test_default_solver_params(self):
        """Default solver parameters."""
        cfg = LPCDConfig()
        assert cfg.alt_steps == 1
        assert 0.0 < cfg.perccorr <= 1.0
        assert 0.0 < cfg.percdamp < 1.0
        assert cfg.use_closed_form is True
        assert cfg.gd_steps >= 1
        assert cfg.gd_batch_size >= 1
        assert cfg.gd_base_lr > 0.0

    def test_default_device(self):
        """Default device is a CUDA device string."""
        cfg = LPCDConfig()
        assert isinstance(cfg.device, str)
        assert cfg.device.startswith("cuda")


class TestLPCDConfigCustomValues:
    """Verify custom values are preserved."""

    def test_all_flags_enabled(self):
        cfg = LPCDConfig(
            enable_qk=True,
            enable_vo=True,
            enable_ud=True,
            enable_residual=False,
        )
        assert cfg.enable_qk is True
        assert cfg.enable_vo is True
        assert cfg.enable_ud is True
        assert cfg.enable_residual is False

    def test_solver_params_overridable(self):
        cfg = LPCDConfig(
            alt_steps=3,
            perccorr=0.25,
            percdamp=0.05,
            use_closed_form=False,
            gd_steps=5,
            gd_batch_size=32,
            gd_base_lr=5e-5,
            device="cpu",
        )
        assert cfg.alt_steps == 3
        assert cfg.perccorr == pytest.approx(0.25)
        assert cfg.percdamp == pytest.approx(0.05)
        assert cfg.use_closed_form is False
        assert cfg.gd_steps == 5
        assert cfg.gd_batch_size == 32
        assert cfg.gd_base_lr == pytest.approx(5e-5)
        assert cfg.device == "cpu"

    def test_is_dataclass_with_expected_fields(self):
        """Catch accidental field removals/renames."""
        names = {f.name for f in fields(LPCDConfig)}
        expected = {
            "enable_qk",
            "enable_vo",
            "enable_ud",
            "enable_residual",
            "alt_steps",
            "perccorr",
            "percdamp",
            "use_closed_form",
            "gd_steps",
            "gd_batch_size",
            "gd_base_lr",
            "device",
        }
        assert expected.issubset(names), f"Missing fields: {expected - names}"


class TestLPCDConfigImport:
    """Verify ``LPCDConfig`` is exposed at the package top level."""

    def test_top_level_import(self):
        import onecomp

        assert hasattr(onecomp, "LPCDConfig")
        assert onecomp.LPCDConfig is LPCDConfig
