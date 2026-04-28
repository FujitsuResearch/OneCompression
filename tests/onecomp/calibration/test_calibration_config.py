"""
Tests for CalibrationConfig dataclass.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from onecomp.calibration import CalibrationConfig


class TestCalibrationConfigDefaults:
    """Verify that CalibrationConfig has the expected default values."""

    def test_default_values(self):
        cfg = CalibrationConfig()
        assert cfg.calibration_dataset == "c4"
        assert cfg.max_length == 2048
        assert cfg.num_calibration_samples == 512
        assert cfg.strategy == "drop_rand"
        assert cfg.seed == 0
        assert cfg.batch_size is None
        assert cfg.num_layers_per_group == 7
        assert cfg.text_key == "text"
        assert cfg.use_quality_filter is False
        assert cfg.max_documents == 10000


class TestCalibrationConfigOverride:
    """Verify that field values can be overridden."""

    def test_override_all_fields(self):
        cfg = CalibrationConfig(
            calibration_dataset="wikitext2",
            max_length=2048,
            num_calibration_samples=256,
            strategy="concat_chunk",
            seed=42,
            batch_size=64,
            num_layers_per_group=14,
            text_key="content",
            use_quality_filter=True,
            max_documents=5000,
        )
        assert cfg.calibration_dataset == "wikitext2"
        assert cfg.max_length == 2048
        assert cfg.num_calibration_samples == 256
        assert cfg.strategy == "concat_chunk"
        assert cfg.seed == 42
        assert cfg.batch_size == 64
        assert cfg.num_layers_per_group == 14
        assert cfg.text_key == "content"
        assert cfg.use_quality_filter is True
        assert cfg.max_documents == 5000

    def test_partial_override(self):
        cfg = CalibrationConfig(max_length=1024, seed=99)
        assert cfg.max_length == 1024
        assert cfg.seed == 99
        assert cfg.calibration_dataset == "c4"
        assert cfg.num_calibration_samples == 512


class TestCalibrationConfigTopLevelExport:
    """Verify CalibrationConfig is importable from top-level onecomp."""

    def test_import_from_onecomp(self):
        from onecomp import CalibrationConfig as CC

        assert CC is CalibrationConfig
