"""
Tests for calibration_data_loader.prepare_calibration_dataset routing logic.

Uses mocks to avoid downloading datasets.

Copyright 2025-2026 Fujitsu Ltd.
"""

import logging
from unittest.mock import patch, MagicMock

import pytest
import torch

from onecomp.calibration import CalibrationConfig
from onecomp.calibration.calibration_data_loader import (
    prepare_calibration_dataset,
    _KNOWN_DATASET_NAMES,
)
from onecomp.utils import add_model_specific_inputs
from onecomp.calibration.chunking import _VALID_CALIBRATION_STRATEGIES


class TestRouting:
    """Verify that prepare_calibration_dataset dispatches to the correct loader."""

    @patch("onecomp.calibration.calibration_data_loader.c4.prepare_calibration_data")
    def test_none_routes_to_c4(self, mock_c4):
        fake_result = {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }
        mock_c4.return_value = fake_result
        tokenizer = MagicMock()

        result = prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            calibration_config=CalibrationConfig(),
            model=MagicMock(),
        )
        mock_c4.assert_called_once()
        assert result is fake_result

    @patch("onecomp.calibration.calibration_data_loader.c4.prepare_calibration_data")
    def test_c4_explicit_routes_to_c4(self, mock_c4):
        mock_c4.return_value = {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }
        tokenizer = MagicMock()

        prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            calibration_config=CalibrationConfig(calibration_dataset="c4"),
            model=MagicMock(),
        )
        mock_c4.assert_called_once()

    @patch("onecomp.calibration.calibration_data_loader.c4.prepare_calibration_data")
    def test_c4_case_insensitive(self, mock_c4):
        mock_c4.return_value = {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }
        tokenizer = MagicMock()

        prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            calibration_config=CalibrationConfig(calibration_dataset="C4"),
            model=MagicMock(),
        )
        mock_c4.assert_called_once()

    @patch("onecomp.calibration.calibration_data_loader.wikitext.prepare_calibration_data")
    def test_wikitext2_routes_to_wikitext(self, mock_wt):
        mock_wt.return_value = {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }
        tokenizer = MagicMock()

        prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            calibration_config=CalibrationConfig(calibration_dataset="wikitext2"),
            model=MagicMock(),
        )
        mock_wt.assert_called_once()

    @patch("os.path.exists", return_value=True)
    @patch("onecomp.calibration.calibration_data_loader.custom.prepare_calibration_data")
    def test_local_path_routes_to_custom(self, mock_custom, mock_exists):
        mock_custom.return_value = {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }
        tokenizer = MagicMock()

        prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            calibration_config=CalibrationConfig(
                calibration_dataset="/data/my_corpus.txt",
            ),
            model=MagicMock(),
        )
        mock_custom.assert_called_once()

    @patch("os.path.exists", return_value=False)
    @patch("onecomp.calibration.calibration_data_loader._load_from_hub")
    def test_unknown_name_routes_to_hub(self, mock_hub, mock_exists):
        mock_hub.return_value = {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }
        tokenizer = MagicMock()

        prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            calibration_config=CalibrationConfig(
                calibration_dataset="username/my-dataset",
            ),
            model=MagicMock(),
        )
        mock_hub.assert_called_once()


class TestInvalidStrategy:
    """Verify that an unknown strategy raises ValueError."""

    def test_invalid_strategy_raises(self):
        tokenizer = MagicMock()
        with pytest.raises(ValueError, match="Unknown calibration strategy"):
            prepare_calibration_dataset(
                tokenizer=tokenizer,
                device=torch.device("cpu"),
                calibration_config=CalibrationConfig(
                    strategy="nonexistent_strategy",
                ),
                model=MagicMock(),
            )


class TestFinalizeCalibrationInputs:
    """Verify that add_model_specific_inputs adds model-specific fields."""

    def _make_inputs(self):
        return {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }

    def _make_model(self, model_type):
        model = MagicMock()
        model.config.model_type = model_type
        return model

    def test_gemma4_adds_mm_token_type_ids(self):
        inputs = self._make_inputs()
        model = self._make_model("gemma4")

        result = add_model_specific_inputs(inputs, model)

        assert "mm_token_type_ids" in result
        assert result["mm_token_type_ids"].shape == result["input_ids"].shape
        assert result["mm_token_type_ids"].dtype == torch.long
        assert (result["mm_token_type_ids"] == 0).all()

    def test_non_gemma4_does_not_add_mm_token_type_ids(self):
        inputs = self._make_inputs()
        model = self._make_model("llama")

        result = add_model_specific_inputs(inputs, model)

        assert "mm_token_type_ids" not in result

    @patch("onecomp.calibration.calibration_data_loader.c4.prepare_calibration_data")
    def test_gemma4_end_to_end(self, mock_c4):
        """mm_token_type_ids is present after full prepare_calibration_dataset."""
        mock_c4.return_value = self._make_inputs()
        model = self._make_model("gemma4")

        result = prepare_calibration_dataset(
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            calibration_config=CalibrationConfig(),
            model=model,
        )

        assert "mm_token_type_ids" in result
        assert result["mm_token_type_ids"].shape == result["input_ids"].shape


class TestKnownConstants:
    """Verify module-level constants."""

    def test_known_dataset_names(self):
        assert "c4" in _KNOWN_DATASET_NAMES
        assert "wikitext2" in _KNOWN_DATASET_NAMES

    def test_valid_strategies(self):
        expected = {"concat_chunk", "concat_chunk_align", "concat_rand", "drop_head", "drop_rand"}
        assert set(_VALID_CALIBRATION_STRATEGIES) == expected
