"""Tests for WikiText calibration document ordering."""

from unittest.mock import MagicMock, patch

import datasets
import torch

from onecomp.calibration.wikitext import load_wikitext_texts, prepare_calibration_data


class TestWikiTextDocumentShuffle:
    def test_default_preserves_non_empty_source_order(self):
        data = datasets.Dataset.from_dict({"text": ["first", "", "second", "third"]})

        with patch(
            "onecomp.calibration.wikitext.load_or_prepare",
            return_value={"train": data},
        ):
            texts = load_wikitext_texts()

        assert texts == ["first", "second", "third"]

    def test_seeded_shuffle_is_reproducible(self):
        data = datasets.Dataset.from_dict({"text": [f"text-{index}" for index in range(20)]})

        with patch(
            "onecomp.calibration.wikitext.load_or_prepare",
            return_value={"train": data},
        ):
            first = load_wikitext_texts(seed=42, shuffle_documents=True)
            repeated = load_wikitext_texts(seed=42, shuffle_documents=True)
            other_seed = load_wikitext_texts(seed=99, shuffle_documents=True)

        assert first == repeated
        assert first != other_seed

    @patch("onecomp.calibration.wikitext.prepare_from_texts")
    @patch("onecomp.calibration.wikitext.load_wikitext_texts")
    def test_drop_strategy_fallback_is_preserved(self, mock_load, mock_prepare):
        mock_load.return_value = ["text"]
        mock_prepare.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        prepare_calibration_data(
            MagicMock(),
            torch.device("cpu"),
            32,
            4,
            "drop_rand",
            42,
            shuffle_documents=True,
        )

        assert mock_load.call_args.kwargs["shuffle_documents"] is True
        assert mock_prepare.call_args.args[5] == "concat_rand"
