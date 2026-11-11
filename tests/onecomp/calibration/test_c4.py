"""Tests for C4 calibration text selection without downloading data."""

from unittest.mock import patch

import datasets
import torch

from onecomp.calibration.c4 import (
    load_c4_for_n_samples_min_length,
    load_c4_texts,
)


class _LengthTokenizer:
    def __call__(self, text, return_tensors=None):
        del return_tensors
        length = int(text.split(":", maxsplit=1)[0])
        return {"input_ids": torch.arange(length).unsqueeze(0)}


def _dataset(texts):
    return datasets.Dataset.from_dict({"text": texts})


class TestC4DocumentShuffle:
    def test_default_preserves_source_order(self):
        data = _dataset([f"text-{index}" for index in range(10)])

        with patch("onecomp.calibration.c4._get_c4_train", return_value=data):
            texts = load_c4_texts(None, 4)

        assert texts == ["text-0", "text-1", "text-2", "text-3"]

    def test_seeded_shuffle_is_reproducible_and_changes_selection(self):
        data = _dataset([f"text-{index}" for index in range(20)])

        with patch("onecomp.calibration.c4._get_c4_train", return_value=data):
            first = load_c4_texts(None, 5, seed=42, shuffle_documents=True)
            repeated = load_c4_texts(None, 5, seed=42, shuffle_documents=True)
            other_seed = load_c4_texts(None, 5, seed=99, shuffle_documents=True)

        assert first == repeated
        assert first != other_seed
        assert first != [f"text-{index}" for index in range(5)]

    def test_shuffle_precedes_minimum_length_filter(self):
        data = _dataset(["2:short-0", "8:long-0", "3:short-1", "9:long-1", "10:long-2"])

        with patch("onecomp.calibration.c4._get_c4_train", return_value=data):
            texts = load_c4_for_n_samples_min_length(
                _LengthTokenizer(),
                num_samples=2,
                min_length=8,
                seed=7,
                shuffle_documents=True,
            )

        assert len(texts) == 2
        assert all(int(text.split(":", maxsplit=1)[0]) >= 8 for text in texts)
