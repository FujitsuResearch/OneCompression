"""
Tests for calibration chunking strategies.

Uses a lightweight tokenizer (GPT-2) to avoid heavy model downloads.

Copyright 2025-2026 Fujitsu Ltd.
"""

import logging

import pytest
import torch
from transformers import AutoTokenizer

from onecomp.calibration.chunking import (
    prepare_from_texts,
    chunk_single_document,
    chunk_concat,
    chunk_concat_rand,
    _VALID_CALIBRATION_STRATEGIES,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def long_texts():
    """Generate texts long enough for chunking (each ~200+ tokens)."""
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning is transforming modern computing and "
        "enabling new applications across many industries. "
    )
    return [base * 20 for _ in range(10)]


@pytest.fixture(scope="module")
def short_texts():
    """Texts too short for single-document strategies with max_length=64."""
    return ["Hello world.", "Short sentence.", "Another brief text."]


LOGGER = logging.getLogger(__name__)


class TestPrepareFromTexts:
    """Test the unified prepare_from_texts entry point."""

    @pytest.mark.parametrize("strategy", list(_VALID_CALIBRATION_STRATEGIES))
    def test_all_strategies_produce_correct_shape(self, tokenizer, long_texts, strategy):
        max_length = 32
        num_samples = 4
        result = prepare_from_texts(
            long_texts,
            tokenizer,
            torch.device("cpu"),
            max_length,
            num_samples,
            strategy,
            seed=0,
            logger=LOGGER,
        )
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[1] == max_length
        assert result["attention_mask"].shape == result["input_ids"].shape

    def test_drop_rand_seed_reproducibility(self, tokenizer, long_texts):
        kwargs = dict(
            texts=long_texts,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            max_length=32,
            num_calibration_samples=4,
            strategy="drop_rand",
            logger=LOGGER,
        )
        r1 = prepare_from_texts(**kwargs, seed=42)
        r2 = prepare_from_texts(**kwargs, seed=42)
        r3 = prepare_from_texts(**kwargs, seed=99)

        assert torch.equal(r1["input_ids"], r2["input_ids"])
        assert not torch.equal(r1["input_ids"], r3["input_ids"])

    def test_concat_rand_seed_reproducibility(self, tokenizer, long_texts):
        kwargs = dict(
            texts=long_texts,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            max_length=32,
            num_calibration_samples=4,
            strategy="concat_rand",
            logger=LOGGER,
        )
        r1 = prepare_from_texts(**kwargs, seed=42)
        r2 = prepare_from_texts(**kwargs, seed=42)
        r3 = prepare_from_texts(**kwargs, seed=99)

        assert torch.equal(r1["input_ids"], r2["input_ids"])
        assert not torch.equal(r1["input_ids"], r3["input_ids"])


class TestDropStrategies:
    """Tests specific to drop_head and drop_rand."""

    def test_drop_head_takes_prefix(self, tokenizer, long_texts):
        max_length = 32
        result = chunk_single_document(
            long_texts,
            tokenizer,
            torch.device("cpu"),
            max_length,
            2,
            "drop_head",
            0,
            LOGGER,
        )
        assert result["input_ids"].shape == (2, max_length)

    def test_drop_rand_returns_correct_count(self, tokenizer, long_texts):
        result = chunk_single_document(
            long_texts,
            tokenizer,
            torch.device("cpu"),
            32,
            5,
            "drop_rand",
            0,
            LOGGER,
        )
        assert result["input_ids"].shape[0] == 5

    def test_short_texts_raise_value_error(self, tokenizer, short_texts):
        with pytest.raises(ValueError, match="Not enough calibration samples"):
            chunk_single_document(
                short_texts,
                tokenizer,
                torch.device("cpu"),
                64,
                5,
                "drop_head",
                0,
                LOGGER,
            )


class TestConcatStrategies:
    """Tests specific to concat_chunk and concat_chunk_align."""

    def test_concat_chunk_creates_maximum_chunks(self, tokenizer, long_texts):
        result = chunk_concat(
            long_texts,
            tokenizer,
            torch.device("cpu"),
            32,
            4,
            align_chunks=False,
            logger=LOGGER,
        )
        assert result["input_ids"].shape[1] == 32
        assert result["input_ids"].shape[0] > 0

    def test_concat_chunk_align_limits_chunks(self, tokenizer, long_texts):
        num_samples = 4
        result = chunk_concat(
            long_texts,
            tokenizer,
            torch.device("cpu"),
            32,
            num_samples,
            align_chunks=True,
            logger=LOGGER,
        )
        assert result["input_ids"].shape[0] <= num_samples

    def test_concat_rand_correct_count(self, tokenizer, long_texts):
        num_samples = 6
        result = chunk_concat_rand(
            long_texts,
            tokenizer,
            torch.device("cpu"),
            32,
            num_samples,
            seed=0,
            logger=LOGGER,
        )
        assert result["input_ids"].shape[0] == num_samples
