"""Tests for model card generation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import pytest

from onecomp.export import generate_model_card

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

RECIPE = {
    "method": "AutoBit + QEP",
    "wbits": 4,
    "groupsize": 128,
}

RESULTS = {
    "perplexity (wikitext2)": 8.1234,
    "accuracy (lambada)": 0.6543,
}


def test_frontmatter():
    card = generate_model_card(MODEL_ID, RECIPE, RESULTS)
    frontmatter = card.split("---")[1]
    assert "license: apache-2.0" in frontmatter
    assert f"base_model: {MODEL_ID}" in frontmatter
    for tag in ("onecomp", "quantized", "gptq"):
        assert f"  - {tag}" in frontmatter


def test_frontmatter_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    card = generate_model_card(MODEL_ID, RECIPE, RESULTS)
    data = yaml.safe_load(card.split("---")[1])
    assert data["license"] == "apache-2.0"
    assert data["base_model"] == MODEL_ID
    assert data["tags"] == ["onecomp", "quantized", "gptq"]


def test_license_override():
    card = generate_model_card(MODEL_ID, {"license": "mit"})
    assert "license: mit" in card
    # The license key is consumed by the frontmatter, not the recipe table.
    assert "| license |" not in card


def test_recipe_table():
    card = generate_model_card(MODEL_ID, RECIPE, RESULTS)
    assert "## Quantization Recipe" in card
    assert "| method | AutoBit + QEP |" in card
    assert "| wbits | 4 |" in card
    assert "| groupsize | 128 |" in card


def test_results_table():
    card = generate_model_card(MODEL_ID, RECIPE, RESULTS)
    assert "## Evaluation Results" in card
    assert "| perplexity (wikitext2) | 8.1234 |" in card
    assert "| accuracy (lambada) | 0.6543 |" in card


def test_empty_sections_omitted():
    card = generate_model_card(MODEL_ID)
    assert "## Quantization Recipe" not in card
    assert "## Evaluation Results" not in card
    assert "## Usage" in card


def test_usage_section_uses_model_name():
    card = generate_model_card(MODEL_ID)
    assert 'load_quantized_model("TinyLlama-1.1B-Chat-v1.0")' in card
