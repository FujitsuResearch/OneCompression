"""Tests for model card generation.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from unittest.mock import Mock

import huggingface_hub
import pytest

import onecomp
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


def test_usage_section_uses_repo_id():
    card = generate_model_card(
        MODEL_ID,
        repo_id="your-name/tinyllama-onecomp",
    )

    assert 'model_dir = snapshot_download(repo_id="your-name/tinyllama-onecomp")' in card
    assert "model, tokenizer = load_quantized_model(model_dir)" in card
    assert 'load_quantized_model("your-name/tinyllama-onecomp")' not in card


def test_local_model_path_is_not_linked(tmp_path):
    card = generate_model_card(str(tmp_path))

    assert f"https://huggingface.co/{tmp_path}" not in card
    assert f"`{tmp_path}`" in card


def test_repo_id_does_not_replace_base_model():
    card = generate_model_card(
        MODEL_ID,
        repo_id="your-name/tinyllama-onecomp",
    )
    assert f"base_model: {MODEL_ID}" in card


def test_repo_usage_downloads_local_snapshot_before_loading(monkeypatch):
    repo_id = "your-name/tinyllama-onecomp"
    model_dir = "/tmp/downloaded-tinyllama"

    snapshot_download = Mock(return_value=model_dir)
    load_quantized_model = Mock(return_value=(object(), object()))

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        snapshot_download,
    )
    monkeypatch.setattr(
        onecomp,
        "load_quantized_model",
        load_quantized_model,
    )

    card = generate_model_card(MODEL_ID, repo_id=repo_id)
    usage_code = card.split("```python\n", 1)[1].split("\n```", 1)[0]

    compile(usage_code, "<generated-model-card-usage>", "exec")
    exec(usage_code, {})

    snapshot_download.assert_called_once_with(repo_id=repo_id)
    load_quantized_model.assert_called_once_with(model_dir)
