"""Unit tests for the OneComp MCP tools.

``estimate_memory`` is validated against hand-computed values from a
known ``config.json`` (a tiny Qwen2 config written to a temp directory,
plus an optional real Qwen2.5-0.5B-Instruct directory).
``recommend_recipe`` is checked for monotonicity in the VRAM budget,
and the long-running ``quantize`` / ``evaluate`` wrappers are exercised
with mocks.  Everything runs on CPU without network access.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
import math
import os
from unittest.mock import MagicMock, patch

import pytest

from onecomp.mcp import tools
from onecomp.mcp.tools import ModelProfile, estimate_memory, evaluate, quantize, recommend_recipe

_BYTES_PER_GB = 1e9

# Tiny Qwen2-style config: small enough to instantiate on the meta
# device instantly, with GQA (num_key_value_heads < num_attention_heads)
# so the KV-cache hand calculation is non-trivial.
TINY_QWEN2_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "float16",
}

# Optional: point at a real Qwen2.5-0.5B-Instruct directory to also
# validate against the full-size config.
QWEN25_DIR = os.environ.get("ONECOMP_QWEN25_DIR", "")


@pytest.fixture(name="tiny_model_dir")
def _tiny_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(TINY_QWEN2_CONFIG), encoding="utf-8")
    return str(tmp_path)


def _expected_kv_gb(config, context_len, batch_size, kv_bytes=2):
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    elements = (
        2
        * config["num_hidden_layers"]
        * config["num_key_value_heads"]
        * head_dim
        * context_len
        * batch_size
    )
    return elements * kv_bytes / _BYTES_PER_GB


# ------------------------------------------------------------------
# estimate_memory
# ------------------------------------------------------------------
def test_estimate_memory_kv_cache_matches_hand_calculation(tiny_model_dir):
    """KV cache: 2 (K,V) x layers x kv_heads x head_dim x tokens x 2 bytes."""
    result = estimate_memory(tiny_model_dir, context_len=1024, batch_size=2)
    expected_kv = _expected_kv_gb(TINY_QWEN2_CONFIG, context_len=1024, batch_size=2)
    assert result["kv_cache_gb"] == pytest.approx(expected_kv, rel=1e-12)
    assert result["num_hidden_layers"] == 2
    assert result["num_key_value_heads"] == 2
    assert result["head_dim"] == 16


def test_estimate_memory_fp16_weights(tiny_model_dir):
    """Without wbits, all parameters are FP16 (2 bytes each)."""
    result = estimate_memory(tiny_model_dir, context_len=1024)
    assert result["wbits"] == 16
    assert result["effective_bits_per_param"] == 16
    assert result["weight_gb"] == pytest.approx(result["num_params"] * 2 / _BYTES_PER_GB)
    assert result["total_gb"] == pytest.approx(result["weight_gb"] + result["kv_cache_gb"])


def test_estimate_memory_quantized_weights(tiny_model_dir):
    """4-bit, groupsize 128: eff = 4 + (16 + 4)/128 = 4.15625 bits/param."""
    result = estimate_memory(tiny_model_dir, wbits=4, context_len=1024)
    assert result["effective_bits_per_param"] == pytest.approx(4.15625)

    num_params = result["num_params"]
    quantizable = int(num_params * 0.95)
    expected_weight_bits = quantizable * 4.15625 + (num_params - quantizable) * 16
    assert result["weight_gb"] == pytest.approx(expected_weight_bits / 8 / _BYTES_PER_GB)
    # Quantised weights must be smaller than FP16 weights.
    fp16 = estimate_memory(tiny_model_dir, context_len=1024)
    assert result["weight_gb"] < fp16["weight_gb"]


@pytest.mark.skipif(not os.path.isdir(QWEN25_DIR), reason="ONECOMP_QWEN25_DIR not set")
def test_estimate_memory_qwen25_0_5b_matches_hand_calculation():
    """Hand-computed values for the real Qwen2.5-0.5B-Instruct config.

    config.json: 24 layers, 2 KV heads, head_dim 64 (hidden 896 / 14
    heads), 494,032,768 parameters (tied embeddings counted once).
    """
    result = estimate_memory(QWEN25_DIR, context_len=4096, batch_size=1)
    assert result["num_params"] == 494032768
    # 2 * 24 * 2 * 64 * 4096 * 1 * 2 bytes = 0.050331648 GB
    assert result["kv_cache_gb"] == pytest.approx(0.050331648, rel=1e-12)
    # FP16 weights: 494032768 * 2 bytes = 0.988065536 GB
    assert result["weight_gb"] == pytest.approx(0.988065536, rel=1e-12)


# ------------------------------------------------------------------
# recommend_recipe
# ------------------------------------------------------------------
_LLAMA7B_PROFILE = ModelProfile(
    model_id="dummy/llama-7b",
    num_params=6_738_415_616,
    num_hidden_layers=32,
    num_key_value_heads=32,
    head_dim=128,
)


def _recommend_7b(total_vram_gb, **kwargs):
    with patch.object(tools, "load_model_profile", return_value=_LLAMA7B_PROFILE):
        return recommend_recipe("dummy/llama-7b", total_vram_gb, **kwargs)


def test_recommend_recipe_monotone_in_vram():
    """More VRAM must never lead to fewer recommended bits."""
    budgets = [3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    bits = [_recommend_7b(v)["wbits"] for v in budgets]
    assert bits == sorted(bits)
    assert bits[0] < bits[-1], "expected different recipes across the budget range"


def test_recommend_recipe_fits_budget():
    result = _recommend_7b(8.0)
    assert result["fits"] is True
    assert result["estimated_total_gb"] <= result["budget_gb"]
    assert result["method"] == "gptq+qep"
    assert result["wbits"] in (8, 6, 5, 4, 3, 2)
    assert result["rationale"]


def test_recommend_recipe_reports_when_nothing_fits():
    result = _recommend_7b(0.5)
    assert result["fits"] is False
    assert result["wbits"] == 2
    assert "not fit" in result["rationale"]


def test_recommend_recipe_memory_target_uses_fewer_bits():
    quality = _recommend_7b(24.0, target="quality")
    memory = _recommend_7b(24.0, target="memory")
    assert memory["wbits"] <= quality["wbits"]
    assert memory["wbits"] >= 3


# ------------------------------------------------------------------
# quantize / evaluate (mocked: real runs are long)
# ------------------------------------------------------------------
def test_quantize_delegates_to_auto_run():
    with patch("onecomp.runner.Runner.auto_run") as auto_run:
        result = quantize(
            model_id="some/model",
            save_dir="/tmp/out",
            wbits=4,
            total_vram_gb=8.0,
            qep=True,
            evaluate=False,
        )
    auto_run.assert_called_once_with(
        model_id="some/model",
        wbits=4,
        total_vram_gb=8.0,
        qep=True,
        evaluate=False,
        save_dir="/tmp/out",
    )
    assert result["status"] == "completed"
    assert result["save_dir"] == "/tmp/out"


def test_evaluate_loads_model_and_computes_perplexity():
    fake_model, fake_tokenizer = MagicMock(), MagicMock()
    with (
        patch(
            "onecomp.quantized_model_loader.QuantizedModelLoader.load_quantized_model",
            return_value=(fake_model, fake_tokenizer),
        ) as loader,
        patch("onecomp.utils.calculate_perplexity", return_value=12.5) as ppl,
    ):
        result = evaluate(model_dir="/tmp/saved", num_samples=8)
    loader.assert_called_once_with("/tmp/saved")
    ppl.assert_called_once_with(model=fake_model, tokenizer=fake_tokenizer, max_samples=8)
    assert result["perplexity"] == 12.5
    assert result["num_samples"] == 8


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: estimate_memory("model", wbits=-1), "wbits"),
        (lambda: estimate_memory("model", context_len=0), "context_len"),
        (lambda: estimate_memory("model", batch_size=True), "batch_size"),
        (lambda: recommend_recipe("model", math.nan), "total_vram_gb"),
        (lambda: recommend_recipe("model", 8, target="fast"), "target"),
        (lambda: quantize("model", "", wbits=4), "save_dir"),
        (lambda: quantize("model", "out", qep=1), "qep"),
        (lambda: evaluate("model", num_samples=0), "num_samples"),
    ],
)
def test_mcp_tools_reject_invalid_domains_before_heavy_work(call, match):
    with pytest.raises(ValueError, match=match):
        call()
