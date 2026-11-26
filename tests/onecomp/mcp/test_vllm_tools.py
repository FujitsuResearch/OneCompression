"""Unit tests for the ``vllm_engine_args`` MCP tool.

Everything runs on CPU with temp directories: the tool only reads
``config.json``, so no model weights, GPU, or vLLM install is needed.
The stdio-visible behaviour is covered through :class:`MCPServer` with
``io.StringIO`` streams.

Copyright 2025-2026 Fujitsu Ltd.
"""

import io
import json

import pytest

from onecomp.mcp.protocol import MCPServer
from onecomp.mcp.tools import build_tools, vllm_engine_args


def _write_config(tmp_path, quantization_config=None):
    config = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


def test_unquantised_model(tmp_path):
    """No quantization_config: supported, no plugin, auto KV dtype."""
    result = vllm_engine_args(_write_config(tmp_path))
    assert result["quant_method"] is None
    assert result["needs_onecomp_plugin"] is False
    assert result["vllm_supported"] is True
    assert result["kv_cache_dtype"] == "auto"
    assert result["engine_kwargs"] == {"model": result["model_dir"]}
    assert result["serve_command"] == f"vllm serve {result['model_dir']}"


def test_mixed_gptq_needs_plugin(tmp_path):
    """mixed_gptq is served by the OneComp plugin."""
    result = vllm_engine_args(
        _write_config(
            tmp_path,
            {
                "quant_method": "mixed_gptq",
                "group_size": -1,
                "quantization_bits": [
                    {"self_attn.q_proj": {"bits": 4, "method": "gptq", "group_size": 128}},
                    {"self_attn.q_proj": {"bits": 8, "method": "gptq", "group_size": 128}},
                ],
            },
        )
    )
    assert result["needs_onecomp_plugin"] is True
    assert "gptq" in result["plugin"]
    assert result["vllm_supported"] is True
    assert result["quantization_bits_summary"] == {
        "num_layers": 2,
        "bits": [4, 8],
        "group_sizes": [128],
    }


def test_plain_gptq_is_native(tmp_path):
    """Plain gptq needs no OneComp plugin."""
    result = vllm_engine_args(_write_config(tmp_path, {"quant_method": "gptq", "bits": 4}))
    assert result["needs_onecomp_plugin"] is False
    assert result["vllm_supported"] is True


def test_rotated_model_is_unsupported(tmp_path):
    """Rotation-preprocessed models cannot be served with vLLM."""
    result = vllm_engine_args(_write_config(tmp_path, {"quant_method": "gptq", "rotated": True}))
    assert result["vllm_supported"] is False
    assert any("rotation" in w.lower() or "Hadamard" in w for w in result["warnings"])


def test_unknown_quant_method_warns(tmp_path):
    result = vllm_engine_args(_write_config(tmp_path, {"quant_method": "kiwi"}))
    assert result["vllm_supported"] is False
    assert any("kiwi" in w for w in result["warnings"])


def test_fp8_kv_cache(tmp_path):
    """fp8 KV precision adds kv_cache_dtype and calculate_kv_scales."""
    result = vllm_engine_args(_write_config(tmp_path), kv_cache_precision="fp8")
    assert result["kv_cache_dtype"] == "fp8"
    assert result["engine_kwargs"]["kv_cache_dtype"] == "fp8"
    assert result["engine_kwargs"]["calculate_kv_scales"] is True
    assert "--kv-cache-dtype fp8 --calculate-kv-scales" in result["serve_command"]


def test_invalid_kv_precision_raises(tmp_path):
    with pytest.raises(ValueError):
        vllm_engine_args(_write_config(tmp_path), kv_cache_precision="int4")


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vllm_engine_args(str(tmp_path / "nowhere"))


# ------------------------------------------------------------------
# stdio protocol integration
# ------------------------------------------------------------------
def _serve(messages):
    lines = "".join(json.dumps(m) + "\n" for m in messages)
    reader = io.StringIO(lines)
    writer = io.StringIO()
    MCPServer(reader, writer, tools=build_tools()).serve_forever()
    return [json.loads(line) for line in writer.getvalue().splitlines()]


def test_tool_is_listed_and_callable_over_stdio(tmp_path):
    """tools/list exposes vllm_engine_args and tools/call executes it."""
    model_dir = _write_config(tmp_path, {"quant_method": "mixed_gptq", "quantization_bits": []})
    responses = _serve(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "vllm_engine_args",
                    "arguments": {"model_dir": model_dir, "kv_cache_precision": "fp8"},
                },
            },
        ]
    )
    names = [t["name"] for t in responses[0]["result"]["tools"]]
    assert "vllm_engine_args" in names
    assert responses[1]["result"]["isError"] is False
    payload = json.loads(responses[1]["result"]["content"][0]["text"])
    assert payload["needs_onecomp_plugin"] is True
    assert payload["kv_cache_dtype"] == "fp8"


def test_tool_error_is_reported_not_fatal(tmp_path):
    """A missing directory becomes an isError result, not a protocol error."""
    responses = _serve(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "vllm_engine_args",
                    "arguments": {"model_dir": str(tmp_path / "missing")},
                },
            },
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ]
    )
    assert responses[0]["result"]["isError"] is True
    assert "FileNotFoundError" in responses[0]["result"]["content"][0]["text"]
    assert responses[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}
