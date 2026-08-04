"""Unit tests for onecomp.cpu.serve (no model / llama.cpp runtime needed).

Covers path resolution, OneComp-checkpoint detection and the OpenAI response
shaping helpers.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json

import pytest


def test_resolve_existing_gguf(tmp_path):
    from onecomp.cpu.serve import resolve_to_gguf

    g = tmp_path / "m.gguf"
    g.write_bytes(b"GGUF")
    assert resolve_to_gguf(str(g)) == str(g)


def test_resolve_missing_gguf(tmp_path):
    from onecomp.cpu.serve import resolve_to_gguf

    with pytest.raises(FileNotFoundError):
        resolve_to_gguf(str(tmp_path / "nope.gguf"))


def test_resolve_dir_with_gguf(tmp_path):
    from onecomp.cpu.serve import resolve_to_gguf

    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    assert resolve_to_gguf(str(tmp_path)).endswith("model.gguf")


def test_resolve_plain_dir_rejected(tmp_path):
    from onecomp.cpu.serve import resolve_to_gguf

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError):
        resolve_to_gguf(str(tmp_path))


def test_is_onecomp_checkpoint(tmp_path):
    from onecomp.cpu.serve import _is_onecomp_checkpoint

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert _is_onecomp_checkpoint(str(plain)) is False

    quant = tmp_path / "quant"
    quant.mkdir()
    (quant / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "gptq"}})
    )
    assert _is_onecomp_checkpoint(str(quant)) is True


def test_completion_response_shapes():
    from onecomp.cpu.serve import _completion_response

    chat = _completion_response("hi", chat=True, cid="x", created=1, model_id="m")
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"]["content"] == "hi"
    assert chat["choices"][0]["finish_reason"] == "stop"

    comp = _completion_response("hi", chat=False, cid="y", created=1, model_id="m")
    assert comp["object"] == "text_completion"
    assert comp["choices"][0]["text"] == "hi"
