"""Unit tests for the auxiliary-file copy in ``Runner.save_quantized_model``.

These tests pin the behaviour of the whitelist-based copy (issue #64,
Issue 1) without running an actual quantization pipeline.

* ``_resolve_source_model_dir`` returns local directories as-is.
* ``_copy_auxiliary_files`` copies ``*.json`` / ``*.jinja`` files but
  skips weight tensors, weight index files, ``config.json``,
  ``generation_config.json`` and any file already present in the
  destination.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from logging import getLogger
from types import SimpleNamespace

from onecomp.runner import Runner


def _make_runner_stub(model_path: str):
    """Build a ``Runner`` instance that bypasses ``__init__``.

    Only the attributes touched by the auxiliary-file helpers are
    initialised so the test does not need a real model or tokenizer.
    """
    runner = Runner.__new__(Runner)
    runner.logger = getLogger("test_save_quantized_aux_files")
    runner.model_config = SimpleNamespace(
        get_model_id_or_path=lambda: model_path,
    )
    return runner


def _write(path, content: str = ""):
    path.write_text(content, encoding="utf-8")


def test_resolve_source_model_dir_returns_local_path(tmp_path):
    src = tmp_path / "src_model"
    src.mkdir()
    runner = _make_runner_stub(str(src))

    assert runner._resolve_source_model_dir() == str(src)


def test_resolve_source_model_dir_none_when_unset(tmp_path):
    runner = _make_runner_stub(None)
    assert runner._resolve_source_model_dir() is None


def test_copy_auxiliary_files_copies_json_and_jinja(tmp_path):
    src = tmp_path / "src_model"
    dst = tmp_path / "save_dir"
    src.mkdir()
    dst.mkdir()

    _write(src / "preprocessor_config.json", '{"do_resize": true}')
    _write(src / "processor_config.json", "{}")
    _write(src / "special_tokens_map.json", "{}")
    _write(src / "chat_template.jinja", "{{ messages }}")
    _write(src / "model-00001-of-00002.safetensors", "WEIGHTS_BIN")
    _write(src / "model-00002-of-00002.safetensors", "WEIGHTS_BIN")
    _write(src / "model.safetensors.index.json", "{}")
    _write(src / "pytorch_model.bin.index.json", "{}")
    _write(src / "pytorch_model.bin", "WEIGHTS_BIN")
    _write(src / "config.json", "{}")
    _write(src / "generation_config.json", "{}")

    runner = _make_runner_stub(str(src))

    copied = runner._copy_auxiliary_files(str(src), str(dst))

    assert copied == 4
    assert (dst / "preprocessor_config.json").is_file()
    assert (dst / "processor_config.json").is_file()
    assert (dst / "special_tokens_map.json").is_file()
    assert (dst / "chat_template.jinja").is_file()

    assert not (dst / "model-00001-of-00002.safetensors").exists()
    assert not (dst / "model-00002-of-00002.safetensors").exists()
    assert not (dst / "model.safetensors.index.json").exists()
    assert not (dst / "pytorch_model.bin.index.json").exists()
    assert not (dst / "pytorch_model.bin").exists()
    assert not (dst / "config.json").exists()
    assert not (dst / "generation_config.json").exists()

    assert (dst / "preprocessor_config.json").read_text(encoding="utf-8") == (
        '{"do_resize": true}'
    )


def test_copy_auxiliary_files_does_not_overwrite_existing(tmp_path, caplog):
    src = tmp_path / "src_model"
    dst = tmp_path / "save_dir"
    src.mkdir()
    dst.mkdir()

    _write(src / "tokenizer_config.json", '{"from": "src"}')
    _write(dst / "tokenizer_config.json", '{"from": "dst"}')

    runner = _make_runner_stub(str(src))
    with caplog.at_level("INFO", logger=runner.logger.name):
        copied = runner._copy_auxiliary_files(str(src), str(dst))

    assert copied == 0
    assert (dst / "tokenizer_config.json").read_text(encoding="utf-8") == ('{"from": "dst"}')
    # Pre-existing destination files (typically the just-written
    # ``tokenizer.save_pretrained`` outputs) must produce a matter-of-fact
    # log line so the auxiliary-copy step is auditable alongside the
    # ``Copied %s`` entries.
    assert any(
        "tokenizer_config.json" in record.message and "Using existing" in record.message
        for record in caplog.records
    )


def test_copy_auxiliary_files_handles_missing_src_dir(tmp_path):
    src = tmp_path / "no_such_dir"
    dst = tmp_path / "save_dir"
    dst.mkdir()

    runner = _make_runner_stub(str(src))
    copied = runner._copy_auxiliary_files(str(src), str(dst))
    assert copied == 0


def test_copy_auxiliary_files_skips_subdirectories(tmp_path):
    src = tmp_path / "src_model"
    dst = tmp_path / "save_dir"
    src.mkdir()
    dst.mkdir()

    nested = src / "nested_dir"
    nested.mkdir()
    _write(nested / "preprocessor_config.json", "{}")
    _write(src / "preprocessor_config.json", "{}")

    runner = _make_runner_stub(str(src))
    copied = runner._copy_auxiliary_files(str(src), str(dst))

    assert copied == 1
    assert (dst / "preprocessor_config.json").is_file()
    assert not (dst / "nested_dir").exists()
