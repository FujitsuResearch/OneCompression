"""Tests for the ``--format gguf`` CLI path (Runner and export mocked).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import os
import sys

import pytest

import onecomp.export
import onecomp.runner
from onecomp import cli


class _FakeRunner:
    """Stands in for the runner returned by ``Runner.auto_run``."""

    def __init__(self, calls):
        self._calls = calls

    def save_dequantized_model(self, path):
        self._calls["dequant_dir"] = path
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            f.write("{}")


@pytest.fixture(name="calls")
def fixture_calls(monkeypatch, tmp_path):
    """Mock ``Runner.auto_run`` and ``export_gguf``; record their inputs."""
    calls = {}

    def fake_auto_run(**kwargs):
        calls["auto_run"] = kwargs
        if kwargs.get("save_dir"):
            os.makedirs(kwargs["save_dir"], exist_ok=True)
        return _FakeRunner(calls)

    def fake_export_gguf(model_dir, config):
        calls["export_src"] = model_dir
        calls["export_config"] = config
        return config.out_path

    monkeypatch.setattr(onecomp.runner.Runner, "auto_run", fake_auto_run)
    monkeypatch.setattr(onecomp.export, "export_gguf", fake_export_gguf)
    return calls


def test_cli_gguf_exports_from_dequantized_weights(calls, monkeypatch, tmp_path):
    save_dir = str(tmp_path / "model-quant")
    monkeypatch.setattr(
        sys, "argv", ["onecomp", "org/model", "--save-dir", save_dir, "--format", "gguf"]
    )
    cli.main()

    # The GGUF file must be built from the dequantized FP16 directory,
    # not from the packed-quantized save directory.
    assert calls["export_src"] == calls["dequant_dir"]
    assert calls["export_src"] != save_dir
    assert calls["export_config"].out_path == os.path.join(save_dir, "model-quant-f16.gguf")
    assert calls["export_config"].name == "model-quant"
    # The temporary dequantized directory is removed afterwards.
    assert not os.path.exists(calls["dequant_dir"])


def test_cli_gguf_requires_save_dir(calls, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["onecomp", "org/model", "--format", "gguf"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "auto_run" not in calls


def test_cli_default_format_skips_export(calls, monkeypatch, tmp_path):
    save_dir = str(tmp_path / "model-quant")
    monkeypatch.setattr(sys, "argv", ["onecomp", "org/model", "--save-dir", save_dir])
    cli.main()
    assert "export_src" not in calls
    assert calls["auto_run"]["save_dir"] == save_dir
