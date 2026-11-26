"""Compatibility tests for lazy top-level public imports."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "public_names"),
    [
        (".export", ["GGUFExportConfig", "export_gguf", "generate_model_card", "push_to_hub"]),
        (".quantizer", ["MDBF"]),
        (".quantized_model_loader", ["StreamingUnsupportedError"]),
        (".post_process", ["GlobalPTQ", "GlobalPTQDistributed", "RouterFineTuning"]),
        (".utils", ["cleanup_memory", "empty_cache", "get_default_device"]),
    ],
)
def test_legacy_public_imports_resolve_lazily(module_name, public_names):
    repository = Path(__file__).resolve().parents[2]
    script = """
import json
import sys
from types import ModuleType

import onecomp

module_name, public_names = json.loads(sys.argv[1])
owner_name = "onecomp" + module_name
assert owner_name not in sys.modules
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
assert "matplotlib" not in sys.modules

owner = ModuleType(owner_name)
expected = {name: object() for name in public_names}
for name, value in expected.items():
    setattr(owner, name, value)
sys.modules[owner_name] = owner

for name, value in expected.items():
    assert name in onecomp.__all__, name
    assert name in dir(onecomp), name
    assert name not in vars(onecomp), name
    imported = __import__("onecomp", fromlist=[name])
    assert getattr(imported, name) is value, name
    assert vars(onecomp)[name] is value, name
    delattr(owner, name)
    assert getattr(onecomp, name) is value, name

assert "torch" not in sys.modules
assert "transformers" not in sys.modules
assert "matplotlib" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, json.dumps([module_name, public_names])],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
