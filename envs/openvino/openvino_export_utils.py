"""Shared helpers for exporting OneComp GPTQ checkpoints to OpenVINO IR.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def read_checkpoint_quantization(model_path: Path) -> dict[str, Any]:
    """Return the checkpoint quantization metadata without modifying it."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config was not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError(f"quantization_config is missing from {config_path}")
    return quantization


def modules_shape(modules: Any) -> str:
    """Describe modules_in_block_to_quantize as flat, nested, empty, or invalid."""
    if not isinstance(modules, list):
        return "invalid"
    if not modules:
        return "empty"
    if all(isinstance(name, str) for name in modules):
        return "flat"
    if all(
        isinstance(group, list) and group and all(isinstance(name, str) for name in group)
        for group in modules
    ):
        return "nested"
    return "invalid"


@contextmanager
def temporary_model_path_for_openvino_export(model_path: Path) -> Iterator[Path]:
    """Yield a model path with flat GPTQ module metadata normalized.

    Some OneComp checkpoints store modules_in_block_to_quantize as List[str],
    while current Transformers and Optimum expect List[List[str]]. For that
    shape, a temporary checkpoint copy is patched and removed after use.
    The source checkpoint is never changed.
    """
    source = model_path.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Checkpoint directory was not found: {source}")

    quantization = read_checkpoint_quantization(source)
    modules_key = "modules_in_block_to_quantize"
    if modules_key not in quantization:
        raise ValueError(f"{modules_key} is missing from {source / 'config.json'}")

    modules = quantization[modules_key]
    shape = modules_shape(modules)
    if shape == "invalid":
        raise ValueError("modules_in_block_to_quantize has an unsupported shape")
    if shape != "flat":
        yield source
        return

    with tempfile.TemporaryDirectory(prefix="onecomp_ov_export_compat_") as temp_root:
        temporary_model = Path(temp_root) / source.name
        shutil.copytree(source, temporary_model, symlinks=True)

        temporary_config_path = temporary_model / "config.json"
        with temporary_config_path.open(encoding="utf-8") as config_file:
            temporary_config = json.load(config_file)
        temporary_config["quantization_config"]["modules_in_block_to_quantize"] = [modules]
        with temporary_config_path.open("w", encoding="utf-8") as config_file:
            json.dump(temporary_config, config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")

        print(
            "[INFO] Using a temporary checkpoint copy with "
            "modules_in_block_to_quantize normalized from flat to nested."
        )
        print(f"[INFO] Temporary checkpoint: {temporary_model}")
        yield temporary_model


def save_openvino_tokenizer(tokenizer: Any, output_dir: Path) -> None:
    """Save OpenVINO tokenizer and detokenizer models for OpenVINO GenAI."""
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer

    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")


def validate_ir_files(output_dir: Path) -> list[str]:
    """Parse every exported non-tokenizer IR with OpenVINO Core."""
    import openvino as ov

    excluded = {"openvino_tokenizer.xml", "openvino_detokenizer.xml"}
    model_files = sorted(path for path in output_dir.glob("*.xml") if path.name not in excluded)
    if not model_files:
        raise FileNotFoundError(f"No model IR XML was generated in {output_dir}")

    core = ov.Core()
    for model_file in model_files:
        core.read_model(model_file)
    return [path.name for path in model_files]
