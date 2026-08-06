"""Locate / fetch llama.cpp tooling needed for GGUF export.

For the direct GPTQ -> GGUF path we only need ``convert_hf_to_gguf.py`` (pure
Python) to build a metadata + tokenizer "skeleton" GGUF; no C++ build is
required. For the fallback dequantize path we additionally need the
``llama-quantize`` binary, which is looked up on PATH / via env vars.

Resolution order for the llama.cpp source tree:
  1. ``$LLAMA_CPP_DIR`` (an existing checkout),
  2. a local cache at ``$ONECOMP_CACHE_DIR/llama.cpp`` (default ~/.cache/onecomp),
  3. a shallow ``git clone`` from GitHub.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from logging import getLogger
from typing import List, Optional

logger = getLogger(__name__)

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_VERSION = "b9370"


def _cache_root() -> str:
    root = os.environ.get("ONECOMP_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "onecomp"
    )
    os.makedirs(root, exist_ok=True)
    return root


def get_llama_cpp_dir(auto_clone: bool = True) -> str:
    """Return a path to a llama.cpp source tree, cloning it if necessary."""
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir and os.path.isfile(os.path.join(env_dir, "convert_hf_to_gguf.py")):
        return env_dir

    cache_dir = os.path.join(_cache_root(), "llama.cpp")
    if os.path.isfile(os.path.join(cache_dir, "convert_hf_to_gguf.py")):
        return cache_dir

    if not auto_clone:
        raise FileNotFoundError(
            "llama.cpp not found. Set $LLAMA_CPP_DIR to a checkout, or allow auto_clone."
        )

    logger.info("Cloning llama.cpp (shallow) into %s", cache_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", LLAMA_CPP_VERSION, LLAMA_CPP_REPO, cache_dir],
        check=True,
    )
    return cache_dir


def get_convert_script(auto_clone: bool = True) -> str:
    """Return the path to ``convert_hf_to_gguf.py``."""
    script = os.path.join(get_llama_cpp_dir(auto_clone=auto_clone), "convert_hf_to_gguf.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {script}")
    return script


def run_convert_hf_to_gguf(
    model_dir: str,
    out_gguf: str,
    outtype: str = "f16",
    extra_args: Optional[List[str]] = None,
) -> str:
    """Run llama.cpp's convert_hf_to_gguf.py on a dense HF model directory.

    Uses the current interpreter and prepends the cloned ``gguf-py`` to
    ``PYTHONPATH`` so the converter runs against its own bundled gguf version.
    """
    llama_dir = get_llama_cpp_dir()
    script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    cmd = [
        sys.executable,
        script,
        model_dir,
        "--outfile",
        out_gguf,
        "--outtype",
        outtype,
    ]
    if extra_args:
        cmd += extra_args

    env = dict(os.environ)
    gguf_py = os.path.join(llama_dir, "gguf-py")
    if os.path.isdir(gguf_py):
        env["PYTHONPATH"] = gguf_py + os.pathsep + env.get("PYTHONPATH", "")

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    if not os.path.isfile(out_gguf):
        raise RuntimeError(f"convert_hf_to_gguf did not produce {out_gguf}")
    return out_gguf


def find_llama_quantize() -> Optional[str]:
    """Locate the ``llama-quantize`` binary (env var / PATH / cached build)."""
    env_bin = os.environ.get("LLAMA_QUANTIZE_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    for name in ("llama-quantize", "quantize"):
        found = shutil.which(name)
        if found:
            return found
    cache_dir = os.path.join(_cache_root(), "llama.cpp")
    for candidate in (
        os.path.join(cache_dir, "build", "bin", "llama-quantize"),
        os.path.join(cache_dir, "llama-quantize"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _require_llama_quantize() -> str:
    binary = find_llama_quantize()
    if binary is None:
        raise FileNotFoundError(
            "llama-quantize not found. Set $LLAMA_QUANTIZE_BIN, put it on PATH, or build llama.cpp.\n"
            "The direct GPTQ->GGUF path does not need this binary.\n"
            "How to build llama.cpp\n"
            "  uv pip install cmake\n"
            "  . .venv/bin/activate\n"
            f"  git clone --depth 1 -b {LLAMA_CPP_VERSION} {LLAMA_CPP_REPO}\n"
            "  cd llama.cpp\n"
            "  cmake -B build -S . -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON -DGGML_LTO=ON\n"
            "  cmake --build build -j$(($(nproc)/2))\n"
            "  cmake --install build\n"
            "  pip install -r requirements.txt\n"
            "export LLAMA_QUANTIZE_BIN='$PWD/llama.cpp/build/bin/llama-quantize'"
        )
    return binary


def run_llama_quantize(in_gguf: str, out_gguf: str, qtype: str) -> str:
    """Quantize an f16 GGUF to ``qtype`` (e.g. Q4_K_M) via llama-quantize."""
    binary = _require_llama_quantize()
    cmd = [binary, in_gguf, out_gguf, qtype]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_gguf


def run_llama_quantize_per_tensor(
    in_gguf: str,
    out_gguf: str,
    tensor_types: "dict[str, str]",
    default_type: str = "Q8_0",
    embedding_type: str = "F16",
    output_type: str = "F16",
) -> str:
    """Quantize selected tensors to explicit per-tensor GGUF types.

    Used by the mixed-bit plugin to K-quantize only the fallback (2/3-bit /
    act-order) layers.  ``llama-quantize`` ignores per-tensor overrides when the
    positional ftype is ``F16`` (it takes a pure-copy fast path), so the default
    must be a *quantized* type.  Tensors that the caller intends to overwrite
    afterwards (the lossless direct layers) are harmlessly quantized to
    ``default_type`` and later replaced; the token-embedding and output tensors
    are pinned to ``embedding_type`` / ``output_type`` (f16) so they are not
    degraded.

    Args:
        in_gguf: Source GGUF (the dequantized f16 skeleton).
        out_gguf: Destination GGUF.
        tensor_types: ``{gguf_tensor_name: ggml_type_name}`` explicit overrides
            (e.g. ``{"blk.0.ffn_down.weight": "Q2_K"}``).
        default_type: ftype applied to tensors with no explicit override.
        embedding_type: type for ``token_embd`` (kept f16 by default).
        output_type: type for the ``output``/lm_head tensor (kept f16 by default).

    Returns:
        ``out_gguf``.
    """
    import tempfile

    binary = _require_llama_quantize()
    cmd = [binary, "--token-embedding-type", embedding_type, "--output-tensor-type", output_type]
    tt_file = None
    if tensor_types:
        tt_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tensortypes.txt", delete=False, encoding="utf-8"
        )
        for name, ggml_type in tensor_types.items():
            tt_file.write(f"{name}={ggml_type}\n")
        tt_file.close()
        cmd += ["--tensor-type-file", tt_file.name]
    cmd += [in_gguf, out_gguf, default_type]
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        if tt_file is not None:
            os.unlink(tt_file.name)
    if not os.path.isfile(out_gguf):
        raise RuntimeError(f"llama-quantize did not produce {out_gguf}")
    return out_gguf
