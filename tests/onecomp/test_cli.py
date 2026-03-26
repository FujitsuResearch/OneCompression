"""End-to-end tests for the ``onecomp`` CLI.

Verifies that ``onecomp TinyLlama/...`` and common flag combinations
run without errors.

The default full-run test always executes (GPU required).
Other variant tests are skipped by default; set ``RUN_CLI_VARIANT_TESTS=1``
to enable them::

    RUN_CLI_VARIANT_TESTS=1 pytest tests/onecomp/test_cli.py -v

Copyright 2025-2026 Fujitsu Ltd.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

TIMEOUT = 1200

_skip_variant = pytest.mark.skipif(
    not os.environ.get("RUN_CLI_VARIANT_TESTS"),
    reason="CLI variant test skipped by default. Set RUN_CLI_VARIANT_TESTS=1 to run.",
)


_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


def _run_onecomp(*args, timeout=TIMEOUT):
    """Run ``python -m onecomp`` in a subprocess.

    Uses the current interpreter directly to avoid ``uv run`` triggering
    an implicit ``uv sync`` that could modify the virtual environment.
    ``cwd`` is set to the project root so that ``tests/onecomp/`` does not
    shadow the real ``onecomp`` package on ``sys.path``.
    """
    cmd = [sys.executable, "-m", "onecomp", *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=_PROJECT_ROOT,
    )


# ------------------------------------------------------------------
# Full default run (always enabled, requires GPU)
# ------------------------------------------------------------------
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_default_full_run(tmp_path):
    """``onecomp TinyLlama/...`` with all defaults (AutoBit + QEP + eval + save).

    This mirrors the documented basic usage and requires a CUDA GPU.
    """
    save_dir = str(tmp_path / "full_run")
    result = _run_onecomp(
        MODEL_ID,
        "--save-dir",
        save_dir,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (tmp_path / "full_run").exists(), "Save directory was not created"


# ------------------------------------------------------------------
# Smoke test (fast, no model loading)
# ------------------------------------------------------------------
@_skip_variant
def test_version():
    """``onecomp --version`` exits cleanly."""
    result = _run_onecomp("--version", timeout=30)
    assert result.returncode == 0
    assert "onecomp" in result.stdout


# ------------------------------------------------------------------
# Quantize-only variant tests (--no-eval --save-dir none)
# ------------------------------------------------------------------
@_skip_variant
def test_wbits4_qep_cpu():
    """Fixed 4-bit + QEP on CPU, skip eval and save."""
    result = _run_onecomp(
        MODEL_ID,
        "--wbits",
        "4",
        "--device",
        "cpu",
        "--no-eval",
        "--save-dir",
        "none",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@_skip_variant
def test_wbits4_no_qep_cpu():
    """Fixed 4-bit without QEP on CPU, skip eval and save."""
    result = _run_onecomp(
        MODEL_ID,
        "--wbits",
        "4",
        "--no-qep",
        "--device",
        "cpu",
        "--no-eval",
        "--save-dir",
        "none",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@_skip_variant
def test_wbits3_cpu():
    """Fixed 3-bit + QEP on CPU, skip eval and save."""
    result = _run_onecomp(
        MODEL_ID,
        "--wbits",
        "3",
        "--device",
        "cpu",
        "--no-eval",
        "--save-dir",
        "none",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@_skip_variant
def test_vram_estimation_cpu():
    """AutoBit with explicit VRAM budget on CPU, skip eval and save."""
    result = _run_onecomp(
        MODEL_ID,
        "--total-vram-gb",
        "0.8",
        "--device",
        "cpu",
        "--no-eval",
        "--save-dir",
        "none",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@_skip_variant
def test_custom_groupsize_cpu():
    """Custom groupsize on CPU, skip eval and save."""
    result = _run_onecomp(
        MODEL_ID,
        "--wbits",
        "4",
        "--groupsize",
        "64",
        "--device",
        "cpu",
        "--no-eval",
        "--save-dir",
        "none",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@_skip_variant
def test_save_quantized_model(tmp_path):
    """Quantize and save to a temp directory."""
    save_dir = str(tmp_path / "quantized")
    result = _run_onecomp(
        MODEL_ID,
        "--wbits",
        "4",
        "--device",
        "cpu",
        "--no-eval",
        "--save-dir",
        save_dir,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert (tmp_path / "quantized").exists(), "Save directory was not created"
