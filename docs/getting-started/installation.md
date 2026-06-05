# Installation

This page describes how to install Fujitsu One Compression (OneComp).

## Requirements

- Python 3.12 or later (< 3.14)
- PyTorch (CPU, CUDA, or MPS on macOS)

## For Users (pip)

### Step 1: Install PyTorch

Install the appropriate version of PyTorch for your system.

=== "CPU only"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

=== "CUDA 11.8"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

=== "CUDA 12.1"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

=== "CUDA 12.4"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

=== "CUDA 12.6"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

=== "CUDA 12.8"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

=== "CUDA 13.0"

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    ```

=== "macOS (MPS)"

    On macOS, install PyTorch from PyPI (default wheels include MPS support).
    You do **not** need the CUDA index URLs above.

    ```bash
    pip install torch torchvision torchaudio
    ```

    Verify MPS:

    ```python
    import torch
    print(torch.backends.mps.is_available())
    ```

    Then install OneComp (step 2 below). GPTQ quantization and Hugging Face
    `generate()` inference on MPS are supported; vLLM serving requires Linux with
    an NVIDIA GPU. An editable install from a git clone is **not** required for
    MPS use — see [For Developers (pip)](#for-developers-pip) only if you are
    contributing to OneComp.

    For usage (`device="mps"`, VRAM budget, limitations), see the
    [macOS / MPS guide](../user-guide/mps.md).

Check your CUDA version (Linux / Windows with NVIDIA GPU):

```bash
nvcc --version
# or
nvidia-smi
```

Verify PyTorch GPU support (CUDA):

```python
import torch
print(torch.cuda.is_available())
```

### Step 2: Install OneComp

```bash
pip install onecomp
```

To enable visualization features (matplotlib), install with the `visualize` extra:

```bash
pip install onecomp[visualize]
```

## For Developers (uv -- recommended)

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is a fast Python package and project manager written in Rust.
It provides deterministic, reproducible environments via its lockfile.

```bash
# Install uv (macOS or Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/FujitsuResearch/OneCompression.git
cd OneCompression
```

The `uv sync` command creates a virtual environment and installs all dependencies.

### Linux (CUDA quantization / vLLM)

```bash
uv sync --extra cu128 --extra dev --extra visualize
```

The `--extra cu128` option installs the CUDA-enabled version of PyTorch (along with `torchvision` from the same CUDA index).
Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, or `cu130`.
PyTorch will be automatically downloaded by `uv`, so you do not need to install it beforehand.

### macOS (development / MPS inference)

```bash
uv sync --extra mps --extra dev --extra visualize
```

On macOS, use `--extra mps` only. CUDA extras (`cu118`–`cu130`), `--extra cpu` (Linux-only),
and `--extra vllm` are not supported on macOS.
After `uv sync`, you can run GPTQ quantization and Hugging Face `generate()` inference on MPS;
vLLM serving still requires Linux with an NVIDIA GPU.
See the [macOS / MPS guide](../user-guide/mps.md) for device placement and usage details.

Adding `--extra dev` installs development tools (black, pytest, pylint).
Adding `--extra visualize` installs matplotlib for visualization features.

To use vLLM for serving quantized models on Linux, add `--extra vllm` together with `--extra cu130`:

```bash
uv sync --extra cu130 --extra dev --extra visualize --extra vllm
```

!!! note "vLLM requires the `cu130` extra"
    Recent vLLM releases depend on `torch>=2.10`, whose wheels are only published for the `cu130` index. The `--extra vllm` declaration in `pyproject.toml` therefore conflicts with `cpu`, `mps`, `cu118`, `cu121`, `cu124`, `cu126`, and `cu128`; combining any of these with `--extra vllm` is rejected by `uv` at lock time.

!!! warning
    Do **not** install vLLM with `uv pip install vllm` after `uv sync`. Packages installed via `uv pip` are not tracked by the lockfile and will be removed or overwritten by subsequent `uv sync` or `uv run` commands. Always use `--extra vllm` instead.

### Running Commands

=== "uv run (no activation needed)"

    ```bash
    uv run onecomp --version
    uv run onecomp TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    uv run pytest tests/ -v
    uv run python example/example_gptq.py
    ```

=== "Traditional virtualenv"

    ```bash
    source .venv/bin/activate
    onecomp --version
    onecomp TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    pytest tests/ -v
    python example/example_gptq.py
    ```

## For Developers (pip)

!!! note
    The editable install below is for developing OneComp from a local clone.
    **macOS users who only want MPS inference or quantization should use the
    [For Users (pip)](#for-users-pip) flow** (`pip install torch` then
    `pip install onecomp` from PyPI); `pip install -e` is not needed for MPS.

```bash
git clone https://github.com/FujitsuResearch/OneCompression.git
cd OneCompression

# First, install PyTorch for your environment
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Then install onecomp with development dependencies
pip install -e ".[dev]"
```

Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, or `cu130`.
On macOS, install PyTorch from PyPI instead (see [macOS (MPS)](#step-1-install-pytorch) above).

## Building Documentation Locally

`--extra docs` alone is enough. PyTorch extras (`mps`, `cu*`, `cpu`) are not required
to build or serve the documentation.

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
