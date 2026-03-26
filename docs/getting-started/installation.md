# Installation

This page describes how to install Fujitsu One Compression (OneComp).

## Requirements

- Python 3.12 or later (< 3.14)
- PyTorch (CPU or CUDA)

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

Check your CUDA version:

```bash
nvcc --version
# or
nvidia-smi
```

Verify PyTorch GPU support:

```python
import torch
print(torch.cuda.is_available())
```

### Step 2: Install OneComp

```bash
pip install git+https://github.com/FujitsuResearch/OneCompression.git
```

## For Developers (uv -- recommended)

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is a fast Python package and project manager written in Rust.
It provides deterministic, reproducible environments via its lockfile.

```bash
# Install uv (macOS or Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/FujitsuResearch/OneCompression.git
cd OneCompression
uv sync --extra cu128 --extra dev
```

The `uv sync` command creates a virtual environment and installs all dependencies (including `torchvision` from the same CUDA index as PyTorch).
Replace `cu128` with the appropriate CUDA variant for your system: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, or `cu128`.

Adding `--extra dev` installs development tools (black, pytest, pylint, matplotlib).

To use vLLM for serving quantized models, add `--extra vllm`:

```bash
uv sync --extra cu128 --extra dev --extra vllm
```

!!! warning
    Do **not** install vLLM with `uv pip install vllm` after `uv sync`. Packages installed via `uv pip` are not tracked by the lockfile and will be removed or overwritten by subsequent `uv sync` or `uv run` commands. Always use `--extra vllm` instead.

### Running Commands

=== "uv run (no activation needed)"

    ```bash
    uv run onecomp --version
    uv run onecomp TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    uv run pytest tests/ -v
    uv run python example/example1.py
    ```

=== "Traditional virtualenv"

    ```bash
    source .venv/bin/activate
    onecomp --version
    onecomp TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    pytest tests/ -v
    python example/example1.py
    ```

## For Developers (pip)

```bash
git clone https://github.com/FujitsuResearch/OneCompression.git
cd OneCompression

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install onecomp with development dependencies
pip install -e ".[dev]"
```

## Building Documentation Locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
