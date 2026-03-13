# Fujitsu Open Source Model Compression Software
- Release Plan
  - NeurIPS paper version release (December 2, 2025)
  - Official version release (March, 2026)
- Citation
```
@inproceedings{
arai2025quantization,
title={Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization},
author={Yamato Arai and Yuma Ichikawa},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=a3l3K9khbL}
}
```

---

> **Below is a draft of the new README**

---

# Python package for LLM compression 

This is a Python package currently under development.

It has not been officially released yet and may behave unstably.

## 📦 Features

(TBD) 

## 🔧 Installation

### for users (pip)

#### 1. Install PyTorch

Please install the appropriate version of PyTorch.

#### ✅ CPU-only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### ✅ CUDA-enabled

Choose the appropriate CUDA version for your system:

| CUDA Version | Installation Command |
|--------------|------------------------|
| CUDA 11.8    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| CUDA 12.1    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 12.4    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.6    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` |
| CUDA 12.8    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |

Check your CUDA version:
```bash
nvcc --version
```

or
```bash
nvidia-smi
```

Verify PyTorch GPU support:
```python
import torch
print(torch.cuda.is_available())
```

#### 2. Install `onecomp`

Once PyTorch is installed, you can install `onecomp`:

**for users**

```bash
pip install git+<git repository URL>
```

### for developers (uv : recommended)

#### Install `uv`

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is a fast Python package and project manager written in Rust.
It offers a drop-in replacement for pip and pip-tools while also managing virtual environments and Python installations.
With its Rust-based dependency resolver and the `uv.lock` lockfile, uv provides deterministic and reproducible environments across development machines and CI pipelines.

```bash
# install uv (for macOS or Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <git repository URL>
cd onecomp
uv sync --extra cu128 --extra dev
```

The `uv sync` command creates a Python virtual environment and installs all dependent libraries.

The `--extra cu128` option installs the CUDA-enabled version of PyTorch.
Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, or `cu128`.
PyTorch will be automatically downloaded by `uv`, so you do not need to install it beforehand.

Adding `--extra dev` installs additional packages for development.

#### Running commands (uv environment)

In the environment created by `uv sync`, you can run commands in two ways:

##### Option 1: Use `uv run` (no activation needed)

```bash
uv run pytest tests/ -v
uv run python example/example1.py
uv run black --check onecomp/
```

##### Option 2: Activate the virtual environment (traditional approach)

```bash
source .venv/bin/activate
pytest tests/ -v
python example/example1.py
black --check onecomp/
```

### for developers (pip)

```bash
git clone <git repository URL>

cd onecomp
# First, install PyTorch with CUDA support for your environment
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Then install onecomp with development dependencies
pip install -e ".[dev]"
```

Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, or `cu128`.


## 🚀 Example

See [example/example1.py](./example/example1.py) and [example/example2.py](./example/example2.py) for more details.


## 📄 License

See [LICENSE](./LICENSE) for more details.

## Citation

```
@inproceedings{
arai2025quantization,
title={Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization},
author={Yamato Arai and Yuma Ichikawa},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=a3l3K9khbL}
}
```
