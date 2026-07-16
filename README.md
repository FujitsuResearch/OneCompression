# Fujitsu One Compression

Fujitsu One Compression (OneComp) is a Python package for LLM compression.

<p align="center">
  <img src="docs/assets/onecomp.gif" alt="OneComp" />
</p>

## ⚡ Just one line.

```bash
onecomp <generative AI>
```

**That's all you need.** OneComp detects your GPU VRAM, picks the best bit-width per layer, quantizes with error propagation, evaluates, and saves — fully automatic.

```bash
# Example
onecomp meta-llama/Llama-2-7b-hf
```

Or from Python:

```python
from onecomp import Runner

Runner.auto_run(model_id="meta-llama/Llama-2-7b-hf")
```

## 📖 Documentation

Full documentation is available at **[https://FujitsuResearch.github.io/OneCompression/](https://FujitsuResearch.github.io/OneCompression/)**.

## 📦 Features

- **Quantization Error Propagation (QEP)**: A post-training quantization method that corrects quantization errors by propagating them to subsequent layers, improving the accuracy of quantized LLMs. See [Arai & Ichikawa, NeurIPS 2025](https://openreview.net/forum?id=a3l3K9khbL) for details. The original reference implementation is available at [FujitsuResearch/qep](https://github.com/FujitsuResearch/qep).
- **Layer-Projected Coordinate Descent (LPCD)**: A unified Post Training Quantization (PTQ) framework that extends layer-wise quantization to arbitrary submodules by optimising relaxed objectives and projecting the solutions with layer-wise quantizers. See [Ichikawa et al., 2025](https://arxiv.org/abs/2512.01546) for details.
- **vLLM Plugin Integration**: Serve OneComp-quantized models with [vLLM](https://docs.vllm.ai/) via built-in plugins for DBF and Mixed-GPTQ quantization methods. Pair with [Open WebUI](https://github.com/open-webui/open-webui) for a ChatGPT-like chat experience on your local machine.
- **AutoBit**: Mixed-precision quantization with ILP-based bitwidth assignment. Automatically estimates the target bitwidth from available VRAM and assigns per-layer bitwidths to minimize quantization error under the memory budget.
- **JointQ**: Joint quantization method that optimizes weight assignments and scale parameters simultaneously for improved quantization accuracy. Supports group-wise quantization (e.g., 4-bit, groupsize=128).
- **Block-wise PTQ**: Post-quantization block-wise distillation that minimises intermediate-representation MSE against an FP16 teacher model at Transformer-block granularity. Includes Phase 1 (greedy per-block optimisation) and Phase 2 CBQ (cross-block sliding-window optimisation). Supports GPTQ, DBF, and OneBit quantizers.
- **LoRA SFT Post-Process**: Fine-tune quantized models with LoRA adapters for accuracy recovery or domain-specific knowledge injection. Supports SFT loss, teacher distillation, and intermediate block alignment.
- **Rotation Preprocessing**: SpinQuant/OstQuant-based rotation preprocessing that reduces quantization error by learning optimal rotation matrices before quantization. Rotation/scaling matrices are absorbed into model weights, with online Hadamard hooks automatically registered at load time. Supports Llama and Qwen3 architectures.
- **Web Dashboard (HPC)**: A browser-based dashboard for launching quantization jobs, deploying models, and validating chat-based inference in HPC environments. See [dashboard/README.md](dashboard/README.md) for details.
- (TBD)

## 🤖 Supported Models

OneComp has been verified with the following model architectures.
Other Hugging Face-compatible models may work but are currently untested.

| # | Architecture | Verified Models | Status |
|---|-------------|-----------------|--------|
| 1 | Llama | TinyLlama, Llama-2, Llama-3 | ✅ Verified |
| 2 | Qwen3 | Qwen3-0.6B ~ 32B | ✅ Verified |
| 3 | Gemma | Gemma 2, Gemma 3, Gemma 4  | ✅ Verified |


> **Note:** Support for additional architectures is planned. Contributions and test reports are welcome.

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
| CUDA 13.0    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130` |

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

#### ✅ macOS (MPS)

On macOS, install PyTorch from PyPI (default wheels include MPS support). You do **not** need the CUDA index URLs above.

```bash
pip install torch torchvision torchaudio
```

Verify MPS:
```python
import torch
print(torch.backends.mps.is_available())
```

Then install OneComp from PyPI (see step 2 below). GPTQ quantization and Hugging Face `generate()` inference on MPS are supported; vLLM serving requires Linux with an NVIDIA GPU. An editable install from a git clone is **not** required for MPS use — see [for developers (pip)](#for-developers-pip) only if you are contributing to OneComp.

> **MPS device placement (GPTQ vs QEP)**  
> With `device="mps"`, calibration and model forward passes can run on the GPU. The bottleneck is usually not missing Cholesky ops on MPS (recent PyTorch builds implement them); the implementation splits work as follows:
>
> - GPTQ (`run_gptq`): Hessian and weights are moved to CPU for the full column-wise loop (including inverse-Hessian Cholesky). If that loop stayed on MPS, `quantize()` would call `maxq.item()` once per column; each call triggers **per-column host sync** (wait for pending MPS ops, then read one scalar—not a full Hessian/weight copy every column)—often several times slower than CPU on Apple Silicon (e.g. ~4× in internal benchmarks with PyTorch 2.12). Keeping GPTQ on CPU avoids that overhead. With `mse=True`, `find_params` also calls `quantize()` in a grid loop and benefits from the same CPU placement.
> - QEP weight correction (`adjust_weight`, when QEP correction runs—typically `qep=True` with error propagation enabled): Per-layer work stays on MPS (e.g. `weight @ delta_hatX`, diagonal damping). Only the Cholesky solve uses CPU via `_safe_cholesky_and_solve` (one solve per layer, not per column); moving all of QEP to CPU does not materially improve speed. The subsequent GPTQ step still uses the CPU path above.
>
> DBF-based AutoBit fallback and multi-GPU quantization are not supported on MPS.

#### 2. Install `onecomp`

Once PyTorch is installed, you can install `onecomp`:

```bash
pip install onecomp
```

To enable multi-GPU training features (DeepSpeed), install with the `distributed` extra:

```bash
pip install "onecomp[distributed]"
```

### for developers (uv : recommended)

#### Install `uv`

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is a fast Python package and project manager written in Rust.
It offers a drop-in replacement for pip and pip-tools while also managing virtual environments and Python installations.
With its Rust-based dependency resolver and the `uv.lock` lockfile, uv provides deterministic and reproducible environments across development machines and CI pipelines.

```bash
# install uv (for macOS or Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/FujitsuResearch/OneCompression.git
cd OneCompression
```

The `uv sync` command creates a Python virtual environment and installs all dependent libraries.

#### Linux (CUDA quantization / vLLM)

```bash
uv sync --extra cu128 --extra dev --extra visualize
```

The `--extra cu128` option installs the CUDA-enabled version of PyTorch (along with `torchvision` from the same CUDA index).
Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, or `cu130`.
PyTorch will be automatically downloaded by `uv`, so you do not need to install it beforehand.

#### macOS (development / MPS inference)

```bash
uv sync --extra mps --extra dev --extra visualize
```

On macOS, use `--extra mps` only. CUDA extras (`cu118`–`cu130`), `--extra cpu` (Linux-only), and `--extra vllm` are not supported on macOS.
After `uv sync`, you can run GPTQ quantization and Hugging Face `generate()` inference on MPS; vLLM serving still requires Linux with an NVIDIA GPU.
See the **MPS device placement (GPTQ vs QEP)** note under [macOS (MPS)](#macos-mps) above for why GPTQ runs on CPU while QEP correction uses MPS.

Adding `--extra dev` installs development tools (black, pre-commit, pytest, pylint).
Adding `--extra visualize` installs matplotlib for visualization features.
Adding `--extra distributed` installs DeepSpeed for multi-GPU training.
Adding `--extra hydra` installs `hydra-core` for the example scripts and `model_validation/` runners that use Hydra-based configuration.

To use vLLM for serving quantized models on Linux, add `--extra vllm` together with `--extra cu130`:

```bash
uv sync --extra cu130 --extra dev --extra visualize --extra vllm
```

> **Note:** `--extra vllm` is only compatible with `--extra cu130`. Recent vLLM releases require `torch>=2.10`, whose wheels are only published for the `cu130` index. Combining `--extra vllm` with `cpu` / `mps` / `cu118` / `cu121` / `cu124` / `cu126` / `cu128` is rejected by `uv` at lock time.

> **Note:** `--extra vllm` may take a long time on the first run if a pre-built `xformers` wheel is not available for your Python/CUDA combination (e.g. Python 3.13). Using Python 3.12 typically avoids this.

#### Running commands (uv environment)

In the environment created by `uv sync`, you can run commands in two ways:

##### Option 1: Use `uv run` (no activation needed)

```bash
uv run pytest tests/ -v
uv run python example/example_gptq.py
uv run black --check onecomp/
```

##### Option 2: Activate the virtual environment (traditional approach)

```bash
source .venv/bin/activate
pytest tests/ -v
python example/example_gptq.py
black --check onecomp/
```

### for developers (pip)

> **Note:** The editable install below is for developing OneComp from a local clone. **macOS users who only want MPS inference or quantization should use the [for users (pip)](#for-users-pip) flow** (`pip install torch` then `pip install onecomp` from PyPI); `pip install -e` is not needed for MPS.

```bash
git clone <git repository URL>
cd OneCompression

# First, install PyTorch with CUDA support for your environment
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Then install onecomp with development dependencies
pip install -e ".[dev]"
```

Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, or `cu130`.

#### Pre-commit

After installing development dependencies (`--extra dev` with uv, or `pip install -e ".[dev]"` with pip), register the Git hooks once:

```bash
pre-commit install
```

On every `git commit`, the following checks run automatically:

| Hook | Description |
|------|-------------|
| **black** | Code formatting (line length 99) |
| **isort** | Import sorting |
| **no-japanese** | Forbid Japanese characters in text files (`.md` and `.gitignore` are excluded) |
| **copyright-header** | Verify the Fujitsu copyright header in Python files |
| **no-email-address** | Forbid email addresses in Python files |

Common commands:

```bash
# Run hooks on staged files only
pre-commit run

# Run hooks on all files (useful after first install or config changes)
pre-commit run --all-files

# Run a specific hook
pre-commit run black --all-files

# With uv (no activation needed)
uv run pre-commit run --all-files
```


### Building Documentation Locally

`--extra docs` alone is sufficient (no PyTorch `mps` / `cu*` extra required):

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 📓 Tutorial Notebook

Interactive walkthrough in Jupyter or [Google Colab](https://colab.research.google.com/github/FujitsuResearch/OneCompression/blob/main/notebook/01_tutorial.ipynb)
— RTN visualization, `Runner.auto_run`, and vLLM chat inference.

See [`notebook/README.md`](./notebook/README.md) for local setup, or the
[Tutorial Notebook guide](https://FujitsuResearch.github.io/OneCompression/getting-started/tutorial-notebook/) in the docs.

## 🚀 Examples

| Category | Script | Description |
|----------|--------|-------------|
| Tutorial | [01_tutorial.ipynb](./notebook/01_tutorial.ipynb) | Interactive notebook (Jupyter / Colab) |
| Quantization | [example_gptq.py](./example/example_gptq.py) | GPTQ quantization |
| | [example_qep_gptq.py](./example/example_qep_gptq.py) | GPTQ + QEP (error propagation) |
| | [example_lpcd_gptq.py](./example/example_lpcd_gptq.py) | GPTQ + QEP + LPCD quantization |
| | [example_jointq.py](./example/example_jointq.py) | JointQ quantization |
| | [example_autobit.py](./example/example_autobit.py) | AutoBit mixed-precision quantization |
| | [example_auto_run.py](./example/example_auto_run.py) | AutoBit with automatic VRAM estimation |
| Calibration | [example_custom_calibration.py](./example/example_custom_calibration.py) | Custom calibration dataset with CalibrationConfig |
| Save / Load | [example_save_load.py](./example/example_save_load.py) | Save and load quantized models |
| | [example_reload_post_process_resave.py](./example/post_process/example_reload_post_process_resave.py) | Reload, post-process, and re-save a quantized checkpoint |
| Rotation Preprocessing | [example_llama_preprocess_rtn.py](./example/pre_process/example_llama_preprocess_rtn.py) | Rotation preprocessing + RTN (TinyLlama) |
| | [example_preprocess_save_load.py](./example/pre_process/example_preprocess_save_load.py) | Save and load rotation-preprocessed quantized models |
| Post-Process | [example_blockwise_ptq.py](./example/post_process/example_blockwise_ptq.py) | Block-wise PTQ via Runner.run() with packed buffers (default) |
| | [example_global_ptq.py](./example/post_process/example_global_ptq.py) | Global PTQ with packed buffers (default), safetensors save |
| | [example_global_ptq_distributed.py](./example/post_process/example_global_ptq_distributed.py) | Global PTQ with DeepSpeed multi-GPU |
| | [example_lora_sft.py](./example/post_process/example_lora_sft.py) | LoRA SFT post-quantization fine-tuning |
| | [example_lora_sft_knowledge.py](./example/post_process/example_lora_sft_knowledge.py) | LoRA SFT knowledge injection |
| vLLM | [example_gptq_vllm_inference.py](./example/vllm_inference/example_gptq_vllm_inference.py) | GPTQ + QEP quantization and vLLM inference |
| | [example_jointq_vllm_inference.py](./example/vllm_inference/example_jointq_vllm_inference.py) | JointQ quantization and vLLM inference |
| | [example_autobit_vllm_inference.py](./example/vllm_inference/example_autobit_vllm_inference.py) | AutoBit quantization and vLLM inference |

## 🔌 vLLM Inference

OneComp-quantized models can be served with [vLLM](https://docs.vllm.ai/): GPTQ, JointQ, and RTN models use vLLM's built-in GPTQ plugin, while DBF and Mixed-GPTQ models are served via OneComp's own plugins.
Combined with [Open WebUI](https://github.com/open-webui/open-webui), you can chat with your quantized model through a ChatGPT-like browser interface — entirely on your local machine.

```bash
# uv users (vLLM requires cu130; see Installation for details)
uv sync --extra cu130 --extra vllm

# pip users
pip install vllm
```

See the [vLLM Inference guide](https://FujitsuResearch.github.io/OneCompression/user-guide/vllm-inference/) for details, including Open WebUI setup instructions.


## 📄 License

See [LICENSE](./LICENSE) for more details.

## Citation

OneComp technical report:

```
@misc{ichikawa2026onecomponelinerevolutiongenerative,
      title={OneComp: One-Line Revolution for Generative AI Model Compression}, 
      author={Yuma Ichikawa and Keiji Kimura and Akihiro Yoshida and Yudai Fujimoto and Hiroki Tokura and Yamato Arai and Yoshiyuki Ishii and Yusei Kawakami and Genki Shikada and Achille Jacquemond and Yoshihiko Fujisawa and Katsuki Fujisawa and Takumi Honda and Akira Sakai},
      year={2026},
      eprint={2603.28845},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.28845}, 
}
```

QEP (Quantization Error Propagation):

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

LPCD (Layer-Projected Coordinate Descent):

```
@article{ichikawa2025lpcd,
title={LPCD: Unified Framework from Layer-Wise to Submodule Quantization},
author={Yuma Ichikawa and Yudai Fujimoto and Akira Sakai},
journal={arXiv preprint arXiv:2512.01546},
year={2025},
url={https://arxiv.org/abs/2512.01546}
}
```
