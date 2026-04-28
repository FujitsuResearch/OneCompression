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

#### 2. Install `onecomp`

Once PyTorch is installed, you can install `onecomp`:

```bash
pip install onecomp
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
uv sync --extra cu128 --extra dev --extra visualize
```

The `uv sync` command creates a Python virtual environment and installs all dependent libraries.

The `--extra cu128` option installs the CUDA-enabled version of PyTorch (along with `torchvision` from the same CUDA index).
Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, or `cu130`.
PyTorch will be automatically downloaded by `uv`, so you do not need to install it beforehand.

Adding `--extra dev` installs development tools (black, pytest, pylint).
Adding `--extra visualize` installs matplotlib for visualization features.
Adding `--extra hydra` installs `hydra-core` for the example scripts and `model_validation/` runners that use Hydra-based configuration.

To use vLLM for serving quantized models, add `--extra vllm` together with `--extra cu130`:

```bash
uv sync --extra cu130 --extra dev --extra visualize --extra vllm
```

> **Note:** `--extra vllm` is only compatible with `--extra cu130`. Recent vLLM releases require `torch>=2.10`, whose wheels are only published for the `cu130` index. Combining `--extra vllm` with `cpu` / `cu118` / `cu121` / `cu124` / `cu126` / `cu128` is rejected by `uv` at lock time.

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

```bash
git clone <git repository URL>
cd OneCompression

# First, install PyTorch with CUDA support for your environment
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Then install onecomp with development dependencies
pip install -e ".[dev]"
```

Replace `cu128` with the appropriate variant for your environment: `cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, or `cu130`.


### Building Documentation Locally

```bash
uv sync --extra cu128 --extra dev --extra docs
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 🚀 Examples

| Category | Script | Description |
|----------|--------|-------------|
| Quantization | [example_gptq.py](./example/example_gptq.py) | GPTQ quantization |
| | [example_qep_gptq.py](./example/example_qep_gptq.py) | GPTQ + QEP (error propagation) |
| | [example_lpcd_gptq.py](./example/example_lpcd_gptq.py) | GPTQ + QEP + LPCD quantization |
| | [example_jointq.py](./example/example_jointq.py) | JointQ quantization |
| | [example_autobit.py](./example/example_autobit.py) | AutoBit mixed-precision quantization |
| | [example_auto_run.py](./example/example_auto_run.py) | AutoBit with automatic VRAM estimation |
| Calibration | [example_custom_calibration.py](./example/example_custom_calibration.py) | Custom calibration dataset with CalibrationConfig |
| Save / Load | [example_save_load.py](./example/example_save_load.py) | Save and load quantized models |
| Rotation Preprocessing | [example_llama_preprocess_rtn.py](./example/pre_process/example_llama_preprocess_rtn.py) | Rotation preprocessing + RTN (TinyLlama) |
| | [example_preprocess_save_load.py](./example/pre_process/example_preprocess_save_load.py) | Save and load rotation-preprocessed quantized models |
| Post-Process | [example_blockwise_ptq.py](./example/post_process/example_blockwise_ptq.py) | Block-wise PTQ (GPTQ + Phase 1 & CBQ) |
| | [example_lora_sft.py](./example/post_process/example_lora_sft.py) | LoRA SFT post-quantization fine-tuning |
| | [example_lora_sft_knowledge.py](./example/post_process/example_lora_sft_knowledge.py) | LoRA SFT knowledge injection |
| vLLM | [example_gptq_vllm_inference.py](./example/vllm_inference/example_gptq_vllm_inference.py) | GPTQ + QEP quantization and vLLM inference |
| | [example_autobit_vllm_inference.py](./example/vllm_inference/example_autobit_vllm_inference.py) | AutoBit quantization and vLLM inference |

## 🔌 vLLM Inference

OneComp-quantized models can be served with [vLLM](https://docs.vllm.ai/) via built-in plugins (DBF, Mixed-GPTQ).
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
