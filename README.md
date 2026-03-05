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

### 1. Install PyTorch

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

### 2. Install `onecomp`

Once PyTorch is installed, you can install `onecomp`:

**for users**

```bash
pip install git+<git repository URL>
```

**for developers**

```bash
git clone <git repository URL>
pip install -e ".[develop]"
```

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
