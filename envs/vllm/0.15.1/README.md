# Legacy vLLM Environment Setup (vLLM 0.15.1)

## Prerequisites

This environment is verified with the following setup:

- **OS**: Linux (x86_64)
- **Python**: 3.12 (the commands below assume the `python3.12` executable is on your `PATH`)
- **CUDA wheel variant**: `cu130` (the pinned `torch` and `vllm` wheels are built for CUDA 13.0)
- **GPU**: NVIDIA GPU with a driver compatible with CUDA 13.0 wheels

If `python3.12` is not available, install it first, e.g. with [pyenv](https://github.com/pyenv/pyenv) or your distribution's package manager (Ubuntu: `sudo apt install python3.12 python3.12-venv`).
All package versions are pinned in `requirements.txt` and installed with `--no-deps`, so use Python 3.12 to match this tested environment.

## Setup & Run

Run the following commands from the repository root to create and use the legacy vLLM environment:

```bash
python3.12 -m venv envs/vllm/0.15.1/.venv
source envs/vllm/0.15.1/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps -r envs/vllm/0.15.1/requirements.txt
python example/vllm_inference/example_gptq_vllm_inference.py
```
