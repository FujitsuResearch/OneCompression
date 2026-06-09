#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv-vllm"

echo "==> Creating vLLM venv at ${VENV_DIR}"
uv venv --python 3.12 "${VENV_DIR}"

echo "==> Installing vLLM + torch (CUDA 12.8)"
uv pip install --python "${VENV_DIR}/bin/python" \
    torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${VENV_DIR}/bin/python" vllm

echo "==> Verifying"
"${VENV_DIR}/bin/python" -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"

echo "==> Done. Set ONECOMP_VLLM_PYTHON=${VENV_DIR}/bin/python if needed."
