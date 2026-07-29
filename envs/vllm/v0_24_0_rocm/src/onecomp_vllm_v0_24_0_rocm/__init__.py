"""ROCm-only workaround for vLLM 0.24.0 TritonW4A16LinearKernel.

Copyright 2025-2026 Fujitsu Ltd.

The entry point installed via ``pip install``
(declared in this package's ``pyproject.toml``) is
:func:`onecomp_vllm_v0_24_0_rocm.patch.apply`.
"""

from .patch import apply

__all__ = ["apply"]
