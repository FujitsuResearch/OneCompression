"""
Resolve installed vLLM package paths for in-place patches.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from pathlib import Path


def vllm_root() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent


def vllm_file(*parts: str) -> Path:
    return vllm_root().joinpath(*parts)
