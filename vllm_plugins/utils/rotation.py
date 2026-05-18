"""Helpers for rotation-preprocessed model metadata in vLLM plugins.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from onecomp.pre_process.rotation_utils import is_online_hadamard_target
from onecomp.pre_process.hadamard_utils import get_hadK, matmul_hadU_cuda

try:
    from vllm.model_executor.layers.linear import LinearMethodBase
except ImportError:  # pragma: no cover - exercised only without vLLM installed.
    class LinearMethodBase:  # type: ignore[no-redef]
        """Fallback base so metadata-only imports still work without vLLM."""

        def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
            raise NotImplementedError

        def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError


@dataclass(frozen=True)
class RotationMetadata:
    """Rotation-related metadata persisted in quantization_config.

    The save path already stores these values under model.config.quantization_config.
    vLLM plugins should read this helper instead of re-parsing raw dict keys.
    """

    rotated: bool = False
    fp32_had: bool = False

    @classmethod
    def from_quant_config(cls, config: dict[str, Any] | None) -> "RotationMetadata":
        if not isinstance(config, dict):
            return cls()
        return cls(
            rotated=bool(config.get("rotated", False)),
            fp32_had=bool(config.get("fp32_had", False)),
        )

    def requires_hadamard(self, prefix: str) -> bool:
        """Return whether a module prefix should receive the online Hadamard transform."""
        return self.rotated and is_online_hadamard_target(prefix)


def apply_online_hadamard(x: torch.Tensor, *, fp32_had: bool, cache_owner: Any) -> torch.Tensor:
    """Apply the online Hadamard transform used by rotation-preprocessed down_proj layers."""
    cached = getattr(cache_owner, "_onecomp_hadamard_cache", None)
    if cached is None or cached[1] != x.shape[-1]:
        had_k, block_size = get_hadK(x.shape[-1])
        setattr(cache_owner, "_onecomp_hadamard_cache", (had_k, x.shape[-1], block_size))
    else:
        had_k, _dim, block_size = cached

    x_dtype = x.dtype
    if fp32_had:
        return matmul_hadU_cuda(x.float(), had_k, block_size).to(x_dtype)
    return matmul_hadU_cuda(x, had_k, block_size)


class RotatedLinearMethod(LinearMethodBase):
    """LinearMethod wrapper that applies the online Hadamard transform before matmul."""

    def __init__(self, base_method: LinearMethodBase, *, fp32_had: bool):
        self.base_method = base_method
        self.fp32_had = fp32_had

    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
        return self.base_method.create_weights(layer, *weight_args, **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        process = getattr(self.base_method, "process_weights_after_loading", None)
        if process is not None:
            process(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = apply_online_hadamard(x, fp32_had=self.fp32_had, cache_owner=layer)
        return self.base_method.apply(layer, x, bias)


def maybe_wrap_rotation_method(
    method: LinearMethodBase,
    *,
    prefix: str,
    metadata: RotationMetadata,
) -> LinearMethodBase:
    """Wrap a vLLM LinearMethod when the target module requires online Hadamard."""
    if metadata.requires_hadamard(prefix):
        return RotatedLinearMethod(method, fp32_had=metadata.fp32_had)
    return method