"""Helpers for rotation-preprocessed model metadata in vLLM plugins.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from onecomp.pre_process.hadamard_utils import get_hadK, matmul_hadU_cuda
from onecomp.pre_process.rotation_utils import is_online_hadamard_target

try:
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.distributed.utils import split_tensor_along_last_dim
    from vllm.model_executor.layers.linear import LinearMethodBase

    try:
        from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
    except ImportError:
        WEIGHT_LOADER_V2_SUPPORTED = None
    try:
        from vllm.model_executor.layers.linear import (
            register_weight_loader_v2_supported_method as _vllm_register_weight_loader_v2_supported_method,
        )
    except ImportError:
        _vllm_register_weight_loader_v2_supported_method = None

    def register_weight_loader_v2_supported_method(cls):
        if WEIGHT_LOADER_V2_SUPPORTED is not None:
            if cls.__name__ not in WEIGHT_LOADER_V2_SUPPORTED:
                WEIGHT_LOADER_V2_SUPPORTED.append(cls.__name__)
            return cls
        if _vllm_register_weight_loader_v2_supported_method is not None:
            return _vllm_register_weight_loader_v2_supported_method(cls)
        return cls

except ImportError:  # pragma: no cover - exercised only without vLLM installed.

    def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        raise RuntimeError("vLLM tensor-model-parallel all_gather is unavailable.")

    def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
    ):
        chunks = torch.chunk(tensor, num_partitions, dim=tensor.dim() - 1)
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in chunks)
        return chunks

    class LinearMethodBase:  # type: ignore[no-redef]
        """Fallback base so metadata-only imports still work without vLLM."""

        def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
            raise NotImplementedError

        def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError

    # Mirror the names bound in the vLLM branch so module-level helpers (e.g.
    # _base_method_uses_weight_loader_v2) resolve without a NameError when vLLM
    # is absent and only the metadata helpers are used.
    WEIGHT_LOADER_V2_SUPPORTED = None
    _vllm_register_weight_loader_v2_supported_method = None

    def register_weight_loader_v2_supported_method(cls):
        return cls


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


def make_hadamard_forward_pre_hook(*, fp32_had: bool):
    """Build a layer pre-hook for layers that still receive the full activation tensor."""

    def hook(layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        if not inputs:
            return inputs
        x = apply_online_hadamard(inputs[0], fp32_had=fp32_had, cache_owner=layer)
        return (x, *inputs[1:])

    return hook


def _base_method_uses_weight_loader_v2(method: LinearMethodBase) -> bool:
    """Return whether vLLM would select ``weight_loader_v2`` for ``method``.

    vLLM decides between the v1 and v2 weight loaders purely from the
    LinearMethod's *class name* (``LinearBase`` checks
    ``quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED``). A
    transparent wrapper that only forwards ``create_weights`` to a base method
    must therefore mirror the base method's membership: forcing v2 onto a
    v1-only base whose ``create_weights`` uses the passed-in ``weight_loader``
    and builds plain ``Parameter`` objects would break weight loading (the v2
    loader calls ``load_row_parallel_weight`` etc., which such params lack).
    """
    if WEIGHT_LOADER_V2_SUPPORTED is None:
        # v2 registry not introspectable in this vLLM build; assume the modern
        # v2 path so v2-native bases (GPTQMarlin, Unquantized, ...) keep loading.
        return True
    return method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED


class RotatedLinearMethod(LinearMethodBase):
    """LinearMethod wrapper that installs the online Hadamard at layer entry.

    This base variant is intentionally *not* registered as
    weight_loader_v2-supported: it is used to wrap base methods that rely on
    vLLM's v1 weight loader, so vLLM keeps selecting v1 for the wrapped layer.
    :class:`RotatedLinearMethodV2` below is the registered variant used when the
    wrapped base method is itself v2-native.
    """

    def __init__(self, base_method: LinearMethodBase, *, fp32_had: bool):
        self.base_method = base_method
        self.fp32_had = fp32_had

    def _requires_tp_gather(self, layer: torch.nn.Module) -> bool:
        tp_size = getattr(layer, "tp_size", 1) or 1
        return bool(getattr(layer, "input_is_parallel", False)) and tp_size > 1

    def _get_tp_metadata(self, layer: torch.nn.Module) -> tuple[int, int]:
        tp_rank = getattr(layer, "tp_rank", None)
        tp_size = getattr(layer, "tp_size", None)
        if tp_rank is None or tp_size is None:
            raise RuntimeError(
                "RotatedLinearMethod requires tp_rank and tp_size on tensor-parallel layers."
            )
        if tp_size <= 1:
            raise RuntimeError("Tensor-parallel Hadamard path requires tp_size > 1.")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise RuntimeError(
                f"Invalid tensor-parallel metadata: tp_rank={tp_rank}, tp_size={tp_size}."
            )
        return tp_rank, tp_size

    def _apply_tp_hadamard(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        tp_rank, tp_size = self._get_tp_metadata(layer)
        # The Hadamard transform mixes the full intermediate dimension, so row
        # parallel inputs must be gathered before applying it. This deliberately
        # adds one all_gather before RowParallelLinear's normal output reduce.
        full_x = tensor_model_parallel_all_gather(x, dim=-1)
        transformed_full_x = apply_online_hadamard(
            full_x,
            fp32_had=self.fp32_had,
            cache_owner=layer,
        )
        local_shards = split_tensor_along_last_dim(
            transformed_full_x,
            num_partitions=tp_size,
            contiguous_split_chunks=True,
        )
        return local_shards[tp_rank]

    def _ensure_hadamard_pre_hook(self, layer: torch.nn.Module) -> None:
        if self._requires_tp_gather(layer):
            return
        if getattr(layer, "_onecomp_hadamard_prehook_installed", False):
            return
        handle = layer.register_forward_pre_hook(
            make_hadamard_forward_pre_hook(fp32_had=self.fp32_had)
        )
        setattr(layer, "_onecomp_hadamard_prehook_installed", True)
        setattr(layer, "_onecomp_hadamard_prehook_handle", handle)

    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
        self._ensure_hadamard_pre_hook(layer)
        return self.base_method.create_weights(layer, *weight_args, **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._ensure_hadamard_pre_hook(layer)
        process = getattr(self.base_method, "process_weights_after_loading", None)
        if process is not None:
            process(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._requires_tp_gather(layer):
            x = self._apply_tp_hadamard(layer, x)
        return self.base_method.apply(layer, x, bias, *args, **kwargs)


@register_weight_loader_v2_supported_method
class RotatedLinearMethodV2(RotatedLinearMethod):
    """weight_loader_v2 variant used when the wrapped base method is v2-native.

    Registering this class name in vLLM's ``WEIGHT_LOADER_V2_SUPPORTED`` is what
    makes vLLM keep using the v2 weight loader for the wrapped base method (e.g.
    ``GPTQMarlinLinearMethod``/``UnquantizedLinearMethod``), whose
    ``create_weights`` builds v2 ``BasevLLMParameter`` weights.
    """


def maybe_wrap_rotation_method(
    method: LinearMethodBase,
    *,
    prefix: str,
    metadata: RotationMetadata,
) -> LinearMethodBase:
    """Wrap a vLLM LinearMethod when the target module requires online Hadamard.

    The wrapper mirrors the base method's weight_loader_v2 support so that
    wrapping never flips vLLM's v1/v2 loader selection for the underlying
    method (see :func:`_base_method_uses_weight_loader_v2`).
    """
    if not metadata.requires_hadamard(prefix):
        return method
    wrapper_cls = (
        RotatedLinearMethodV2
        if _base_method_uses_weight_loader_v2(method)
        else RotatedLinearMethod
    )
    return wrapper_cls(method, fp32_had=metadata.fp32_had)
