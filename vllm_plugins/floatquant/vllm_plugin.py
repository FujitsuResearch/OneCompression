"""vLLM plugin for OneComp FloatQuant/FP8 fake-quant checkpoints.

Copyright 2025-2026 Fujitsu Ltd.

Registers the dedicated ``onecomp_fake_quant`` quantization method for
checkpoints saved by ``onecomp.quantizer.floatquant.FloatQuant`` with
``checkpoint_format="fake_quant"``:

    "quantization_config": {
        "quant_method": "onecomp_fake_quant",
        "fmt": "nvfp4",
        "block_size": 16,
        "use_hessian": false,
        "checkpoint_format": "fake_quant"
    }

Fake-quant checkpoints already contain the dequantized FP16 weights, so
every Linear layer maps to ``UnquantizedLinearMethod``; the plugin's job is
only to make vLLM accept the config instead of rejecting the unknown
``quant_method``.

The dedicated name deliberately does NOT collide with vLLM's built-in
``fp8`` / ``mxfp4`` handlers, so native FP8 checkpoints (e.g. written by
``save_vllm_fp8_model``) and external MXFP4 models keep loading through
vLLM's built-ins even while this plugin is installed.  Legacy fake-quant
checkpoints that still carry ``quant_method="nvfp4"|"mxfp4"|"fp8"`` should
have their ``config.json`` updated to the dedicated name (the weights are
unchanged plain FP16).
"""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from vllm_plugins.floatquant.config import resolve_fake_quant_config

logger = init_logger(__name__)


class _FakeQuantConfigBase(QuantizationConfig):
    """Shared config for FP16 fake-quant checkpoints (weights unquantized)."""

    METHOD: str = ""

    def __init__(self, fmt: str, block_size, use_hessian: bool):
        super().__init__()
        self.fmt = fmt
        self.block_size = block_size
        self.use_hessian = use_hessian
        logger.info(
            "%s fake-quant checkpoint: FP16 weights (fmt=%s, block_size=%s, use_hessian=%s)",
            self.METHOD,
            fmt,
            block_size,
            use_hessian,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(fmt={self.fmt}, block_size={self.block_size}, "
            f"use_hessian={self.use_hessian})"
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "_FakeQuantConfigBase":
        resolved = resolve_fake_quant_config(config)
        return cls(
            fmt=resolved["fmt"],
            block_size=resolved["block_size"],
            use_hessian=resolved["use_hessian"],
        )

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        # Fake-quant weights are plain FP16 Linear weights; nothing to scan.
        pass

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        del prefix
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


@register_quantization_config("onecomp_fake_quant")
class OneCompFakeQuantConfig(_FakeQuantConfigBase):
    """OneComp fake-quant checkpoints (FP16 weights; ``fmt`` records the format)."""

    METHOD = "onecomp_fake_quant"


def register_vllm_plugin():
    """Entry point for ``vllm.general_plugins``; registration is a side
    effect of importing this module."""
    logger.info("Registered the onecomp_fake_quant quantization method with vLLM")
