"""vLLM plugin for mixed-bitwidth GPTQ inference.

Copyright 2025-2026 Fujitsu Ltd.

Registers ``mixed_gptq`` as a quantization method.  Supports per-module,
per-layer quantization config via the ``quantization_bits`` format:

    "quantization_config": {
        "quant_method": "mixed_gptq",
        "quantization_bits": [
            {   // layer 0
                "self_attn.q_proj": {"bits": 4, "method": "gptq", "group_size": 128},
                "self_attn.k_proj": {"bits": 4, "method": "gptq", "group_size": 128},
                "mlp.down_proj":    {"bits": 8, "method": "gptq", "group_size": -1},
                ...
            },
            ...  // layer 1, 2, ...
        ],
        "group_size": -1,       // global fallback
        "desc_act": false,
        "sym": true
    }

Per-module ``group_size`` is resolved in order:
  1. ``mod_cfg["group_size"]``           (direct per-module)
  2. ``mod_cfg["params"]["group_size"]``  (extensible dict)
  3. global ``group_size`` from top-level config   (fallback)

Kernel dispatch per module (automatic, Marlin preferred):
  - bits in GPTQ_MARLIN_SUPPORTED_BITS (4/8) + sym=True (explicit) + desc_act=False:
    GPTQ Marlin kernel (try first, fastest)
  - bits 4/8 + sym unknown/false: GPTQ Exllama kernel (fallback if Marlin fails)
  - bits 2/3:             GPTQ Exllama kernel (gptq_v2 format)
  - bits 0 or not listed: UnquantizedLinearMethod
"""

from typing import Any, Optional

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
)

from vllm_plugins.gptq.constants import GPTQ_MARLIN_SUPPORTED_BITS, should_use_gptq_marlin
from vllm_plugins.utils.module import (
    _lookup_module_config,
    _parse_layer_and_module,
    _validate_quant_config_within_shard,
)
from vllm_plugins.utils.rotation import RotationMetadata, maybe_wrap_rotation_method

logger = init_logger(__name__)


@register_quantization_config("mixed_gptq")
class MixedGPTQConfig(QuantizationConfig):

    def __init__(
        self,
        quantization_bits,
        group_size=-1,
        desc_act=False,
        sym=False,
        lm_head_quantized=False,
        checkpoint_format="gptq_v2",
        rotation_metadata: Optional[RotationMetadata] = None,
    ):
        super().__init__()
        self.quantization_bits = quantization_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.sym = sym
        self.lm_head_quantized = lm_head_quantized
        self.checkpoint_format = checkpoint_format
        self.rotation_metadata = (
            rotation_metadata if rotation_metadata is not None else RotationMetadata()
        )

        all_bits = set()
        all_methods = set()
        for layer_cfg in quantization_bits:
            for mod_cfg in layer_cfg.values():
                all_bits.add(mod_cfg.get("bits", 0))
                all_methods.add(mod_cfg.get("method", "gptq"))
        logger.info(
            "MixedGPTQConfig: %d layers, bits=%s, methods=%s",
            len(quantization_bits),
            sorted(all_bits),
            sorted(all_methods),
        )

    def __repr__(self):
        return (
            f"MixedGPTQConfig(layers={len(self.quantization_bits)}, "
            f"group_size={self.group_size}, desc_act={self.desc_act})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "mixed_gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MixedGPTQConfig":
        quantization_bits = config.get("quantization_bits", [])

        if not quantization_bits and "layer_bits" in config:
            layer_bits = config["layer_bits"]
            quantization_bits = [{"_all": {"bits": b, "method": "gptq"}} for b in layer_bits]

        group_size = config.get("group_size", -1)
        desc_act = config.get("desc_act", False)
        sym = config.get("sym", False)
        lm_head_quantized = config.get("lm_head", False)
        checkpoint_format = config.get("checkpoint_format", "gptq_v2")
        return cls(
            quantization_bits=quantization_bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            lm_head_quantized=lm_head_quantized,
            checkpoint_format=checkpoint_format,
            rotation_metadata=RotationMetadata.from_quant_config(config),
        )

    def _resolve_group_size(self, mod_cfg: dict) -> int:
        """Resolve group_size: per-module direct > params > global."""
        if mod_cfg is not None:
            if "group_size" in mod_cfg:
                return mod_cfg["group_size"]
            additional = mod_cfg.get("params", {})
            if isinstance(additional, dict) and "group_size" in additional:
                return additional["group_size"]
        return self.group_size

    def _make_gptq_config(self, bits: int, group_size: Optional[int] = None) -> GPTQConfig:
        gs = group_size if group_size is not None else self.group_size
        return GPTQConfig(
            weight_bits=bits,
            group_size=gs,
            desc_act=self.desc_act,
            lm_head_quantized=self.lm_head_quantized,
            dynamic={},
            checkpoint_format=self.checkpoint_format,
        )

    def _make_marlin_config(self, bits: int, group_size: Optional[int] = None) -> GPTQMarlinConfig:
        gs = group_size if group_size is not None else self.group_size
        return GPTQMarlinConfig(
            weight_bits=bits,
            group_size=gs,
            desc_act=self.desc_act,
            is_sym=self.sym,
            lm_head_quantized=self.lm_head_quantized,
            dynamic={},
            full_config={
                "bits": bits,
                "group_size": gs,
                "desc_act": self.desc_act,
                "sym": self.sym,
            },
        )

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        # Override to prevent base class from scanning safetensors metadata.
        # MixedGPTQConfig uses quantization_bits for per-layer dispatch,
        # so modules_in_block_to_quantize is not needed.
        # Signature updated for vLLM>=0.20 (added hf_config kwarg).
        pass

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None

        layer_idx, module_suffix = _parse_layer_and_module(prefix)
        if layer_idx is None:
            return maybe_wrap_rotation_method(
                UnquantizedLinearMethod(),
                prefix=prefix,
                metadata=self.rotation_metadata,
            )

        mod_cfg = _lookup_module_config(self.quantization_bits, layer_idx, module_suffix)
        if mod_cfg is None:
            return maybe_wrap_rotation_method(
                UnquantizedLinearMethod(),
                prefix=prefix,
                metadata=self.rotation_metadata,
            )

        if not _validate_quant_config_within_shard(
            self.quantization_bits, layer_idx, module_suffix
        ):
            raise ValueError(
                f"Detected some but not all shards of {prefix} "
                "are quantized. All shards of fused layers "
                "to have the same precision."
            )

        bits = mod_cfg.get("bits", 0)
        method = mod_cfg.get("method", "gptq")
        group_size = self._resolve_group_size(mod_cfg)

        if bits == 0:
            return maybe_wrap_rotation_method(
                UnquantizedLinearMethod(),
                prefix=prefix,
                metadata=self.rotation_metadata,
            )

        int_bits = int(bits)

        if should_use_gptq_marlin(
            bits=int_bits,
            sym=self.sym,
            desc_act=self.desc_act,
            method=method,
        ):
            try:
                cfg = self._make_marlin_config(int_bits, group_size)
                return maybe_wrap_rotation_method(
                    GPTQMarlinLinearMethod(cfg),
                    prefix=prefix,
                    metadata=self.rotation_metadata,
                )
            except Exception:
                pass

        if method == "gptq" and int_bits in (2, 3, 4, 8):
            return maybe_wrap_rotation_method(
                GPTQLinearMethod(self._make_gptq_config(int_bits, group_size)),
                prefix=prefix,
                metadata=self.rotation_metadata,
            )

        logger.warning(
            "mixed_gptq: unsupported method=%s bits=%s at %s, " "falling back to unquantized",
            method,
            bits,
            prefix,
        )
        return maybe_wrap_rotation_method(
            UnquantizedLinearMethod(),
            prefix=prefix,
            metadata=self.rotation_metadata,
        )


def register_vllm_plugin():
    register_quantization_config(MixedGPTQConfig)
