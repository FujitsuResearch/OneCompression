"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import glob
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from .quantizer.dbf.dbf_layer import DoubleBinaryLinear
from .quantizer.gptq.gptq_layer import GPTQLinear


class QuantizedModelLoader:
    """Loader for quantized models saved by onecomp (GPTQ, DBF, etc.)."""

    @classmethod
    def load_quantized_model(
        cls,
        save_directory: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = True,
    ) -> Tuple[Any, Any]:
        """Load a quantized model and tokenizer from a directory saved by onecomp.

        The directory must contain:
        - config.json (with quantization_config)
        - tokenizer files
        - model.safetensors (quantized layers: qweight/scales for GPTQ, scaling0/bp for DBF)

        Quantization parameters (quant_method, bits, group_size, etc.) are read from
        config.json and quantized layers are reconstructed directly from the safetensors
        state_dict. No quantization_results.pt is needed.

        Args:
            save_directory: Path to the saved model directory.
            torch_dtype: Model dtype (default: torch.float16).
            device_map: Device placement (default: "auto").
            trust_remote_code: Passed to from_pretrained.
            local_files_only: Passed to from_pretrained.

        Returns:
            (model, tokenizer)

        Example:
            >>> model, tokenizer = QuantizedModelLoader.load_quantized_model("./tinyllama_gptq3")
        """
        save_directory = os.path.abspath(save_directory)
        if not os.path.isdir(save_directory):
            raise FileNotFoundError(f"Saved model directory not found: {save_directory}")

        config_dict, quant_config = cls._load_config_and_quant_config(save_directory)
        model = cls._build_empty_model_from_config(config_dict, torch_dtype)

        # Load state_dict from safetensors
        state_dict = cls._load_state_dict_from_dir(save_directory)

        # Replace quantized layers with empty modules
        cls._replace_quantized_layers(model, state_dict, quant_config)

        # Load all weights (quantized + non-quantized) in one go
        model.load_state_dict(state_dict, strict=False, assign=True)

        # Device placement
        if device_map:
            try:
                from accelerate import dispatch_model, infer_auto_device_map

                device_map_resolved = infer_auto_device_map(model)
                model = dispatch_model(model, device_map=device_map_resolved)
            except ImportError:
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            save_directory,
            local_files_only=local_files_only,
        )

        return model, tokenizer

    @staticmethod
    def _load_config_and_quant_config(save_directory: str) -> Tuple[Dict, Dict]:
        """Load config.json and return (config_dict, quant_config) with validation.

        Raises:
            FileNotFoundError: If config.json is missing.
            ValueError: If quantization_config, quant_method, or
                modules_in_block_to_quantize is missing.
        """
        config_path = os.path.join(save_directory, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found in {save_directory}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        quant_config = config_dict.get("quantization_config")
        if quant_config is None:
            raise ValueError(
                "No quantization config found in config.json. "
                "Expected 'quantization_config'."
            )
        if quant_config.get("quant_method") is None:
            raise ValueError("quant_method not found in quantization config.")
        if "modules_in_block_to_quantize" not in quant_config:
            raise ValueError(
                "quantization_config must contain 'modules_in_block_to_quantize'."
            )

        return config_dict, quant_config

    @staticmethod
    def _build_empty_model_from_config(
        config_dict: Dict,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        """Build an empty CausalLM model from config_dict.

        Raises:
            ValueError: If model_type is missing or not in CONFIG_MAPPING.
        """
        clean_config = dict(config_dict)
        clean_config.pop("quantization_config", None)

        model_type = clean_config.get("model_type")
        if not model_type or model_type not in CONFIG_MAPPING:
            raise ValueError(
                f"Cannot build config: model_type={model_type!r} not in CONFIG_MAPPING."
            )

        dtype = torch_dtype if torch_dtype is not None else torch.float16
        config_cls = CONFIG_MAPPING[model_type]
        model_config = config_cls.from_dict(clean_config) # to build empty model from config
        return AutoModelForCausalLM.from_config(model_config, torch_dtype=dtype)

    @staticmethod
    def _set_module_by_name(
        model: torch.nn.Module, full_name: str, module: torch.nn.Module
    ) -> None:
        """Replace the submodule at *full_name* (dotted path) with *module*."""
        name_to_module = dict(model.named_modules())
        parent_name, _, child_name = full_name.rpartition(".")
        parent = name_to_module.get(parent_name, model)
        setattr(parent, child_name, module)

    @staticmethod
    def _load_state_dict_from_dir(directory: str) -> dict:
        """Load all tensors from *.safetensors in *directory*.

        Raises:
            FileNotFoundError: If no *.safetensors files are found in *directory*.
        """
        state_dict: dict = {}
        safetensors_files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
        if safetensors_files:
            for f in safetensors_files:
                state_dict.update(load_file(f))
        if not state_dict:
            raise FileNotFoundError(
                f"No model weights found in {directory}. "
                "Expected *.safetensors files."
            )
        return state_dict

    @staticmethod
    def _replace_quantized_layers(model, state_dict: dict, quant_config: dict):
        """Replace ``nn.Linear`` with empty quantized modules for layers in config.

        *quant_config* must contain ``modules_in_block_to_quantize`` (list of layer
        names). Modules are created with zero buffers of the right shape; *state_dict*
        is left unchanged so the caller can ``load_state_dict(state_dict)`` once to
        fill all weights.
        """
        quant_method = quant_config["quant_method"]
        quantized_names = sorted(quant_config["modules_in_block_to_quantize"])
        name_to_module = dict(model.named_modules())

        for name in quantized_names:
            prefix = name + "."
            layer_sd = {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

            linear = name_to_module[name]
            in_features, out_features = linear.in_features, linear.out_features

            if quant_method == "gptq":
                quantized_module = GPTQLinear.from_saved_state(
                    layer_sd,
                    in_features=in_features,
                    out_features=out_features,
                    wbits=quant_config.get("bits", quant_config.get("wbits")),
                    groupsize=quant_config.get("group_size", quant_config.get("groupsize", -1)),
                    actorder=quant_config.get("desc_act", quant_config.get("actorder", False)),
                    empty=True,
                )
            elif quant_method == "dbf":
                quantized_module = DoubleBinaryLinear.from_saved_state(
                    layer_sd,
                    in_features=in_features,
                    out_features=out_features,
                    empty=True,
                )
            else:
                raise ValueError(f"Unknown quant_method: {quant_method}")

            QuantizedModelLoader._set_module_by_name(model, name, quantized_module)
