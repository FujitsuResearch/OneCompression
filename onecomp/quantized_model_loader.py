"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from .quantizer.dbf.config import resolve_dbf_layer_bits
from .quantizer.dbf.dbf_layer import DoubleBinaryLinear
from .quantizer.gptq.config import resolve_gptq_layer_wbits, resolve_gptq_layer_group_size
from .quantizer.gptq.gptq_layer import GPTQLinear
from .utils.quant_config import get_quant_param


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
        """Load a quantized model and tokenizer from a safetensors directory.

        The directory must contain:
        - config.json (with quantization_config)
        - tokenizer files
        - model.safetensors (quantized layers: qweight/scales for GPTQ, scaling0/bp for DBF)

        Quantization parameters (quant_method, bits, group_size, etc.) are read from
        config.json and quantized layers are reconstructed directly from the safetensors
        state_dict. No quantization_results.pt is needed.

        For models saved with post-processing modifications (e.g. LoRA adapters),
        use :meth:`load_quantized_model_pt` instead.

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

    @classmethod
    def load_quantized_model_pt(
        cls,
        save_directory: str,
        *,
        device_map: str = "auto",
        local_files_only: bool = True,
    ) -> Tuple[Any, Any]:
        """Load a quantized model and tokenizer saved as a PyTorch .pt file.

        Use this method to load models saved by
        :meth:`Runner.save_quantized_model_pt`, which preserves custom
        module types (e.g. ``LoRAGPTQLinear`` from LoRA post-processing).

        The directory must contain:
        - ``model.pt`` (serialized with ``torch.save``)
        - Tokenizer files

        Args:
            save_directory: Path to the saved model directory.
            device_map: Device placement (default: ``"auto"``).
                Set to ``""`` or ``None`` to skip device placement.
            local_files_only: Passed to ``AutoTokenizer.from_pretrained``.

        Returns:
            (model, tokenizer)

        Example:
            >>> model, tokenizer = QuantizedModelLoader.load_quantized_model_pt(
            ...     "./quantized_model_lora"
            ... )
        """
        save_directory = os.path.abspath(save_directory)
        if not os.path.isdir(save_directory):
            raise FileNotFoundError(f"Saved model directory not found: {save_directory}")

        model_path = os.path.join(save_directory, "model.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"model.pt not found in {save_directory}. "
                "This directory may have been saved with save_quantized_model() "
                "(safetensors format); use load_quantized_model() instead."
            )

        model = torch.load(model_path, map_location="cpu", weights_only=False)

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
                "No quantization config found in config.json. " "Expected 'quantization_config'."
            )
        if quant_config.get("quant_method") is None:
            raise ValueError("quant_method not found in quantization config.")

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
        model_config = config_cls.from_dict(clean_config)
        try:
            return AutoModelForCausalLM.from_config(model_config, torch_dtype=dtype)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_config(model_config, torch_dtype=dtype)

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
                f"No model weights found in {directory}. " "Expected *.safetensors files."
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
        # mixed_* use the same tensor format as the base method (e.g. mixed_gptq -> gptq)
        if quant_method and quant_method.startswith("mixed_"):
            effective_method = quant_method[len("mixed_") :]
        else:
            effective_method = quant_method

        # Validate that all entries in quantization_bits use the same quant method.
        # Per-layer method switching is not supported; raise early with a clear message.
        quantization_bits_list = quant_config.get("quantization_bits")
        if quantization_bits_list:
            methods_found: set = set()
            for layer_cfg in quantization_bits_list:
                for mod_cfg in layer_cfg.values():
                    if isinstance(mod_cfg, dict) and "method" in mod_cfg:
                        methods_found.add(mod_cfg["method"])
            if len(methods_found) > 1:  # TODO: support mixed methods
                raise ValueError(
                    "Mixed quantization methods across layers are not supported. "
                    f"Found methods: {sorted(methods_found)}. "
                    "All layers must use the same quantization method."
                )

        if "modules_in_block_to_quantize" not in quant_config:
            raise ValueError(
                "modules_in_block_to_quantize is required in quantization_config "
                "but was not found."
            )
        module_list = quant_config["modules_in_block_to_quantize"]
        if not module_list:
            return  # nothing to replace

        quantization_bits_list = quant_config.get("quantization_bits") or []
        if quant_method and quant_method.startswith("mixed_") and quantization_bits_list:
            # Build from quantization_bits; use module_list[0] to infer layer name prefix
            first_name = module_list[0]
            prefix_match = re.match(r"^(.+\.layers)\.\d+\.", first_name)
            prefix = prefix_match.group(1) if prefix_match else "model.layers"
            quantized_names = sorted(
                f"{prefix}.{i}.{suffix}"
                for i, layer_cfg in enumerate(quantization_bits_list)
                if isinstance(layer_cfg, dict)
                for suffix in layer_cfg
            )
        else:
            quantized_names = sorted(module_list)

        name_to_module = dict(model.named_modules())

        for name in quantized_names:
            if name not in name_to_module:
                # Layer name not found in model skeleton; skip rather than crash.
                continue

            prefix = name + "."
            layer_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

            linear = name_to_module[name]
            in_features, out_features = linear.in_features, linear.out_features

            if effective_method == "gptq":
                layer_wbits = resolve_gptq_layer_wbits(name, quant_config)
                layer_groupsize = resolve_gptq_layer_group_size(name, quant_config)
                quantized_module = GPTQLinear.from_saved_state(
                    layer_sd,
                    in_features=in_features,
                    out_features=out_features,
                    wbits=layer_wbits,
                    groupsize=layer_groupsize,
                    actorder=get_quant_param(quant_config, "desc_act", "actorder", default=False),
                    empty=True,
                    checkpoint_format=get_quant_param(
                        quant_config, "checkpoint_format", default="gptq"
                    ),
                )
            elif effective_method == "dbf":
                layer_target_bits = resolve_dbf_layer_bits(name, quant_config)
                quantized_module = DoubleBinaryLinear.from_saved_state(
                    layer_sd,
                    in_features=in_features,
                    out_features=out_features,
                    empty=True,
                    target_bits=layer_target_bits,
                )
            else:
                raise ValueError(
                    f"Unknown quant_method: {quant_method} (effective: {effective_method})"
                )

            QuantizedModelLoader._set_module_by_name(model, name, quantized_module)
