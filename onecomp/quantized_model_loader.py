"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import glob
import json
import os
import re
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from .quantizer.dbf.config import resolve_dbf_layer_bits
from .quantizer.dbf.dbf_layer import DoubleBinaryLinear
from .quantizer.gptq.config import resolve_gptq_layer_group_size, resolve_gptq_layer_wbits
from .quantizer.gptq.gptq_layer import GPTQLinear
from .quantizer.onebit.onebit_layer import OneBitLinear
from .utils.device import get_default_device
from .utils.dtype import needs_bfloat16
from .utils.lora import LORA_ADAPTER_SUBDIR
from .utils.quant_config import get_quant_param

logger = getLogger(__name__)


class QuantizedModelLoader:
    """Loader for quantized models saved by onecomp (GPTQ, DBF, OneBit, etc.)."""

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

        If the directory additionally contains a PEFT-format LoRA adapter
        sidecar (``adapter_model.safetensors`` + ``adapter_config.json``), the
        matching ``GPTQLinear`` layers are automatically re-wrapped with
        ``LoRAGPTQLinear`` populated from the sidecar. This lets
        ``runner.save_quantized_model`` → ``load_quantized_model`` round-trip
        models produced by a LoRA post-process such as ``PostProcessLoraSFT``.

        For legacy models saved via ``torch.save`` (``.pt`` format), use
        :meth:`load_quantized_model_pt` instead.

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
        if needs_bfloat16(save_directory):
            torch_dtype = torch.bfloat16
        model = cls._build_empty_model_from_config(config_dict, torch_dtype)

        # Load state_dict from safetensors
        state_dict = cls._load_state_dict_from_dir(save_directory)

        # Align checkpoint key prefixes with the empty model built from config.
        # Gemma3 VLMs are a common case: weights saved from from_pretrained
        # use model.language_model.model.layers. (language_model is a
        # ForCausalLM wrapper) while from_config exposes
        # model.language_model.layers.* directly.
        state_dict = cls._remap_state_dict_keys(state_dict, model)

        # Replace quantized layers with empty modules
        cls._replace_quantized_layers(model, state_dict, quant_config)

        # Load all weights (quantized + non-quantized) in one go
        model.load_state_dict(state_dict, strict=False, assign=True)

        # ``assign=True`` swaps Parameter objects in place, which breaks the
        # weight sharing established by ``from_config`` for models with
        # ``tie_word_embeddings=True``.  Concretely, ``embed_tokens.weight``
        # gets replaced by the bf16 tensor from the checkpoint while
        # ``lm_head.weight`` keeps its original (often fp16) tensor, leading
        # to a dtype mismatch at the final ``F.linear`` call during
        # generation.  Re-tie when (a) the config tree still asks for it
        # (multi-config VLMs such as Llama 3.2-Vision place the flag in
        # ``text_config`` rather than at the top level, so we walk the
        # nested configs) and (b) ``lm_head`` is still a plain
        # ``nn.Linear`` -- if it has been replaced by a quantized layer
        # (e.g. ``GPTQLinear``) it has no ``weight`` attribute to retie
        # and tying would be meaningless.
        if cls._should_retie_word_embeddings(model.config):
            lm_head = getattr(model, "lm_head", None)
            if isinstance(lm_head, torch.nn.Linear):
                model.tie_weights()
                logger.info("Re-tied lm_head to embed_tokens after assign-load")

        # Safety net: ``load_state_dict(..., assign=True)`` only replaces
        # parameters whose key in the checkpoint exactly matches the model's
        # ``named_parameters`` path.  For some VLMs (e.g. Cohere2Vision's
        # ``multi_modal_projector``) the path prefix differs between the
        # checkpoint and the model class produced by ``from_config``, so a
        # subset of params silently keeps the empty-model dtype.  Together
        # with the config-based dtype default in
        # ``_build_empty_model_from_config`` this normalises any remaining
        # fp16 tensors of non-quantized modules to ``target_dtype``.  fp32
        # params (e.g. fp32 LayerNorm in mixed-precision models) are left
        # untouched, and quantized layers are skipped so that GPTQ scales
        # and similar fp16 metadata are preserved.
        target_dtype = (
            torch_dtype if torch_dtype is not None else cls._resolve_dtype_from_config(config_dict)
        )
        if target_dtype is None:
            target_dtype = torch.float16
        converted = cls._cast_fp16_to_target_dtype(model, target_dtype)
        if converted:
            logger.info(
                "Cast %d non-quantized fp16 tensor(s) to %s: %s",
                len(converted),
                target_dtype,
                converted,
            )

        cls._load_generation_config(model, save_directory)

        # Register Hadamard hooks for rotation-preprocessed models
        if quant_config.get("rotated", False):
            from .pre_process.rotation_utils import (
                collect_down_proj_types,
                register_online_hadamard_hooks,
            )

            fp32_had = quant_config.get("fp32_had", False)
            down_proj_types = collect_down_proj_types(model)

            hooks = register_online_hadamard_hooks(
                model,
                layers_cls=down_proj_types,
                fp32_had=fp32_had,
            )
            logger.info(
                "Registered Hadamard pre-hooks on %d down_proj layers (fp32_had=%s)",
                len(hooks),
                fp32_had,
            )

        # Re-apply LoRA adapter from PEFT-format sidecar if present.
        # This must run while the model is still on CPU, before dispatch_model,
        # so LoRA wrappers are included in the device map traversal below.
        cls._apply_lora_adapters_from_sidecar(model, save_directory)

        # Device placement
        if device_map:
            try:
                from accelerate import dispatch_model, infer_auto_device_map

                device_map_resolved = infer_auto_device_map(model)
                model = dispatch_model(model, device_map=device_map_resolved)
            except ImportError:
                model = model.to(get_default_device())

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
                model = model.to(get_default_device())

        tokenizer = AutoTokenizer.from_pretrained(
            save_directory,
            local_files_only=local_files_only,
        )

        return model, tokenizer

    @staticmethod
    def _load_generation_config(model: torch.nn.Module, save_directory: str) -> None:
        """Attach ``generation_config.json`` when present in the save directory.

        ``AutoModel*.from_config`` builds an empty model without the
        checkpoint's generation defaults.  Multimodal Gemma 4 models rely on
        fields such as ``suppress_tokens`` to block modality delimiter tokens
        during text-only ``generate()``.
        """
        gen_config_path = os.path.join(save_directory, "generation_config.json")
        if not os.path.isfile(gen_config_path):
            return
        model.generation_config = GenerationConfig.from_pretrained(save_directory)
        logger.info("Loaded generation_config.json from %s", save_directory)

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
    def _cast_fp16_to_target_dtype(model: torch.nn.Module, target_dtype: torch.dtype) -> List[str]:
        """Cast fp16 params/buffers of non-quantized modules to ``target_dtype``.

        Quantized layers (``GPTQLinear``, ``DoubleBinaryLinear``,
        ``OneBitLinear``) are skipped so their fp16 metadata (e.g. GPTQ
        ``scales``, OneBit ``a``/``b`` scaling vectors) is preserved.
        Only fp16 tensors are cast: fp32 params (e.g. fp32 LayerNorm in
        mixed-precision models) and other dtypes are left untouched.

        Args:
            model: The model whose parameters/buffers should be normalised.
            target_dtype: Destination dtype.  When equal to
                ``torch.float16`` this is a no-op.

        Returns:
            Fully-qualified names of every parameter / buffer whose
            dtype was actually converted (e.g. ``"model.layers.0.mlp.
            down_proj.weight"``).  An empty list means nothing needed
            casting (or ``target_dtype == torch.float16``).  The list
            form makes it easy for tests and operators to inspect
            which submodules were touched by the safety net.
        """
        converted: List[str] = []
        if target_dtype == torch.float16:
            return converted
        skip_types = (GPTQLinear, DoubleBinaryLinear, OneBitLinear)
        for mod_name, mod in model.named_modules():
            if isinstance(mod, skip_types):
                continue
            for p_name, p in mod.named_parameters(recurse=False):
                if p.dtype == torch.float16:
                    p.data = p.data.to(target_dtype)
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    converted.append(full_name)
            for b_name, b in mod.named_buffers(recurse=False):
                if b.dtype == torch.float16:
                    b.data = b.data.to(target_dtype)
                    full_name = f"{mod_name}.{b_name}" if mod_name else b_name
                    converted.append(full_name)
        return converted

    @classmethod
    def _should_retie_word_embeddings(cls, config: Any) -> bool:
        """Return True if any nesting level of ``config`` requests weight tying.

        Single-config language models (e.g. Llama, Qwen) expose
        ``tie_word_embeddings`` directly on ``model.config``.  Multi-
        config VLMs vary: ``gemma-4`` puts the flag at the top level
        but ``llama3.2-vlm-torchtune`` and other torchtune-derived
        checkpoints place it inside ``text_config`` only, so the naive
        ``getattr(model.config, "tie_word_embeddings", False)`` would
        miss the tying request and skip the re-tie that
        ``load_state_dict(..., assign=True)`` necessitates.

        We walk the config tree shallowly: any direct sub-attribute
        that itself exposes ``tie_word_embeddings`` is inspected.  The
        check is intentionally non-recursive past one level because
        HuggingFace nests language sub-configs at most one level deep
        in practice (``text_config``, ``language_config`` etc.) and a
        deeper recursion would risk being confused by unrelated
        sub-objects.

        Args:
            config: A ``transformers.PretrainedConfig``-like object
                (anything supporting ``getattr``).

        Returns:
            ``True`` if ``tie_word_embeddings`` is truthy at the top
            level or on any direct sub-attribute that itself looks
            like a config (i.e. carries a ``tie_word_embeddings``
            attribute).  ``False`` otherwise.
        """
        if getattr(config, "tie_word_embeddings", False):
            return True
        try:
            sub_items = vars(config).items()
        except TypeError:
            return False
        for _, value in sub_items:
            # Duck-type check: only descend into things that themselves
            # carry the flag, so we don't accidentally walk unrelated
            # auxiliary objects (e.g. tokenizer caches) that happen to
            # be stored on the config.
            if hasattr(value, "tie_word_embeddings") and getattr(
                value, "tie_word_embeddings", False
            ):
                return True
        return False

    @staticmethod
    def _resolve_dtype_from_config(
        config_dict: Dict,
    ) -> Optional[torch.dtype]:
        """Read ``torch_dtype`` / ``dtype`` from a config dict.

        Accepts both the JSON-serialised string form (e.g. ``"bfloat16"``)
        and a real ``torch.dtype`` value.  Returns ``None`` when the field
        is missing, ``"auto"``, or otherwise unresolvable.
        """
        for key in ("torch_dtype", "dtype"):
            val = config_dict.get(key)
            if isinstance(val, torch.dtype):
                return val
            if isinstance(val, str) and val and val != "auto":
                resolved = getattr(torch, val, None)
                if isinstance(resolved, torch.dtype):
                    return resolved
        return None

    @classmethod
    def _build_empty_model_from_config(
        cls,
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

        # Default to the dtype recorded in config.json so the empty model
        # starts in the same dtype as the saved checkpoint.  This avoids
        # leaving non-quantized submodules at the hard-coded fp16 default
        # if ``load_state_dict(..., assign=True)`` cannot find their key
        # in the state_dict (e.g. tied or path-shifted VLM submodules).
        if torch_dtype is None:
            torch_dtype = cls._resolve_dtype_from_config(clean_config)
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

    @classmethod
    def _remap_state_dict_keys(cls, state_dict: dict, model: torch.nn.Module) -> dict:
        """Rewrite checkpoint keys so they match model parameter paths.

        Quantized models are saved from a from_pretrained instance whose
        submodule naming can differ from the from_config model built at
        load time.  Without remapping, load_state_dict(..., assign=True)
        silently skips mismatched keys and leaves layers at their empty-model
        initial values (often all zeros for quantized buffers).

        Remapping runs before _replace_quantized_layers, so the empty
        model still exposes nn.Linear.weight rather than GPTQ buffers
        (``qweight``, ``scales``, …).  Known prefix rewrites are therefore
        applied from checkpoint key patterns alone; they must not require the
        destination key to already exist in model.named_parameters().

        Args:
            state_dict: Tensors loaded from *.safetensors.
            model: Empty model returned by _build_empty_model_from_config.

        Returns:
            A new dict with keys renamed where a unique target exists in
            model.  Unmatched keys are kept under their original names
            so strict=False loading can still proceed.
        """
        model_keys = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
        if not any(
            cls._apply_known_state_dict_key_rewrites(key) is not None for key in state_dict
        ) and all(key in model_keys for key in state_dict):
            return state_dict

        remapped: dict = {}
        rewrite_count = 0
        for ckpt_key, tensor in state_dict.items():
            if ckpt_key in model_keys:
                remapped[ckpt_key] = tensor
                continue

            target_key = cls._resolve_state_dict_key(ckpt_key, model_keys)
            if target_key is not None and target_key != ckpt_key:
                remapped[target_key] = tensor
                rewrite_count += 1
            else:
                remapped[ckpt_key] = tensor

        if rewrite_count:
            logger.info(
                "Remapped %d state_dict key(s) to match model module paths",
                rewrite_count,
            )
        return remapped

    @staticmethod
    def _apply_known_state_dict_key_rewrites(ckpt_key: str) -> Optional[str]:
        """Return a rewritten key for known save/load prefix drift, else None."""
        if ".language_model.model." in ckpt_key:
            return ckpt_key.replace(".language_model.model.", ".language_model.", 1)
        if ckpt_key.startswith("language_model.model."):
            return "model." + ckpt_key.replace("language_model.model.", "language_model.", 1)
        return None

    @staticmethod
    def _resolve_state_dict_key(ckpt_key: str, model_keys: set) -> Optional[str]:
        """Return the remapped key for ckpt_key, or None if unknown."""
        rewritten = QuantizedModelLoader._apply_known_state_dict_key_rewrites(ckpt_key)
        if rewritten is not None:
            return rewritten

        # Suffix fallback for other prefix drift: only when the suffix maps
        # uniquely onto the current model (non-quantized params/buffers).
        match = re.search(r"(layers\.\d+(?:\..+)*)$", ckpt_key)
        if match:
            suffix = match.group(1)
            hits = [name for name in model_keys if name.endswith(suffix)]
            if len(hits) == 1:
                return hits[0]
        return None

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
    def _resolve_name_by_layer_suffix(
        name: str,
        candidates: Dict[str, Any],
        *,
        on_ambiguous: str = "first",
    ) -> Optional[str]:
        """Resolve *name* against *candidates* by exact or layer-suffix match.

        ``on_ambiguous`` controls what happens when the suffix matches more than
        one candidate:

        - ``"first"`` (default): keep the quantized-layer load path best-effort
          by preferring a single ``language_model`` hit, then falling back to
          ``hits[0]``. This is intended for VLM tied/shared submodules that
          point at the *same* weights.
        - ``"error"``: raise ``ValueError``. Required for the LoRA re-wrap path,
          where colliding candidates can be *distinct* layers (e.g. the same
          ``layers.N.<suffix>`` under both ``language_model`` and ``vision``),
          so ambiguity is rejected before applying the ``language_model``
          preference.
        """
        if name in candidates:
            return name

        # For VLMs with tied/shared submodules, only the prefix may differ.
        match = re.search(r"(layers\.\d+\..+)$", name)
        if not match:
            return None
        suffix = match.group(1)
        hits = [candidate for candidate in candidates if candidate.endswith(suffix)]
        if len(hits) > 1:
            if on_ambiguous == "error":
                logger.warning(
                    "Ambiguous suffix %s for %s: %s",
                    suffix,
                    name,
                    hits,
                )
                raise ValueError(
                    f"Ambiguous layer-suffix match for {name!r}: suffix {suffix!r} "
                    f"matches multiple candidates {hits}. Refusing to guess which "
                    "layer to target."
                )
            lang_hits = [candidate for candidate in hits if "language_model" in candidate]
            if len(lang_hits) == 1:
                hits = lang_hits
            else:
                logger.warning(
                    "Ambiguous suffix %s for %s: %s",
                    suffix,
                    name,
                    hits,
                )
        return hits[0] if hits else None

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

        # For VLMs with tied/shared submodules (e.g. Gemma3), the
        # named_modules() path may differ from the state_dict key prefix.
        # Build a suffix -> state_dict prefix map to handle this.
        sd_prefix_map: dict[str, str] = {}
        for key in state_dict:
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                sd_prefix_map.setdefault(parts[0], parts[0])

        def _get_layer_sd(name: str) -> dict:
            prefix = name + "."
            result = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if result:
                return result
            # Fallback: match by layer suffix (e.g. "layers.0.self_attn.q_proj")
            alt_name = QuantizedModelLoader._resolve_name_by_layer_suffix(name, sd_prefix_map)
            if alt_name:
                alt_prefix = alt_name + "."
                return {
                    k[len(alt_prefix) :]: v
                    for k, v in state_dict.items()
                    if k.startswith(alt_prefix)
                }
            return {}

        for name in quantized_names:
            if name not in name_to_module:
                continue

            layer_sd = _get_layer_sd(name)

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
            elif effective_method == "onebit":
                quantized_module = OneBitLinear.from_saved_state(
                    layer_sd,
                    in_features=in_features,
                    out_features=out_features,
                    empty=True,
                )
            else:
                raise ValueError(
                    f"Unknown quant_method: {quant_method} (effective: {effective_method})"
                )

            QuantizedModelLoader._set_module_by_name(model, name, quantized_module)

    @staticmethod
    def _apply_lora_adapters_from_sidecar(model, save_directory: str) -> int:
        """Re-wrap GPTQLinear layers with LoRAGPTQLinear from a PEFT-format sidecar.

        Looks for ``adapter_model.safetensors`` + ``adapter_config.json`` under
        ``save_directory/lora_adapter/``. If both are present, each referenced
        GPTQLinear layer is replaced in-place with a ``LoRAGPTQLinear`` wrapper
        populated with the saved LoRA weights.

        For backward compatibility, also checks the legacy top-level layout
        (``save_directory/adapter_model.safetensors``) used by an earlier
        version of :meth:`Runner.save_quantized_model`.

        Returns:
            int: Number of layers wrapped (0 if no adapter sidecar was found).
        """
        adapter_dir = os.path.join(save_directory, LORA_ADAPTER_SUBDIR)
        adapter_weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not (os.path.isfile(adapter_weights_path) and os.path.isfile(adapter_config_path)):
            # Fallback to legacy top-level layout.
            legacy_weights = os.path.join(save_directory, "adapter_model.safetensors")
            legacy_config = os.path.join(save_directory, "adapter_config.json")
            if os.path.isfile(legacy_weights) and os.path.isfile(legacy_config):
                adapter_weights_path = legacy_weights
                adapter_config_path = legacy_config
            else:
                return 0

        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)

        lora_r = int(adapter_config["r"])
        lora_alpha = int(adapter_config["lora_alpha"])
        lora_dropout = float(adapter_config.get("lora_dropout", 0.0))

        adapter_sd = load_file(adapter_weights_path)

        peft_prefix = "base_model.model."
        per_layer: Dict[str, Dict[str, torch.Tensor]] = {}

        for key, tensor in adapter_sd.items():
            if not key.startswith(peft_prefix):
                logger.warning("Skipping unexpected adapter key %s", key)
                continue
            body = key[len(peft_prefix) :]
            # Support both the plain form "<path>.lora_A.weight" and PEFT's
            # adapter-name form "<path>.lora_A.default.weight".
            if body.endswith(".lora_A.weight"):
                layer_path = body[: -len(".lora_A.weight")]
                per_layer.setdefault(layer_path, {})["A"] = tensor
            elif body.endswith(".lora_B.weight"):
                layer_path = body[: -len(".lora_B.weight")]
                per_layer.setdefault(layer_path, {})["B"] = tensor
            elif body.endswith(".lora_A.default.weight"):
                layer_path = body[: -len(".lora_A.default.weight")]
                per_layer.setdefault(layer_path, {})["A"] = tensor
            elif body.endswith(".lora_B.default.weight"):
                layer_path = body[: -len(".lora_B.default.weight")]
                per_layer.setdefault(layer_path, {})["B"] = tensor
            else:
                logger.warning("Skipping unrecognized adapter key %s", key)

        if not per_layer:
            return 0

        # Inline import to avoid pulling post_process into module-import time
        # and to sidestep any circular-import risk.
        from .post_process.post_process_lora_sft import LoRAGPTQLinear

        name_to_module = dict(model.named_modules())
        wrapped = 0
        for layer_path, ab in per_layer.items():
            if "A" not in ab or "B" not in ab:
                logger.warning(
                    "Adapter layer %s missing lora_A or lora_B; skipping",
                    layer_path,
                )
                continue
            # Fail fast on ambiguity: unlike the quantized-layer path, colliding
            # candidates here can be distinct layers, so mis-wrapping would pass
            # the wrapped-count check below undetected.
            resolved_layer_path = QuantizedModelLoader._resolve_name_by_layer_suffix(
                layer_path,
                name_to_module,
                on_ambiguous="error",
            )
            if resolved_layer_path is None:
                logger.warning(
                    "Adapter references layer %s not found in model; skipping",
                    layer_path,
                )
                continue
            if resolved_layer_path != layer_path:
                logger.info(
                    "Resolved adapter layer %s -> %s by suffix match",
                    layer_path,
                    resolved_layer_path,
                )
            base_layer = name_to_module[resolved_layer_path]
            if not isinstance(base_layer, GPTQLinear):
                logger.warning(
                    "Adapter layer %s is %s, expected GPTQLinear; skipping",
                    resolved_layer_path,
                    type(base_layer).__name__,
                )
                continue

            wrapper = LoRAGPTQLinear(
                base_layer=base_layer,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            with torch.no_grad():
                wrapper.lora_A.weight.copy_(ab["A"].to(wrapper.lora_A.weight.dtype))
                wrapper.lora_B.weight.copy_(ab["B"].to(wrapper.lora_B.weight.dtype))
            # Match the base layer's device so the wrapper and base share placement.
            base_device = base_layer.qweight.device
            wrapper.to(base_device)
            QuantizedModelLoader._set_module_by_name(model, resolved_layer_path, wrapper)
            wrapped += 1

        if wrapped < len(per_layer):
            expected = len(per_layer)
            skipped = expected - wrapped
            raise ValueError(
                "Failed to apply LoRA adapter sidecar fully: "
                f"applied {wrapped}/{expected} layer(s), skipped {skipped}. "
                "See preceding WARNING logs for skipped layer names and reasons."
            )

        logger.info(
            "Re-wrapped %d GPTQLinear layers with LoRAGPTQLinear from adapter sidecar",
            wrapped,
        )
        return wrapped
