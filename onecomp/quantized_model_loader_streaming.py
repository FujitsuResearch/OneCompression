"""Low-memory load path of :class:`~onecomp.quantized_model_loader.QuantizedModelLoader`.

The safetensors headers become a state dict of meta tensors, the existing
loader steps resolve and validate it on a meta skeleton, and only then are the
tensors read one at a time onto their final device. A checkpoint outside the
supported scope is reported as :class:`StreamingUnsupported` before any tensor
value is read.

Copyright 2025-2026 Fujitsu Ltd.
"""

import contextlib
import glob
import os
from dataclasses import dataclass
from itertools import chain
from typing import Any, Optional

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from .quantized_model_loader import QuantizedModelLoader
from .utils.dtype import needs_bfloat16
from .utils.lora import LORA_ADAPTER_SUBDIR

_BUILTIN_CLASSES_BY_MODEL_TYPE = {
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    "llama": (LlamaConfig, LlamaForCausalLM),
}
_QUANT_METHODS = ("gptq", "mixed_gptq")
_LORA_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class StreamingUnsupported:
    """Why a checkpoint needs the legacy loader, found before any tensor value is read."""

    reason: str


@dataclass
class StreamingPlan:
    """Meta skeleton in its final layout and dtypes, plus the tensors to read into it.

    Attributes:
        model: Model whose checkpoint tensors are still meta placeholders.
        tensors: ``(model key, file, checkpoint key)`` of every tensor to read.
        destination: Device that receives every tensor.
        device_map: Value for ``model.hf_device_map``; ``None`` leaves it unset.
        retie: Whether ``lm_head`` shares the input embedding weight.
    """

    model: torch.nn.Module
    tensors: list[tuple[str, str, str]]
    destination: torch.device
    device_map: Optional[dict[str, Any]]
    retie: bool


def check_streaming_scope(
    save_directory: str,
    config_dict: dict[str, Any],
    quant_config: dict[str, Any],
    device_map: Any,
) -> Optional[StreamingUnsupported]:
    """Check from the config and file names alone whether a checkpoint can be streamed.

    Args:
        save_directory: Saved model directory.
        config_dict: Parsed ``config.json``.
        quant_config: Its validated ``quantization_config``.
        device_map: The caller's ``device_map``.

    Returns:
        ``None`` if the checkpoint is a candidate, otherwise the reason it is not.
    """
    model_type = config_dict.get("model_type")
    if model_type not in _BUILTIN_CLASSES_BY_MODEL_TYPE:
        return StreamingUnsupported(f"model_type {model_type!r} is not supported")
    # The model builder resolves both classes from these registries. Checking them
    # before construction keeps custom constructors out of the planning step.
    config_class, model_class = _BUILTIN_CLASSES_BY_MODEL_TYPE[model_type]
    if (
        CONFIG_MAPPING[model_type] is not config_class
        or MODEL_FOR_CAUSAL_LM_MAPPING[config_class] is not model_class
    ):
        return StreamingUnsupported(f"model_type {model_type!r} is registered to a custom class")
    quant_method = quant_config["quant_method"]
    if quant_method not in _QUANT_METHODS:
        return StreamingUnsupported(f"quant_method {quant_method!r} is not supported")
    if not quant_config["modules_in_block_to_quantize"]:
        return StreamingUnsupported("the model has no quantized layers")
    if quant_config.get("rotated", False):
        return StreamingUnsupported("rotated models are not supported")
    # Any trace of a sidecar, even a dangling link or half of the file pair, is
    # left to the legacy loader's own handling.
    for directory in (save_directory, os.path.join(save_directory, LORA_ADAPTER_SUBDIR)):
        if any(os.path.lexists(os.path.join(directory, name)) for name in _LORA_ADAPTER_FILES):
            return StreamingUnsupported("a LoRA adapter sidecar is present")
    if device_map is not None and not isinstance(device_map, str):
        return StreamingUnsupported(
            f"device_map of type {type(device_map).__name__} is not supported"
        )
    return None


def read_meta_state_dict(
    save_directory: str,
) -> tuple[dict[str, torch.Tensor], dict[str, str]] | StreamingUnsupported:
    """Build meta tensors from the headers of the root ``*.safetensors`` files.

    The files are selected as the legacy loader selects them, so an index file
    is never read and a stale or broken one has no effect.

    Args:
        save_directory: Saved model directory.

    Returns:
        ``(meta_state, files)``: a meta tensor with the stored shape and dtype for
        each key, and the file that holds each key. Otherwise, the reason the
        checkpoint cannot be streamed.
    """
    meta_state: dict[str, torch.Tensor] = {}
    files: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(save_directory, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in meta_state:
                    # The legacy loader keeps the last file's tensor; leave that to it.
                    return StreamingUnsupported(f"tensor {key!r} is stored in several files")
                tensor_slice = reader.get_slice(key)
                dtype = _SAFETENSORS_DTYPES.get(tensor_slice.get_dtype())
                if dtype is None:
                    return StreamingUnsupported(
                        f"tensor {key!r} has unsupported dtype {tensor_slice.get_dtype()}"
                    )
                meta_state[key] = torch.empty(tensor_slice.get_shape(), dtype=dtype, device="meta")
                files[key] = path
    return meta_state, files


def plan_streaming_load(
    save_directory: str,
    config_dict: dict[str, Any],
    quant_config: dict[str, Any],
    torch_dtype: Optional[torch.dtype],
    device_map: Any,
) -> StreamingPlan | StreamingUnsupported:
    """Run the legacy load steps on a meta skeleton to plan a load without reading values.

    Key resolution, layer replacement, the critical-key and shape checks, the
    re-tie and the dtype rules are those of the legacy loader, applied to meta
    tensors built from the checkpoint headers.

    Args:
        save_directory: Absolute path of the saved model directory.
        config_dict: Parsed ``config.json``.
        quant_config: Its validated ``quantization_config``.
        torch_dtype: As in ``load_quantized_model``.
        device_map: As in ``load_quantized_model``.

    Returns:
        The plan, or the reason the checkpoint needs the legacy loader.

    Raises:
        FileNotFoundError: If the root ``*.safetensors`` files hold no tensor.
        RuntimeError: For the legacy quantized-layer, shape and critical-key errors.
        ValueError: For the legacy error on mixed per-layer quantization methods.
    """
    unsupported = check_streaming_scope(save_directory, config_dict, quant_config, device_map)
    if unsupported is not None:
        return unsupported
    header = read_meta_state_dict(save_directory)
    if isinstance(header, StreamingUnsupported):
        return header
    meta_state, files = header
    if not meta_state:
        raise FileNotFoundError(
            f"No model weights found in {save_directory}. Expected *.safetensors files."
        )

    if needs_bfloat16(save_directory):
        torch_dtype = torch.bfloat16
    with init_empty_weights(include_buffers=False):
        model = QuantizedModelLoader._build_empty_model_from_config(config_dict, torch_dtype)
    # The registry check covers from_config; this one covers a replaced builder.
    _, model_class = _BUILTIN_CLASSES_BY_MODEL_TYPE[config_dict["model_type"]]
    if type(model) is not model_class:
        return StreamingUnsupported(f"the model was built as {type(model).__name__}")

    # Key resolution moves the tensor objects without copying them, so their ids
    # still identify the checkpoint key after it.
    checkpoint_key_of = {id(tensor): key for key, tensor in meta_state.items()}
    state = QuantizedModelLoader._replace_quantized_layers(model, meta_state, quant_config)
    state = QuantizedModelLoader._remap_state_dict_keys(state, model)
    incompat = model.load_state_dict(state, strict=False, assign=True)
    retie = QuantizedModelLoader._should_retie_lm_head(model)
    missing = [key for key in incompat.missing_keys if not (retie and key == "lm_head.weight")]
    unexpected = incompat.unexpected_keys
    QuantizedModelLoader._raise_on_critical_keys(missing, unexpected)
    if missing or unexpected:
        # Meta placeholders have no initial values to keep, unlike the legacy model.
        return StreamingUnsupported(
            f"{len(missing)} model key(s) are missing from the checkpoint and "
            f"{len(unexpected)} checkpoint key(s) are unexpected"
        )
    if retie:
        model.tie_weights()
    target_dtype = (
        torch_dtype
        or QuantizedModelLoader._resolve_dtype_from_config(config_dict)
        or torch.float16
    )
    QuantizedModelLoader._cast_fp16_to_target_dtype(model, target_dtype)
    tensors = [
        (key, files[checkpoint_key_of[id(tensor)]], checkpoint_key_of[id(tensor)])
        for key, tensor in state.items()
        # The re-tied head takes the embedding values, as in the legacy loader.
        if not (retie and key == "lm_head.weight")
    ]

    if not device_map:
        return StreamingPlan(model, tensors, torch.device("cpu"), None, retie)
    resolved_map = infer_auto_device_map(model)
    devices = set(resolved_map.values())
    if len(devices) > 1:
        return StreamingUnsupported(f"the inferred placement spans {sorted(map(str, devices))}")
    (device,) = devices
    if isinstance(device, int) and torch.cuda.is_available():
        destination = torch.device("cuda", device)
    elif device == "cpu":
        destination = torch.device("cpu")
    else:
        return StreamingUnsupported(f"the inferred placement is {device!r}")
    return StreamingPlan(model, tensors, destination, resolved_map, retie)


def materialize(plan: StreamingPlan) -> torch.nn.Module:
    """Read the planned tensors one at a time onto the destination and check the result.

    Args:
        plan: Result of :func:`plan_streaming_load`.

    Returns:
        The planned model with every tensor on ``plan.destination``.

    Raises:
        RuntimeError: If a tensor is left on the meta device or a quantized
            buffer is empty, non-finite or all zero.
    """
    model = plan.model
    with contextlib.ExitStack() as stack:
        file_readers: dict[str, Any] = {}
        for key, path, checkpoint_key in plan.tensors:
            if plan.destination.type == "cpu":
                # Tensors placed on CPU stay views of the file mapping. One reader per file
                # gives them one mapping per file, as the legacy loader has, instead of a
                # mapping of the whole file per tensor.
                if path not in file_readers:
                    file_readers[path] = stack.enter_context(
                        safe_open(path, framework="pt", device="cpu")
                    )
                tensor = file_readers[path].get_tensor(checkpoint_key)
            else:
                # A reader per tensor unmaps the file once the tensor is copied, which keeps
                # the host memory used for reading at about the largest tensor.
                with safe_open(path, framework="pt", device="cpu") as reader:
                    tensor = reader.get_tensor(checkpoint_key)
            # Without a dtype the value is cast to the placeholder's final dtype.
            set_module_tensor_to_device(
                model, key, plan.destination, value=tensor, clear_cache=False
            )
            del tensor
    if plan.retie:
        model.tie_weights()
    leftovers = [
        name
        for name, tensor in chain(model.named_parameters(), model.named_buffers())
        if tensor.is_meta
    ]
    if leftovers:
        raise RuntimeError(f"Tensors left on the meta device after loading: {leftovers[:20]}")
    # Buffers built outside the meta context, such as the RoPE frequencies, are on CPU.
    model.to(plan.destination)
    # Checked after placement so that the existing module checks run unchanged; checking
    # before the transfer would need a per-tensor version of them. On a GPU, their FP32
    # copies need GPU memory beyond the model.
    QuantizedModelLoader._assert_quantized_modules_loaded(model)
    if plan.device_map is not None:
        model.hf_device_map = dict(plan.device_map)
    return model
