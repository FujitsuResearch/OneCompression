"""Reconstruct a dense Hugging Face model from an OneComp checkpoint.

Used for (a) the dequantize -> convert_hf_to_gguf -> llama-quantize fallback path
and (b) building a metadata/tokenizer "skeleton" GGUF when the original
full-precision model is not available locally.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import os
from collections.abc import Mapping
from glob import glob
from logging import getLogger
from typing import Dict

import torch

from onecomp.cpu.export.checkpoint import (
    UNSUPPORTED_METHODS,
    dequantize_layer,
    iter_gptq_layers,
    load_quant_config,
    read_quant_meta,
)

logger = getLogger(__name__)

_QUANT_SUFFIXES = (".qweight", ".scales", ".qzeros", ".g_idx", ".perm")
# DBF (DoubleBinaryLinear) tensors; the dense weight is reconstructed by a forward.
_DBF_SUFFIXES = (".scaling0", ".scaling2", ".scaling4", ".bp1", ".bp3")
# MDBF tensors are nested under one submodule per path.
_MDBF_MARKER = ".paths.0.A_sign_packed"


def _dequantize_dbf_layers(model, state, torch_dtype):
    """Reconstruct dense ``{name}.weight`` for every DBF layer via identity-forward.

    DBF stores a binary factorization (scaling0/2/4 + packed bp1/bp3); the
    effective dense weight ``W`` is recovered as ``layer(eye(in)).T`` using
    OneComp's own ``DoubleBinaryLinear`` so the math matches inference exactly.
    Returns ``(dense_weights, consumed_keys)``.
    """
    import torch

    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear

    modules = dict(model.named_modules())
    dense: Dict[str, torch.Tensor] = {}
    consumed = set()
    bp1_keys = sorted(k for k in state if k.endswith(".bp1"))
    for bk in bp1_keys:
        name = bk[: -len(".bp1")]
        target = modules.get(name)
        if target is None or not hasattr(target, "in_features"):
            logger.warning("DBF layer %s not found in dense model; skipped", name)
            continue
        in_f, out_f = int(target.in_features), int(target.out_features)
        lsd = {k.split(".")[-1]: state[k] for k in state if k.startswith(name + ".")}
        bias = lsd.pop("bias", None)  # keep bias out of the identity-forward
        layer = DoubleBinaryLinear.from_saved_state(lsd, in_f, out_f).eval()
        with torch.no_grad():
            eye = torch.eye(in_f, dtype=torch.float16)
            w = layer(eye).T.contiguous()  # (out, in) = W, no bias
        dense[f"{name}.weight"] = w.to(torch_dtype)
        if bias is not None:
            dense[f"{name}.bias"] = bias.to(torch_dtype)
        for suffix in _DBF_SUFFIXES:
            consumed.add(name + suffix)
        consumed.add(name + ".bias")
    if dense:
        logger.info("Dequantized %d DBF layers", len(dense))
    return dense, consumed


def _check_mdbf_shapes(
    layer_state_dict: Mapping[str, torch.Tensor],
    layer_name: str,
    in_features: int,
    out_features: int,
) -> None:
    """Reject MDBF shapes that can silently reconstruct an invalid weight.

    Args:
        layer_state_dict: Checkpoint tensors for one MDBF layer.
        layer_name: Layer name used in error messages.
        in_features: Dense layer input width.
        out_features: Dense layer output width.

    Raises:
        KeyError: If a required MDBF tensor is absent.
        ValueError: If factor shapes do not match the dense layer.
    """
    from onecomp.quantizer.mdbf.mdbf_layer import mdbf_path_indices

    def _raise_shape(path_index: int, tensor_name: str, actual: object, expected: object) -> None:
        """Raise a consistent shape validation error."""
        raise ValueError(
            f"Invalid MDBF shape for {layer_name}.paths.{path_index}.{tensor_name}: "
            f"expected {expected}, got {actual}."
        )

    for path_index in sorted(mdbf_path_indices(layer_state_dict)):
        prefix = f"paths.{path_index}."
        q_u = layer_state_dict[prefix + "Q_U_amp"]
        if q_u.ndim != 2:
            _raise_shape(path_index, "Q_U_amp", tuple(q_u.shape), "a 2-D tensor")

        rank, amplitude_rank = (int(dim) for dim in q_u.shape)
        if rank <= 0 or amplitude_rank <= 0:
            _raise_shape(
                path_index,
                "Q_U_amp",
                tuple(q_u.shape),
                "positive rank and amplitude dimensions",
            )

        # Packed byte counts pin down rank for production widths of at least 8.
        expected_packed_sizes = {
            "A_sign_packed": (out_features * rank + 7) // 8,
            "B_sign_packed": (rank * in_features + 7) // 8,
        }
        for tensor_name, expected_size in expected_packed_sizes.items():
            actual_size = layer_state_dict[prefix + tensor_name].numel()
            if actual_size != expected_size:
                _raise_shape(path_index, tensor_name, actual_size, expected_size)

        expected_shapes = {
            "A_amp": (out_features, amplitude_rank),
            "B_amp": (in_features, amplitude_rank),
            "Q_V_amp": (rank, amplitude_rank),
        }
        for tensor_name, expected_shape in expected_shapes.items():
            actual_shape = tuple(layer_state_dict[prefix + tensor_name].shape)
            if actual_shape != expected_shape:
                _raise_shape(path_index, tensor_name, actual_shape, expected_shape)

        # from_saved_state rebuilds these buffers from the dense layer widths;
        # checkpoint values are validation inputs, not reconstruction inputs.
        expected_sign_shapes = {
            "_A_sign_shape": (out_features, rank),
            "_B_sign_shape": (rank, in_features),
        }
        for tensor_name, expected_shape in expected_sign_shapes.items():
            tensor = layer_state_dict.get(prefix + tensor_name)
            if tensor is None:
                continue
            tensor_shape = tuple(tensor.shape)
            if tensor_shape != (2,):
                _raise_shape(path_index, tensor_name, tensor_shape, (2,))

            actual_shape = tuple(int(dim) for dim in tensor.tolist())
            if actual_shape != expected_shape:
                _raise_shape(path_index, tensor_name, actual_shape, expected_shape)


def _dequantize_mdbf_layers(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    torch_dtype: torch.dtype,
    save_directory: str,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Reconstruct dense weights for every MDBF layer.

    Args:
        model: Dense model exposing the target linear modules.
        state: Flat checkpoint state dict.
        torch_dtype: Output weight dtype.
        save_directory: Checkpoint directory containing quantization metadata.

    Returns:
        Dense tensors and checkpoint keys consumed during reconstruction.

    Raises:
        KeyError: If a required MDBF tensor is absent.
        ValueError: If MDBF metadata, path or bias presence, or factor shapes
            are inconsistent with the dense model.
        RuntimeError: If an MDBF layer has no matching dense module.
    """
    marker_keys = sorted(key for key in state if key.endswith(_MDBF_MARKER))
    if not marker_keys:
        return {}, set()

    from onecomp.quantizer.mdbf.config import resolve_mdbf_paths
    from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear

    modules = dict(model.named_modules())
    expected_paths = resolve_mdbf_paths(load_quant_config(save_directory))
    dense: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for marker_key in marker_keys:
        name = marker_key[: -len(_MDBF_MARKER)]
        target = modules.get(name)
        # Treat either an absent target or one without Linear-compatible input
        # dimensions as a mapping failure. Only the absent case is invisible to
        # _reject_unfilled_weights(), but neither can be reconstructed safely.
        if target is None or not hasattr(target, "in_features"):
            raise RuntimeError(
                f"MDBF layer {name!r} from {save_directory} has no matching "
                "nn.Linear in the dense model; its weight cannot be exported."
            )

        in_features = int(target.in_features)
        out_features = int(target.out_features)
        prefix = name + "."
        # Preserve paths.{p} in relative keys so tensors from different paths
        # do not collide as they would if only the final component were kept.
        layer_state_dict = {
            key[len(prefix) :]: tensor for key, tensor in state.items() if key.startswith(prefix)
        }

        MultipathMDBFLinear.validate_saved_state(
            layer_state_dict,
            layer_name=name,
            expected_paths=expected_paths,
            expects_bias=getattr(target, "bias", None) is not None,
        )
        _check_mdbf_shapes(layer_state_dict, name, in_features, out_features)

        layer = MultipathMDBFLinear.from_saved_state(
            layer_state_dict, in_features, out_features
        ).eval()
        with torch.no_grad():
            # Accumulate over the factor rank and across paths in fp32, then cast.
            weight = layer.get_weight(torch.float32)
        dense[f"{name}.weight"] = weight.to(torch_dtype)
        bias = layer_state_dict.get("bias")
        if bias is not None:
            dense[f"{name}.bias"] = bias.to(torch_dtype)
        consumed.update(key for key in state if key.startswith(prefix))

    logger.info("Dequantized %d MDBF layers", len(marker_keys))
    return dense, consumed


def _reject_unfilled_weights(
    missing: list[str], retied: set[str], save_directory: str, quant_method: str
) -> None:
    """Fail when ``load_state_dict(strict=False)`` left a weight at its random init.

    ``strict=False`` is needed because the checkpoint legitimately lacks
    non-persistent buffers, but it equally swallows a whole quantizer's worth of
    unreconstructed weights.  Only ``.weight`` / ``.bias`` keys are checked, so
    buffers stay exempt while any dense tensor that found no source is loud.

    Args:
        missing: ``missing_keys`` from ``load_state_dict``.
        retied: Keys since restored by ``tie_weights()``.
        save_directory: Checkpoint directory, for the error message.
        quant_method: Checkpoint's ``quant_method``, for the error message.

    Raises:
        RuntimeError: If any weight/bias key had no source.
    """
    unfilled = sorted(
        key for key in missing if key.endswith((".weight", ".bias")) and key not in retied
    )
    if not unfilled:
        return
    raise RuntimeError(
        f"{len(unfilled)} tensor(s) in {save_directory} (quant_method={quant_method!r}) "
        f"had no source and would be exported as random init: {unfilled[:8]}"
        f"{' ...' if len(unfilled) > 8 else ''}. This layout has no dense "
        "reconstruction implemented in onecomp.cpu.export.dequantize."
    )


def dequantize_to_hf(
    save_directory: str,
    output_directory: str,
    torch_dtype: torch.dtype = torch.float16,
) -> str:
    """Write a dense HF model to ``output_directory``.

    Args:
        save_directory: A supported OneComp quantized model directory.
        output_directory: Destination directory for the dense HF model.
        torch_dtype: dtype of the reconstructed dense weights.

    Returns:
        ``output_directory``.

    Raises:
        KeyError: If a required quantized tensor is absent.
        ValueError: If the quantization method is unsupported, or MDBF
            metadata, paths, bias presence, or factor shapes are invalid.
        RuntimeError: If a quantized layer cannot be mapped to the dense model,
            or a weight or bias has no checkpoint source.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    # Guard here as well as in ``plan_export``: this is a public entry point and
    # is also reached via ``export_via_dequantize`` / the skeleton builder.
    meta = read_quant_meta(save_directory)
    if meta.quant_method in UNSUPPORTED_METHODS:
        raise ValueError(
            f"quant_method={meta.quant_method!r} has no dense reconstruction "
            "implemented in dequantize_to_hf; its tensors would be dropped and the "
            "result would carry randomly initialised weights."
        )

    os.makedirs(output_directory, exist_ok=True)

    config = AutoConfig.from_pretrained(save_directory)
    # Drop quantization metadata so the rebuilt model is a plain dense model.
    if hasattr(config, "quantization_config"):
        config.quantization_config = None
    try:
        delattr(config, "quantization_config")
    except AttributeError:
        pass

    logger.info("Building empty dense model from config (%s)", config.model_type)
    model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)

    state: Dict[str, torch.Tensor] = {}
    for shard in sorted(glob(os.path.join(save_directory, "*.safetensors"))):
        state.update(load_file(shard, device="cpu"))

    dense_state: Dict[str, torch.Tensor] = {}
    quant_keys = set()
    n_layers = 0
    if meta.is_gptq_family:
        for layer in iter_gptq_layers(save_directory):
            dense_state[layer.weight_key] = dequantize_layer(layer).to(torch_dtype)
            for suffix in _QUANT_SUFFIXES:
                quant_keys.add(layer.name + suffix)
            n_layers += 1
        logger.info("Dequantized %d GPTQ-family layers", n_layers)

    # DBF layers (and any DBF layers mixed into an autobit checkpoint).
    dbf_dense, dbf_consumed = _dequantize_dbf_layers(model, state, torch_dtype)
    dense_state.update(dbf_dense)
    quant_keys |= dbf_consumed

    # MDBF layers use a nested paths.{p}.* layout.
    mdbf_dense, mdbf_consumed = _dequantize_mdbf_layers(model, state, torch_dtype, save_directory)
    dense_state.update(mdbf_dense)
    quant_keys |= mdbf_consumed

    for key, tensor in state.items():
        if key in quant_keys:
            continue
        dense_state[key] = tensor.to(torch_dtype) if tensor.is_floating_point() else tensor

    # Rotated models keep an online Hadamard on down_proj that llama.cpp cannot
    # apply; fold its inverse into the weight so the GGUF needs no online op. Run
    # this *after* every source (GPTQ/DBF/raw fp16) has populated dense_state so
    # an unquantized down_proj is de-folded too.
    if meta.rotated:
        from onecomp.cpu.export.rotation import defold_rotated_dense_state

        defold_rotated_dense_state(dense_state, fp32_had=meta.fp32_had)

    missing, unexpected = model.load_state_dict(dense_state, strict=False, assign=True)
    if unexpected:
        logger.warning("Unexpected keys when loading dense state: %s", unexpected[:8])

    # ``assign=True`` swaps in new parameter objects, which severs the
    # embed_tokens <-> lm_head sharing that tied-embedding models (e.g. Qwen2.5,
    # Gemma) rely on. Without re-tying, lm_head keeps its random init and the
    # exported model emits garbage. Re-establish the tie when the checkpoint did
    # not carry a separate lm_head weight.
    retied = set()
    if getattr(model.config, "tie_word_embeddings", False) and not any(
        k.endswith("lm_head.weight") for k in dense_state
    ):
        model.tie_weights()
        retied = {k for k in missing if k.endswith("lm_head.weight")}
        logger.info("Re-tied lm_head to embed_tokens (tie_word_embeddings=True)")

    _reject_unfilled_weights(missing, retied, save_directory, meta.quant_method)

    model.save_pretrained(output_directory, safe_serialization=True)
    _copy_tokenizer(save_directory, output_directory)
    logger.info("Wrote dense HF model to %s", output_directory)
    return output_directory


def _copy_tokenizer(src: str, dst: str) -> None:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(src)
        tok.save_pretrained(dst)
    except Exception as exc:  # pragma: no cover - tokenizer is best-effort
        logger.warning("Could not copy tokenizer from %s: %s", src, exc)
