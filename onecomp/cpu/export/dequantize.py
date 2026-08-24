"""Reconstruct a dense (fp16) Hugging Face model from an OneComp GPTQ checkpoint.

Used for (a) the dequantize -> convert_hf_to_gguf -> llama-quantize fallback path
and (b) building a metadata/tokenizer "skeleton" GGUF when the original
full-precision model is not available locally.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import os
from glob import glob
from logging import getLogger
from typing import Dict

import torch

from onecomp.cpu.export.checkpoint import dequantize_layer, iter_gptq_layers, read_quant_meta

logger = getLogger(__name__)

_QUANT_SUFFIXES = (".qweight", ".scales", ".qzeros", ".g_idx", ".perm")
# DBF (DoubleBinaryLinear) tensors; the dense weight is reconstructed by a forward.
_DBF_SUFFIXES = (".scaling0", ".scaling2", ".scaling4", ".bp1", ".bp3")


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


def dequantize_to_hf(
    save_directory: str,
    output_directory: str,
    torch_dtype: torch.dtype = torch.float16,
) -> str:
    """Write a dense HF model (dequantized GPTQ weights) to ``output_directory``.

    Args:
        save_directory: An OneComp quantized model directory (gptq/mixed_gptq).
        output_directory: Destination directory for the dense HF model.
        torch_dtype: dtype of the reconstructed dense weights.

    Returns:
        ``output_directory``.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

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
    meta = read_quant_meta(save_directory)
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
    if getattr(model.config, "tie_word_embeddings", False) and not any(
        k.endswith("lm_head.weight") for k in dense_state
    ):
        model.tie_weights()
        logger.info("Re-tied lm_head to embed_tokens (tie_word_embeddings=True)")

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
