"""Helpers for block-wise PTQ.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yoshiyuki Ishii

Ported from qep-dev/src/blockwise_quantization/run_blockwise_ptq.py
and adapted for onecomp-lab's quantized model structure.

Key differences from qep-dev:
  - Calibration data: onecomp uses prepare_calibration_dataset()
    returning {"input_ids": (N, seq_len), "attention_mask": (N, seq_len)}
    instead of qep-dev's get_loaders() returning list of (input_ids, labels).
  - Model structure: onecomp uses HuggingFace AutoModelForCausalLM;
    no vision-LLM wrappers, no model.seqlen attribute.
  - Quantized layers: isinstance() detection instead of attribute flags.
"""

import gc
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)


class _CatcherStop(Exception):
    """Raised by Catcher to abort the forward pass after capturing inputs."""


# ---------------------------------------------------------------------------
# Tensor transfer helpers (same as qep-dev)
# ---------------------------------------------------------------------------


def layer_kwargs_to_device(layer_kwargs: dict, dev: torch.device) -> dict:
    """Transfer all tensors in layer_kwargs to *dev* (detached)."""
    out = {}
    for k, v in layer_kwargs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to(dev)
        elif isinstance(v, tuple) and v and isinstance(v[0], torch.Tensor):
            out[k] = tuple(t.detach().to(dev) for t in v)
        else:
            out[k] = v
    return out


def deep_clone_layer_kwargs(layer_kwargs: dict) -> dict:
    """Deep-clone layer_kwargs so tensor mutations don't propagate."""
    out = {}
    for k, v in layer_kwargs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, tuple) and v and isinstance(v[0], torch.Tensor):
            out[k] = tuple(t.clone() for t in v)
        else:
            out[k] = v
    return out


def layer_forward_single(layer, inp_cpu, layer_kwargs_gpu, dev):
    """Forward one sample through *layer* with CPU -> GPU -> CPU transfer."""
    inp_gpu = inp_cpu.unsqueeze(0).to(dev)
    raw = layer(inp_gpu, **layer_kwargs_gpu)
    out = raw[0] if isinstance(raw, tuple) else raw
    if out.dim() == 3 and out.size(0) == 1:
        out = out.squeeze(0)
    result = out.cpu()
    del inp_gpu, raw, out
    return result


# ---------------------------------------------------------------------------
# Model structure helpers
# ---------------------------------------------------------------------------


def get_transformer_layers(model):
    """Return the nn.ModuleList of Transformer blocks.

    Supports Llama, Mistral, Qwen2, Qwen3, Gemma, GPT-NeoX, OPT,
    and Vision-Language Models (Qwen3-VL, Qwen2.5-VL, etc.).
    """
    # Standard LLM: model.model.layers (Llama, Mistral, Qwen, Gemma, ...)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # VLM: model.model.language_model.layers (Qwen3-VL, Qwen2.5-VL, ...)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    # GPT-NeoX (Pythia)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    # OPT
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


def _get_language_model_backbone(model):
    """Return the language-model backbone (for embed_tokens / rotary_emb).

    qep-dev equivalent: utils.model_utils.get_language_model()
    """
    # Standard LLM: model.model has embed_tokens directly
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model
    # VLM: model.model.language_model has embed_tokens (Qwen3-VL, Qwen2.5-VL, ...)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "embed_tokens")
    ):
        return model.model.language_model
    # GPT-NeoX (Pythia)
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    # OPT
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder
    raise ValueError(f"Cannot find language model backbone in {type(model).__name__}")


# ---------------------------------------------------------------------------
# Layer input collection
# ---------------------------------------------------------------------------
#
# qep-dev version: collect_layer_inputs(model, language_model, dataloader, nsamples, dev)
#   - dataloader: list of (input_ids, labels) from get_loaders()
#   - uses model.seqlen for pre-allocation
#   - forward: forward_model(batch[0].to(dev))
#
# onecomp-lab version below:
#   - calibration_inputs: {"input_ids": (N, seq_len), "attention_mask": (N, seq_len)}
#     from prepare_calibration_dataset()
#   - seq_len from calibration_inputs shape
#   - forward: model(input_ids=..., attention_mask=...)
#


def collect_layer_inputs(model, layers, calibration_inputs, dev):
    """Collect first-layer inputs via Catcher hook. Returns CPU tensors.

    Args:
        model: Full model (e.g. LlamaForCausalLM) on CPU.
        layers: Transformer block ModuleList (e.g. model.model.layers).
        calibration_inputs: dict with "input_ids" (N, seq_len) and
            "attention_mask" (N, seq_len), both on CPU.
        dev: GPU device for forward passes.

    Returns:
        inps: list of Tensors, each (seq_len, hidden_size), on CPU.
        layer_kwargs_cpu: dict of kwargs (attention_mask, position_ids, etc.)
            on CPU.
    """
    if not layers:
        raise ValueError("layers is empty; cannot collect layer inputs")

    # --- Save and disable use_cache (same as qep-dev) ---
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        use_cache = model.config.text_config.use_cache
        model.config.text_config.use_cache = False
    else:
        use_cache = model.config.use_cache
        model.config.use_cache = False

    # --- Move embedding layers to GPU ---
    backbone = _get_language_model_backbone(model)
    # Architecture-dependent embedding attributes:
    #   Llama/Mistral/Qwen2/Qwen3/Gemma/Phi/OPT: embed_tokens
    #   GPT-NeoX (Pythia):                  embed_in
    #   OPT:                                embed_positions, project_in
    #   Llama/Mistral (some versions):      rotary_emb (at backbone level)
    _moved_embed_attrs = []
    for attr in ("embed_tokens", "embed_in", "embed_positions", "project_in", "rotary_emb"):
        if hasattr(backbone, attr) and getattr(backbone, attr) is not None:
            setattr(backbone, attr, getattr(backbone, attr).to(dev))
            _moved_embed_attrs.append(attr)

    layers[0] = layers[0].to(dev)

    # --- Pre-allocate ---
    nsamples = calibration_inputs["input_ids"].shape[0]
    seq_len = calibration_inputs["input_ids"].shape[1]

    # hidden_size from model config
    if hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_size = model.config.text_config.hidden_size
    else:
        raise ValueError("Cannot determine hidden_size from model.config")

    dtype = next(iter(model.parameters())).dtype
    inps_gpu = torch.zeros((nsamples, seq_len, hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "layer_kwargs": {}}

    # --- Catcher hook (same pattern as qep-dev) ---
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            for attr in (
                "layer_idx",
                "self_attn",
                "mlp",
                "attention_type",
                "sliding_window",
                "attention_dropout",
                "layer_type",
            ):
                if hasattr(module, attr):
                    setattr(self, attr, getattr(module, attr))

        def forward(self, inp, **kwargs):
            inps_gpu[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"].update(kwargs)
            raise _CatcherStop

    layers[0] = Catcher(layers[0])

    # --- Forward each sample ---
    # onecomp difference: use model(input_ids=..., attention_mask=...)
    # instead of qep-dev's forward_model(batch[0].to(dev))
    for j in range(nsamples):
        try:
            model(
                input_ids=calibration_inputs["input_ids"][j : j + 1].to(dev),
                attention_mask=calibration_inputs["attention_mask"][j : j + 1].to(dev),
            )
        except _CatcherStop:
            pass

    layers[0] = layers[0].module

    # --- Move everything back to CPU ---
    layers[0] = layers[0].cpu()
    for attr in _moved_embed_attrs:
        setattr(backbone, attr, getattr(backbone, attr).cpu())

    inps = [inps_gpu[j].cpu() for j in range(nsamples)]
    del inps_gpu

    try:
        from transformers.cache_utils import Cache as _CacheBase
    except ImportError:
        _CacheBase = None

    layer_kwargs_cpu = {}
    for k, v in cache["layer_kwargs"].items():
        if k in ("past_key_value", "past_key_values"):
            continue
        if _CacheBase is not None and isinstance(v, _CacheBase):
            continue
        if isinstance(v, torch.Tensor):
            layer_kwargs_cpu[k] = v.detach().cpu()
        elif isinstance(v, tuple) and v and isinstance(v[0], torch.Tensor):
            layer_kwargs_cpu[k] = tuple(t.detach().cpu() for t in v)
        else:
            layer_kwargs_cpu[k] = v

    gc.collect()
    torch.cuda.empty_cache()

    # --- Restore use_cache ---
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        model.config.text_config.use_cache = use_cache
    else:
        model.config.use_cache = use_cache

    return inps, layer_kwargs_cpu


# ---------------------------------------------------------------------------
# Auto-detect quantisation method per block
# ---------------------------------------------------------------------------
#
# qep-dev equivalent: _auto_detect_quantization_strategy()
#   - Detects GPTQLinear via hasattr(mod, 'is_gptq_quantized')
#   - Detects DBF via _is_dbf_sequential() (5-stage nn.Sequential)
#
# onecomp-lab version:
#   - Detects GPTQLinear via isinstance()
#   - Detects DoubleBinaryLinear via isinstance() (not nn.Sequential)
#   - Detects OneBitLinear via isinstance()
#


def auto_detect_quantization_strategy(layers):
    """Return per-block method string: "gptq" / "dbf" / "onebit" / None (FP16)."""
    from ...quantizer.gptq.gptq_layer import GPTQLinear
    from ...quantizer.dbf.dbf_layer import DoubleBinaryLinear
    from ...quantizer.onebit.onebit_layer import OneBitLinear

    strategy = []
    for layer in layers:
        method = None
        for _name, mod in layer.named_modules():
            if isinstance(mod, GPTQLinear):
                method = "gptq"
                break
            if isinstance(mod, DoubleBinaryLinear):
                method = "dbf"
                break
            if isinstance(mod, OneBitLinear):
                method = "onebit"
                break
        strategy.append(method)
    return strategy
