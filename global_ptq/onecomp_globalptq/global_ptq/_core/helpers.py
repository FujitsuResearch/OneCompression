"""Helpers for global PTQ.

Model structure inspection, quantization method detection, and STE utilities.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

import logging

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def find_target_modules(
    model: nn.Module,
    target_class: type,
) -> List[Tuple[str, nn.Module]]:
    """Return all modules of *target_class* as ``(name, module)`` pairs."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, target_class)
    ]


def get_language_model_backbone(model: nn.Module) -> nn.Module:
    """Return the language-model backbone (for layers / embed_tokens).

    Matches the implementation in ``onecomp.post_process._blockwise.helpers``.
    Supports Llama, Mistral, Qwen, Gemma, GPT-NeoX, OPT, and VLMs.
    """
    # Standard LLM: model.model (Llama, Mistral, Qwen, Gemma, ...)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    # VLM: model.model.language_model (Qwen3-VL, Qwen2.5-VL, ...)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model
    # GPT-NeoX (Pythia)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox
    # OPT
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder

    # Fallback: search for language_model/text_model suffixes (for newer VLMs)
    _VLM_TEXT_SUFFIXES = ("language_model", "text_model")
    for name, mod in model.named_modules():
        if any(name.endswith(s) for s in _VLM_TEXT_SUFFIXES):
            return mod

    return model


def get_logits(output) -> torch.Tensor:
    """Extract logits from a model output (CausalLMOutput or tuple)."""
    return output.logits if hasattr(output, "logits") else output[0]


def detect_quantization_method(
    model: nn.Module,
) -> Tuple[Optional[str], List[Tuple[str, nn.Module]]]:
    """Auto-detect the quantization method applied to *model*.

    Returns:
        (method, modules) where *method* is ``"gptq"``, ``"dbf"``,
        ``"mdbf"``, or ``None``, and *modules* is the list of
        ``(name, module)`` pairs for the detected quantized layers.

    Priority: GPTQ > DBF > MDBF.  When multiple types coexist a warning is
    emitted and only the highest-priority layers are returned.
    """
    from onecomp.quantizer.gptq.gptq_layer import GPTQLinear
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear

    from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear

    gptq_modules = find_target_modules(model, GPTQLinear)
    dbf_modules = find_target_modules(model, DoubleBinaryLinear)
    mdbf_modules = find_target_modules(model, MultipathMDBFLinear)

    if gptq_modules and (dbf_modules or mdbf_modules):
        logger.warning(
            "Mixed GPTQ + DBF/MDBF model detected (gptq=%d, dbf=%d, mdbf=%d). "
            "Global PTQ currently optimises GPTQ layers only; "
            "other layers will be skipped.",
            len(gptq_modules), len(dbf_modules), len(mdbf_modules),
        )
    if gptq_modules:
        return "gptq", gptq_modules
    if dbf_modules:
        return "dbf", dbf_modules
    if mdbf_modules:
        return "mdbf", mdbf_modules
    return None, []


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------


def enable_gradient_checkpointing(model: nn.Module) -> bool:
    """Enable gradient checkpointing to reduce GPU memory usage.

    Returns ``True`` if checkpointing was successfully enabled.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        return True
    backbone = get_language_model_backbone(model)
    if backbone is not None and hasattr(backbone, "gradient_checkpointing"):
        backbone.gradient_checkpointing = True
        return True
    return False


def disable_gradient_checkpointing(model: nn.Module) -> None:
    """Disable gradient checkpointing."""
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    else:
        backbone = get_language_model_backbone(model)
        if backbone is not None and hasattr(backbone, "gradient_checkpointing"):
            backbone.gradient_checkpointing = False


def remove_input_require_grads(model: nn.Module) -> None:
    """Safely remove the input_require_grads hook if it exists.

    This is necessary to avoid pickling errors when saving the model,
    as enable_input_require_grads() registers a local function as a hook.
    """
    # 1. Standard Peft/Transformers removal method
    if hasattr(model, "remove_input_require_grads"):
        model.remove_input_require_grads()

    # 2. Handle stored in attribute (standard Transformers)
    if hasattr(model, "_input_require_grads_hook"):
        try:
            model._input_require_grads_hook.remove()
        except Exception:
            pass
        delattr(model, "_input_require_grads_hook")

    # 3. Aggressive cleanup of input embeddings hooks (fallback)
    if hasattr(model, "get_input_embeddings"):
        try:
            embeddings = model.get_input_embeddings()
            if embeddings is not None and hasattr(embeddings, "_forward_hooks"):
                # Iterate over hooks and remove any that look like make_inputs_require_grads
                for hook_id in list(embeddings._forward_hooks.keys()):
                    hook_fn = embeddings._forward_hooks[hook_id]
                    # Check function name or string representation
                    if "make_inputs_require_grads" in str(hook_fn):
                        del embeddings._forward_hooks[hook_id]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# STE helpers
# ---------------------------------------------------------------------------


def smooth_ste_round(
    x: torch.Tensor,
    min_val: int,
    max_val: int,
    k: float = 10.0,
) -> torch.Tensor:
    """Forward: round + clamp,  backward: smooth sigmoid approximation."""
    x_clamped = x.clamp(min_val, max_val)
    hard = x_clamped.round()
    frac = x_clamped - x_clamped.floor()
    soft = x_clamped.floor() + torch.sigmoid(k * (frac - 0.5))
    return soft + (hard - soft).detach()


def smooth_sign_ste(
    x: torch.Tensor,
    k: float = 100.0,
) -> torch.Tensor:
    """Forward: sign(x),  backward: tanh(k*x).

    Used for differentiable binary weight approximation in DBF.
    The forward pass produces hard {-1, +1} values (zero maps to +1),
    while the backward pass uses the smooth ``tanh(k*x)`` surrogate
    so that gradients flow through binary decisions.

    Note:
        The default ``k=100.0`` is suitable for ``smooth_ste_round`` where
        values lie far from the transition boundary.  DBF and MDBF binary
        weights use their respective ``dbf_ste_k`` and ``mdbf_ste_k`` values
        (both default to 2.0) to avoid gradient saturation in ``tanh(k*x)``.
    """
    hard = x.sign()
    hard[hard == 0] = 1
    soft = torch.tanh(k * x)
    return soft + (hard - soft).detach()
