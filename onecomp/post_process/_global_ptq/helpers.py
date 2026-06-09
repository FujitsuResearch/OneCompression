"""Helpers for global PTQ.

Model structure inspection and quantization method detection.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_language_model_backbone(model: nn.Module) -> nn.Module:
    """Return the language-model sub-module for VLMs, or *model* itself.

    For standard CausalLMs this is a no-op.  For VLMs (Qwen3-VL, Gemma3,
    etc.) this returns the ``language_model`` or ``text_model`` child so
    that downstream code can access ``.layers`` / ``.model.layers``.
    """
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
        (method, modules) where *method* is ``"gptq"``, ``"dbf"``, or
        ``None``, and *modules* is the list of ``(name, module)`` pairs
        for the detected quantized layers.

    When both GPTQ and DBF layers are present (mixed quantization),
    a warning is emitted and only GPTQ layers are returned.
    """
    from ...quantizer.dbf.dbf_layer import DoubleBinaryLinear
    from ...quantizer.gptq.gptq_layer import GPTQLinear

    gptq_modules = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, GPTQLinear)
    ]
    dbf_modules = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, DoubleBinaryLinear)
    ]

    if gptq_modules and dbf_modules:
        logger.warning(
            "Mixed GPTQ + DBF model detected (gptq=%d, dbf=%d). "
            "Global PTQ currently optimises GPTQ layers only; "
            "DBF layers will be skipped.",
            len(gptq_modules),
            len(dbf_modules),
        )
    if gptq_modules:
        return "gptq", gptq_modules
    if dbf_modules:
        return "dbf", dbf_modules
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
