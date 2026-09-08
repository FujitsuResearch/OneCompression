"""Loss functions for global PTQ.

KL divergence, next-token prediction, entropy regularisation, and
intermediate-layer cosine similarity loss with forward-hook management.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KL divergence loss
# ---------------------------------------------------------------------------


def compute_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL divergence from teacher to student on the vocabulary distribution.

    ``D_KL(teacher || student)`` computed on the logits.
    """
    lt = (teacher_logits / temperature).float()
    ls = (student_logits / temperature).float()
    t_log_probs = F.log_softmax(lt, dim=-1)
    s_log_probs = F.log_softmax(ls, dim=-1)

    if attention_mask is not None:
        # Compute per-token KL divergence
        kl = F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction="none")
        # Sum over vocabulary, then mask and average over non-padding tokens
        kl = kl.sum(dim=-1)
        kl = (kl * attention_mask).sum() / attention_mask.sum().clamp(min=1.0)
    else:
        kl = F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction="batchmean")

    return kl * (temperature ** 2)


# ---------------------------------------------------------------------------
# Next-token prediction loss
# ---------------------------------------------------------------------------


def compute_ntp_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Next token prediction loss (shifted cross-entropy).

    Computes standard causal LM loss where position *t* predicts
    token at position *t+1*.  When used with ``w_ntp > 0`` and
    ``w_distill = 0``, this enables pure QAT (no teacher dependency).
    """
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Entropy regularisation
# ---------------------------------------------------------------------------


def compute_entropy_loss(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Negative entropy of the student prediction distribution.

    Minimising this loss maximises entropy, which prevents the quantised
    model from becoming overconfident on the small calibration set and
    improves generalisation to unseen text.
    """
    logits_f = (logits / temperature).float()
    p = F.softmax(logits_f, dim=-1)
    log_p = F.log_softmax(logits_f, dim=-1)
    entropy = -(p * log_p).sum(dim=-1).mean()
    return -entropy


# ---------------------------------------------------------------------------
# Intermediate-layer cosine similarity loss (with hook management)
# ---------------------------------------------------------------------------


def setup_intermediate_hooks(language_model: nn.Module) -> dict:
    """Register forward hooks on each transformer layer to capture hidden states.

    Returns a hooks dict with ``"outputs"`` (list of captured tensors)
    and ``"handles"`` (list of hook handles for removal).
    """
    hooks: dict = {"outputs": [], "handles": []}

    layers = _get_transformer_layers(language_model)
    for layer in layers:
        def _make_hook():
            def hook(_module, _inp, out):
                hooks["outputs"].append(out[0] if isinstance(out, tuple) else out)
            return hook
        hooks["handles"].append(layer.register_forward_hook(_make_hook()))

    return hooks


def clear_hooks(hooks: dict) -> None:
    """Clear captured outputs (call before each forward pass)."""
    hooks["outputs"] = []


def remove_hooks(hooks: dict) -> None:
    """Remove all registered hooks and clear captured outputs."""
    for h in hooks["handles"]:
        h.remove()
    hooks["handles"] = []
    hooks["outputs"] = []


def compute_intermediate_loss(
    student_hooks: dict,
    teacher_hooks: dict,
) -> torch.Tensor:
    """Per-layer cosine similarity loss between student and teacher hidden states.

    For each pair of matching layers the loss is ``1 - cosine_similarity``.
    """
    s_out: List[torch.Tensor] = student_hooks["outputs"]
    t_out: List[torch.Tensor] = teacher_hooks["outputs"]
    if not s_out or not t_out:
        dev = s_out[0].device if s_out else (t_out[0].device if t_out else "cpu")
        return torch.tensor(0.0, device=dev)

    n = min(len(s_out), len(t_out))
    total = torch.tensor(0.0, device=s_out[0].device)
    count = 0
    for i in range(n):
        if s_out[i].shape == t_out[i].shape:
            s_flat = s_out[i].float().reshape(-1, s_out[i].shape[-1])
            t_flat = t_out[i].float().reshape(-1, t_out[i].shape[-1])
            total = total + (1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1)).mean()
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_transformer_layers(language_model: nn.Module) -> nn.ModuleList:
    """Find the ``layers`` attribute on a language model backbone.

    Handles common HuggingFace patterns:
    ``model.layers``, ``model.model.layers``, and direct ``.layers``.
    """
    for attr_path in ("layers", "model.layers"):
        obj = language_model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        f"Cannot find transformer layers on {type(language_model).__name__}. "
        "Expected a `layers` or `model.layers` attribute."
    )
