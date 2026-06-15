"""Loss functions for global PTQ.

KL divergence and next-token prediction losses.

Copyright 2025-2026 Fujitsu Ltd.

Authors: Yoshiyuki Ishii, Keiji Kimura, Yuma Ichikawa

"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# KL divergence loss
# ---------------------------------------------------------------------------


def compute_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence from teacher to student on the vocabulary distribution.

    ``D_KL(teacher || student)`` computed on the last-token logits.
    """
    lt = (teacher_logits / temperature).float()
    ls = (student_logits / temperature).float()
    t_log_probs = F.log_softmax(lt, dim=-1)
    s_log_probs = F.log_softmax(ls, dim=-1)
    kl = F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction="batchmean")
    return kl * (temperature**2)


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
