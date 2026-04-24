"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

import torch
import torch.nn.functional as F
from torch import nn

from onecomp.utils.blockwise import (
    backward_input,
    get_blocks_and_inputs,
    forward_input,
    move_kwargs_to_device,
)
from onecomp.utils.device import get_default_device, empty_cache


def _find_head_modules(model, blocks):
    """Find the norm layer and lm_head that follow the transformer blocks.

    Walks the module tree to locate the parent of blocks, then looks
    for known attribute names used by common HuggingFace architectures.

    Returns:
        tuple[nn.Module, nn.Module]: (norm, lm_head)

    """
    parent = None
    for name, module in model.named_modules():
        if module is blocks:
            parent_name = name.rpartition(".")[0]
            parent = model.get_submodule(parent_name) if parent_name else model
            break

    norm = None
    for attr in ("norm", "final_layer_norm", "ln_f"):
        norm = getattr(parent, attr, None)
        if norm is not None:
            break

    lm_head = None
    for attr in ("lm_head", "embed_out", "output"):
        lm_head = getattr(model, attr, None)
        if lm_head is not None:
            break

    return norm, lm_head


def _map_candidates_to_blocks(blocks, candidates):
    """Map each candidate (name, module) to its parent block index."""
    module_to_block = {}
    for idx, block in enumerate(blocks):
        for m in block.modules():
            module_to_block[m] = idx

    block_to_candidates = {}
    for name, module in candidates:
        idx = module_to_block.get(module)
        if idx is not None:
            block_to_candidates.setdefault(idx, []).append((name, module))
    return block_to_candidates


def collect_activation_stats_blockwise(
    model,
    candidates,
    *,
    num_samples=128,
    seqlen=256,
    use_curvature_b=True,
    batch_size=16,
    device=None,
    logger=None,
):
    """Collect full Gram and curvature matrices via block-wise processing.

    Returns:
        tuple[dict, dict]: (a_diag, b_diag)
    """
    from transformers import AutoTokenizer
    from onecomp.utils.calibration import prepare_calibration_dataset

    if device is None:
        device = get_default_device()

    original_device = next(model.parameters()).device
    if original_device.type != "cpu":
        if logger:
            logger.info("Moving model to CPU for block-wise activation collection")
        model.to("cpu")
        empty_cache(original_device)

    model_id = getattr(model.config, "_name_or_path", None)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    calib_data = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        max_length=seqlen,
        num_calibration_samples=num_samples,
    )

    actual_samples = min(num_samples, calib_data["input_ids"].shape[0])
    model_inputs = {
        "input_ids": calib_data["input_ids"][:actual_samples],
        "attention_mask": calib_data["attention_mask"][:actual_samples],
    }

    blocks, inps, kwargs = get_blocks_and_inputs(model, model_inputs, batch_size)
    kwargs = move_kwargs_to_device(kwargs, device)
    block_to_candidates = _map_candidates_to_blocks(blocks, candidates)

    a_accum = {}
    b_accum = {}
    counts = {}
    for name, module in candidates:
        d_in = module.weight.shape[1]
        d_out = module.weight.shape[0]
        a_accum[name] = torch.zeros(d_in, dtype=torch.float64)
        counts[name] = 0
        if use_curvature_b:
            b_accum[name] = torch.zeros(d_out, dtype=torch.float64)

    if logger:
        tag = "Gram + Curvature" if use_curvature_b else "Gram only"
        logger.info(
            "Block-wise activation stats (%s): %d samples, seqlen=%d, " "%d blocks, %d layers",
            tag,
            actual_samples,
            seqlen,
            len(blocks),
            len(candidates),
        )

    # Collect a_diag
    saved_inps = [inps]

    for block_idx, block in enumerate(blocks):
        block.to(device)

        hooks = []
        for name, module in block_to_candidates.get(block_idx, []):
            hooks.append(module.register_forward_hook(_make_fwd_hook(name, a_accum, counts)))

        inps = forward_input(inps, block, kwargs, batch_size, device)
        saved_inps.append(inps)

        for h in hooks:
            h.remove()
        block.cpu()
        empty_cache(device)

    # Collect b_diag
    if use_curvature_b:
        norm, lm_head = _find_head_modules(model, blocks)
        if norm is None or lm_head is None:
            raise RuntimeError(
                "Cannot compute curvature: "
                f"norm={'found' if norm else 'NOT found'}, "
                f"lm_head={'found' if lm_head else 'NOT found'}. "
                "The model may use non-standard module names. "
                "Set use_curvature_b=False to skip curvature estimation."
            )
        input_ids = model_inputs["input_ids"]

        grad = _compute_loss_grad(
            saved_inps[-1],
            norm,
            lm_head,
            input_ids,
            device,
        )

        for block_idx in range(len(blocks) - 1, -1, -1):
            block = blocks[block_idx]
            block.to(device)

            orig_grad_flags = {}
            for p in block.parameters():
                orig_grad_flags[p] = p.requires_grad
                p.requires_grad_(True)

            hooks = []
            for name, module in block_to_candidates.get(block_idx, []):
                hooks.append(module.register_full_backward_hook(_make_bwd_hook(name, b_accum)))

            grad = backward_input(
                saved_inps[block_idx],
                block,
                grad,
                kwargs,
                batch_size,
                device,
            )

            for p, flag in orig_grad_flags.items():
                p.requires_grad_(flag)
            for h in hooks:
                h.remove()
            block.cpu()
            empty_cache(device)

    a_diag = {}
    b_diag = {}
    for name in a_accum:
        cnt = max(counts[name], 1)
        a_diag[name] = (a_accum[name] / cnt).float()
        if use_curvature_b:
            b_diag[name] = (b_accum[name] / cnt).float()
        else:
            b_diag[name] = None

    if original_device.type != "cpu":
        if logger:
            logger.info("Restoring model to %s", original_device)
        model.to(original_device)

    return a_diag, b_diag


def _make_fwd_hook(key, A_accum, counts):
    def hook(_mod, inp, _out):
        x = inp[0].detach().float()
        x_flat = x.reshape(-1, x.shape[-1])
        A_accum[key].add_((x_flat**2).sum(dim=0).cpu().double())
        counts[key] += x_flat.shape[0]

    return hook


def _make_bwd_hook(key, B_accum):
    def hook(_mod, _grad_in, grad_out):
        g = grad_out[0]
        if g is None:
            return
        g = g.detach().float()
        g_flat = g.reshape(-1, g.shape[-1])
        B_accum[key].add_((g_flat**2).sum(dim=0).cpu().double())

    return hook


def _compute_loss_grad(final_hidden, norm, lm_head, input_ids, device):
    all_grads = []

    norm.to(device)
    lm_head.to(device)

    for i in range(final_hidden.shape[0]):
        out_i = final_hidden[i : i + 1].to(device)
        out_i = out_i.detach().requires_grad_(True)
        ids_i = input_ids[i : i + 1].to(device)

        with torch.enable_grad():
            normed = norm(out_i)
            logits = lm_head(normed)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids_i[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            loss.backward()

        all_grads.append(out_i.grad.cpu())

    norm.cpu()
    lm_head.cpu()
    empty_cache(device)

    return torch.cat(all_grads)
