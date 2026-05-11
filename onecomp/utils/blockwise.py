"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Yudai Fujimoto, Akihiro Yoshida, Yuma Ichikawa

"""

from logging import getLogger

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

logger = getLogger(__name__)


def _get_blocks(
    model: nn.Module,
) -> nn.ModuleList:
    """Get the language-model transformer blocks in the model.

    For VLMs (e.g., Qwen3-VL, Gemma3) that contain both a vision encoder
    and a language model, this returns the language-model decoder blocks
    only.  For standard CausalLMs the behaviour is unchanged.

    The detection works by looking for a ``language_model`` sub-module in
    the model tree.  If found, the search for ``GradientCheckpointingLayer``
    blocks is restricted to that sub-module so that vision-encoder blocks
    are never returned.

    Args:
        model (nn.Module): The model to analyze.

    Raises:
        RuntimeError: If transformer blocks are not found.

    Returns:
        nn.ModuleList: The list of transformer blocks.
    """
    # Sub-module name suffixes that indicate a language-model backbone inside a VLM.
    # "language_model": Qwen3-VL, Gemma3, LLaVA
    # "text_model": InternVL and similar architectures
    _VLM_TEXT_SUFFIXES = ("language_model", "text_model")

    search_root = model
    for name, mod in model.named_modules():
        if any(name.endswith(s) for s in _VLM_TEXT_SUFFIXES):
            search_root = mod
            logger.info("Using text submodel: %s (%s)", name, type(mod).__name__)
            break

    for module in search_root.modules():
        if isinstance(module, nn.ModuleList):
            if len(module) > 0 and isinstance(module[0], GradientCheckpointingLayer):
                return module

    raise RuntimeError("Transformer blocks not found.")


class StopForward(Exception):
    """An exception to stop the forward pass after capturing activations."""

    pass


class Catcher(nn.Module):
    """A wrapper module to capture input activations and keyword arguments.

    Attribute access is proxied to the wrapped module so that model code
    that reads layer attributes (e.g. ``attention_type``) before calling
    ``forward()`` does not raise ``AttributeError``.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.inp = None
        self.kwargs = {}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    # *args should be gotten from the model such as per_layer_input for gemma4,
    # but it is calculated and added in _compute_per_layer_inputs() and prepare_block_kwargs(), respectively.
    def forward(self, inp: torch.Tensor, *args, **kwargs: dict):
        self.inp = inp.clone()
        self.kwargs.update(kwargs)
        raise StopForward()


_PER_LAYER_INPUTS_KEY = "_per_layer_inputs"
_POS_EMB_MAP_KEY = "_position_embeddings_map"
_ATTN_MASK_MAP_KEY = "_attention_mask_map"


def _find_blocks_parent(model, blocks):
    """Return the parent module that directly owns blocks.

    For Gemma4 this is the Gemma4TextModel that owns .layers,
    .rotary_emb, and the config with layer_types.
    """
    for name, module in model.named_modules():
        if module is blocks:
            parent_name = name.rpartition(".")[0]
            return model.get_submodule(parent_name) if parent_name else model
    return None


@torch.no_grad()
def _compute_per_layer_inputs(model, blocks, input_ids, batch_size):
    """Compute per-layer input embeddings for all calibration samples.

    Gemma4 supply a per-layer embedding (per_layer_input) as 
    an extra positional argument to each decoder layer.
    This function detects such models, computes the tensor
    [N, seq, num_layers, hidden_per_layer] from input_ids in
    batch_size chunks (to limit GPU memory).
    For models that do not use this mechanism, returns None.
    """
    if not getattr(blocks[0], "hidden_size_per_layer_input", 0):
        return None

    for _name, module in model.named_modules():
        if not (
            hasattr(module, "get_per_layer_inputs")
            and hasattr(module, "project_per_layer_inputs")
        ):
            continue
        for child in module.modules():
            if child is blocks:
                chunks = []
                for ids_batch in input_ids.split(batch_size):
                    embeds = module.embed_tokens(ids_batch)
                    pli = module.get_per_layer_inputs(ids_batch, embeds)
                    chunks.append(
                        module.project_per_layer_inputs(embeds, pli).cpu()
                    )
                return torch.cat(chunks)

    logger.warning(
        "Blocks expect per_layer_input but no provider module was found. "
        "Block outputs may be incorrect."
    )
    return None


@torch.no_grad()
def get_blocks_and_inputs(
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[nn.ModuleList, torch.Tensor, dict[str, torch.Tensor]]:
    """Get the transformer blocks and their input activations.

    Keyword arguments (``kwargs``) are captured with a **single sample**
    so that they are batch-size-independent.  This avoids shape mismatches
    when the same ``kwargs`` are later reused with varying batch sizes in
    ``make_grouped_module`` (batch=1), ``compute_hessian_and_crossterm``
    and ``forward_input``.

    For models that pass per-layer embeddings as a positional argument
    (e.g. Gemma4's ``per_layer_input``), the full per-sample tensor is
    pre-computed and stored under ``kwargs["_per_layer_inputs"]``.
    ``forward_input`` / ``backward_input`` will pick the correct
    per-block, per-batch slice automatically.

    Args:
        model (nn.Module): The model to analyze.
        model_inputs (dict[str, torch.Tensor]): The input tensors for the model.
        batch_size (int): The batch size for computing input activations.

    Returns:
        tuple[nn.ModuleList, torch.Tensor, dict[str, torch.Tensor]]:
        The list of transformer blocks, the input activations, and the keyword arguments.
    """

    blocks = _get_blocks(model)

    # Detect models with heterogeneous layer types (e.g. Gemma4 with
    # full_attention / sliding_attention)
    blocks_parent = _find_blocks_parent(model, blocks)
    layer_types = getattr(getattr(blocks_parent, "config", None), "layer_types", None)
    unique_layer_types = set(layer_types) if layer_types else set()
    has_mixed_types = len(unique_layer_types) > 1

    rotary_hook_handle = None
    pos_emb_map: dict[str, tuple[torch.Tensor, ...]] = {}
    if has_mixed_types:
        rotary_emb = getattr(blocks_parent, "rotary_emb", None)
        if rotary_emb is not None:
            def _capture_rotary(_mod, args, output):
                lt = args[2] if len(args) > 2 else None
                if lt is not None:
                    pos_emb_map[lt] = tuple(t.clone() for t in output)
            rotary_hook_handle = rotary_emb.register_forward_hook(_capture_rotary)

    # replace the first transformer block with a input catcher.
    blocks[0] = Catcher(blocks[0])

    inp_ids = model_inputs["input_ids"]
    model_kwargs = {k: v for k, v in model_inputs.items() if k != "input_ids"}
    model_kwargs["use_cache"] = False

    # Capture kwargs with batch=1 so they stay batch-independent.
    # expand_kwargs_batch() will later expand them to match each forward call.
    single_kwargs = {
        k: v[:1] if isinstance(v, torch.Tensor) and v.dim() >= 1 else v
        for k, v in model_kwargs.items()
    }
    logger.info("Capturing batch-independent kwargs with single sample.")
    try:
        _ = model(inp_ids[:1], **single_kwargs)
    except StopForward:
        pass

    if rotary_hook_handle is not None:
        rotary_hook_handle.remove()

    kwargs = dict(blocks[0].kwargs)  # shallow-copy before next loop overwrites
    blocks[0].inp = None  # release single-sample activation (no longer needed)

    # Now capture block inputs for all calibration samples.
    block_inps = []
    for inp in inp_ids.split(batch_size):
        try:
            _ = model(inp, **model_kwargs)
        except StopForward:
            block_inps.append(blocks[0].inp.cpu())

    inps = torch.cat(block_inps)

    # restore the original transformer block
    blocks[0] = blocks[0].module

    # Pre-compute per-layer inputs for some models (e.g. Gemma4).
    pli = _compute_per_layer_inputs(model, blocks, inp_ids, batch_size)
    if pli is not None:
        kwargs[_PER_LAYER_INPUTS_KEY] = pli

    # Store per-type position embeddings and attention masks when the model
    # uses heterogeneous layer types (full_attention and sliding_attention).
    if len(pos_emb_map) > 1:
        kwargs[_POS_EMB_MAP_KEY] = pos_emb_map

    # Store per-type attention masks when the model uses heterogeneous layer types.
    if has_mixed_types and blocks_parent is not None:
        attn_mask_map = _compute_per_type_attention_masks(
            blocks_parent, kwargs, unique_layer_types,
        )
        if attn_mask_map is not None:
            kwargs[_ATTN_MASK_MAP_KEY] = attn_mask_map

    return (blocks, inps, kwargs)


def _compute_per_type_attention_masks(blocks_parent, kwargs, unique_layer_types):
    """Compute attention masks for each layer type.

    Uses create_causal_mask / create_sliding_window_causal_mask
    from transformers to produce the correct mask per layer type.
    """
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    position_ids = kwargs.get("position_ids")
    if position_ids is None:
        return None

    config = blocks_parent.config
    seq_len = position_ids.shape[-1]
    device = position_ids.device
    dtype = next(blocks_parent.parameters()).dtype
    dummy_embeds = torch.zeros(1, seq_len, config.hidden_size, device=device, dtype=dtype)
    attn_mask_1d = torch.ones(1, seq_len, device=device, dtype=torch.long)

    # tmp: Gemma4 only has full_attention and sliding_attention layer types.
    _mask_creators = {
        "full_attention": create_causal_mask,
        "sliding_attention": create_sliding_window_causal_mask,
    }

    mask_map = {}
    for lt in unique_layer_types:
        creator = _mask_creators.get(lt)
        if creator is not None:
            mask_map[lt] = creator(
                config, dummy_embeds, attn_mask_1d,
                past_key_values=None, position_ids=position_ids,
            )
        else:
            logger.warning("No mask creator found for layer type: %s", lt)

    return mask_map if len(mask_map) > 1 else None


def move_kwargs_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: move_kwargs_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_kwargs_to_device(v, device) for v in x]
    elif isinstance(x, tuple):
        return tuple(move_kwargs_to_device(v, device) for v in x)
    else:
        return x


def expand_kwargs_batch(kwargs, batch_size):
    """Expand batch=1 tensors in kwargs to the given batch size.

    Block-level kwargs are captured with batch=1 to keep them
    batch-independent.  Before forwarding a block with a larger batch,
    every tensor whose first dimension is 1 is expanded via
    ``Tensor.expand`` (a zero-copy view) so that models whose internal
    operations require matching batch dimensions receive correctly
    shaped inputs.

    Args:
        kwargs: Block-level keyword arguments (may contain nested
            dicts, tuples, and lists).
        batch_size (int): Target batch size.

    Returns:
        dict: kwargs with expanded tensors.
    """
    if batch_size <= 1:
        # shallow copy for enabling prepare_block_kwargs to use pop() in safely
        return dict(kwargs)

    def _expand(v):
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 1:
            return v.expand(batch_size, *v.shape[1:])
        elif isinstance(v, tuple):
            return tuple(_expand(t) for t in v)
        elif isinstance(v, list):
            return [_expand(t) for t in v]
        elif isinstance(v, dict):
            return {k: _expand(val) for k, val in v.items()}
        return v

    return {k: _expand(v) for k, v in kwargs.items()}


def prepare_block_kwargs(batch_kwargs, block, pli, offset, batch_size, device):
    """Adjust batch_kwargs so that they are correct for block.

    Three following concerns are handled for Gemma4:
    1. per_layer_input: token-dependent, per-layer embedding sliced 
    from the full _per_layer_inputs tensor.
    2. position_embeddings: differ between full_attention and sliding_attention layer types.
    3. attention_mask: causal vs sliding-window mask.
    """
    # 1) per_layer_input
    batch_kwargs.pop(_PER_LAYER_INPUTS_KEY, None)
    if pli is not None:
        layer_idx = getattr(block, "layer_idx", None)
        if layer_idx is not None:
            batch_kwargs["per_layer_input"] = (
                pli[offset : offset + batch_size, :, layer_idx, :].to(device)
            )

    # 2) Per-type position embeddings
    pos_map = batch_kwargs.pop(_POS_EMB_MAP_KEY, None)
    if pos_map is not None:
        layer_type = getattr(block, "layer_type", None) or getattr(
            getattr(block, "self_attn", None), "layer_type", None
        )
        if layer_type and layer_type in pos_map:
            batch_kwargs["position_embeddings"] = pos_map[layer_type]

    # 3) Per-type attention mask
    mask_map = batch_kwargs.pop(_ATTN_MASK_MAP_KEY, None)
    if mask_map is not None:
        layer_type = getattr(block, "layer_type", None) or getattr(
            getattr(block, "self_attn", None), "layer_type", None
        )
        if layer_type and layer_type in mask_map:
            batch_kwargs["attention_mask"] = mask_map[layer_type]

    return batch_kwargs


@torch.no_grad()
def forward_input(
    inps: torch.Tensor,
    block: nn.Module,
    kwargs: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Forward the input through the block

    Args:
        inps (torch.Tensor): activation inputs of the block
        block (nn.Module): Transformer block to forward the input
        kwargs (dict[str, torch.Tensor]): other keyword arguments for the block forward
        batch_size (int): Batch size for forwarding
        device (torch.device): Device to move the input

    Returns:
        torch.Tensor: The output of the block
    """
    pli = kwargs.get(_PER_LAYER_INPUTS_KEY)
    next_inps = []
    offset = 0
    for inp in inps.split(batch_size):
        bs = inp.shape[0]
        batch_kwargs = expand_kwargs_batch(kwargs, bs)
        batch_kwargs = prepare_block_kwargs(batch_kwargs, block, pli, offset, bs, device)
        out = block(inp.to(device), **batch_kwargs)
        out = out[0] if isinstance(out, tuple) else out
        next_inps.append(out.cpu())
        offset += bs
    return torch.cat(next_inps)


def backward_input(
    inps: torch.Tensor,
    block: nn.Module,
    grad: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Backward through a block, returning grad w.r.t. its input.

    Runs forward + backward on each mini-batch so that only one batch
    lives on device at a time.

    Args:
        inps: Input activations that were fed into block during forward.
        block: Transformer block to differentiate through.
        grad: Upstream gradient (same leading dims as inps).
        kwargs: Extra keyword arguments forwarded to the block.
        batch_size: Mini-batch size.
        device: Device to run the computation on.

    Returns:
        Gradient of the loss w.r.t. inps (on CPU).
    """
    pli = kwargs.get(_PER_LAYER_INPUTS_KEY)
    all_inp_grads = []

    for j in range(0, inps.shape[0], batch_size):
        inp_batch = inps[j : j + batch_size].to(device)
        inp_batch = inp_batch.detach().requires_grad_(True)
        grad_batch = grad[j : j + batch_size].to(device)
        bs = inp_batch.shape[0]
        batch_kwargs = expand_kwargs_batch(kwargs, bs)
        batch_kwargs = prepare_block_kwargs(batch_kwargs, block, pli, j, bs, device)

        with torch.enable_grad():
            out = block(inp_batch, **batch_kwargs)
            out = out[0] if isinstance(out, tuple) else out
            out.backward(grad_batch)

        all_inp_grads.append(inp_batch.grad.cpu())

    block.zero_grad()
    return torch.cat(all_inp_grads)
