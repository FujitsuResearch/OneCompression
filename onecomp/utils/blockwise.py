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

    def forward(self, inp: torch.Tensor, **kwargs: dict):
        self.inp = inp.clone()
        self.kwargs.update(kwargs)
        raise StopForward()


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

    Args:
        model (nn.Module): The model to analyze.
        model_inputs (dict[str, torch.Tensor]): The input tensors for the model.
        batch_size (int): The batch size for computing input activations.

    Returns:
        tuple[nn.ModuleList, torch.Tensor, dict[str, torch.Tensor]]:
        The list of transformer blocks, the input activations, and the keyword arguments.
    """

    blocks = _get_blocks(model)

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

    return (blocks, inps, kwargs)


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
        return kwargs

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
    next_inps = []
    for inp in inps.split(batch_size):
        batch_kwargs = expand_kwargs_batch(kwargs, inp.shape[0])
        out = block(inp.to(device), **batch_kwargs)
        out = out[0] if isinstance(out, tuple) else out
        next_inps.append(out.cpu())
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
    all_inp_grads = []

    for j in range(0, inps.shape[0], batch_size):
        inp_batch = inps[j : j + batch_size].to(device)
        inp_batch = inp_batch.detach().requires_grad_(True)
        grad_batch = grad[j : j + batch_size].to(device)
        batch_kwargs = expand_kwargs_batch(kwargs, inp_batch.shape[0])

        with torch.enable_grad():
            out = block(inp_batch, **batch_kwargs)
            out = out[0] if isinstance(out, tuple) else out
            out.backward(grad_batch)

        all_inp_grads.append(inp_batch.grad.cpu())

    block.zero_grad()
    return torch.cat(all_inp_grads)
