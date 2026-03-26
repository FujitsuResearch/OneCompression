"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Yudai Fujimoto, Akihiro Yoshida

"""

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer


def _get_blocks(
    model: nn.Module,
) -> nn.ModuleList:
    """Get the list of transformer blocks in the model.

    Args:
        model (nn.Module): The model to analyze.

    Raises:
        RuntimeError: If transformer blocks are not found.

    Returns:
        nn.ModuleList: The list of transformer blocks.
    """
    for module in model.modules():
        if isinstance(module, nn.ModuleList):
            if len(module) > 0 and isinstance(module[0], GradientCheckpointingLayer):
                return module

    raise RuntimeError("Transformer blocks not found.")


class StopForward(Exception):
    """An exception to stop the forward pass after capturing activations."""

    pass


class Catcher(nn.Module):
    """A wrapper module to capture input activations and keyword arguments."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.inp = None
        self.kwargs = {}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # return model-specific attributes such as Qwen3's attention_type
            return getattr(self.module, name)

    def forward(self, inp: torch.Tensor, **kwargs):
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

    block_inps = []

    # forward model to capture the input activations of the first block
    with torch.no_grad():
        for inp in inp_ids.split(batch_size):
            try:
                _ = model(inp, **model_kwargs)
            except StopForward:
                block_inps.append(blocks[0].inp.cpu())

    inps = torch.cat(block_inps)
    kwargs = blocks[0].kwargs

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
        out = block(inp.to(device), **kwargs)
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

        with torch.enable_grad():
            out = block(inp_batch, **kwargs)
            out = out[0] if isinstance(out, tuple) else out
            out.backward(grad_batch)

        all_inp_grads.append(inp_batch.grad.cpu())

    block.zero_grad()
    return torch.cat(all_inp_grads)
