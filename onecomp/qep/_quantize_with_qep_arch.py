"""
Architecture-aware Quantization with QEP Module

Copyright 2025-2026 Fujitsu Ltd.

Author: Yudai Fujimoto

An architecture-aware implementation that exploits model structure to
reduce redundant forward passes and memory consumption.
For example, in Llama-like architectures, QKV layers share the same
input activations, so they can be captured once and reused.


"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import math
import copy
from logging import getLogger
from collections import OrderedDict

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from onecomp.model_config import ModelConfig
from onecomp.qep._qep_config import QEPConfig
from onecomp.quantizer._quantizer import Quantizer
from onecomp.utils import prepare_calibration_dataset

logger = getLogger(__name__)


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


def _make_tensor_id(x: torch.Tensor):
    ptr = x.untyped_storage().data_ptr()
    return (id(x), ptr)


def make_grouped_module(
    block: nn.Module,
    inps: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    device: torch.device,
) -> list[list[nn.Module]]:
    """Make groups of modules that share the same input activations.

    Args:
        block (nn.Module): The transformer block to analyze for grouping modules.
        inps (torch.Tensor): The input activations.
        kwargs (dict[str, torch.Tensor]): The keyword arguments for the forward pass.
        device (torch.device): The device to use for computation.

    Returns:
        list[list[nn.Module]]: A list of groups, where each group is a list of modules
        that share the same input activations.
    """
    groups = OrderedDict()

    def hook(module, inp, _):
        key = _make_tensor_id(inp[0] if isinstance(inp, tuple) else inp)
        if key not in groups:
            groups[key] = [module]
        else:
            groups[key].append(module)

    handlers = []
    for module in block.modules():
        if isinstance(module, nn.Linear):
            handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        inp = inps[0].unsqueeze(0).to(device)
        _ = block(inp, **kwargs)

    for handler in handlers:
        handler.remove()

    return list(groups.values())


@torch.no_grad()
def compute_hessian_and_crossterm(
    block_q: nn.Module,
    block_f: nn.Module,
    module_q: nn.Module,
    module_f: nn.Module,
    inps_q: torch.Tensor,
    inps_f: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        block_q (nn.Module): The Transformer block to be quantized.
        block_f (nn.Module): The Transformer block containing full-precision weights.
        module_q (nn.Module): The layer to be quantized.
        module_f (nn.Module): The layer containing the full-precision weight.
        inps_q (torch.Tensor): The input activations for the quantized block.
        inps_f (torch.Tensor): The input activations for the full-precision block.
        kwargs (dict[str, torch.Tensor]): The keyword arguments for the forward pass.
        batch_size (int): The batch size for computation.
        device (torch.device): The device to use for computation.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The Hessian matrix and the cross-term matrix.
    """

    # set input hook
    dest = {}

    def make_hook(name):
        def hook(module, inp, out):
            dest[name] = inp[0] if isinstance(inp, tuple) else inp

        return hook

    handlers = [
        module_q.register_forward_hook(make_hook("q")),
        module_f.register_forward_hook(make_hook("f")),
    ]

    # Initialize Hessian and cross-term matrix
    hidden_dim = module_q.in_features
    H = torch.zeros((hidden_dim, hidden_dim), device=device)
    C = torch.zeros((hidden_dim, hidden_dim), device=device)

    # compute Hessian and crossterm
    N = inps_q.size(0)
    nsamples = 0

    # compute Hessian and crossterm in batches
    for first in range(0, N, batch_size):
        last = min(first + batch_size, N)

        _ = block_q(inps_q[first:last].to(device), **kwargs)
        _ = block_f(inps_f[first:last].to(device), **kwargs)

        x_q = dest["q"].view(-1, hidden_dim).float()
        x_f = dest["f"].view(-1, hidden_dim).float()

        tmp = x_q.size(0)

        H *= nsamples / (nsamples + tmp)
        C *= nsamples / (nsamples + tmp)
        nsamples += tmp

        # Scaling
        x_q_scaled = math.sqrt(2 / nsamples) * x_q
        x_f_scaled = math.sqrt(2 / nsamples) * x_f

        H += x_q_scaled.t() @ x_q_scaled
        C += (x_f_scaled - x_q_scaled).t() @ x_q_scaled

    for handler in handlers:
        handler.remove()

    return (H, C)


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


@torch.no_grad()
def run_quantize_with_qep_arch(
    model_config: ModelConfig,
    quantizer: Quantizer,
    qep_config: QEPConfig,
    calibration_dataset,
    max_length: int,
    num_calibration_samples: int,
    calibration_strategy: str,
    calibration_seed: int,
):
    """Run architecture-aware quantization with QEP.

    Exploits model structure to avoid redundant forward passes.
    For example, in Llama-like architectures, Q/K/V projections in the
    same attention block share the same input activations, so the
    activation capture can be performed once per block instead of once
    per layer.

    Args:
        model_config (ModelConfig): Model configuration for loading the model
            and tokenizer.
        quantizer (Quantizer): The quantizer to use.
        qep_config (QEPConfig): Configuration for QEP
            (percdamp, perccorr, exclude_layer_keywords).
        calibration_dataset: Calibration dataset. If None, a default dataset
            is loaded.
        max_length (int): Maximum sequence length for calibration inputs.
        num_calibration_samples (int): Number of calibration samples.
        calibration_strategy (str): Strategy for preparing calibration inputs.
        calibration_seed (int): Random seed for calibration data preparation.

    """

    # TODO: Parameterize when necessary
    batch_size = 16

    model = model_config.load_model(device_map="cpu")
    tokenizer = model_config.load_tokenizer()
    device = qep_config.device

    model_inputs = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        calibration_dataset=calibration_dataset,
        max_length=max_length,
        num_calibration_samples=num_calibration_samples,
        strategy=calibration_strategy,
        seed=calibration_seed,
        logger=logger,
    )

    # Setup the quantizer
    quantizer.setup(model)

    # 1. Prepare transformer blocks and their inputs
    blocks, inps, kwargs = get_blocks_and_inputs(model, model_inputs, batch_size)

    inps_q = inps
    inps_f = inps.clone()
    kwargs = move_kwargs_to_device(kwargs, device)

    logger.info("Quantizing the model using %s", quantizer.name)

    # 2. For each target transformer block, perform the following sequentially
    for block_idx, block in enumerate(blocks):

        logger.info(
            "Processing : %2d-th Transformer Block -------------------------------------------------",
            block_idx + 1,
        )

        block_q = block.to(device)
        block_f = copy.deepcopy(block_q)

        groups_q = make_grouped_module(block_q, inps_q, kwargs, device)
        groups_f = make_grouped_module(block_f, inps_f, kwargs, device)

        # 3. For each group of layers, perform the following sequentially
        for group_q, group_f in zip(groups_q, groups_f):

            logger.info(
                "Processing group of layers: %s",
                ", ".join([quantizer.module_to_name.get(m, "N/A") for m in group_q]),
            )

            # 3-1. compute hessian and cross-term matrix
            H, delta_hatX = compute_hessian_and_crossterm(
                block_q,
                block_f,
                group_q[0],
                group_f[0],
                inps_q,
                inps_f,
                kwargs,
                batch_size,
                device,
            )

            # 3-2, For each module in the group, perform weight correction and quantization
            for module in group_q:

                # Skip layers not registered for quantization
                if module not in quantizer.module_to_name:
                    logger.info("Skipping layer (not in quantization targets)")
                    continue

                name = quantizer.module_to_name[module]

                # Determine whether to apply weight correction (QEP).
                # Layers matching exclude_layer_keywords are quantized
                # but without error-propagation weight correction.
                # Note: hessian is always passed because it is needed for
                # quantization itself (e.g., GPTQ uses it via flag_hessian).
                exclude = any(kw in name for kw in qep_config.exclude_layer_keywords)
                layer_delta = None if exclude else delta_hatX.clone()

                logger.info(
                    "Processing layer: %s %s=================================================",
                    name,
                    "(no weight correction) " if exclude else "",
                )

                # Quantize the weights of the target layer
                quantizer.quantize_with_qep(
                    module,
                    quant_input_activation=None,
                    original_input_activation=None,
                    percdamp=qep_config.percdamp,
                    perccorr=qep_config.perccorr,
                    hessian=H.clone(),
                    delta_hatX=layer_delta,
                )

                # Update the weights of the target layer
                dtype = module.weight.data.dtype
                module.weight.data = (
                    quantizer.results[name].dequantized_weight.to(device).to(dtype)
                )

        # forward input to the next block
        inps_q = forward_input(inps_q, block_q, kwargs, batch_size, device)
        inps_f = forward_input(inps_f, block_f, kwargs, batch_size, device)

        # free memory
        block_q.cpu()
        torch.cuda.empty_cache()

    quantizer.execute_post_processing()
