"""
Architecture-aware Quantization with QEP Module

Copyright 2025-2026 Fujitsu Ltd.

Author: Yudai Fujimoto, Yuma Ichikawa

An architecture-aware implementation that exploits model structure to
reduce redundant forward passes and memory consumption.
For example, in Llama-like architectures, QKV layers share the same
input activations, so they can be captured once and reused.


"""

import math
import copy
from logging import getLogger
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from onecomp.calibration import CalibrationConfig, prepare_calibration_dataset
from onecomp.model_config import ModelConfig
from onecomp.qep._qep_config import QEPConfig
from onecomp.quantizer._quantizer import Quantizer
from onecomp.utils.blockwise import (
    _PER_LAYER_INPUTS_KEY,
    prepare_block_kwargs,
    get_blocks_and_inputs,
    forward_input,
    move_kwargs_to_device,
    expand_kwargs_batch,
)

logger = getLogger(__name__)


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
    # Store (module, raw_tensor) pairs.
    captured: list[tuple[nn.Module, torch.Tensor]] = []

    def hook(module, inp, _):
        tensor = inp[0] if isinstance(inp, tuple) else inp
        # Keep the tensor reference alive to prevent GC from reusing its id().
        captured.append((module, tensor))

    handlers = []
    for module in block.modules():
        if isinstance(module, nn.Linear):
            handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        inp = inps[0].unsqueeze(0).to(device)
        pli = kwargs.get(_PER_LAYER_INPUTS_KEY)
        block_kwargs = expand_kwargs_batch(kwargs, 1)
        block_kwargs = prepare_block_kwargs(block_kwargs, block, pli, 0, 1, device)
        _ = block(inp, **block_kwargs)

    for handler in handlers:
        handler.remove()

    # Group by tensor identity: modules that received the exact same
    # Python object share one input activation.  This avoids the false
    # positives that value-based comparison (torch.equal)
    groups: list[list[nn.Module]] = []
    tid_to_idx: dict[int, int] = {}
    for module, tensor in captured:
        tid = id(tensor)
        if tid in tid_to_idx:
            groups[tid_to_idx[tid]].append(module)
        else:
            tid_to_idx[tid] = len(groups)
            groups.append([module])

    del captured

    # Split groups by in_features so that modules with different input
    # dimensions (e.g. MoE router vs expert down_proj) stay separate.
    result = []
    for group in groups:
        by_dim = OrderedDict()
        for module in group:
            dim = module.in_features
            if dim not in by_dim:
                by_dim[dim] = []
            by_dim[dim].append(module)
        result.extend(by_dim.values())

    return result


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
    pli = kwargs.get(_PER_LAYER_INPUTS_KEY)

    # compute Hessian and crossterm in batches
    for first in range(0, N, batch_size):
        last = min(first + batch_size, N)
        bs = last - first
        batch_kwargs = expand_kwargs_batch(kwargs, bs)
        batch_kwargs = prepare_block_kwargs(batch_kwargs, block_q, pli, first, bs, device)

        _ = block_q(inps_q[first:last].to(device), **batch_kwargs)
        _ = block_f(inps_f[first:last].to(device), **batch_kwargs)

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
def _compute_per_module_hessians(
    block: nn.Module,
    modules: list[nn.Module],
    inps: torch.Tensor,
    kwargs: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[nn.Module, torch.Tensor | None]:
    """Compute independent Hessians for each module via shared forward passes.

    Used for MoE expert layers where the standard cross-term computation
    is invalid (the router in quantized vs full-precision blocks may route
    different tokens to the same expert).  Each module gets its own Hessian
    built solely from the quantized block's activations.
    """
    dest: dict[int, torch.Tensor] = {}

    def _make_hook(key):
        def hook(_, inp, __):
            dest[key] = inp[0] if isinstance(inp, tuple) else inp
        return hook

    handlers = [m.register_forward_hook(_make_hook(i)) for i, m in enumerate(modules)]

    hessians: dict[int, torch.Tensor] = {}
    nsamples: dict[int, int] = {}
    for i, m in enumerate(modules):
        dim = m.in_features
        hessians[i] = torch.zeros((dim, dim), device=device)
        nsamples[i] = 0

    N = inps.size(0)
    pli = kwargs.get(_PER_LAYER_INPUTS_KEY)

    for first in range(0, N, batch_size):
        last = min(first + batch_size, N)
        bs = last - first
        batch_kwargs = expand_kwargs_batch(kwargs, bs)
        batch_kwargs = prepare_block_kwargs(batch_kwargs, block, pli, first, bs, device)
        _ = block(inps[first:last].to(device), **batch_kwargs)

        for i, m in enumerate(modules):
            if i not in dest:
                continue
            x = dest[i].view(-1, m.in_features).float()
            tmp = x.size(0)
            if tmp == 0:
                continue
            hessians[i] *= nsamples[i] / (nsamples[i] + tmp)
            nsamples[i] += tmp
            x_scaled = math.sqrt(2 / nsamples[i]) * x
            hessians[i] += x_scaled.t() @ x_scaled

        dest.clear()

    for h in handlers:
        h.remove()

    return {
        modules[i]: (hessians[i] if nsamples[i] > 0 else None)
        for i in range(len(modules))
    }


@torch.no_grad()
def run_quantize_with_qep_arch(
    model_config: ModelConfig,
    quantizer: Quantizer,
    qep_config: QEPConfig,
    calibration_config: CalibrationConfig,
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
        calibration_config (CalibrationConfig): Calibration parameters.

    """

    # TODO: Parameterize when necessary
    batch_size = 16

    model = model_config.load_model(device_map="cpu")
    tokenizer = model_config.load_tokenizer()
    device = qep_config.device

    model_inputs = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        calibration_config=calibration_config,
        model=model,
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

    # Build set of remaining target names for early termination.
    # Restrict to modules that actually reside within the language-model blocks so
    # that VLM vision-encoder layers (registered by quantizer.setup but unreachable
    # via the block loop) do not prevent early termination.
    block_modules = {m for block in blocks for m in block.modules()}
    remaining_targets = {
        name for module, name in quantizer.module_to_name.items() if module in block_modules
    }

    # 2. For each target transformer block, perform the following sequentially
    for block_idx, block in enumerate(blocks):

        logger.info(
            "Processing : %2d-th Transformer Block -------------------------------------------------",
            block_idx + 1,
        )

        # Early termination: stop once all target layers are quantized
        if not remaining_targets:
            logger.info("All target layers quantized -- skipping remaining blocks.")
            break

        block_q = block.to(device)
        block_f = copy.deepcopy(block_q)

        groups_q = make_grouped_module(block_q, inps_q, kwargs, device)

        # Build name→module map for block_f, then align groups_f to
        # groups_q by module name.  Using make_grouped_module on
        # block_f independently can produce a different group ordering
        # because tensor identity (used for grouping) is
        # non-deterministic across deepcopy + forward.
        name_to_module_f = {
            name: mod for name, mod in block_f.named_modules() if isinstance(mod, nn.Linear)
        }
        name_to_module_q = {
            name: mod for name, mod in block_q.named_modules() if isinstance(mod, nn.Linear)
        }
        groups_f = [
            [
                name_to_module_f[next(n for n, m2 in name_to_module_q.items() if m2 is m)]
                for m in gq
            ]
            for gq in groups_q
        ]

        # Partition groups into regular and MoE-expert groups.
        # Expert layers need per-module Hessians because MoE routing
        # produces different token subsets in block_q vs block_f.
        regular_pairs: list[tuple[list, list]] = []
        expert_modules_q: list[nn.Module] = []

        for group_q, group_f in zip(groups_q, groups_f):
            targets = [m for m in group_q if m in quantizer.module_to_name]
            if not targets:
                continue
            is_expert = any(
                ".experts." in quantizer.module_to_name[m] for m in targets
            )
            if is_expert:
                expert_modules_q.extend(targets)
            else:
                regular_pairs.append((group_q, group_f))

        # 3. Process regular (non-expert) groups with full QEP
        for group_q, group_f in regular_pairs:

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
                try:
                    dtype = module.weight.data.dtype
                    module.weight.data = (
                        quantizer.results[name].compute_dequantized_weight().to(device).to(dtype)
                    )
                except (ValueError, NotImplementedError):
                    logger.error(
                        "Failed to compute dequantized weight for %s. "
                        "Keeping original weights.",
                        name,
                    )
                remaining_targets.discard(name)

        # 4. Process MoE expert layers with per-module Hessians (no cross-term)
        if expert_modules_q:
            logger.info(
                "Computing per-expert Hessians for %d MoE expert layers "
                "(weight correction disabled for expert layers)",
                len(expert_modules_q),
            )
            expert_hessians = _compute_per_module_hessians(
                block_q, expert_modules_q, inps_q, kwargs, batch_size, device,
            )
            for module_q in expert_modules_q:
                name = quantizer.module_to_name[module_q]
                H = expert_hessians[module_q]
                if H is None:
                    logger.warning(
                        "Expert layer %s received no tokens during calibration; skipping",
                        name,
                    )
                    remaining_targets.discard(name)
                    continue

                logger.info(
                    "Processing layer: %s (no weight correction) =================================================",
                    name,
                )
                quantizer.quantize_with_qep(
                    module_q,
                    quant_input_activation=None,
                    original_input_activation=None,
                    percdamp=qep_config.percdamp,
                    perccorr=qep_config.perccorr,
                    hessian=H,
                    delta_hatX=None,
                )
                try:
                    dtype = module_q.weight.data.dtype
                    module_q.weight.data = (
                        quantizer.results[name].compute_dequantized_weight().to(device).to(dtype)
                    )
                except (ValueError, NotImplementedError):
                    logger.error(
                        "Failed to compute dequantized weight for %s. "
                        "Keeping original weights.",
                        name,
                    )
                remaining_targets.discard(name)

        # forward input to the next block
        inps_q = forward_input(inps_q, block_q, kwargs, batch_size, device)
        inps_f = forward_input(inps_f, block_f, kwargs, batch_size, device)

        # Compute MSE between quantized and full-precision block outputs
        mse = F.mse_loss(inps_q.float(), inps_f.float()).item()
        logger.info(f"[INFO] Layer {block_idx + 1} MSE: {mse:.6e}")

        # free memory
        block_q.cpu()
        torch.cuda.empty_cache()

    quantizer.execute_post_processing()
