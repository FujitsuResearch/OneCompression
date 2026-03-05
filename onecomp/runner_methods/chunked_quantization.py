"""
Chunked Calibration Quantization Module

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

Quantization module for large-scale calibration data.
Splits calibration data and accumulates X^T X across multiple forward passes,
reducing memory usage while performing quantization.

Multiple quantizers can be specified simultaneously, sharing the X^T X
accumulation (Phase 1) to reduce the number of forward passes during
benchmark comparisons.

Processing flow:
1. Prepare calibration data on CPU
2. Load the model and set up quantizers
3. Split layers into groups, and for each group:
   Phase 1: Forward pass per chunk -> accumulate X^T X in FP64 (once only)
   Phase 2: Quantize using X^T X with each quantizer
   Phase 3: Compute quantization errors with each quantizer

"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import time
from logging import getLogger
from typing import List

import torch

from onecomp.model_config import ModelConfig
from onecomp.quantizer._quantizer import Quantizer, QuantizationResult
from onecomp.utils import prepare_calibration_dataset


logger = getLogger(__name__)


# =============================================================================
# Main: run_chunked_quantization
# =============================================================================


def run_chunked_quantization(
    model_config: ModelConfig,
    quantizers: List[Quantizer],
    calibration_dataset,
    max_length: int,
    num_calibration_samples: int,
    calibration_strategy: str,
    calibration_seed: int,
    calibration_batch_size: int,
    num_layers_per_group: int,
):
    """Run quantization for large-scale calibration data.

    When multiple quantizers are specified, X^T X accumulation (Phase 1) is shared,
    and quantization (Phase 2) and error computation (Phase 3) are performed
    sequentially for each quantizer.

    Args:
        model_config (ModelConfig): Model configuration.
        quantizers (list[Quantizer]): List of quantizers. Each quantizer must have
            flag_hessian=True or flag_xtx=True.
        calibration_dataset (datasets.Dataset): Calibration dataset.
        max_length (int): Maximum sequence length.
        num_calibration_samples (int): Number of calibration samples.
        calibration_strategy (str): Calibration strategy.
        calibration_seed (int): Random seed.
        calibration_batch_size (int): Number of sentences per chunk.
        num_layers_per_group (int): Number of layers to process simultaneously.

    Note:
        Results are stored directly in each quantizer.results.
    """

    # Load model
    model = model_config.load_model()
    tokenizer = model_config.load_tokenizer()
    input_device = next(model.parameters()).device

    # Prepare calibration data on CPU (to save GPU memory)
    inputs = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        calibration_dataset=calibration_dataset,
        max_length=max_length,
        num_calibration_samples=num_calibration_samples,
        strategy=calibration_strategy,
        seed=calibration_seed,
        logger=logger,
    )
    total_samples = inputs["input_ids"].shape[0]

    logger.info(
        "Chunked calibration: total_samples=%d, batch_size=%d, num_chunks=%d",
        total_samples,
        calibration_batch_size,
        (total_samples + calibration_batch_size - 1) // calibration_batch_size,
    )

    quantizer_names = [q.name for q in quantizers]
    logger.info("Quantizers: %s", ", ".join(quantizer_names))

    # Set up each quantizer (build module_to_name)
    for quantizer in quantizers:
        quantizer.setup(model)

    # Verify that all quantizers target the same layers
    layer_names_list = [list(q.module_to_name.values()) for q in quantizers]
    if not all(names == layer_names_list[0] for names in layer_names_list[1:]):
        raise ValueError(
            "All quantizers must target the same layers. Got:\n"
            + "\n".join(f"  {q.name}: {names}" for q, names in zip(quantizers, layer_names_list))
        )

    # List all target layers
    all_layers = list(quantizers[0].module_to_name.items())  # [(module, name), ...]

    logger.info(
        "Chunked calibration: %d layers total, %d layers/group, %d groups",
        len(all_layers),
        num_layers_per_group,
        (len(all_layers) + num_layers_per_group - 1) // num_layers_per_group,
    )

    for group_start in range(0, len(all_layers), num_layers_per_group):
        group = all_layers[group_start : group_start + num_layers_per_group]
        group_names = [name for _, name in group]
        num_groups = (len(all_layers) + num_layers_per_group - 1) // num_layers_per_group

        group_idx = group_start // num_layers_per_group + 1
        layers_list = "\n".join(f"  - {name}" for name in group_names)
        logger.info(
            "Processing layer group %d/%d (%d layers):\n%s",
            group_idx,
            num_groups,
            len(group),
            layers_list,
        )

        # --- Phase 1: Forward per chunk -> accumulate X^T X (FP64) ---
        xtx_dict, nsamples = accumulate_xtx(
            model=model,
            inputs=inputs,
            input_device=input_device,
            group=group,
            calibration_batch_size=calibration_batch_size,
        )

        # --- Phase 2 & 3: Quantize and compute errors for each quantizer ---
        for quantizer in quantizers:
            logger.info("Quantizing with %s ...", quantizer.name)
            quantize_group(quantizer, group, xtx_dict, nsamples)

            if quantizer.calc_quant_error:
                record_quantization_errors(quantizer, group, xtx_dict, nsamples)

        # Release X^T X
        xtx_dict.clear()

    # Post-processing
    for quantizer in quantizers:
        quantizer.execute_post_processing()


# =============================================================================
# Phase 1: Accumulate X^T X
# =============================================================================


def accumulate_xtx(
    model,
    inputs,
    input_device,
    group,
    calibration_batch_size,
):
    """Accumulate X^T X in FP64 on GPU by forwarding in chunks.

    Within the hook, X^T X is computed and accumulated directly from activations,
    eliminating the need to store activations and improving GPU memory efficiency.

    Args:
        model: Model for forward passes.
        inputs: Calibration data on CPU.
            {"input_ids": (total_samples, max_length),
             "attention_mask": (total_samples, max_length)}
        input_device: Model's input device (GPU).
        group: List of target layers [(module, name), ...].
        calibration_batch_size: Number of sentences per chunk.

    Returns:
        xtx_dict: Dict[name, torch.Tensor]
            X^T X for each layer (FP64, CPU), shape (in_features, in_features).
        nsamples: int
            Number of samples (= total_samples * seq_len). Common across all layers.
    """

    total_samples = inputs["input_ids"].shape[0]

    # Dictionary for accumulating X^T X (accumulate on GPU, move to CPU at the end)
    xtx_dict = {}  # name -> Tensor (FP64, GPU)
    nsamples = 0

    # Compute and accumulate X^T X directly within the hook
    def make_accumulate_hook(name):
        def hook(_module, input, _output):  # pylint: disable=redefined-builtin
            # input: (batch, seq_len, hidden_size) -> (batch * seq_len, hidden_size)
            matrix_x = (input[0] if isinstance(input, tuple) else input).detach()
            matrix_x = matrix_x.reshape(-1, matrix_x.shape[-1]).to(torch.float64)
            xtx_chunk = matrix_x.T @ matrix_x  # (hidden_size, hidden_size), FP64, GPU

            if name not in xtx_dict:
                xtx_dict[name] = xtx_chunk
            else:
                xtx_dict[name] += xtx_chunk

        return hook

    # Register hooks
    handles = []
    for module, name in group:
        handle = module.register_forward_hook(make_accumulate_hook(name))
        handles.append(handle)

    # Forward per chunk -> accumulate X^T X within hooks
    num_chunks = (total_samples + calibration_batch_size - 1) // calibration_batch_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * calibration_batch_size
        chunk_end = min(chunk_start + calibration_batch_size, total_samples)

        # Move chunk to GPU
        chunk_inputs = {k: v[chunk_start:chunk_end].to(input_device) for k, v in inputs.items()}

        # Forward pass (X^T X is accumulated within hooks)
        with torch.no_grad():
            model(**chunk_inputs)

        # Add to nsamples (batch * seq_len, common across all layers)
        chunk_size = chunk_end - chunk_start
        seq_len = inputs["input_ids"].shape[1]
        nsamples += chunk_size * seq_len

        # Free memory
        del chunk_inputs
        torch.cuda.empty_cache()

        logger.info(
            "  Chunk %d/%d done (samples %d-%d)",
            chunk_idx + 1,
            num_chunks,
            chunk_start,
            chunk_end,
        )

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Move X^T X to CPU
    xtx_dict = {name: xtx.cpu() for name, xtx in xtx_dict.items()}

    return xtx_dict, nsamples


# =============================================================================
# Phase 2: X^T X -> Hessian -> Quantization
# =============================================================================


def quantize_group(quantizer, group, xtx_dict, nsamples):
    """Quantize each layer in the group using accumulated X^T X.

    flag_hessian=True (GPTQ, DBF, etc.):
        Compute Hessian and pass to quantize_layer: H = (2 / nsamples) * X^T X
    flag_xtx=True (JointQ, etc.):
        Pass X^T X and nsamples directly to quantize_layer.

    Args:
        quantizer: Quantizer instance.
        group: List of target layers [(module, name), ...].
        xtx_dict: Dict[name, torch.Tensor]
            X^T X for each layer (FP64, CPU), shape (in_features, in_features).
        nsamples: int
            Number of samples (= total_samples * seq_len).
    """

    for module, name in group:
        logger.info("Quantizing layer: %s", name)
        start_time = time.time()

        if quantizer.flag_xtx:
            # Pass X^T X directly (JointQ, etc.)
            # TODO: If quantize_layer modifies matrix_XX in-place,
            #       xtx_dict[name] will be corrupted. This is not an issue
            #       currently, but consider using .clone() in case the
            #       quantize_layer implementation changes in the future.
            result = quantizer.quantize_layer(
                module, input=None, matrix_XX=xtx_dict[name], dim_n=nsamples
            )
        elif quantizer.flag_hessian:
            # X^T X -> Hessian: H = (2 / nsamples) * X^T X
            device = module.weight.device
            hessian = (2.0 / nsamples) * xtx_dict[name]
            hessian = hessian.to(dtype=quantizer.hessian_dtype, device=device)
            result = quantizer.quantize_layer(module, input=None, hessian=hessian)
            del hessian
        else:
            # Hessian/X^T X not required (RTN, etc.)
            result = quantizer.quantize_layer(module)

        end_time = time.time()

        # Backward compatibility: convert Tensor to QuantizationResult if needed
        if isinstance(result, torch.Tensor):
            result = QuantizationResult(dequantized_weight=result)
        result.quantization_time = end_time - start_time

        quantizer.results[name] = result
        torch.cuda.empty_cache()


# =============================================================================
# Phase 3: Record Quantization Errors
# =============================================================================


def record_quantization_errors(quantizer, group, xtx_dict, nsamples):
    """Compute and record quantization errors using X^T X.

    Computes the following two types of errors:

    1. Weight error: ||W - W_hat||^2_F
    2. Output error: ||delta_W X^T||^2_F = tr(delta_W * X^T X * delta_W^T)
       where delta_W = W - W_hat

    Args:
        quantizer: Quantizer instance.
        group: List of target layers [(module, name), ...].
        xtx_dict: Dict[name, torch.Tensor]
            X^T X for each layer (FP64, CPU), shape (in_features, in_features).
        nsamples: int
            Number of samples (= total_samples * seq_len).
    """

    for module, name in group:
        result = quantizer.results[name]
        dequantized_weight = result.dequantized_weight

        # Weight error: ||W - W_hat||^2_F
        (
            result.weight_squared_error,
            result.mean_weight_squared_error,
            result.relative_weight_squared_error,
        ) = quantizer.calculate_weight_quantization_error(module, dequantized_weight)

        # Output error: ||delta_W X^T||^2_F = tr(delta_W * X^T X * delta_W^T)
        device = module.weight.device
        matrix_W = module.weight.data.detach().to(device).to(torch.float64)
        delta_W = matrix_W - dequantized_weight.to(device).to(torch.float64)
        xtx = xtx_dict[name].to(device=device, dtype=torch.float64)

        # tr(delta_W * XtX * delta_W^T) = sum((delta_W @ XtX) * delta_W)
        temp = delta_W @ xtx  # (out_features, in_features)
        output_squared_error = (temp * delta_W).sum().item()
        del temp

        # ||WX^T||^2_F = tr(W * X^T X * W^T) = sum((W @ XtX) * W)
        temp_w = matrix_W @ xtx  # (out_features, in_features)
        original_output_norm_squared = (temp_w * matrix_W).sum().item()
        del temp_w, xtx, matrix_W

        num_elements = delta_W.shape[0] * nsamples  # out_features * nsamples
        mean_output_squared_error = output_squared_error / num_elements
        del delta_W

        result.output_squared_error = output_squared_error
        result.mean_output_squared_error = mean_output_squared_error

        # Relative output error = ||delta_W X^T||^2_F / ||WX^T||^2_F
        result.relative_output_squared_error = (
            output_squared_error / original_output_norm_squared
            if original_output_norm_squared > 0
            else None
        )

        torch.cuda.empty_cache()
