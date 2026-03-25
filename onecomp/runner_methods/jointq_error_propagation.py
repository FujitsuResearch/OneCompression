"""
JointQ Error Propagation Module

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import time
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Any, List, Optional

import torch

from jointq import quantize_advanced
from onecomp.quantizer.jointq._jointq import JointQResult
from onecomp.utils import capture_input_activations


@dataclass
class JointQErrorPropResult(JointQResult):
    """JointQ Error Propagation quantization result class.

    Inherits from JointQResult and adds Error Propagation-specific parameters.

    Attributes:
        (Inherited from JointQResult)
        dequantized_weight: Dequantized weight (FP16, CPU).
        bits: Quantization bit width.
        symmetric: Whether symmetric quantization was used.
        group_size: Group size.
        scale: Scale factor.
        zero_point: Zero point.
        assignment: Integer assignment.
        quantization_time: Time taken for Step 1 (per-layer quantization) in seconds.

        (Error Propagation-specific)
        quantization_time_step2: Time taken for Step 2 (Error Propagation re-quantization) in seconds.
        skip_step2: Whether Step 2 re-quantization was skipped.
    """

    quantization_time_step2: float = None
    skip_step2: bool = None


def run_jointq_error_propagation(
    model,
    inputs: Dict[str, Any],
    current_results: Dict[str, Any],
    logger=None,
    max_layers: int = None,
    skip_threshold_increase: float = 0.01,
    skip_threshold_error: float = 0.01,
    skip_threshold_amplification: float = 5.0,
    device=None,
    batch_size: int = None,
    variation_scale: float = 0.1,
    variation_cap: float = 0.05,
    degradation_threshold: float = 0.1,
    max_iter: int = 10,
    log_level: int = 0,
    exclude_layer_keywords: Optional[List[str]] = None,
) -> None:
    """Run quantization using JointQ error propagation.

    A model-architecture-agnostic generic implementation.
    It consumes extra CPU memory and involves redundant forward passes.
    Exploiting model architecture to avoid redundant forward passes would enable
    faster quantization.

    Current procedure:
    1. Save the original model's input activations to CPU.
    2. For each target layer l, perform the following in order:
    2-1. Save the quantized model's input activations for layer l to CPU.
    2-2. Quantize the weights of layer l in the quantized model.
    2-3. Update the weights of layer l in the quantized model.

    TODO: Implement architecture-aware quantization. (Starting with generic naive implementation)

    Args:
        model: PyTorch model (already loaded).
        inputs: Calibration inputs (dict containing "input_ids" and "attention_mask").
        current_results: Dictionary to store quantization results (updated in-place).
        logger: Logger (uses the module logger if None).
        max_layers: Maximum number of layers to process (None for all layers, for testing).
        skip_threshold_increase: Skip threshold for relative error increase (default: 0.01).
        skip_threshold_error: Skip threshold for relative cumulative error (default: 0.01).
        skip_threshold_amplification (float): Skip threshold for error amplification ratio.
            Re-quantize if amplification exceeds this value even when g_relative is small
            (default: 5.0).
        device: Device for computation (uses each layer's device if None).
        batch_size (int): Batch size (default: None, solves the optimization problem all at once).
        variation_scale (float): Scaling factor from degradation rate to variation rate (default: 0.1).
        variation_cap (float): Upper bound for maximum variation rate (default: 0.05).
        degradation_threshold (float): Degradation rate threshold; variation rate is 0 below this (default: 0.1).
        max_iter (int): Maximum iterations for quantize_advanced (default: 10).
        log_level (int): Log level for quantize_advanced (default: 0).
        exclude_layer_keywords (list[str]): List of keywords for layers to exclude from Step 2.
            If any keyword is found in a layer name, that layer is excluded from Step 2
            re-quantization (Step 1 results are used as-is).
            None means all layers are targeted (default: None).
    """
    if logger is None:
        logger = getLogger(__name__)

    logger.info("Running JointQ error propagation...")

    # Get target layer names from current_results keys
    target_layer_names = set(current_results.keys())

    # Build module_to_name (only layers included in current_results)
    module_to_name = {
        module: name for name, module in model.named_modules() if name in target_layer_names
    }

    # Set for determining layers excluded by exclude_layer_keywords
    exclude_layer_names = set()
    if exclude_layer_keywords:
        for name in list(module_to_name.values()):
            if any(kw in name for kw in exclude_layer_keywords):
                exclude_layer_names.add(name)
        logger.info(
            "Excluding %d layers from step2 by keywords %s: %s",
            len(exclude_layer_names),
            exclude_layer_keywords,
            sorted(exclude_layer_names),
        )

    # For testing: limit the number of layers
    if max_layers is not None:
        module_to_name = dict(list(module_to_name.items())[:max_layers])
        logger.info("Limiting to %d layers for testing", max_layers)

    # Step 1: Save the original model's input activations to CPU
    # (Excluded layers do not need Step 2, but are also excluded from original activation capture)
    step2_module_to_name = {
        m: n for m, n in module_to_name.items() if n not in exclude_layer_names
    }
    logger.info("Step 1: Capturing original input activations...")
    original_input_activations = capture_input_activations(
        model=model,
        inputs=inputs,
        module_to_name=step2_module_to_name,
        logger=logger,
    )

    # Step 2: For each layer, perform the following in order
    for module, name in module_to_name.items():
        # Excluded layers only have Step 1 weights applied (Step 2 skipped)
        if name in exclude_layer_names:
            logger.info(
                " ========= Applying step1 weight (excluded from step2): %s =========",
                name,
            )
            result = current_results[name]
            # TODO: Migrate to compute_dequantized_weight() API
            current_results[name] = JointQErrorPropResult(
                dequantized_weight=result.dequantized_weight,
                quantization_time=result.quantization_time,
                bits=result.bits,
                symmetric=result.symmetric,
                group_size=result.group_size,
                scale=result.scale,
                zero_point=result.zero_point,
                assignment=result.assignment,
                quantization_time_step2=0,
                skip_step2=True,
            )
            dtype = module.weight.data.dtype
            module_device = module.weight.data.device
            module.weight.data = result.dequantized_weight.to(module_device).to(
                dtype
            )  # TODO: Migrate to compute_dequantized_weight()
            continue

        logger.info(
            " ========= Processing layer: %s =========",
            name,
        )

        # 2-1. Save input activations for the target layer only to CPU
        quant_input_activation = capture_input_activations(
            model=model,
            inputs=inputs,
            module_to_name={module: name},
            logger=logger,
        )[name]

        # 2-2. Quantize the weights of the target layer
        current_results[name] = quantize_layer_weights(
            module,
            quant_input_activation,
            original_input_activations[name],
            current_results[name],
            skip_threshold_increase=skip_threshold_increase,
            skip_threshold_error=skip_threshold_error,
            skip_threshold_amplification=skip_threshold_amplification,
            device=device,
            batch_size=batch_size,
            variation_scale=variation_scale,
            variation_cap=variation_cap,
            degradation_threshold=degradation_threshold,
            max_iter=max_iter,
            log_level=log_level,
            logger=logger,
        )

        # 2-3. Update the weights of the target layer
        dtype = module.weight.data.dtype
        module_device = module.weight.data.device
        module.weight.data = (
            current_results[name].dequantized_weight.to(module_device).to(dtype)
        )  # TODO: Migrate to compute_dequantized_weight()

        del quant_input_activation

    del original_input_activations


def quantize_layer_weights(
    module,
    quant_input_activation,
    original_input_activation,
    current_result,
    skip_threshold_increase=0.01,
    skip_threshold_error=0.01,
    skip_threshold_amplification=5.0,
    device=None,
    batch_size=None,
    variation_scale=0.1,
    variation_cap=0.05,
    degradation_threshold=0.1,
    max_iter=10,
    log_level=0,
    logger=None,
):
    """Quantize the weights of a target layer (Step 2-2).

    The current_result contains a per-layer quantized result. Specifically, it stores
    the feasible solution W_quant* of:
    min f(W_quant), f(W_quant) := ||W_orig X_orig - W_quant X_orig||^2_F

    This function calls a module that solves the following optimization problem using
    the above feasible solution W_quant* as the initial solution:
    min g(W_quant), g(W_quant) := ||W_orig X_orig - W_quant X_quant||^2_F

    W_quant* is not updated (skipped) in the following cases:
    1. Relative cumulative error is small enough and amplification ratio is below threshold
       (g_relative < skip_threshold_error and (g-f)/f < skip_threshold_amplification)
    2. Error increase rate is small ((g - f) / f < skip_threshold_increase)

    Args:
        module: Target PyTorch module.
        quant_input_activation (torch.Tensor): Input activation of the layer in the quantized model.
        original_input_activation (torch.Tensor): Input activation of the layer in the original model.
        current_result (JointQResult): Quantization result for the target layer.
        skip_threshold_increase (float): Skip threshold for error increase rate (default: 0.01).
        skip_threshold_error (float): Skip threshold for relative cumulative error (default: 0.01).
        skip_threshold_amplification (float): Skip threshold for error amplification ratio.
            Re-quantize if amplification exceeds this value even when g_relative is small
            (default: 5.0).
        device: Device for computation (uses module.weight.data's device if None).
        batch_size (int): Batch size (default: None, solves the optimization problem all at once).
        variation_scale (float): Scaling factor from degradation rate to variation rate (default: 0.1).
        variation_cap (float): Upper bound for maximum variation rate (default: 0.05).
        degradation_threshold (float): Degradation rate threshold; variation rate is 0 below this (default: 0.1).
        max_iter (int): Maximum iterations for quantize_advanced (default: 10).
        log_level (int): Log level for quantize_advanced (default: 0).
        logger: Logger (uses the module logger if None).
    """
    if logger is None:
        logger = getLogger(__name__)

    # Determine device
    if device is None:
        device = module.weight.data.device

    # Convert to FP64 and reshape to (total_samples, in_features)
    flat_quant_input = quant_input_activation.reshape(
        -1, quant_input_activation.shape[-1]
    ).double()
    flat_original_input = original_input_activation.reshape(
        -1, original_input_activation.shape[-1]
    ).double()
    original_weight_cpu = module.weight.data.detach().cpu().double()
    original_weight = original_weight_cpu.to(device)

    # Note:
    # flat_quant_input and flat_original_input are on CPU.
    # original_weight is on GPU. (Transferred via CPU to avoid GPU-to-GPU P2P transfer)

    # Step 1: Skip decision
    skip, f_row_errors, g_row_errors = should_skip_requantization(
        original_weight=original_weight,
        current_result=current_result,
        flat_original_input=flat_original_input,
        flat_quant_input=flat_quant_input,
        skip_threshold_increase=skip_threshold_increase,
        skip_threshold_error=skip_threshold_error,
        skip_threshold_amplification=skip_threshold_amplification,
        logger=logger,
    )
    if skip:
        # TODO: Migrate to compute_dequantized_weight() API
        return JointQErrorPropResult(
            dequantized_weight=current_result.dequantized_weight,
            quantization_time=current_result.quantization_time,
            bits=current_result.bits,
            symmetric=current_result.symmetric,
            group_size=current_result.group_size,
            scale=current_result.scale,
            zero_point=current_result.zero_point,
            assignment=current_result.assignment,
            quantization_time_step2=0,
            skip_step2=True,
        )

    # Step 2: Re-quantization
    # Step 2-1: Preparation
    # matrix_Y: W_orig @ X_orig^T (out_features, total_samples)
    device = original_weight.device
    matrix_Y = (original_weight @ flat_original_input.to(device).T).cpu()
    del original_weight

    # Step 2-2: Compute maximum variation rate
    max_variation_rate = compute_max_variation_rate(
        f_row_errors=f_row_errors,
        g_row_errors=g_row_errors,
        variation_scale=variation_scale,
        variation_cap=variation_cap,
        degradation_threshold=degradation_threshold,
        logger=logger,
    ).cpu()
    del f_row_errors, g_row_errors

    start_time = time.time()
    solution = quantize_advanced(
        matrix_Y=matrix_Y,
        matrix_X=flat_quant_input,
        init_scale=current_result.scale,
        init_zero_point=current_result.zero_point,
        init_assignment=current_result.assignment,
        bits=current_result.bits,
        symmetric=current_result.symmetric,
        group_size=current_result.group_size,
        batch_size=batch_size,
        device=device,
        max_variation_rate=max_variation_rate,
        max_iter=max_iter,
        log_level=log_level,
    )
    quantization_time_step2 = time.time() - start_time

    # Get dequantized weight
    dequantized_weight = solution.get_dequantized_weight_matrix()

    # Return the dequantized weight in the same shape and dtype as the original
    dequantized_weight = (
        dequantized_weight.reshape(module.weight.shape).to(module.weight.data.dtype).cpu()
    )

    # Get quantized result (scale, assignment, zero_point)
    scale, assignment, zero_point = solution.get_quantized_result()

    # Compute and log g_error, g_relative and actual variation rate after re-quantization
    log_requantization_stats(
        original_weight=original_weight_cpu.to(device),
        dequantized_weight=dequantized_weight,
        current_result=current_result,
        flat_original_input=flat_original_input,
        flat_quant_input=flat_quant_input,
        device=device,
        logger=logger,
    )
    del original_weight_cpu

    return JointQErrorPropResult(
        dequantized_weight=dequantized_weight,
        quantization_time=current_result.quantization_time,
        bits=current_result.bits,
        symmetric=current_result.symmetric,
        group_size=current_result.group_size,
        scale=scale.cpu(),
        zero_point=zero_point.cpu(),
        assignment=assignment.cpu(),
        quantization_time_step2=quantization_time_step2,
        skip_step2=False,
    )


def should_skip_requantization(
    original_weight,
    current_result,
    flat_original_input,
    flat_quant_input,
    skip_threshold_increase=0.01,
    skip_threshold_error=0.01,
    skip_threshold_amplification=5.0,
    logger=None,
):
    """Determine whether re-quantization should be skipped.

    Skip if any of the following conditions hold:
    1. Relative cumulative error is small enough and amplification ratio is below threshold
       (g_relative < skip_threshold_error and (g-f)/f < skip_threshold_amplification)
    2. Error increase rate is small ((g - f) / f < skip_threshold_increase)

    Args:
        original_weight (torch.Tensor): Original model weight W_orig.
        current_result (JointQResult): Quantization result for the target layer.
        flat_original_input (torch.Tensor): Original model's input activation X_orig (reshaped).
            (total_samples, in_features)
        flat_quant_input (torch.Tensor): Quantized model's input activation X_quant (reshaped).
            (total_samples, in_features)
        skip_threshold_increase (float): Skip threshold for error increase rate (default: 0.01).
        skip_threshold_error (float): Skip threshold for relative cumulative error (default: 0.01).
        skip_threshold_amplification (float): Skip threshold for error amplification ratio (default: 5.0).
            Re-quantize if amplification exceeds this value even when g_relative is small.
        logger: Logger (uses the module logger if None).

    Returns:
        tuple: (skip, f_row_errors, g_row_errors)
            skip (bool): True if re-quantization should be skipped.
            f_row_errors (torch.Tensor): Per-row quantization squared errors (out_features,).
            g_row_errors (torch.Tensor): Per-row cumulative squared errors (out_features,).
    """
    if logger is None:
        logger = getLogger(__name__)

    # Compute the dequantized weight
    dequantized_weight = current_result.compute_dequantized_weight(
        device=original_weight.device
    ).double()

    # Compute quantization error f(W_quant*) and cumulative error g(W_quant*)
    f_error, g_error, f_relative, g_relative, f_row_errors, g_row_errors = (
        compute_quantization_errors(
            original_weight=original_weight,
            dequantized_weight=dequantized_weight,
            flat_original_input=flat_original_input,
            flat_quant_input=flat_quant_input,
        )
    )

    # Skip decision
    relative_error_increase = (g_error - f_error) / f_error
    logger.info(
        "f_error=%.4e (%.4f%%), g_error=%.4e (%.4f%%), (g-f)/f=%.4f",
        f_error,
        f_relative * 100,
        g_error,
        g_relative * 100,
        relative_error_increase,
    )

    if (
        g_relative < skip_threshold_error
        and relative_error_increase < skip_threshold_amplification
    ):
        logger.info(
            "Skipping re-quantization (g_relative %.4f%% < %.4f%% and increase %.4f < %.4f)",
            g_relative * 100,
            skip_threshold_error * 100,
            relative_error_increase,
            skip_threshold_amplification,
        )
        return True, f_row_errors, g_row_errors

    if relative_error_increase < skip_threshold_increase:
        logger.info(
            "Skipping re-quantization (error increase %.4f < %.4f)",
            relative_error_increase,
            skip_threshold_increase,
        )
        return True, f_row_errors, g_row_errors

    return False, f_row_errors, g_row_errors


def compute_quantization_errors(
    original_weight,
    dequantized_weight,
    flat_original_input,
    flat_quant_input,
    compute_f=True,
):
    """Compute quantization error and cumulative error (Step 1-2).

    Args:
        original_weight (torch.Tensor): Original model weight W_orig (out_features, in_features).
        dequantized_weight (torch.Tensor): Dequantized weight W_quant* (out_features, in_features).
        flat_original_input (torch.Tensor): Original model's input activation X_orig (reshaped).
            (total_samples, in_features)
        flat_quant_input (torch.Tensor): Quantized model's input activation X_quant (reshaped).
            (total_samples, in_features)
        compute_f (bool): Whether to compute f_error (quantization error) (default: True).
            If False, f_error, f_relative, and f_row_errors return None.

    Returns:
        tuple: (f_error, g_error, f_relative, g_relative, f_row_errors, g_row_errors)
            f_error: Quantization error f(W_quant*) = ||W_orig X_orig - W_quant* X_orig||^2_F (None if compute_f=False)
            g_error: Cumulative error g(W_quant*) = ||W_orig X_orig - W_quant* X_quant||^2_F
            f_relative: Relative quantization error f_error / ||W_orig X_orig||^2_F (None if compute_f=False)
            g_relative: Relative cumulative error g_error / ||W_orig X_orig||^2_F
            f_row_errors: Per-row quantization squared errors (out_features,) (None if compute_f=False)
            g_row_errors: Per-row cumulative squared errors (out_features,)
    """

    device = original_weight.device
    total_samples = flat_original_input.shape[0]
    out_features = original_weight.shape[0]

    # Use batch processing for memory efficiency
    batch_size = min(total_samples, 128)

    f_error = 0.0 if compute_f else None
    g_error = 0.0
    original_output_norm_sq = 0.0  # ||W_orig X_orig||^2_F

    # Per-row squared errors
    f_row_errors = (
        torch.zeros(out_features, device=device, dtype=torch.float64) if compute_f else None
    )
    g_row_errors = torch.zeros(out_features, device=device, dtype=torch.float64)

    for i in range(0, total_samples, batch_size):
        # batch_X_orig_T: (in_features, batch_size)
        batch_X_orig_T = flat_original_input[i : i + batch_size].T.to(device)
        # batch_X_quant_T: (in_features, batch_size)
        batch_X_quant_T = flat_quant_input[i : i + batch_size].T.to(device)

        # W_orig @ X_orig^T: (out_features, batch_size)
        original_output = original_weight @ batch_X_orig_T

        # ||W_orig X_orig||^2_F
        original_output_norm_sq += torch.sum(original_output.pow(2)).item()

        # f_error: ||W_orig X_orig - W_quant* X_orig||^2_F
        if compute_f:
            f_diff = original_output - dequantized_weight @ batch_X_orig_T
            f_diff_sq = f_diff.pow(2)
            f_error += torch.sum(f_diff_sq).item()
            f_row_errors += torch.sum(f_diff_sq, dim=1)  # Accumulate per-row squared errors
            del f_diff, f_diff_sq

        # g_error: ||W_orig X_orig - W_quant* X_quant||^2_F
        g_diff = original_output - dequantized_weight @ batch_X_quant_T
        g_diff_sq = g_diff.pow(2)
        g_error += torch.sum(g_diff_sq).item()
        g_row_errors += torch.sum(g_diff_sq, dim=1)  # Accumulate per-row squared errors

        del batch_X_orig_T, batch_X_quant_T, original_output, g_diff, g_diff_sq

    # Compute relative errors
    f_relative = f_error / original_output_norm_sq if compute_f else None
    g_relative = g_error / original_output_norm_sq

    return f_error, g_error, f_relative, g_relative, f_row_errors, g_row_errors


def compute_max_variation_rate(
    f_row_errors,
    g_row_errors,
    variation_scale=0.1,
    variation_cap=0.05,
    degradation_threshold=0.1,
    epsilon=1e-12,
    logger=None,
):
    """Compute the maximum variation rate.

    Computes how far from W_quant the re-quantization can deviate.
    Given the initial solution tilde_w_i for row i, the variation rate is defined as:
        variation_i := ||w_quant_i - tilde_w_i||^2_2 / ||tilde_w_i||^2_2
    quantize_advanced optimizes under the constraint variation_i <= max_variation_rate_i.

    Definition:
        degradation_i := (g_row_errors_i - f_row_errors_i) / (f_row_errors_i + epsilon)
        variation_i = clip(variation_scale * degradation_i, 0, variation_cap)
                      if degradation_i > degradation_threshold else 0

    Concrete examples with default parameters (variation is 1/10 of degradation):
        degradation <= 10%  -> variation = 0%   (below threshold -> no change)
        degradation = 11%  -> variation = 1.1% (0.1 * 0.11)
        degradation = 20%  -> variation = 2.0% (0.1 * 0.20)
        degradation = 30%  -> variation = 3.0% (0.1 * 0.30)
        degradation >= 50%  -> variation = 5.0% (capped by variation_cap)

    Args:
        f_row_errors (torch.Tensor): Per-row quantization squared errors (out_features,).
        g_row_errors (torch.Tensor): Per-row cumulative squared errors (out_features,).
        variation_scale (float): Scaling factor from degradation rate to variation rate (default: 0.1).
        variation_cap (float): Upper bound for maximum variation rate (default: 0.05).
        degradation_threshold (float): Degradation rate threshold; variation rate is 0 below this (default: 0.1).
        epsilon (float): Small value to avoid division by zero (default: 1e-12).
        logger: Logger (uses the module logger if None).

    Returns:
        torch.Tensor: Maximum variation rate variation_i (out_features,).
    """
    if logger is None:
        logger = getLogger(__name__)

    # Compute degradation rate of squared errors
    degradation = (g_row_errors - f_row_errors) / (f_row_errors + epsilon)

    # variation_i = clip(variation_scale * degradation_i, 0, variation_cap)
    #               if degradation_i > degradation_threshold else 0
    variation = torch.where(
        degradation > degradation_threshold,
        torch.clamp(variation_scale * degradation, min=0.0, max=variation_cap),
        torch.zeros_like(degradation),
    )

    # Log output
    num_active = (variation > 0).sum().item()
    logger.info(
        "Variation: active=%d/%d, deg=%.4f, var=%.6f/%.6f (mean/max)",
        num_active,
        variation.shape[0],
        degradation.mean().item(),
        variation.mean().item(),
        variation.max().item(),
    )

    return variation


def log_requantization_stats(
    original_weight,
    dequantized_weight,
    current_result,
    flat_original_input,
    flat_quant_input,
    device,
    logger=None,
):
    """Compute and log the cumulative error and actual variation rate after re-quantization.

    Args:
        original_weight (torch.Tensor): Original model weight (on device, float64).
        dequantized_weight (torch.Tensor): Re-quantized dequantized weight (CPU).
        current_result (JointQResult): Quantization result before re-quantization.
        flat_original_input (torch.Tensor): Original model's input activation X_orig (reshaped).
            (total_samples, in_features)
        flat_quant_input (torch.Tensor): Quantized model's input activation X_quant (reshaped).
            (total_samples, in_features)
        device: Device for computation.
        logger: Logger (uses the module logger if None).
    """
    if logger is None:
        logger = getLogger(__name__)

    new_dequantized_weight = dequantized_weight.to(device).double()

    _, g_error_new, _, g_relative_new, _, _ = compute_quantization_errors(
        original_weight=original_weight,
        dequantized_weight=new_dequantized_weight,
        flat_original_input=flat_original_input,
        flat_quant_input=flat_quant_input,
        compute_f=False,
    )

    # Compute actual variation rate: variation_i = ||w_new_i - w_old_i||^2_2 / ||w_old_i||^2_2
    old_dequantized_weight = current_result.compute_dequantized_weight(device=device).double()
    diff_sq = (new_dequantized_weight - old_dequantized_weight).pow(2).sum(dim=1)
    old_norm_sq = old_dequantized_weight.pow(2).sum(dim=1)
    actual_variation = diff_sq / (old_norm_sq + 1e-12)

    num_changed = (actual_variation > 0).sum().item()
    logger.info(
        "After step2: g_error=%.4e (%.4f%%)",
        g_error_new,
        g_relative_new * 100,
    )
    logger.info(
        "After step2: variation: changed=%d/%d, min=%.4f%%, mean=%.4f%%, max=%.4f%%",
        num_changed,
        actual_variation.shape[0],
        actual_variation.min().item() * 100,
        actual_variation.mean().item() * 100,
        actual_variation.max().item() * 100,
    )

    del original_weight, new_dequantized_weight, old_dequantized_weight
