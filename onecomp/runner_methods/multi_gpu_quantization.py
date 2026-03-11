"""
Multi-GPU Quantization Module (Multi-threaded version)

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Phase 1: Capture - Capture activations for all layers in a single thread
Phase 2: Quantize - Parallel quantization using multiple threads

"""

from logging import getLogger
import time
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

import torch

from onecomp.quantizer._quantizer import QuantizationResult
from onecomp.utils import check_activations


logger = getLogger(__name__)


# =============================================================================
# Serialization helpers
# =============================================================================


def get_model_config_dict(model_config) -> Dict[str, Any]:
    """Convert ModelConfig to dict (for future multi-process support)."""
    return {
        "model_id": model_config.model_id,
        "path": model_config.path,
        "dtype": model_config.dtype,
        "device": model_config.device,
    }


def get_quantizer_config_dict(quantizer) -> Dict[str, Any]:
    """Convert Quantizer settings to dict (for future multi-process support)."""
    config = asdict(quantizer)
    # Exclude internal state
    config.pop("module_to_name", None)
    config.pop("results", None)
    return config


def get_calibration_config_dict(
    calibration_dataset,
    max_length: int,
    num_calibration_samples: int,
    calibration_strategy: str,
    calibration_seed: int,
) -> Dict[str, Any]:
    """Convert calibration settings to dict."""
    return {
        "calibration_dataset": calibration_dataset,
        "max_length": max_length,
        "num_calibration_samples": num_calibration_samples,
        "calibration_strategy": calibration_strategy,
        "calibration_seed": calibration_seed,
    }


# =============================================================================
# Phase 1: Capture Phase
# =============================================================================


def run_capture_phase(
    model_config,
    quantizer,
    calibration_dataset,
    max_length: int,
    num_calibration_samples: int,
    calibration_strategy: str,
    calibration_seed: int,
) -> Dict[str, Any]:
    """Phase 1: Capture input activations and weights for all layers.

    Args:
        model_config: Model configuration.
        quantizer: Quantizer instance.
        calibration_dataset: Calibration dataset.
        max_length: Maximum sequence length.
        num_calibration_samples: Number of calibration samples.
        calibration_strategy: Calibration strategy.
        calibration_seed: Random seed.

    Returns:
        Dict containing:
            - "layer_data": Dict[layer_name, {"weight": Tensor, "input_activation": Tensor}]
            - "layer_names": List of layer names in order
    """
    from onecomp.utils import prepare_calibration_dataset

    logger.info("=== Phase 1: Capture ===")
    start_time = time.time()

    # Load model (follows model_config.device settings)
    model = model_config.load_model()
    tokenizer = model_config.load_tokenizer()

    # Get the device for placing input data
    input_device = next(model.parameters()).device

    # Prepare calibration data
    inputs = prepare_calibration_dataset(
        tokenizer=tokenizer,
        device=input_device,
        calibration_dataset=calibration_dataset,
        max_length=max_length,
        num_calibration_samples=num_calibration_samples,
        strategy=calibration_strategy,
        seed=calibration_seed,
        logger=logger,
    )

    # Set up quantizer and get target layers
    quantizer.setup(model)

    # Store data for each layer
    layer_data = {}
    layer_names = []

    def capture_hook(module, input, output):  # pylint: disable=redefined-builtin
        """Forward hook: Capture input activations and weights, save to CPU."""
        name = quantizer.module_to_name[module]
        logger.info("Capturing layer: %s", name)

        # Get input activation
        if isinstance(input, tuple):
            input_activation = input[0].detach().cpu()
        else:
            input_activation = input.detach().cpu()

        # Get weight
        weight = module.weight.data.detach().cpu()

        layer_data[name] = {
            "weight": weight,
            "input_activation": input_activation,
        }
        layer_names.append(name)

    # Register hooks
    handles = []
    for module in quantizer.module_to_name.keys():
        handle = module.register_forward_hook(capture_hook)
        handles.append(handle)

    # Run forward pass
    logger.info("Running forward pass to capture activations...")
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # =============================================================
    # Check phase: Abort if any captured activation is all-zeros
    # When ModelConfig.device is "auto", captured activations may be all-zeros.
    # The following function raises RuntimeError if any activation is all-zeros.
    # =============================================================
    try:
        check_activations(
            {name: data["input_activation"] for name, data in layer_data.items()},
        )
    except RuntimeError as e:
        logger.error("Capture failed: %s", e)
        raise

    # Release model
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info("Capture phase completed: %d layers in %.2f sec", len(layer_data), elapsed)

    return {
        "layer_data": layer_data,
        "layer_names": layer_names,
    }


# =============================================================================
# Phase 2: Quantization Phase
# =============================================================================


def run_quantization_phase(
    layer_data: Dict[str, Dict],
    layer_names: List[str],
    quantizer,
    gpu_ids: List[int],
) -> Dict[str, Dict]:
    """Phase 2: Parallel quantization using multiple threads.

    Args:
        layer_data: Layer data (containing weight and input_activation).
        layer_names: List of layer names (order preserved).
        quantizer: Quantizer instance.
        gpu_ids: List of GPU IDs to use.

    Returns:
        Dict of quantization results.
    """
    logger.info("=== Phase 2: Quantization (multi-threaded) ===")
    logger.info("Using GPUs: %s for %d layers", gpu_ids, len(layer_names))
    start_time = time.time()

    # Force PyTorch lazy initialization upfront (avoid multi-thread race conditions)
    # torch.linalg.solve's internal initialization can cause
    # "lazy wrapper should be called at most once" errors when called from multiple threads simultaneously
    # Note: Not required for quantizers other than JointQ, but the overhead is negligible (a few ms),
    #       so it is always executed without conditional branching
    for gpu_id in gpu_ids:
        device = torch.device(f"cuda:{gpu_id}")
        dummy_A = torch.randn(2, 2, device=device, dtype=torch.float64)
        dummy_b = torch.randn(2, 1, device=device, dtype=torch.float64)
        _ = torch.linalg.solve(dummy_A, dummy_b)
        del dummy_A, dummy_b
    torch.cuda.synchronize()
    logger.info("Initialized torch.linalg on all GPUs")

    results = {}
    lock = threading.Lock()

    def quantize_single_layer(layer_name: str, device: torch.device, gpu_id: int):
        """Quantize a single layer on the specified GPU."""

        weight = layer_data[layer_name]["weight"]
        activation = layer_data[layer_name]["input_activation"]

        # Create a dummy module
        # weight.device is used to specify the computation device:
        #   - GPTQ: Device for Hessian computation and quantization
        #   - JointQ: Fallback for the device parameter
        dummy_module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        dummy_module.weight.data = weight.to(device)

        # activation is passed as-is on CPU
        # - GPTQ: Moved to GPU as needed within calculate_hessian
        # - JointQ: Moved to CPU within quantize_layer

        # Hessian computation + quantization
        layer_start = time.time()
        hessian = None
        if quantizer.flag_hessian:
            hessian = quantizer.calculate_hessian(dummy_module, activation)
        quant_result = quantizer.quantize_layer(dummy_module, activation, hessian=hessian)
        layer_elapsed = time.time() - layer_start

        # Backward compatibility: convert Tensor to QuantizationResult if needed
        if isinstance(quant_result, torch.Tensor):
            quant_result = QuantizationResult(dequantized_weight=quant_result)

        # Set quantization time
        quant_result.quantization_time = layer_elapsed

        # Move dequantized_weight to CPU (if still on GPU)
        if quant_result.dequantized_weight.is_cuda:
            quant_result.dequantized_weight = quant_result.dequantized_weight.cpu()

        # Compute quantization error (if calc_quant_error=True)
        if quantizer.calc_quant_error:
            # Output quantization error
            (
                quant_result.output_squared_error,
                quant_result.mean_output_squared_error,
                quant_result.relative_output_squared_error,
            ) = quantizer.calculate_output_quantization_error(
                dummy_module, activation, quant_result.dequantized_weight
            )

            # Weight quantization error
            (
                quant_result.weight_squared_error,
                quant_result.mean_weight_squared_error,
                quant_result.relative_weight_squared_error,
            ) = quantizer.calculate_weight_quantization_error(
                dummy_module, quant_result.dequantized_weight
            )

        with lock:
            results[layer_name] = quant_result
            logger.info("  %s on GPU %d: %.2f sec", layer_name, gpu_id, layer_elapsed)

        # Free memory
        del dummy_module
        if hessian is not None:
            del hessian
        torch.cuda.empty_cache()

    # Shared task queue (dynamic work-stealing approach)
    # Process heavier layers first (LPT: Longest Processing Time first)
    # Layers with more elements take longer to process
    sorted_layer_names = sorted(
        layer_names,
        key=lambda name: layer_data[name]["weight"].numel(),
        reverse=True,  # Descending order (heaviest layers first)
    )

    task_queue: queue.Queue = queue.Queue()
    for layer_name in sorted_layer_names:
        task_queue.put(layer_name)

    def gpu_worker(gpu_id: int):
        """GPU worker: Fetch tasks from queue and process (idle GPUs pick up next task)."""
        # Set current device at the start of the thread
        # Required when JointQ etc. implicitly use the current device
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        logger.info("GPU %d worker started", gpu_id)
        task_count = 0
        while True:
            try:
                layer_name = task_queue.get_nowait()
                logger.info("GPU %d got task: %s", gpu_id, layer_name)
            except queue.Empty:
                break  # Exit when queue is empty
            try:
                quantize_single_layer(layer_name, device, gpu_id)
            except Exception as e:
                logger.error("GPU %d failed on task %s: %s", gpu_id, layer_name, e)
                import traceback

                logger.error("Traceback:\n%s", traceback.format_exc())
                raise  # Re-raise exception to stop the job
            task_queue.task_done()
            task_count += 1
        logger.info("GPU %d worker finished (%d tasks completed)", gpu_id, task_count)

    # One thread per GPU, process until queue is empty
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(gpu_worker, gpu_id) for gpu_id in gpu_ids]
        # Wait for all tasks to complete (exceptions will be raised here)
        for f in futures:
            f.result()

    elapsed = time.time() - start_time
    logger.info("Quantization phase completed in %.2f sec", elapsed)

    # Reorder results according to layer_names order
    ordered_results = {name: results[name] for name in layer_names}

    return ordered_results


# =============================================================================
# Main Entry Point
# =============================================================================


def run_multi_gpu_quantization(
    model_config,
    quantizer,
    calibration_dataset,
    max_length: int,
    num_calibration_samples: int,
    calibration_strategy: str,
    calibration_seed: int,
    gpu_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Main entry point for multi-GPU quantization.

    Args:
        model_config: Model configuration.
        quantizer: Quantizer instance.
        calibration_dataset: Calibration dataset.
        max_length: Maximum sequence length.
        num_calibration_samples: Number of calibration samples.
        calibration_strategy: Calibration strategy.
        calibration_seed: Random seed.
        gpu_ids: List of GPU IDs to use (all GPUs if None).

    Returns:
        Dict containing "results" with quantization results for each layer
    """
    total_start = time.time()

    # Set GPU ID list
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    logger.info("Multi-GPU quantization started with GPUs: %s", gpu_ids)

    # Phase 1: Capture
    capture_result = run_capture_phase(
        model_config=model_config,
        quantizer=quantizer,
        calibration_dataset=calibration_dataset,
        max_length=max_length,
        num_calibration_samples=num_calibration_samples,
        calibration_strategy=calibration_strategy,
        calibration_seed=calibration_seed,
    )

    # Phase 2: Parallel quantization
    results = run_quantization_phase(
        layer_data=capture_result["layer_data"],
        layer_names=capture_result["layer_names"],
        quantizer=quantizer,
        gpu_ids=gpu_ids,
    )

    total_elapsed = time.time() - total_start
    logger.info("Total time: %.2f sec", total_elapsed)

    return {"results": results}
