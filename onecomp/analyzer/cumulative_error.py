"""Cumulative error analysis for quantized models.

Cumulative error: ||W_orig @ X_orig - W_quant @ X_quant||^2_F

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

"""

from logging import getLogger

import torch


def analyze_cumulative_error(model, inputs, quantization_results, layer_keywords=None):
    """Analyze cumulative quantization error.

    Args:
        model: PyTorch model object.
        inputs: Calibration inputs (dict with "input_ids", "attention_mask").
        quantization_results: Dictionary mapping layer names to QuantizationResult.
        layer_keywords: List of keywords to filter layers.
            Only layers containing any of the keywords are analyzed.
            Default: ["mlp.down_proj"]
            Example: ["mlp.down_proj"] or ["q_proj", "k_proj"]

    Returns:
        dict: Layer name -> cumulative squared error
    """
    logger = getLogger(__name__)

    if layer_keywords is None:
        layer_keywords = ["mlp.down_proj"]

    target_layer_names = [
        name
        for name in quantization_results.keys()
        if any(keyword in name for keyword in layer_keywords)
    ]
    logger.info(
        "Filtering layers by keywords %s: %d layers", layer_keywords, len(target_layer_names)
    )

    # Step 1: Capture original outputs (W_orig @ X_orig)
    logger.info("Capturing original outputs...")
    original_outputs = _capture_layer_outputs(model, inputs, target_layer_names)

    # Step 2: Update ALL layers in quantization_results (not just filtered ones)
    all_layer_names = list(quantization_results.keys())
    logger.info("Updating all %d layers to quantized weights...", len(all_layer_names))
    _update_weights(model, quantization_results, all_layer_names)

    # Step 3: Forward on quantized model and compute errors directly (no storage)
    logger.info("Computing cumulative errors...")
    results = _compute_cumulative_errors(
        model, inputs, target_layer_names, original_outputs, quantization_results
    )

    return results


def _update_weights(model, quantization_results, target_layer_names):
    """Update model weights to quantized weights."""
    name_to_module = {name: module for name, module in model.named_modules()}
    for name in target_layer_names:
        module = name_to_module.get(name)
        if module is None or not hasattr(module, "weight"):
            continue
        dtype = module.weight.data.dtype
        device = module.weight.data.device
        module.weight.data = quantization_results[name].dequantized_weight.to(device).to(dtype)


def _capture_layer_outputs(model, inputs, target_layer_names):
    """Capture layer outputs during forward pass."""
    name_to_module = {name: module for name, module in model.named_modules()}
    outputs = {}
    module_to_name = {
        name_to_module[name]: name for name in target_layer_names if name in name_to_module
    }

    def capture_output_hook(
        module, input, output
    ):  # pylint: disable=redefined-builtin,unused-argument
        name = module_to_name[module]
        if isinstance(output, tuple):
            outputs[name] = output[0].detach().cpu()
        else:
            outputs[name] = output.detach().cpu()

    handles = []
    for module in module_to_name.keys():
        handles.append(module.register_forward_hook(capture_output_hook))

    with torch.no_grad():
        model(**inputs)

    for handle in handles:
        handle.remove()

    return outputs


def _compute_cumulative_errors(
    model, inputs, target_layer_names, original_outputs, quantization_results
):
    """Compute cumulative errors during forward pass without storing outputs."""
    logger = getLogger(__name__)
    name_to_module = {name: module for name, module in model.named_modules()}
    results = {}
    module_to_name = {
        name_to_module[name]: name for name in target_layer_names if name in name_to_module
    }

    def compute_error_hook(
        module, input, output
    ):  # pylint: disable=redefined-builtin,unused-argument
        name = module_to_name[module]
        if isinstance(output, tuple):
            quant_out = output[0]
        else:
            quant_out = output

        orig_out = original_outputs[name].to(quant_out.device)
        diff = orig_out.float() - quant_out.float()
        squared_error = (diff**2).sum().item()
        numel = diff.numel()
        mean_squared_error = squared_error / numel

        # Get local error from quantization_results
        local_mean_squared_error = quantization_results[name].mean_output_squared_error

        results[name] = {
            "squared_error": squared_error,
            "mean_squared_error": mean_squared_error,
            "local_mean_squared_error": local_mean_squared_error,
        }
        logger.debug(
            "%s: cumulative=%.2e, local=%.2e, ratio=%.2f",
            name,
            mean_squared_error,
            local_mean_squared_error,
            mean_squared_error / local_mean_squared_error if local_mean_squared_error else 0,
        )

    handles = []
    for module in module_to_name.keys():
        handles.append(module.register_forward_hook(compute_error_hook))

    with torch.no_grad():
        model(**inputs)

    for handle in handles:
        handle.remove()

    return results


def plot_cumulative_error(results, output_path=None, layer_keywords=None):
    """Plot cumulative error vs local error as a grouped bar chart.

    Args:
        results: Dictionary mapping layer names to error dict.
            Each error dict contains "mean_squared_error" and "local_mean_squared_error".
        output_path: Path to save the plot. If None, displays the plot.
        layer_keywords: List of keywords used for filtering (for display in title).

    Returns:
        str: Path to the saved plot (if output_path is specified).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        ) from e

    # Extract layer indices and errors
    layer_indices = []
    cumulative_errors = []
    local_errors = []
    for name, error_dict in results.items():
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_indices.append(int(parts[i + 1]))
                    cumulative_errors.append(error_dict["mean_squared_error"])
                    local_errors.append(error_dict["local_mean_squared_error"])
                    break
                except ValueError:
                    pass

    # Sort by layer index
    if layer_indices:
        sorted_data = sorted(zip(layer_indices, cumulative_errors, local_errors))
        layer_indices, cumulative_errors, local_errors = zip(*sorted_data)
    else:
        layer_indices = list(range(len(results)))
        cumulative_errors = [v["mean_squared_error"] for v in results.values()]
        local_errors = [v["local_mean_squared_error"] for v in results.values()]

    # Create grouped bar chart
    x = np.arange(len(layer_indices))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, cumulative_errors, width, label="Cumulative")
    ax.bar(x + width / 2, local_errors, width, label="Local")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Squared Error")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_indices)
    ax.legend()
    ax.set_yscale("log")

    # Build title with keywords
    if layer_keywords:
        title = f"Cumulative vs Local Error ({', '.join(layer_keywords)})"
    else:
        title = "Cumulative vs Local Error"
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        return None
