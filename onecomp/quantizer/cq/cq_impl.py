"""CQ (Clustering Quantization) quantization module

This module provides CQ quantization functionality for neural network weights.
CQ is a 2-value clustering quantization method that splits weights into two clusters
for quantization.

Functions:
    run_cq(layer, each_row): Execute CQ quantization on a layer.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from logging import getLogger
from typing import Union

import torch
import torch.nn as nn
import transformers
import gc

logger = getLogger(__name__)


def quantize(
    x: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Quantize floating point values to integers (CQ method: threshold-based binarization).

    Args:
        x: Input tensor (floating point).
        threshold: Clustering threshold.

    Returns:
        Quantized integer tensor (INT8, {0, 1}).
    """
    return (x > threshold).to(torch.int8)


def dequantize(
    x: torch.Tensor,
    threshold: Union[torch.Tensor, float],
    left_mean: Union[torch.Tensor, float],
    right_mean: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Dequantize quantized integers back to floating point values (CQ method).

    Args:
        x: Original input tensor (used for threshold comparison).
        threshold: Clustering threshold.
        left_mean: Left cluster mean value (corresponds to W <= threshold).
        right_mean: Right cluster mean value (corresponds to W > threshold).

    Returns:
        Dequantized floating point tensor.
    """
    return torch.where(x <= threshold, left_mean, right_mean)


def run_cq(
    layer: torch.nn.Module,
    each_row: bool = True,
) -> dict[str, torch.Tensor]:
    """Execute quantization using CQ.

    Performs 2-value clustering quantization using CQ (Clustering Quantization).
    Sorts weights, finds the split point that minimizes SSE (Sum of Squared Errors),
    and splits into two clusters, quantizing with each cluster's mean value.

    For Conv2d layers, weights are flattened to 2D for processing.
    For transformers.Conv1D layers, weights are transposed for processing.
    All tensors are moved to CPU after processing.
    Weight reconstruction: W = torch.where(quantized_weight == 0, left_mean, right_mean)

    Args:
        layer (torch.nn.Module): Layer module to quantize (Linear, Conv2d, Conv1D, etc.).
        each_row (bool, optional): Whether to quantize row-wise. If True, computes independent
            threshold and cluster means for each row. If False, processes entire weight
            as a single vector to compute global threshold and cluster means. Default is True.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing quantization results with the following keys:
            - "dequantized_weight": Dequantized weights (original shape, original dtype, CPU).
            - "quantized_weight": Quantized weights (indices {0, 1}, INT8, CPU).
            - "threshold": Clustering threshold (scalar or row-wise vector, CPU).
            - "left_mean": Left cluster mean value (scalar or row-wise vector, CPU).
            - "right_mean": Right cluster mean value (scalar or row-wise vector, CPU).
    """

    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    if not each_row:
        # Flatten W into a vector
        vector = torch.flatten(W)
        logger.debug(
            f"vector: shape = {vector.shape}, dtype = {vector.dtype}, device = {vector.device}"
        )

        # Run clustering to get threshold and cluster means
        threshold, left_mean, right_mean = run_clustering(vector, verbose=True)

        # Quantize weights (binarize using threshold)
        # Compute integer indices (0: left_mean, 1: right_mean)
        Q_int = quantize(W, threshold)
        # Dequantize
        Q = dequantize(W, threshold, left_mean, right_mean)

        # Convert scalar values to tensors
        threshold_tensor = torch.tensor([threshold], dtype=torch.float32)
        left_mean_tensor = torch.tensor([left_mean], dtype=torch.float32)
        right_mean_tensor = torch.tensor([right_mean], dtype=torch.float32)

    else:
        # Process each row
        Q = W.clone()
        Q_int = torch.zeros_like(W, dtype=torch.int8)
        thresholds = torch.zeros(W.shape[0], dtype=torch.float32, device=W.device)
        left_means = torch.zeros(W.shape[0], dtype=torch.float32, device=W.device)
        right_means = torch.zeros(W.shape[0], dtype=torch.float32, device=W.device)

        for i in range(W.shape[0]):
            # Get row vector
            vector = W[i, :]

            # Run clustering to get threshold and cluster means
            threshold, left_mean, right_mean = run_clustering(vector)

            # Quantize weights (binarize using threshold)
            Q_int[i, :] = quantize(W[i, :], threshold)
            # Dequantize
            Q[i, :] = dequantize(W[i, :], threshold, left_mean, right_mean)

            # Save row-wise parameters
            thresholds[i] = threshold
            left_means[i] = left_mean
            right_means[i] = right_mean

        threshold_tensor = thresholds
        left_mean_tensor = left_means
        right_mean_tensor = right_means

    if isinstance(layer, transformers.Conv1D):
        Q = Q.t()
        Q_int = Q_int.t()

    # Store quantized weights on CPU
    dequantized_weight = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype).cpu()
    quantized_weight = Q_int.reshape(layer.weight.shape).cpu()

    del W, Q, Q_int
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "dequantized_weight": dequantized_weight,
        "quantized_weight": quantized_weight,
        "threshold": threshold_tensor.cpu(),
        "left_mean": left_mean_tensor.cpu(),
        "right_mean": right_mean_tensor.cpu(),
    }


def run_clustering(
    vector: torch.Tensor,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Run clustering on a vector and return the threshold and cluster means."""

    # Step 1: Sort the vector
    vector, _ = torch.sort(vector)
    device = vector.device

    # Step 2: Compute cumulative sum and cumulative sum of squares
    cumsum = torch.cumsum(vector, dim=0)
    cumsum2 = torch.cumsum(vector**2, dim=0)
    indices = torch.arange(1, vector.shape[0], device=device)

    # SSE of the left cluster (0 to i-1)
    sse_left = cumsum2[:-1] - (cumsum[:-1] ** 2) / indices
    # SSE of the right cluster (i to n-1)
    sse_right = (cumsum2[-1] - cumsum2[:-1]) - ((cumsum[-1] - cumsum[:-1]) ** 2) / (
        vector.shape[0] - indices
    )

    # Compute scores
    scores = sse_left + sse_right
    if verbose:
        logger.debug(
            f"score: shape = {scores.shape}, dtype = {scores.dtype}, device = {scores.device}"
        )

    # Step 3: Get the index and threshold that minimize the score
    best_index = torch.argmin(scores)
    threshold = vector[best_index].item()
    left_mean = (cumsum[best_index] / (best_index + 1)).item()
    right_mean = ((cumsum[-1] - cumsum[best_index]) / (vector.shape[0] - (best_index + 1))).item()

    if verbose:
        logger.debug(f"best_index = {best_index}")
        logger.debug(f"threshold = {threshold}")
        logger.debug(f"left mean = {left_mean}")
        logger.debug(f"right mean = {right_mean}")

    return threshold, left_mean, right_mean
