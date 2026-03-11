"""
Activation capture validation utilities.

Intended use-cases:
- Validate forward-hook captured input activations before running expensive quantization.
- Detect pathological cases such as all-zero activations
  (often caused by device_map="auto" sharding + hooks).

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

from __future__ import annotations

from typing import Dict, List

import torch


def check_activations(
    activations: Dict[str, torch.Tensor],
    max_list_in_message: int = 5,
) -> None:
    """
    Validate captured activations to detect pathological capture failures.

    This function inspects activation tensors collected via forward hooks and
    raises an error if any tensor contains all-zero values. All-zero activations
    typically indicate a capture failure rather than legitimate model behavior,
    commonly caused by ``device_map="auto"`` distributing layers across multiple
    GPUs while hooks only capture data on a single device.

    Args:
        activations: Dictionary mapping layer names to their captured activation
            tensors. Non-tensor values are silently skipped.
        max_list_in_message: Maximum number of problematic layer names to include
            in the error message preview. Defaults to 5.

    Returns:
        None. The function returns silently if all activations are valid.

    Raises:
        RuntimeError: If one or more activation tensors are all-zero. The error
            message includes the count of affected layers, the first affected
            layer name, and suggested workarounds.

    Example:
        >>> activations = {"layer1": torch.randn(2, 3), "layer2": torch.zeros(2, 3)}
        >>> check_activations(activations)
        RuntimeError: Capture failed: some captured input activations are all-zero ...

    Note:
        This check should be performed before expensive downstream operations
        (e.g., quantization calibration) to fail fast on invalid data.
    """
    zero_layers: List[str] = []
    for name, x in activations.items():
        if not isinstance(x, torch.Tensor):
            continue
        if float(x.abs().max().item()) == 0.0:
            zero_layers.append(name)

    if len(zero_layers) == 0:
        return

    preview = ", ".join(zero_layers[:max_list_in_message])
    raise RuntimeError(
        "Capture failed: some captured input activations are all-zero "
        f"(count={len(zero_layers)}; first={zero_layers[0]}; preview={preview}). "
        "Workarounds: avoid device_map='auto' during capture by setting "
        "ModelConfig(device='cuda' or 'cuda:0'), or run capture on a single "
        "visible GPU (CUDA_VISIBLE_DEVICES=0 / Slurm --gres=gpu:1)."
    )
