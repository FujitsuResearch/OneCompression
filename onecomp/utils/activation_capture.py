"""
Activation capture utilities for quantization

Copyright 2026 Fujitsu Ltd.

Author: Keiji Kimura(kimura-keiji@fujitsu.com)

"""

from logging import getLogger

import torch

from .activation_check import check_activations


def capture_input_activations(
    model,
    inputs,
    module_to_name,
    exclude_layer_keywords=None,
    logger=None,
):
    """Capture input activations for specified layers during a forward pass.

    This function registers forward hooks on target layers and captures their
    input activations. Captured activations are moved to CPU to save GPU memory.
    This is typically used for quantization calibration or error analysis.

    Args:
        model (torch.nn.Module): PyTorch model to capture activations from.
        inputs (dict[str, torch.Tensor]): Model inputs as a dictionary.
            For transformer models, typically includes:
            - "input_ids": Token IDs (shape: [batch_size, seq_len])
            - "attention_mask": Attention mask (shape: [batch_size, seq_len])
        module_to_name (dict[torch.nn.Module, str]): Mapping from module objects
            to their layer names. This can be obtained from:
            - `quantizer.module_to_name` (if using OneComp quantizer)
            - Or built manually using `model.named_modules()`
        exclude_layer_keywords (list[str] | None): Keywords for layers to skip.
            Layers whose names contain any of these keywords will not be captured.
            Default is None (capture all layers in module_to_name).
        logger (logging.Logger | None): Logger for progress messages.
            If None, uses the module's default logger.

    Returns:
        dict[str, torch.Tensor]: Mapping of layer names to their input activations.
            Each tensor is on CPU with shape typically [batch_size, seq_len, hidden_dim].

    Example:
        >>> # Using with OneComp quantizer
        >>> activations = capture_input_activations(
        ...     model=model,
        ...     inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        ...     module_to_name=quantizer.module_to_name,
        ...     exclude_layer_keywords=["mlp.down_proj"],
        ... )
        >>> # Access specific layer's activation
        >>> layer0_input = activations["model.layers.0.self_attn.q_proj"]

    Note:
        - Activations are stored on CPU to avoid GPU memory overflow.
        - The function validates captured activations using `check_activations()`.
    """
    if logger is None:
        logger = getLogger(__name__)

    if exclude_layer_keywords is None:
        exclude_layer_keywords = []

    input_activations = {}

    def capture_input_hook(module, input, _):  # pylint: disable=redefined-builtin
        """Forward hook: capture input activations and store them on CPU."""
        name = module_to_name[module]
        logger.info("Capturing input activations for %s...", name)
        if isinstance(input, tuple):
            input_activations[name] = input[0].detach().cpu()
        else:
            input_activations[name] = input.detach().cpu()

    # Register hooks on target layers
    handles = []
    for module, name in module_to_name.items():
        # Skip excluded layers
        if any(keyword in name for keyword in exclude_layer_keywords):
            continue
        handle = module.register_forward_hook(capture_input_hook)
        handles.append(handle)

    # Run a forward pass to capture input activations
    logger.info("Capturing input activations...")
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Validate captured activations
    check_activations(input_activations)

    return input_activations
