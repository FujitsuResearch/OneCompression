"""

Copyright 2025-2026 Fujitsu Ltd.


"""

import torch


def add_model_specific_inputs(inputs, model):
    """Add model-specific fields to a model-input dictionary.

    Gemma 4 requires mm_token_type_ids (multimodal token type ids) even
    for text-only inference.
    cf) https://huggingface.co/google/gemma-4-31B/discussions/3

    Args:
        inputs: Model input dictionary (must contain ``input_ids``).
        model: Model instance.

    Returns:
        dict: The same inputs dict, possibly augmented with extra keys.
    """
    config = getattr(model, "config", None)
    if getattr(config, "model_type", "") == "gemma4":
        inputs["mm_token_type_ids"] = torch.zeros_like(
            inputs["input_ids"], dtype=torch.long
        )
    return inputs
