"""

Copyright 2025-2026 Fujitsu Ltd.

"""

_BFLOAT16_MODEL_KEYWORDS = ("gemma-3", "gemma3", "gemma_3", "gemma-4", "gemma4", "gemma_4")


def needs_bfloat16(model_id_or_path: str) -> bool:
    """Return True if the model requires bfloat16 (e.g. Gemma 3 / 4)."""
    _id = model_id_or_path.lower()
    return any(key in _id for key in _BFLOAT16_MODEL_KEYWORDS)
