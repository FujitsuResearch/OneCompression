"""Copyright 2025-2026 Fujitsu Ltd."""

__all__ = ["AutoDBFForCausalLM"]


def __getattr__(name):
    if name == "AutoDBFForCausalLM":
        from .models.auto import AutoDBFForCausalLM

        return AutoDBFForCausalLM
    raise AttributeError(f"module {__name__} has no attribute {name}")
