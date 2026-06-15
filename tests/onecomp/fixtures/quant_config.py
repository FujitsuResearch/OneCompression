"""Shared quantization_config test helpers.

A single source of truth for the minimal schema-valid ``quantization_config``
used across post_process and utils tests, plus the thin ``model.config``
stand-in that carries it. Keeping one builder here means a schema change is
reflected in every test from one place.

Copyright 2025-2026 Fujitsu Ltd.
"""


def valid_quant_config(**extra) -> dict:
    """Return a minimal schema-valid ``quantization_config``.

    Args:
        **extra: Optional keys merged into (and overriding) the base config,
            e.g. ``valid_quant_config(rotated=True)``.
    """
    config = {
        "quant_method": "gptq",
        "modules_in_block_to_quantize": [["self_attn.q_proj"]],
    }
    config.update(extra)
    return config


class FakeConfig:
    """Stand-in for ``model.config`` carrying a ``quantization_config``."""

    def __init__(self, quantization_config):
        self.quantization_config = quantization_config
