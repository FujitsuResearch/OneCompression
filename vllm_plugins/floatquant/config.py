"""Pure config-resolution logic for the FloatQuant fake-quant vLLM plugin.

Kept free of vLLM imports so the logic is unit-testable on CPU-only
environments (see ``tests/vllm_plugins/floatquant``).

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from __future__ import annotations

#: Canonical method written by ``FloatQuant.get_quant_config`` since the
#: dedicated-name migration; the format lives in ``fmt``.
CANONICAL_FAKE_QUANT_METHOD = "onecomp_fake_quant"

#: Legacy checkpoints (written before the dedicated name) used the format
#: name itself as ``quant_method``.
LEGACY_FAKE_QUANT_METHODS = ("nvfp4", "mxfp4", "fp8")

SUPPORTED_FAKE_QUANT_METHODS = (CANONICAL_FAKE_QUANT_METHOD,) + LEGACY_FAKE_QUANT_METHODS

#: Microscaling formats a fake-quant checkpoint can declare in ``fmt``.
SUPPORTED_FAKE_QUANT_FORMATS = ("nvfp4", "mxfp4", "fp8")


def resolve_fake_quant_config(config: dict) -> dict:
    """Validate and normalize an OneComp FloatQuant fake-quant quantization_config.

    Fake-quant checkpoints store plain dequantized FP16 Linear weights; the
    ``quantization_config`` only records which microscaling format produced
    them.  This helper is the single source of truth for deciding whether a
    checkpoint belongs to the fake-quant path.

    Args:
        config: The ``quantization_config`` dict from ``config.json``.

    Returns:
        dict: Normalized fields ``quant_method``, ``fmt``, ``block_size``
        and ``use_hessian``.

    Raises:
        ValueError: If ``quant_method`` is unsupported or the checkpoint is
            not in fake-quant format (e.g. a native FP8 checkpoint, which
            must be loaded by vLLM's built-in handler without this plugin).
    """
    method = config.get("quant_method")
    if method not in SUPPORTED_FAKE_QUANT_METHODS:
        raise ValueError(
            f"Unsupported quant_method {method!r} for the FloatQuant fake-quant plugin "
            f"(expected one of {SUPPORTED_FAKE_QUANT_METHODS})."
        )

    checkpoint_format = config.get("checkpoint_format")
    if checkpoint_format != "fake_quant":
        raise ValueError(
            f"quant_method {method!r} with checkpoint_format={checkpoint_format!r} is not "
            "an OneComp fake-quant checkpoint. This plugin only handles checkpoints saved "
            "by onecomp (checkpoint_format='fake_quant', FP16 weights). For native "
            f"{method!r} checkpoints, uninstall/disable this plugin so vLLM's built-in "
            "handler is used."
        )

    fmt = config.get("fmt", method)
    if fmt not in SUPPORTED_FAKE_QUANT_FORMATS:
        raise ValueError(
            f"Unsupported fake-quant fmt {fmt!r} "
            f"(expected one of {SUPPORTED_FAKE_QUANT_FORMATS})."
        )
    return {
        "quant_method": CANONICAL_FAKE_QUANT_METHOD,
        "fmt": fmt,
        "block_size": config.get("block_size", config.get("group_size")),
        "use_hessian": bool(config.get("use_hessian", False)),
    }
