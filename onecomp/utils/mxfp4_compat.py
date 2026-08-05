"""
MXFP4 checkpoint compatibility helpers.

Copyright 2025-2026 Fujitsu Ltd.

Some MXFP4 checkpoints (e.g. tokyotech-llm/GPT-OSS-Swallow-*-MXFP4,
saved with transformers 4.57) store the packed FP4 blocks as 3-D
tensors ``[num_experts, rows, groups * 16]`` while the original OpenAI
checkpoints (and transformers >= 5.x) use the 4-D layout
``[num_experts, rows, groups, 16]``.

transformers' ``_convert_moe_packed_tensors`` asserts
``blocks.shape[:-1] == scales.shape`` and therefore fails with an
AssertionError on the flattened 3-D layout when loading with
``Mxfp4Config(dequantize=True)``.

``patch_mxfp4_flat_blocks`` wraps ``convert_moe_packed_tensors`` so that
flattened blocks are reshaped to the expected 4-D layout before
dequantization.  The reshape is a pure view change; the dequantized
weights are bit-identical to those from an equivalent 4-D checkpoint.
"""

from logging import getLogger

_logger = getLogger(__name__)

_PATCH_FLAG = "_onecomp_flat_blocks_patch"


def _normalize_blocks(blocks, scales):
    """Reshape flattened 3-D MXFP4 blocks to the 4-D layout.

    Expected layouts:
        4-D (OpenAI):    blocks [..., rows, groups, 16], scales [..., rows, groups]
        3-D (flattened): blocks [..., rows, groups * 16], scales [..., rows, groups]
    """
    if blocks.ndim == scales.ndim:
        num_groups = scales.shape[-1]
        blocks = blocks.reshape(*blocks.shape[:-1], num_groups, -1)
    return blocks


def patch_mxfp4_flat_blocks(logger=None):
    """Make transformers' MXFP4 dequantization accept flattened 3-D blocks.

    Idempotent; safe to call multiple times.  No-op if the transformers
    version does not expose ``convert_moe_packed_tensors``.
    """
    logger = logger or _logger

    try:
        from transformers.integrations import mxfp4 as hf_mxfp4
    except ImportError:
        logger.warning("transformers.integrations.mxfp4 not found; skipping compat patch.")
        return False

    original = getattr(hf_mxfp4, "convert_moe_packed_tensors", None)
    if original is None:
        logger.warning(
            "convert_moe_packed_tensors not found in transformers; skipping compat patch."
        )
        return False

    if getattr(original, _PATCH_FLAG, False):
        return True

    def convert_moe_packed_tensors(blocks, scales, **kwargs):
        return original(_normalize_blocks(blocks, scales), scales, **kwargs)

    setattr(convert_moe_packed_tensors, _PATCH_FLAG, True)
    hf_mxfp4.convert_moe_packed_tensors = convert_moe_packed_tensors
    logger.info("Patched transformers MXFP4 dequantization to accept flattened 3-D blocks.")
    return True
