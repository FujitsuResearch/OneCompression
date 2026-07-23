"""
Shared FloatQuant configuration constants.

Copyright 2025-2026 Fujitsu Ltd.
"""

SUPPORTED_FORMATS = ("nvfp4", "mxfp4", "fp8")
SUPPORTED_SCALE_TIMINGS = ("auto", "none", "static", "in_loop")
SUPPORTED_SCALE_OBJECTIVES = ("auto", "mse", "diag_wmse", "conditional")
SUPPORTED_SCALE_CANDIDATE_STRATEGIES = ("local", "full", "adaptive")

# Default block sizes per format (-1 means per-channel, used by fp8).
DEFAULT_BLOCK_SIZES = {"nvfp4": 16, "mxfp4": 32, "fp8": -1}

__all__ = [
    "DEFAULT_BLOCK_SIZES",
    "SUPPORTED_FORMATS",
    "SUPPORTED_SCALE_CANDIDATE_STRATEGIES",
    "SUPPORTED_SCALE_OBJECTIVES",
    "SUPPORTED_SCALE_TIMINGS",
]
