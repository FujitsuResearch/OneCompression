"""
Device-related utilities for cross-platform support (CUDA / MPS / CPU).

Copyright 2025-2026 Fujitsu Ltd.

"""

import torch


def get_default_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache(device: torch.device | str | None = None) -> None:
    """Release device memory cache for the given device type.

    Safe to call on any platform — silently does nothing when the
    device backend is not available.
    """
    device_type = torch.device(device if device is not None else get_default_device()).type

    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
