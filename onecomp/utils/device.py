"""
Device-related utilities for cross-platform support (CUDA / MPS / CPU).

Copyright 2025-2026 Fujitsu Ltd.

"""

import gc

import torch


def _has_mps_backend() -> bool:
    return hasattr(torch.backends, "mps")


def is_mps_device(device: torch.device | str | None) -> bool:
    """Return True when device refers to Apple MPS."""
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "mps"
    dev = str(device)
    return dev == "mps" or dev.startswith("mps:")


def get_default_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _has_mps_backend() and torch.backends.mps.is_available():
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
    elif device_type == "mps" and _has_mps_backend():
        empty_cache_fn = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if empty_cache_fn is not None:
            empty_cache_fn()


def cleanup_memory(device: torch.device | str | None = None) -> None:
    """Run a Python GC pass, then release the device memory cache.

    Convenience wrapper for the common ``gc.collect()`` + :func:`empty_cache`
    pair: the collection frees tensors still held by reference cycles so the
    subsequent cache release can actually return their blocks. A full GC pass
    is not free, so prefer bare :func:`empty_cache` inside hot loops (per
    layer / per batch) and use this at coarser boundaries.
    """
    gc.collect()
    empty_cache(device)
