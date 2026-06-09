"""Lightweight HuggingFace Hub helpers used by the API layer.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://huggingface.co/api/models"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_LOCAL_MODEL_ROOT_ENV = "LOCAL_MODEL_ROOT"
_DEFAULT_LOCAL_MODEL_ROOT = "/models"


def _local_model_root() -> Path:
    return Path(
        os.environ.get(_LOCAL_MODEL_ROOT_ENV, _DEFAULT_LOCAL_MODEL_ROOT),
    ).resolve()


def _resolve_local_model_path(identifier: str) -> Path | None:
    """Return a model directory under LOCAL_MODEL_ROOT, or None if not found."""
    root = _local_model_root()
    candidate = Path(identifier.strip())
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / identifier).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved if resolved.is_dir() else None


def resolve_model_identifier(model_id: str) -> str:
    """Normalize *model_id* and map local names to an absolute filesystem path.

    Short identifiers such as ``gemma-2-2b-it`` that exist under
    ``LOCAL_MODEL_ROOT`` are returned as absolute paths so downstream loaders
    (``ModelConfig``, ``transformers``) use the local snapshot instead of
    HuggingFace Hub.
    """
    normalized = model_id.strip()
    if not normalized:
        raise ValueError("Model name must not be empty.")

    local_dir = _resolve_local_model_path(normalized)
    if local_dir is not None:
        return str(local_dir)
    return normalized


def check_model_exists(model_id: str) -> None:
    """Raise ``ValueError`` if ``model_id`` is not resolvable on HuggingFace Hub.

    A model directory under ``LOCAL_MODEL_ROOT`` (default ``/models``) is
    treated as a valid local identifier. This matches the deployment layout
    where already-downloaded models live on shared storage.
    """
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("Model name must not be empty.")

    if _resolve_local_model_path(model_id) is not None:
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    url = f"{_HF_API_BASE}/{model_id}"
    try:
        resp = httpx.get(url, headers=headers, timeout=_TIMEOUT, follow_redirects=True)
    except httpx.HTTPError as exc:
        # Network failure / DNS / proxy issue — surface as a validation error
        # so the user knows we cannot confirm the model.
        logger.warning("HF model lookup failed for %s: %s", model_id, exc)
        raise ValueError(
            f"Could not reach HuggingFace Hub to verify '{model_id}': {exc}. "
            "Check network connectivity (HF_ENDPOINT / proxy) and try again."
        ) from exc

    if resp.status_code == 200:
        return
    if resp.status_code == 404:
        raise ValueError(
            f"Model '{model_id}' was not found on HuggingFace Hub "
            f"(https://huggingface.co/{model_id})."
        )
    if resp.status_code in (401, 403):
        raise ValueError(
            f"Model '{model_id}' exists but requires authentication. "
            "Set HF_TOKEN with read access and retry."
        )

    raise ValueError(
        f"Unexpected response {resp.status_code} from HuggingFace Hub for '{model_id}'."
    )
