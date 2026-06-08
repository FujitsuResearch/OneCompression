"""Lightweight HuggingFace Hub helpers used by the API layer.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_HF_API_BASE = "https://huggingface.co/api/models"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def check_model_exists(model_id: str) -> None:
    """Raise ``ValueError`` if ``model_id`` is not resolvable on HuggingFace Hub.

    A local path that exists on disk is treated as a valid model identifier,
    matching ``transformers.AutoConfig.from_pretrained`` behaviour. This is
    important when callers point at an already-downloaded model directory.
    """
    if not model_id or not model_id.strip():
        raise ValueError("Model name must not be empty.")

    model_id = model_id.strip()

    # Local path takes precedence (matches transformers' resolution order).
    if os.path.isdir(model_id):
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
