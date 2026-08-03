"""Hugging Face Hub upload helpers for quantized models.

Thin wrapper around ``huggingface_hub.HfApi`` that creates the target
repository, writes the model card, and uploads a save directory
(safetensors and/or GGUF artifacts) in one call.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import os
from logging import getLogger
from typing import Optional

logger = getLogger(__name__)


def push_to_hub(
    save_dir: str,
    repo_id: str,
    model_card: Optional[str] = None,
    private: bool = True,
    token: Optional[str] = None,
) -> str:
    """Upload a model save directory to the Hugging Face Hub.

    Args:
        save_dir (str): Local directory containing the model artifacts
            (config, tokenizer, safetensors and/or GGUF files).
        repo_id (str): Target repository (``"user/repo-name"``).
        model_card (str or None): Model card Markdown (see
            ``generate_model_card``).  Written to ``README.md`` inside
            ``save_dir`` before upload unless one already exists.
        private (bool): Create the repository as private (default: True).
        token (str or None): Hugging Face access token.  Falls back to
            the cached login / ``HF_TOKEN`` when ``None``.

    Returns:
        str: URL of the uploaded repository.

    Raises:
        ValueError: If ``save_dir`` is not a directory, or no Hugging
            Face token can be resolved.

    Examples:
        >>> from onecomp.export import generate_model_card, push_to_hub
        >>> card = generate_model_card("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> push_to_hub("./quantized_model", "user/tinyllama-onecomp", card)
        'https://huggingface.co/user/tinyllama-onecomp'
    """
    # Lazy import so that offline use of onecomp.export never requires
    # huggingface_hub to be importable.
    from huggingface_hub import HfApi  # pylint: disable=import-outside-toplevel

    if not os.path.isdir(save_dir):
        raise ValueError(f"save_dir is not a directory: {save_dir}")

    if token is None:
        try:
            from huggingface_hub import get_token  # pylint: disable=import-outside-toplevel
        except ImportError:
            get_token = None
        if get_token is not None and get_token() is None:
            raise ValueError(
                "no Hugging Face token found; pass token=..., run "
                "`huggingface-cli login`, or set the HF_TOKEN "
                "environment variable"
            )

    readme_path = os.path.join(save_dir, "README.md")
    if model_card is not None and not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        logger.info("Wrote model card to %s", readme_path)

    api = HfApi(token=token)
    repo_url = api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
        commit_message=f"Upload quantized model from {os.path.basename(save_dir)}",
    )
    logger.info("Pushed %s to %s", save_dir, repo_id)
    return str(repo_url)
