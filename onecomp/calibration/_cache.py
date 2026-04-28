"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Akihiro Yoshida

"""

import os
from logging import getLogger

import datasets

CALIB_CACHE_DIR = os.environ.get(
    "ONECOMP_CALIB_CACHE",
    os.environ.get("CALIB_DATA_ROOT", None),
)


def load_or_prepare(
    hf_id, subset=None, *, local_name=None, data_files=None, logger=None, **kwargs
):
    """Load a dataset from local cache or download from HuggingFace Hub.

    Priority: local cache (``CALIB_CACHE_DIR/local_name``) >
    HuggingFace Hub download > save to local cache.

    Args:
        hf_id: HuggingFace dataset identifier (e.g. ``"Salesforce/wikitext"``).
        subset: Dataset subset / config name (e.g. ``"wikitext-2-raw-v1"``).
        local_name: Sub-directory name inside ``CALIB_CACHE_DIR``.
            If *None*, caching is skipped.
        data_files: ``data_files`` argument forwarded to ``load_dataset``.
        logger: Logger (optional).
        **kwargs: Extra keyword arguments forwarded to ``load_dataset``.

    Returns:
        ``datasets.DatasetDict`` or ``datasets.Dataset`` depending on
        whether split was passed via *kwargs*.
    """
    if logger is None:
        logger = getLogger(__name__)

    if CALIB_CACHE_DIR and local_name:
        local_path = os.path.join(CALIB_CACHE_DIR, local_name)
        if os.path.isdir(local_path):
            try:
                logger.info("Loading from local cache: %s", local_path)
                return datasets.load_from_disk(local_path)
            except Exception as exc:
                logger.warning(
                    "load_from_disk failed: %s -> downloading from HF Hub",
                    exc,
                )

    logger.info("Downloading %s from Hugging Face Hub ...", hf_id)
    if data_files is not None:
        ds = datasets.load_dataset(
            hf_id,
            subset,
            data_files=data_files,
            trust_remote_code=True,
            **kwargs,
        )
    else:
        ds = datasets.load_dataset(
            hf_id,
            subset,
            trust_remote_code=True,
            **kwargs,
        )

    if CALIB_CACHE_DIR and local_name:
        local_path = os.path.join(CALIB_CACHE_DIR, local_name)
        try:
            os.makedirs(local_path, exist_ok=True)
            logger.info("Saving to local cache: %s", local_path)
            ds.save_to_disk(local_path)
        except Exception as exc:
            logger.warning("save_to_disk failed: %s", exc)

    return ds
