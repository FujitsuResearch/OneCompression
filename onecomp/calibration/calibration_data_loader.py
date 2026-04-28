"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Akihiro Yoshida

"""

import os
from logging import getLogger

import torch

from ..utils import add_model_specific_inputs
from .calibration_config import CalibrationConfig
from .chunking import _VALID_CALIBRATION_STRATEGIES, prepare_from_texts
from ._cache import load_or_prepare
from . import c4
from . import wikitext
from . import custom

_KNOWN_DATASET_NAMES = frozenset(
    {
        "c4",
        "wikitext2",
    }
)


def prepare_calibration_dataset(
    tokenizer,
    device,
    calibration_config: CalibrationConfig,
    model,
    logger=None,
):
    """Prepare calibration data for quantization methods such as GPTQ.

    Processing flow:
        1. Obtain text data from the specified source.
        2. Chunk the texts according to the chosen *strategy*.

    strategy:
        - "concat_chunk": concatenate all texts -> tokenize at once -> equal-length chunks.
            Creates as many chunks as possible.
        - "concat_chunk_align": concatenate all texts -> tokenize at once -> equal-length chunks.
            Fixes the number of chunks to num_calibration_samples.
        - "concat_rand": concatenate all texts -> tokenize at once -> randomly sample
            starting positions to create num_calibration_samples chunks.
        - "drop_head": no cross-document, extract the first max_length tokens.
            Documents with token length < max_length are discarded.
        - "drop_rand": no cross-document, extract a random window of max_length tokens.
            Documents with token length < max_length are discarded.

    About concat_chunk / concat_chunk_align:
        - No padding needed: all chunks have the same length (max_length).
        - Data efficiency: even short texts are fully utilized.
        - Compute efficiency: batch processing is possible.
        - Caveat: document boundaries become ambiguous (different documents may
          coexist in a single chunk).
        - For GPTQ calibration the goal is to collect input-activation statistics
          (Hessian matrix), so semantic coherence across sentences is not important;
          therefore this method works well in practice.

    Args:
        tokenizer: Tokenizer.
        device (torch.device): Device to place tensors on (CPU or GPU).
        calibration_config (CalibrationConfig): Calibration parameters.
        model (torch.nn.Module): Model instance. It is used to add model-specific token-type fields to calibration inputs.
        logger: Logger (optional).

    Returns:
        dict: Model input dictionary.
            - "input_ids": tensor of shape (num_chunks, max_length).
            - "attention_mask": tensor of shape (num_chunks, max_length).
    """
    if logger is None:
        logger = getLogger(__name__)

    calibration_dataset = calibration_config.calibration_dataset
    max_length = calibration_config.max_length
    num_calibration_samples = calibration_config.num_calibration_samples
    strategy = calibration_config.strategy
    seed = calibration_config.seed
    text_key = calibration_config.text_key
    use_quality_filter = calibration_config.use_quality_filter
    max_documents = calibration_config.max_documents

    if strategy not in _VALID_CALIBRATION_STRATEGIES:
        raise ValueError(
            f"Unknown calibration strategy: {strategy!r}. "
            f"Available: {list(_VALID_CALIBRATION_STRATEGIES)}"
        )

    logger.info("Preparing calibration dataset (strategy=%s) ...", strategy)

    name_lower = calibration_dataset.lower() if calibration_dataset else None

    if calibration_dataset is None or name_lower == "c4":
        if calibration_dataset is None:
            logger.info("Calibration dataset is not specified, using default C4 dataset")
        result = c4.prepare_calibration_data(
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            strategy,
            seed,
            use_quality_filter=use_quality_filter,
            logger=logger,
        )
    elif name_lower == "wikitext2":
        result = wikitext.prepare_calibration_data(
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            strategy,
            seed,
            logger=logger,
        )
    elif os.path.exists(calibration_dataset):
        result = custom.prepare_calibration_data(
            calibration_dataset,
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            strategy,
            seed,
            text_key=text_key,
            max_documents=max_documents,
            logger=logger,
        )
    else:
        # --- try as HuggingFace Hub dataset ID ---
        result = _load_from_hub(
            calibration_dataset,
            tokenizer,
            device,
            calibration_config,
            logger=logger,
        )

    return add_model_specific_inputs(result, model)


_TEXT_COLUMN_CANDIDATES = ("text", "content", "sentence", "document", "body", "input")


def _find_text_column(column_names, text_key="text"):
    """Return the first matching text column name, or *None*."""
    if text_key in column_names:
        return text_key
    for col in _TEXT_COLUMN_CANDIDATES:
        if col in column_names:
            return col
    return None


def _load_from_hub(
    hf_id,
    tokenizer,
    device,
    calibration_config: CalibrationConfig,
    logger=None,
):
    """Download a HuggingFace Hub dataset and prepare calibration data.

    Called as a last-resort fallback when *hf_id* is neither a known
    dataset name nor a local path.
    """
    if logger is None:
        logger = getLogger(__name__)

    text_key = calibration_config.text_key
    max_documents = calibration_config.max_documents

    local_name = hf_id.replace("/", "__")

    logger.info(
        "Trying as HuggingFace Hub dataset: %s",
        hf_id,
    )
    try:
        ds = load_or_prepare(hf_id, local_name=local_name, logger=logger)
    except Exception as exc:
        raise ValueError(
            f"Unknown dataset name or path: {hf_id!r}. "
            f"Known names: {sorted(_KNOWN_DATASET_NAMES)}. "
            f"Tried as HF Hub ID but failed: {exc}"
        ) from exc

    import datasets as _ds  # pylint: disable=import-outside-toplevel

    if isinstance(ds, _ds.DatasetDict):
        data = ds.get("train", ds[list(ds.keys())[0]])
    else:
        data = ds

    col = _find_text_column(data.column_names, text_key)
    if col is None:
        raise ValueError(
            f"No text column found in dataset {hf_id!r}. "
            f"Available columns: {data.column_names}"
        )
    if col != text_key:
        logger.info("Using column '%s' (requested '%s')", col, text_key)

    texts = [t for t in data[col] if t and isinstance(t, str) and t.strip()]
    if max_documents and len(texts) > max_documents:
        texts = texts[:max_documents]

    if not texts:
        raise ValueError(f"No text data found in dataset {hf_id!r}")

    logger.info("Loaded %d texts from HF Hub dataset %s", len(texts), hf_id)
    return prepare_from_texts(
        texts,
        tokenizer,
        device,
        calibration_config.max_length,
        calibration_config.num_calibration_samples,
        calibration_config.strategy,
        calibration_config.seed,
        logger,
    )
