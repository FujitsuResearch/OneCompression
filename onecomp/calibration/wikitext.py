"""
WikiText-2 dataset text loader for calibration.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Akihiro Yoshida
"""

from logging import getLogger

from ._cache import load_or_prepare
from .chunking import prepare_from_texts


def load_wikitext_texts(
    *,
    split="train",
    seed=0,
    shuffle_documents=False,
    logger=None,
):
    """Load WikiText-2 raw texts as a list of non-empty text blocks.

    Args:
        split: Dataset split (``"train"`` or ``"test"``).
        seed: Random seed for document shuffling.
        shuffle_documents: Shuffle documents before removing empty entries.
        logger: Logger (optional).

    Returns:
        list[str]: Non-empty text samples.
    """
    if logger is None:
        logger = getLogger(__name__)

    logger.info("Loading WikiText-2 (split=%s) ...", split)
    ds = load_or_prepare(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        local_name="wikitext2",
        logger=logger,
    )
    dataset = ds[split]
    if shuffle_documents:
        dataset = dataset.shuffle(seed=seed)

    texts = [t for t in dataset["text"] if t.strip()]
    logger.info("Loaded %d non-empty WikiText-2 texts", len(texts))
    return texts


def prepare_calibration_data(
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    *,
    shuffle_documents=False,
    logger=None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Load WikiText-2 texts and chunk them according to *strategy*.

    Returns:
        dict: ``{"input_ids": Tensor, "attention_mask": Tensor}``
    """
    if logger is None:
        logger = getLogger(__name__)

    logger.info("Using WikiText-2 dataset.")
    texts = load_wikitext_texts(
        split="train",
        seed=seed,
        shuffle_documents=shuffle_documents,
        logger=logger,
    )
    if strategy in ("drop_head", "drop_rand"):
        logger.warning(
            "WikiText-2 has many short documents; strategy %r would discard most. "
            "Falling back to concat_rand.",
            strategy,
        )
        strategy = "concat_rand"
    return prepare_from_texts(
        texts,
        tokenizer,
        device,
        max_length,
        num_calibration_samples,
        strategy,
        seed,
        logger,
    )
