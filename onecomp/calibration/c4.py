"""
C4 dataset text loader for calibration.

Provides functions to load text samples from the AllenAI C4 dataset,
with optional quality filtering.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Akihiro Yoshida
"""

from logging import getLogger

from ._cache import load_or_prepare
from .chunking import prepare_from_texts

_C4_DATA_FILES = {"train": "en/c4-train.00001-of-01024.json.gz"}


def _get_c4_train(logger=None):
    """Load the C4 train split via the local-cache-first loader."""
    ds = load_or_prepare(
        "allenai/c4",
        data_files=_C4_DATA_FILES,
        local_name="c4",
        logger=logger,
    )
    return ds["train"]


def check_text_quality(text, min_chars=100):
    """Check text quality for calibration data.

    Criteria:
        1. Sufficient length (>= *min_chars*)
        2. High ASCII ratio (>= 80 %)
        3. Sufficient words (>= 20)
        4. Normal average word length (2–15)
        5. Has sentence-ending punctuation (>= 2)
        6. No repetitive patterns (5 consecutive identical words)
        7. No HTML / JavaScript fragments
        8. Limited special characters (<= 5 %)
        9. Limited URLs (<= 5)
        10. Limited digit ratio (<= 30 %)

    Args:
        text: Text to check.
        min_chars: Minimum character count.

    Returns:
        tuple[bool, str]: ``(is_valid, rejection_reason)``.
    """
    if not text or not isinstance(text, str):
        return False, "empty_or_invalid"

    text = text.strip()

    # 1. Character count check
    if len(text) < min_chars:
        return False, "too_short"

    # 2. ASCII ratio check (English text filter)
    ascii_count = sum(1 for c in text if c.isascii())
    if ascii_count / len(text) < 0.8:
        return False, "low_ascii_ratio"

    # 3. Word count check
    words = text.split()
    if len(words) < 20:
        return False, "too_few_words"

    # 4. Average word length check
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2 or avg_word_len > 15:
        return False, "abnormal_word_length"

    # 5. Punctuation check (proper sentences)
    if sum(1 for c in text if c in ".!?") < 2:
        return False, "no_punctuation"

    # 6. Repetitive pattern detection (5+ consecutive same words = spam)
    for i in range(len(words) - 4):
        if len(set(words[i : i + 5])) == 1:
            return False, "repetitive_pattern"

    # 7. HTML/JavaScript code detection
    text_lower = text.lower()
    for tag in (
        "<script",
        "<style",
        "<div",
        "<span",
        "<table",
        "<!doctype",
        "<html",
        "<head",
        "<body",
    ):
        if tag in text_lower:
            return False, "contains_html"

    # 8. Excessive special characters check
    if sum(1 for c in text if c in "{}[]<>|\\^~`") / len(text) > 0.05:
        return False, "too_many_special_chars"

    # 9. Excessive URLs check
    url_count = (
        text_lower.count("http://") + text_lower.count("https://") + text_lower.count("www.")
    )
    if url_count > 5:
        return False, "too_many_urls"

    # 10. Excessive digits check (price lists, tables)
    if sum(1 for c in text if c.isdigit()) / len(text) > 0.3:
        return False, "too_many_digits"

    return True, "ok"


def load_c4_for_n_samples_min_length(
    tokenizer,
    num_samples,
    min_length,
    *,
    use_filter=False,
    logger=None,
):
    """Collect *num_samples* C4 texts whose token length >= *min_length*.

    Intended for no-cross-document strategies (``drop_head`` / ``drop_rand``).

    Args:
        tokenizer: Tokenizer.
        num_samples (int): Number of samples to collect.
        min_length (int): Minimum token length (shorter texts are discarded).
        use_filter (bool): Apply quality filtering before length check.
        logger: Logger (optional).

    Returns:
        list[str]: Text samples that satisfy the conditions.
    """
    if logger is None:
        logger = getLogger(__name__)

    logger.info(
        "Loading C4 texts: collecting %d samples with >= %d tokens " "(quality_filter=%s) ...",
        num_samples,
        min_length,
        use_filter,
    )

    dataset = _get_c4_train(logger)

    collected: list[str] = []
    scanned = 0
    discarded_short = 0
    discarded_quality = 0

    for item in dataset:
        text = item["text"].strip()
        scanned += 1

        if use_filter:
            is_valid, _ = check_text_quality(text)
            if not is_valid:
                discarded_quality += 1
                continue

        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if len(ids) < min_length:
            discarded_short += 1
            continue

        collected.append(text)
        if len(collected) >= num_samples:
            break

    if len(collected) < num_samples:
        raise ValueError(
            "Could not collect enough long samples from C4: "
            f"collected={len(collected)}, required={num_samples}, "
            f"min_length={min_length}, scanned={scanned}, "
            f"discarded_short={discarded_short}, "
            f"discarded_quality={discarded_quality}"
        )

    logger.info(
        "Collected %d/%d samples (scanned=%d, discarded_short=%d, " "discarded_quality=%d)",
        len(collected),
        num_samples,
        scanned,
        discarded_short,
        discarded_quality,
    )
    return collected


def load_c4_for_aligned_chunks(
    tokenizer,
    num_calibration_samples,
    max_length,
    *,
    use_filter=False,
    logger=None,
):
    """Load enough C4 samples so that the total token count reaches
    *num_calibration_samples* × *max_length*.

    Used by the ``concat_chunk_align`` strategy.

    Args:
        tokenizer: Tokenizer.
        num_calibration_samples (int): Target number of chunks.
        max_length (int): Length of each chunk.
        use_filter (bool): Apply quality filtering.
        logger: Logger (optional).

    Returns:
        list[str]: Text samples.
    """
    if logger is None:
        logger = getLogger(__name__)

    target_tokens = num_calibration_samples * max_length

    logger.info(
        "concat_chunk_align mode: targeting %d chunks (%d tokens, " "quality_filter=%s)",
        num_calibration_samples,
        target_tokens,
        use_filter,
    )

    dataset = _get_c4_train(logger)

    texts: list[str] = []
    total_tokens = 0
    discarded_quality = 0

    for item in dataset:
        text = item["text"].strip()

        if use_filter:
            is_valid, _ = check_text_quality(text)
            if not is_valid:
                discarded_quality += 1
                continue

        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        total_tokens += len(tokens)
        texts.append(text)

        if total_tokens >= target_tokens:
            break

    logger.info(
        "Loaded %d samples with ~%d tokens (target=%d, " "discarded_quality=%d)",
        len(texts),
        total_tokens,
        target_tokens,
        discarded_quality,
    )

    if total_tokens < target_tokens:
        logger.warning(
            "Could not reach target tokens. Loaded %d tokens, " "target was %d tokens.",
            total_tokens,
            target_tokens,
        )

    return texts


def load_c4_texts(
    tokenizer,
    num_samples,
    *,
    use_filter=False,
    logger=None,
):
    """Load *num_samples* C4 texts (simple sequential loading).

    Args:
        tokenizer: Tokenizer (unused when ``use_filter=False``, but kept
            for a consistent interface).
        num_samples (int): Number of samples to load.
        use_filter (bool): Apply quality filtering.
        logger: Logger (optional).

    Returns:
        list[str]: Text samples.
    """
    if logger is None:
        logger = getLogger(__name__)

    dataset = _get_c4_train(logger)

    if not use_filter:
        texts = list(dataset.select(range(min(num_samples, len(dataset))))["text"])
        logger.info("Loaded %d C4 texts", len(texts))
        return texts

    texts: list[str] = []
    discarded = 0
    for item in dataset:
        text = item["text"].strip()
        is_valid, _ = check_text_quality(text)
        if is_valid:
            texts.append(text)
            if len(texts) >= num_samples:
                break
        else:
            discarded += 1

    logger.info(
        "Loaded %d C4 texts (discarded %d by quality filter)",
        len(texts),
        discarded,
    )
    return texts


def prepare_calibration_data(
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    *,
    use_quality_filter=False,
    logger=None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Load C4 texts and chunk them according to *strategy*.

    Args:
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Target number of samples.
        strategy: Chunking strategy.
        seed: Random seed.
        use_quality_filter: Apply text quality filtering.
        logger: Logger (optional).

    Returns:
        dict: ``{"input_ids": Tensor, "attention_mask": Tensor}``
    """
    if logger is None:
        logger = getLogger(__name__)

    logger.info(
        "Using AllenAI C4 dataset (quality_filter=%s).",
        use_quality_filter,
    )

    if strategy in ("drop_head", "drop_rand"):
        texts = load_c4_for_n_samples_min_length(
            tokenizer=tokenizer,
            num_samples=num_calibration_samples,
            min_length=max_length,
            use_filter=use_quality_filter,
            logger=logger,
        )
    elif strategy == "concat_chunk_align":
        texts = load_c4_for_aligned_chunks(
            tokenizer,
            num_calibration_samples,
            max_length,
            use_filter=use_quality_filter,
            logger=logger,
        )
    else:
        texts = load_c4_texts(
            tokenizer,
            num_calibration_samples,
            use_filter=use_quality_filter,
            logger=logger,
        )

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
