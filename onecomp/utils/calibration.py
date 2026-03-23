"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from logging import getLogger

from datasets import load_dataset
import torch

_VALID_CALIBRATION_STRATEGIES = (
    "concat_chunk",
    "concat_chunk_align",
    "drop_head",
    "drop_rand",
)


def load_c4_for_n_samples_min_length(
    tokenizer,
    num_samples,
    min_length,
    logger=None,
):
    """Collect num_samples texts from C4 whose token length is at least min_length.

    Intended for no-cross-document calibration strategies (each sample is
    independently extracted).

    Args:
        tokenizer: Tokenizer.
        num_samples (int): Number of samples to collect.
        min_length (int): Minimum token length (shorter texts are discarded).
        logger: Logger (optional).

    Returns:
        list[str]: Text samples that satisfy the length condition.
    """
    if logger is None:
        logger = getLogger(__name__)

    logger.info(
        "Loading C4 texts until we collect %d samples with >= %d tokens...",
        num_samples,
        min_length,
    )

    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    )

    collected = []
    scanned = 0
    discarded_short = 0

    for item in dataset:
        text = item["text"].strip()
        scanned += 1

        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        # Policy: discard texts shorter than min_length (== min_length is OK)
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
            f"min_length={min_length}, scanned={scanned}, discarded_short={discarded_short}"
        )

    logger.info(
        "Collected %d/%d samples (scanned=%d, discarded_short=%d)",
        len(collected),
        num_samples,
        scanned,
        discarded_short,
    )
    return collected


def load_c4_for_aligned_chunks(
    tokenizer,
    num_calibration_samples,
    max_length,
    logger=None,
):
    """Load enough samples from C4 to achieve the target number of chunks.

    Used for the concat_chunk_align strategy.
    Ensures that the total number of tokens is at least
    num_calibration_samples * max_length so that
    num_chunks == num_calibration_samples.

    Args:
        tokenizer: Tokenizer.
        num_calibration_samples (int): Target number of chunks.
        max_length (int): Maximum length of each chunk.
        logger: Logger (optional).

    Returns:
        list: List of text samples.
    """
    if logger is None:
        logger = getLogger(__name__)

    target_chunks = num_calibration_samples
    target_tokens = target_chunks * max_length

    logger.info(
        "concat_chunk_align mode: targeting %d chunks (%d tokens)",
        target_chunks,
        target_tokens,
    )

    # Load from the C4 dataset (over-read initially).
    # An average C4 sample has roughly 500 tokens, so we estimate the
    # required number of samples and load a bit more than needed.
    estimated_tokens_per_sample = 500
    initial_samples = max(
        num_calibration_samples,
        (target_tokens // estimated_tokens_per_sample) * 2,
    )

    logger.info("Loading initial %d samples from C4 dataset...", initial_samples)
    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    )

    texts = []
    total_tokens = 0
    sample_idx = 0

    # Add samples until the required number of tokens is reached
    while total_tokens < target_tokens and sample_idx < len(dataset):
        text = dataset[sample_idx]["text"].strip()
        # Estimate token count (exact counting is expensive, so we approximate)
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
        total_tokens += len(tokens)
        texts.append(text)
        sample_idx += 1

    logger.info(
        "Loaded %d samples with approximately %d tokens (target: %d tokens)",
        len(texts),
        total_tokens,
        target_tokens,
    )

    if total_tokens < target_tokens:
        logger.warning(
            "Could not reach target tokens. " "Loaded %d tokens, target was %d tokens.",
            total_tokens,
            target_tokens,
        )

    return texts


# =============================================================================
# Private functions: chunking
# =============================================================================


def _chunk_single_document(
    texts,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    logger,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """Extract fixed-length chunks from individual documents (for drop_head / drop_rand).

    Args:
        texts: List of text samples.
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Required number of samples.
        strategy: "drop_head" or "drop_rand".
        seed: Random seed (for drop_rand).
        logger: Logger.

    Returns:
        dict: {"input_ids": tensor, "attention_mask": tensor}
    """
    gen = None
    if strategy == "drop_rand":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

    chunks = []
    scanned = 0
    discarded_short = 0

    for text in texts:
        scanned += 1
        ids = tokenizer(text.strip(), return_tensors="pt")["input_ids"][0]

        # Policy: discard texts shorter than max_length (== max_length is OK)
        if len(ids) < max_length:
            discarded_short += 1
            continue

        if strategy == "drop_head":
            chunk = ids[:max_length]
        else:
            assert strategy == "drop_rand"
            # random window (no cross-document, no padding)
            max_start = len(ids) - max_length
            if max_start == 0:
                start = 0
            else:
                start = int(torch.randint(0, max_start + 1, (1,), generator=gen).item())
            chunk = ids[start : start + max_length]

        chunks.append(chunk)
        if len(chunks) >= num_calibration_samples:
            break

    if len(chunks) < num_calibration_samples:
        raise ValueError(
            "Not enough calibration samples after dropping short texts: "
            f"collected={len(chunks)}, required={num_calibration_samples}, "
            f"max_length={max_length}, scanned={scanned}, discarded_short={discarded_short}."
        )

    input_ids = torch.stack(chunks, dim=0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    logger.info(
        "Created %d single-document chunks of length %d "
        "(scanned=%d, discarded_short=%d, padding=0)",
        input_ids.shape[0],
        input_ids.shape[1],
        scanned,
        discarded_short,
    )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def _chunk_concat(
    texts,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    align_chunks,
    logger,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """Concatenate all texts and split into chunks (for concat_chunk / concat_chunk_align).

    Args:
        texts: List of text samples.
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Required number of samples (used only when align_chunks=True).
        align_chunks: If True, fix the number of chunks to num_calibration_samples.
        logger: Logger.

    Returns:
        dict: {"input_ids": tensor, "attention_mask": tensor}
    """
    # Concatenate all texts and tokenize.
    # Joining with "\n\n" marks document boundaries,
    # although boundaries become ambiguous after chunking.
    all_text = "\n\n".join([text.strip() for text in texts])
    all_tokens = tokenizer(all_text, return_tensors="pt")["input_ids"][0]

    # Split into chunks of max_length.
    # All chunks have the same length, so no padding is needed.
    # Remaining tokens at the end (fewer than max_length) are discarded.
    total_tokens = len(all_tokens)

    if align_chunks:
        # align_chunks mode: fix the number of chunks to num_calibration_samples
        num_chunks = min(num_calibration_samples, total_tokens // max_length)
        if num_chunks < num_calibration_samples:
            logger.warning(
                "Not enough tokens for %d chunks. Using %d chunks instead.",
                num_calibration_samples,
                num_chunks,
            )
    else:
        # Normal mode: create as many chunks as possible
        num_chunks = total_tokens // max_length

    if num_chunks == 0:
        # Edge case: calibration data is too short.
        # This should not normally occur, but pad to a single chunk for safety.
        logger.warning("Calibration data is too short. Using all tokens as a single chunk.")
        num_chunks = 1
        padded_tokens = torch.zeros(max_length, dtype=all_tokens.dtype)
        padded_tokens[:total_tokens] = all_tokens
        input_ids = padded_tokens.unsqueeze(0)
        attention_mask = torch.zeros(max_length, dtype=torch.long).unsqueeze(0)
        attention_mask[:, :total_tokens] = 1
        discarded_tokens = 0
        padded_count = max_length - total_tokens
    else:
        # Normal case: reshape into chunks of max_length.
        # shape: (num_chunks, max_length)
        used_tokens = num_chunks * max_length
        input_ids = all_tokens[:used_tokens].reshape(num_chunks, max_length)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        discarded_tokens = total_tokens - used_tokens
        padded_count = 0

    logger.info(
        "Created %d chunks of length %d (total %d tokens, discarded %d tokens, padded %d tokens)",
        num_chunks,
        max_length,
        total_tokens,
        discarded_tokens,
        padded_count,
    )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


# =============================================================================
# Private functions: per-data-source processing
# =============================================================================


def _prepare_from_c4(
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    logger,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Prepare calibration data from the C4 dataset.

    Args:
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Number of samples.
        strategy: Calibration strategy.
        seed: Random seed.
        logger: Logger.

    Returns:
        dict: {"input_ids": tensor, "attention_mask": tensor}
    """
    logger.info("No calibration dataset provided. Using AllenAI's C4 dataset.")

    # Load data according to the strategy
    if strategy in ("drop_head", "drop_rand"):
        # No-cross-document mode: collect num_calibration_samples documents with token length >= max_length
        texts = load_c4_for_n_samples_min_length(
            tokenizer=tokenizer,
            num_samples=num_calibration_samples,
            min_length=max_length,
            logger=logger,
        )
        return _chunk_single_document(
            texts,
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            strategy,
            seed,
            logger,
        )

    if strategy == "concat_chunk_align":
        # concat_chunk_align: adjust the number of samples so that num_chunks == num_calibration_samples
        texts = load_c4_for_aligned_chunks(tokenizer, num_calibration_samples, max_length, logger)
        return _chunk_concat(
            texts,
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            align_chunks=True,
            logger=logger,
        )

    # concat_chunk: normal mode, load num_calibration_samples samples
    assert strategy == "concat_chunk"
    texts = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(num_calibration_samples))["text"]

    return _chunk_concat(
        texts,
        tokenizer,
        device,
        max_length,
        num_calibration_samples,
        align_chunks=False,
        logger=logger,
    )


def _prepare_from_custom(
    calibration_dataset,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    logger,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Prepare calibration data from a user-provided dataset.

    Args:
        calibration_dataset: List of text samples.
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Number of samples.
        strategy: Calibration strategy.
        seed: Random seed.
        logger: Logger.

    Returns:
        dict: {"input_ids": tensor, "attention_mask": tensor}
    """
    if strategy in ("drop_head", "drop_rand"):
        return _chunk_single_document(
            calibration_dataset,
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            strategy,
            seed,
            logger,
        )

    # concat_chunk / concat_chunk_align
    assert strategy in ("concat_chunk", "concat_chunk_align")
    align_chunks = strategy == "concat_chunk_align"
    return _chunk_concat(
        calibration_dataset,
        tokenizer,
        device,
        max_length,
        num_calibration_samples,
        align_chunks=align_chunks,
        logger=logger,
    )


# =============================================================================
# Public function: entry point
# =============================================================================


def prepare_calibration_dataset(
    tokenizer,
    device,
    calibration_dataset=None,
    max_length=512,
    num_calibration_samples=128,
    strategy="drop_rand",
    seed=0,
    logger=None,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Prepare calibration data for quantization methods such as GPTQ.

    Processing flow:
        1. Obtain data source: use C4 if calibration_dataset is None.
        2. Chunk the data according to the chosen strategy.

    strategy:
        - "concat_chunk": concatenate all texts -> tokenize at once -> equal-length chunks.
            Creates as many chunks as possible.
        - "concat_chunk_align": concatenate all texts -> tokenize at once -> equal-length chunks.
            Fixes the number of chunks to num_calibration_samples.
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
        calibration_dataset (list, optional):
            List of texts for calibration.
            If omitted, the AllenAI C4 dataset is used.
        max_length (int): Maximum length of each chunk.
        num_calibration_samples (int):
            Number of samples when using the default dataset.
        strategy (str):
            Calibration data preparation strategy (see above).
        seed (int):
            Random seed for strategy="drop_rand".
        logger: Logger (optional).

    Returns:
        dict: Model input dictionary.
            - "input_ids": tensor of shape (num_chunks, max_length).
            - "attention_mask": tensor of shape (num_chunks, max_length).
    """
    if logger is None:
        logger = getLogger(__name__)

    if strategy not in _VALID_CALIBRATION_STRATEGIES:
        raise ValueError(
            "Unknown calibration strategy: "
            f"{strategy!r}. Available: {list(_VALID_CALIBRATION_STRATEGIES)}"
        )

    logger.info("Preparing the calibration dataset... (strategy=%s)", strategy)

    if calibration_dataset is None:
        return _prepare_from_c4(
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            strategy,
            seed,
            logger,
        )

    return _prepare_from_custom(
        calibration_dataset,
        tokenizer,
        device,
        max_length,
        num_calibration_samples,
        strategy,
        seed,
        logger,
    )
