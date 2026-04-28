"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Akihiro Yoshida
"""

import random

import torch

_VALID_CALIBRATION_STRATEGIES = (
    "concat_chunk",
    "concat_chunk_align",
    "concat_rand",
    "drop_head",
    "drop_rand",
)


def chunk_single_document(
    texts,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    logger,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Extract fixed-length chunks from individual documents
    (for ``drop_head`` / ``drop_rand``).

    Args:
        texts: List of text samples.
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Required number of samples.
        strategy: ``"drop_head"`` or ``"drop_rand"``.
        seed: Random seed (for ``drop_rand``).
        logger: Logger.

    Returns:
        dict: ``{"input_ids": tensor, "attention_mask": tensor}``
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

        if len(ids) < max_length:
            discarded_short += 1
            continue

        if strategy == "drop_head":
            chunk = ids[:max_length]
        else:
            assert strategy == "drop_rand"
            max_start = len(ids) - max_length
            start = (
                0
                if max_start == 0
                else int(torch.randint(0, max_start + 1, (1,), generator=gen).item())
            )
            chunk = ids[start : start + max_length]

        chunks.append(chunk)
        if len(chunks) >= num_calibration_samples:
            break

    if len(chunks) < num_calibration_samples:
        raise ValueError(
            "Not enough calibration samples after dropping short texts: "
            f"collected={len(chunks)}, required={num_calibration_samples}, "
            f"max_length={max_length}, scanned={scanned}, "
            f"discarded_short={discarded_short}."
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


def chunk_concat(
    texts,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    align_chunks,
    logger,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Concatenate all texts and split into equal-length chunks
    (for ``concat_chunk`` / ``concat_chunk_align``).

    Args:
        texts: List of text samples.
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Required number of samples
            (only used when *align_chunks* is True).
        align_chunks: If True, fix the number of chunks to
            *num_calibration_samples*.
        logger: Logger.

    Returns:
        dict: ``{"input_ids": tensor, "attention_mask": tensor}``
    """
    all_text = "\n\n".join(text.strip() for text in texts)
    all_tokens = tokenizer(all_text, return_tensors="pt")["input_ids"][0]

    total_tokens = len(all_tokens)

    if align_chunks:
        num_chunks = min(num_calibration_samples, total_tokens // max_length)
        if num_chunks < num_calibration_samples:
            logger.warning(
                "Not enough tokens for %d chunks. Using %d chunks instead.",
                num_calibration_samples,
                num_chunks,
            )
    else:
        num_chunks = total_tokens // max_length

    if num_chunks == 0:
        logger.warning("Calibration data is too short. " "Using all tokens as a single chunk.")
        num_chunks = 1
        padded_tokens = torch.zeros(max_length, dtype=all_tokens.dtype)
        padded_tokens[:total_tokens] = all_tokens
        input_ids = padded_tokens.unsqueeze(0)
        attention_mask = torch.zeros(max_length, dtype=torch.long).unsqueeze(0)
        attention_mask[:, :total_tokens] = 1
        discarded_tokens = 0
        padded_count = max_length - total_tokens
    else:
        used_tokens = num_chunks * max_length
        input_ids = all_tokens[:used_tokens].reshape(num_chunks, max_length)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        discarded_tokens = total_tokens - used_tokens
        padded_count = 0

    logger.info(
        "Created %d chunks of length %d " "(total %d tokens, discarded %d, padded %d)",
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


def chunk_concat_rand(
    texts,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    seed,
    logger,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Concatenate all texts, tokenize, then randomly sample positions
    (for ``concat_rand``).

    This matches the standard calibration approach used in GPTQ / AWQ
    literature: join all documents into one long token sequence, then
    draw ``num_calibration_samples`` random windows of ``max_length``.
    Samples may overlap.

    Recommended for datasets composed of many short documents (e.g.
    WikiText-2) where per-document strategies would discard most entries.

    Args:
        texts: List of text samples.
        tokenizer: Tokenizer.
        device: Device to place tensors on.
        max_length: Chunk length.
        num_calibration_samples: Number of random samples to draw.
        seed: Random seed.
        logger: Logger.

    Returns:
        dict: ``{"input_ids": tensor, "attention_mask": tensor}``
    """
    all_text = "\n\n".join(text.strip() for text in texts)
    all_tokens = tokenizer(all_text, return_tensors="pt")["input_ids"]

    L = all_tokens.shape[1]

    if L <= max_length:
        logger.warning(
            "Total tokens (%d) <= max_length (%d). " "Returning a single padded chunk.",
            L,
            max_length,
        )
        padded = torch.zeros(1, max_length, dtype=all_tokens.dtype)
        padded[:, :L] = all_tokens
        input_ids = padded
        attention_mask = torch.zeros(1, max_length, dtype=torch.long)
        attention_mask[0, :L] = 1
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }

    hi = max(0, L - max_length - 1)

    target_tokens = num_calibration_samples * max_length
    if L < target_tokens:
        logger.warning(
            "Total tokens (%d) < target (%d = %d samples x %d length). "
            "Random chunks will overlap significantly.",
            L,
            target_tokens,
            num_calibration_samples,
            max_length,
        )

    random.seed(seed)
    chunks = []
    for _ in range(num_calibration_samples):
        i = random.randint(0, hi)
        chunks.append(all_tokens[0, i : i + max_length])

    input_ids = torch.stack(chunks, dim=0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    logger.info(
        "Created %d random-sliced chunks of length %d " "(total %d tokens, overlap possible)",
        input_ids.shape[0],
        input_ids.shape[1],
        L,
    )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def prepare_from_texts(
    texts,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    logger,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Chunk a ``list[str]`` according to *strategy*.

    This is the shared entry point used by every data-source module
    once raw texts have been collected.
    """
    if strategy in ("drop_head", "drop_rand"):
        return chunk_single_document(
            texts,
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            strategy,
            seed,
            logger,
        )

    if strategy == "concat_rand":
        return chunk_concat_rand(
            texts,
            tokenizer,
            device,
            max_length,
            num_calibration_samples,
            seed,
            logger,
        )

    assert strategy in ("concat_chunk", "concat_chunk_align")
    return chunk_concat(
        texts,
        tokenizer,
        device,
        max_length,
        num_calibration_samples,
        align_chunks=(strategy == "concat_chunk_align"),
        logger=logger,
    )
