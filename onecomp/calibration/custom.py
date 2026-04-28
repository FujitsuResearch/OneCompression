"""
Custom dataset text loader for calibration.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa, Akihiro Yoshida
"""

import ast
import csv
import json
import os
import random
from logging import getLogger
from typing import List, Optional

import datasets

from .chunking import prepare_from_texts


def _decode_text(text) -> str:
    """Decode text from bytes / string literal and clean Wiki40b tags."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    elif isinstance(text, str):
        if text.startswith("b'") or text.startswith('b"'):
            try:
                bytes_obj = ast.literal_eval(text)
                if isinstance(bytes_obj, bytes):
                    text = bytes_obj.decode("utf-8", errors="ignore")
            except (ValueError, SyntaxError):
                pass

    if not isinstance(text, str):
        return ""

    text = text.replace("_START_ARTICLE_", "")
    text = text.replace("_START_SECTION_", "\n")
    text = text.replace("_START_PARAGRAPH_", "\n")
    text = text.replace("_NEWLINE_", "\n")
    return text.strip()


def _find_text_column(column_names, text_key="text") -> Optional[str]:
    """Return the first matching column name from *column_names*."""
    for candidate in (text_key, "text", "content", "sentence", "document", "body"):
        if candidate in column_names:
            return candidate
    return None


def _load_from_parquet_files(
    parquet_files: List[str],
    text_key: str = "text",
    max_documents: int = 10000,
    seed: int = 0,
) -> List[str]:
    """Load texts from one or more Parquet files efficiently.

    Only reads the text column and samples row groups to avoid loading
    multi-GB files entirely into memory.
    """
    import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel

    schema = pq.read_schema(parquet_files[0])
    column_key = _find_text_column(schema.names, text_key)
    if column_key is None:
        raise ValueError(f"No text column found in parquet. Available: {schema.names}")

    texts: list[str] = []
    docs_per_file = max(1, max_documents // len(parquet_files))
    random.seed(seed)

    for pf_path in parquet_files:
        pf = pq.ParquetFile(pf_path)
        total_rows = pf.metadata.num_rows
        need = min(docs_per_file, total_rows)

        if total_rows <= need:
            table = pq.read_table(pf_path, columns=[column_key])
            for val in table.column(column_key):
                s = val.as_py()
                if s and len(s) > 50:
                    texts.append(s)
        else:
            indices = sorted(random.sample(range(total_rows), need))
            n_rg = pf.metadata.num_row_groups
            rg_boundaries: list[tuple[int, int]] = []
            offset = 0
            for rg_idx in range(n_rg):
                n = pf.metadata.row_group(rg_idx).num_rows
                rg_boundaries.append((offset, offset + n))
                offset += n

            needed_rgs: set[int] = set()
            for idx in indices:
                for rg_idx, (start, end) in enumerate(rg_boundaries):
                    if start <= idx < end:
                        needed_rgs.add(rg_idx)
                        break

            rg_data = {}
            for rg_idx in sorted(needed_rgs):
                table = pf.read_row_group(rg_idx, columns=[column_key])
                rg_data[rg_idx] = table.column(column_key)

            for idx in indices:
                for rg_idx, (start, end) in enumerate(rg_boundaries):
                    if start <= idx < end:
                        s = rg_data[rg_idx][idx - start].as_py()
                        if s and len(s) > 50:
                            texts.append(s)
                        break

        if len(texts) >= max_documents:
            break

    return texts[:max_documents]


def _load_from_hf_dataset(
    data_path: str,
    text_key: str = "text",
    max_documents: int = 10000,
    seed: int = 0,
) -> List[str]:
    """Load texts from a HuggingFace Dataset directory."""
    ds = datasets.load_from_disk(data_path)

    if hasattr(ds, "keys"):
        data = ds.get("train", ds[list(ds.keys())[0]])
    else:
        data = ds

    column_key = _find_text_column(data.column_names, text_key)
    if column_key is None:
        raise ValueError(f"No text column found. Available: {data.column_names}")

    total_docs = len(data)
    if total_docs > max_documents:
        random.seed(seed)
        indices = sorted(random.sample(range(total_docs), max_documents))
    else:
        indices = range(total_docs)

    texts: list[str] = []
    for idx in indices:
        decoded = _decode_text(data[int(idx)][column_key])
        if decoded and len(decoded) > 50:
            texts.append(decoded)

    return texts


def load_custom_texts(
    data_path: str,
    *,
    text_key: str = "text",
    max_documents: int = 10000,
    seed: int = 0,
    logger=None,
) -> List[str]:
    """Load text data from various file formats.

    Args:
        data_path: Path to a data file or directory.
        text_key: Column / key name for the text field.
        max_documents: Maximum number of documents to load.
        seed: Random seed for sampling.
        logger: Logger (optional).

    Returns:
        list[str]: Text samples.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
        ValueError: If the format is unsupported or no text is found.
    """
    if logger is None:
        logger = getLogger(__name__)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # Directory handling: parquet files first, then HuggingFace Datasets
    if os.path.isdir(data_path):
        import glob as _glob  # pylint: disable=import-outside-toplevel

        parquet_files = sorted(_glob.glob(os.path.join(data_path, "*.parquet")))
        if not parquet_files:
            parquet_files = sorted(
                _glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True)
            )

        if parquet_files:
            logger.info(
                "Loading from %d parquet file(s) in: %s",
                len(parquet_files),
                data_path,
            )
            return _load_from_parquet_files(
                parquet_files,
                text_key,
                max_documents,
                seed,
            )

        logger.info("Loading as HuggingFace Dataset from: %s", data_path)
        return _load_from_hf_dataset(data_path, text_key, max_documents, seed)

    ext = os.path.splitext(data_path)[1].lower()

    if ext in (".txt", ".text"):
        logger.info("Loading plain text file: %s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "\n\n" in content:
            return [t.strip() for t in content.split("\n\n") if t.strip()]
        return [content.strip()] if content.strip() else []

    if ext == ".json":
        logger.info("Loading JSON file: %s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts: list[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_key in item:
                    texts.append(item[text_key])
                elif isinstance(item, str):
                    texts.append(item)
        elif isinstance(data, dict):
            val = data.get(text_key)
            if isinstance(val, list):
                texts = val
            elif val is not None:
                texts = [val]
        return [t for t in texts if t and isinstance(t, str)]

    if ext in (".jsonl", ".ndjson"):
        logger.info("Loading JSON Lines file: %s", data_path)
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and text_key in obj:
                        texts.append(obj[text_key])
                    elif isinstance(obj, str):
                        texts.append(obj)
                except json.JSONDecodeError:
                    continue
        return [t for t in texts if t and isinstance(t, str)]

    if ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        logger.info("Loading %s file: %s", ext.upper().lstrip("."), data_path)
        texts = []
        with open(data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                col = _find_text_column(row.keys(), text_key)
                if col and row[col]:
                    texts.append(row[col])
        return [t for t in texts if t and isinstance(t, str)]

    if ext == ".parquet":
        logger.info("Loading Parquet file: %s", data_path)
        return _load_from_parquet_files(
            [data_path],
            text_key,
            max_documents,
            seed,
        )

    if ext == ".arrow":
        logger.info("Loading Arrow file: %s", data_path)
        ds = datasets.Dataset.from_file(data_path)
        col = _find_text_column(ds.column_names, text_key)
        if col is None:
            raise ValueError(f"No text column found. Available: {ds.column_names}")
        return [t for t in ds[col] if t and isinstance(t, str)]

    raise ValueError(
        f"Unsupported format: {ext}\n"
        f"Supported: .txt, .json, .jsonl, .csv, .tsv, .parquet, .arrow, "
        f"HuggingFace Dataset directory"
    )


def prepare_calibration_data(
    data_path,
    tokenizer,
    device,
    max_length,
    num_calibration_samples,
    strategy,
    seed,
    *,
    text_key="text",
    max_documents=10000,
    logger=None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """
    Load custom dataset with WikiText-compatible output format.

    Supports: .txt, .json, .jsonl, .csv, .parquet, .arrow, HF Dataset dirs

    Returns:
        dict: ``{"input_ids": Tensor, "attention_mask": Tensor}``
    """
    if logger is None:
        logger = getLogger(__name__)

    logger.info("Using custom dataset: %s", data_path)
    texts = load_custom_texts(
        data_path,
        text_key=text_key,
        max_documents=max_documents,
        seed=seed,
        logger=logger,
    )
    if not texts:
        raise ValueError(f"No text data found in {data_path}")

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
