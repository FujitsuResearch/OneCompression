"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CalibrationConfig:
    """Configuration for calibration data preparation.

    Groups all calibration-related parameters into a single object
    used by :class:`~onecomp.runner.Runner` and
    :class:`~onecomp.quantizer.autobit.AutoBitQuantizer`.

    Attributes:
        calibration_dataset (str):
            Dataset name (``"c4"``, ``"wikitext2"``), a local file path,
            or a HuggingFace Hub dataset ID.  Defaults to ``"c4"``
            (AllenAI C4 dataset).
        max_length (int):
            Maximum token length per calibration chunk.
        num_calibration_samples (int):
            Target number of calibration samples.
        strategy (str):
            Chunking strategy.  One of ``"concat_chunk"``,
            ``"concat_chunk_align"``, ``"concat_rand"``,
            ``"drop_head"``, ``"drop_rand"``.
        seed (int):
            Random seed used by stochastic strategies
            (e.g. ``"drop_rand"``).
        batch_size (int or None):
            Batch size for chunked calibration forward passes.
            ``None`` means a single forward pass with all data.
            Used only by chunked quantization
            (:func:`~onecomp.runner_methods.chunked_quantization.run_chunked_quantization`).
        num_layers_per_group (int):
            Number of layers processed simultaneously in chunked
            calibration mode.
            Used only by chunked quantization
            (:func:`~onecomp.runner_methods.chunked_quantization.run_chunked_quantization`).
        text_key (str):
            Column name used when loading custom or Hub datasets.
        use_quality_filter (bool):
            Apply C4 quality filtering (ignored for non-C4 sources).
        max_documents (int):
            Cap on documents loaded from custom files or Hub datasets.

    Examples:
        Default configuration (C4 dataset)::

            config = CalibrationConfig()

        WikiText-2 with longer sequences::

            config = CalibrationConfig(
                calibration_dataset="wikitext2",
                max_length=2048,
                num_calibration_samples=256,
            )

        Custom file with quality filtering::

            config = CalibrationConfig(
                calibration_dataset="./my_data.jsonl",
                text_key="content",
                use_quality_filter=True,
            )
    """

    calibration_dataset: str = "c4"
    max_length: int = 2048
    num_calibration_samples: int = 512
    strategy: str = "drop_rand"
    seed: int = 0
    batch_size: Optional[int] = None
    num_layers_per_group: int = 7
    text_key: str = "text"
    use_quality_filter: bool = False
    max_documents: int = 10000
