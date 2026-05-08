"""Training arguments for rotation/scaling preprocessing.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami
"""

import os
import tempfile
from dataclasses import dataclass, field

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Hyperparameters for rotation/scaling preprocessing with HuggingFace Trainer.

    Extends ``transformers.TrainingArguments`` with ``SGDG``-specific learning
    rates and momentum values consumed by ``_PreprocessTrainer``.
    Also overrides several inherited defaults (e.g. ``max_steps=800``,
    ``gradient_checkpointing=True``, ``save_strategy="no"``); see the
    class body for the full list of preprocess-specific defaults.

    Attributes:
        rotation_lr: Learning rate for 2-D rotation (Stiefel) parameters.
        scaling_lr: Learning rate for 1-D scaling parameters.
        rotation_momentum: Momentum factor for the rotation parameter group.
        scaling_momentum: Momentum factor for the scaling parameter group.
        seed: Random seed for Trainer data shuffling and training
            reproducibility.  Independent of the ``seed`` argument in
            ``prepare_rotated_model``, which controls rotation matrix
            initialisation and calibration data preparation.
    """

    # Rotation / scaling optimiser parameters (used by _PreprocessTrainer)
    rotation_lr: float = field(default=1.5)
    scaling_lr: float = field(default=1.5)
    rotation_momentum: float = field(default=0)
    scaling_momentum: float = field(default=0)

    # Training
    seed: int = field(default=42)
    max_steps: int = field(default=800)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    weight_decay: float = field(default=0.0)
    lr_scheduler_type: str = field(default="cosine")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)

    # Checkpoints (not needed for rotation training; use a fixed path to
    # avoid creating a new temp directory on every instantiation)
    output_dir: str = field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "onecomp_preprocess")
    )
    save_strategy: str = field(default="no")
    save_safetensors: bool = field(default=False)

    # Logging (avoid conflict with onecomp's logger)
    report_to: str = field(default="none")
    logging_steps: int = field(default=10)
    disable_tqdm: bool = field(default=False)
    log_on_each_node: bool = field(default=False)


def get_training_arguments(train_args_config) -> TrainingArguments:
    """Merge user overrides into ``TrainingArguments`` with dataclass defaults.

    Args:
        train_args_config: Mapping of field names to values (e.g. from
            ``training_args_override`` in ``apply_preprocess_train``). Omitted
            keys keep the defaults declared on ``TrainingArguments``.

    Returns:
        TrainingArguments: Instantiated training configuration for the
        preprocess Trainer.
    """
    return TrainingArguments(**train_args_config)
