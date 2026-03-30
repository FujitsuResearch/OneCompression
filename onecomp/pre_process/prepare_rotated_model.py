"""Rotation preprocessing for quantization.

Apply rotation matrices and scaling diagonals to model weights so that
subsequent quantization (e.g. GPTQ) achieves higher accuracy.

Processing flow:
    1. Load the original model
    2. (Optional) Train rotation / scaling matrices via SpinQuant/OstQuant pipeline
    3. Absorb the learned matrices into the model weights
    4. Save the rotated model with ``save_pretrained``
    5. Return a ``RotatedModelConfig`` pointing at the saved directory

The saved model has the same architecture (``nn.Linear`` layers) as the
original.  When loaded through ``RotatedModelConfig``, online Hadamard
hooks are automatically registered on ``down_proj`` layers.

Inspired by:
    - https://github.com/facebookresearch/SpinQuant/blob/main/optimize_rotation.py
    - https://github.com/BrotherHappy/OSTQuant/blob/main/quant/ost_train.py

Example:

    >>> from onecomp import ModelConfig, Runner, prepare_rotated_model, GPTQ
    >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
    >>> rotated_config = prepare_rotated_model(
    ...     model_config=model_config, save_directory="./rotated_model"
    ... )
    >>> quantizer = GPTQ(wbits=4, groupsize=128)
    >>> runner = Runner(model_config=rotated_config, quantizer=quantizer)
    >>> runner.run()

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura, Yusei Kawakami

"""

from __future__ import annotations

import copy
import os
from logging import getLogger
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM

from ..model_config import ModelConfig
from .rotation_utils import cleanup_memory

if TYPE_CHECKING:
    from ..rotated_model_config import RotatedModelConfig


logger = getLogger(__name__)


def prepare_rotated_model(
    model_config: ModelConfig,
    save_directory: str,
    *,
    rotation: bool = True,
    scaling: bool = False,
    rotation_mode: str = "random",
    scaling_mode: str = "identity",
    seed: int = 0,
    enable_training: bool = True,
    calibration_dataset=None,
    max_length: int = 2048,
    num_calibration_samples: int = 128,
    calibration_strategy: str = "drop_rand",
    wbits: int = 4,
    sym: bool = False,
    groupsize: int = -1,
    fp32_had: bool = False,
    use_sdpa: bool = False,
    training_args_override: dict | None = None,
) -> RotatedModelConfig:
    """Train rotation matrices, apply them to model weights, and save.

    Args:
        model_config: Original model configuration (``model_id`` or ``path``).
        save_directory: Directory to save the rotated model.
        rotation: Whether to apply rotation matrices (R1, R2).
        scaling: Whether to apply scaling diagonals (S_*).
        rotation_mode: ``"random"`` or ``"identity"``.
        scaling_mode: ``"identity"`` | ``"random_ones"`` | ``"random"``.
        seed: Random seed for rotation matrix initialisation and
            calibration data preparation.  Note that the Trainer uses a
            separate seed (``TrainingArguments.seed``, default ``42``) for
            data shuffling and training reproducibility.
        enable_training: If ``True``, train the rotation/scaling matrices;
            otherwise use the randomly initialised matrices directly.
        calibration_dataset: List of texts for calibration.
            If ``None``, the C4 dataset is used (same as ``Runner``).
        max_length: Sequence length for calibration data (default: 2048).
        num_calibration_samples: Number of calibration samples.
            Default matches ``Runner`` (128).
        calibration_strategy: Strategy for preparing calibration inputs
            (``"drop_rand"``, ``"concat_chunk"``, etc.).
            See :func:`~onecomp.utils.calibration.prepare_calibration_dataset`.
        wbits: Weight quantisation bit-width for the RTN proxy during
            training.  Should match the quantizer's ``wbits``.
        sym: Symmetric quantisation for the RTN proxy.
        groupsize: Group size for the RTN proxy (``-1`` = per-channel).
            Should match the quantizer's ``groupsize``.  When positive,
            the value must evenly divide the ``out_features`` of every
            ``nn.Linear`` layer in the model.
        fp32_had: Use FP32 for the online Hadamard transform.
        use_sdpa: Use SDPA attention implementation during training.
        training_args_override: Override ``TrainingArguments`` fields (dict).

    Returns:
        :class:`~onecomp.rotated_model_config.RotatedModelConfig` pointing at
        *save_directory*.

    Examples:
        Basic usage:

        >>> from onecomp import ModelConfig, prepare_rotated_model, GPTQ
        >>> model_config = ModelConfig(model_id="meta-llama/Llama-2-7b-hf")
        >>> rotated_config = prepare_rotated_model(
        ...     model_config=model_config,
        ...     save_directory="./rotated_model",
        ... )

        Without training (random rotation only):

        >>> rotated_config = prepare_rotated_model(
        ...     model_config=model_config,
        ...     save_directory="./rotated_model",
        ...     enable_training=False,
        ... )
    """
    from .train_rotation import (
        PreprocessManager,
        apply_preprocess_eval,
        apply_preprocess_train,
    )
    from ..utils import prepare_calibration_dataset

    model_path = model_config.get_model_id_or_path()

    # 1. Load the original model and tokenizer
    logger.info("Loading original model from %s ...", model_path)
    model = model_config.load_model()
    tokenizer = model_config.load_tokenizer()

    # Free GPU memory — the original model is only needed on CPU from here on.
    if next(model.parameters()).is_cuda:
        logger.info("Moving original model to CPU to free GPU memory ...")
        model.to("cpu")
        cleanup_memory()

    # 2. Generate rotation / scaling tensors
    model_config_hf = copy.deepcopy(model.config)
    model_type = getattr(model_config_hf, "model_type", "")
    manager = PreprocessManager(
        model_config_hf,
        rotation=rotation,
        scaling=scaling,
        rotation_mode=rotation_mode,
        scaling_mode=scaling_mode,
        seed=seed,
    )

    # 3. Training (optional)
    need_train = (rotation or scaling) and enable_training
    if need_train:
        logger.info("Preparing calibration data ...")
        calibration_inputs = prepare_calibration_dataset(
            tokenizer=tokenizer,
            device="cpu",
            calibration_dataset=calibration_dataset,
            max_length=max_length,
            num_calibration_samples=num_calibration_samples,
            strategy=calibration_strategy,
            seed=seed,
        )

        # Release the original model to halve peak CPU memory during training.
        # Training only needs config and model_type (already extracted above);
        # model weights are re-loaded from model_path inside the training pipeline.
        logger.info("Releasing original model to free memory for training ...")
        del model
        cleanup_memory()

        logger.info("Starting rotation training ...")
        apply_preprocess_train(
            model_config_hf,
            model_type,
            manager,
            model_path,
            tokenizer=tokenizer,
            calibration_inputs=calibration_inputs,
            wbits=wbits,
            sym=sym,
            groupsize=groupsize,
            use_sdpa=use_sdpa,
            fp32_had=fp32_had,
            training_args_override=training_args_override,
        )

        # Re-load the original model on CPU for rotation application.
        logger.info("Re-loading original model on CPU ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()

    # 4. Apply rotation to model weights (eval path)
    rotation_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    apply_preprocess_eval(
        model,
        manager,
        fp32_had=fp32_had,
        dev=rotation_dev,
    )

    # 5. Save the rotated model and tokenizer
    os.makedirs(save_directory, exist_ok=True)
    logger.info("Saving rotated model to %s ...", save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Embed rotation metadata into config.json
    import json

    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["fp32_had"] = fp32_had
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Rotation preprocessing complete. Saved to %s", save_directory)

    del model
    cleanup_memory()

    # Lazy import to avoid circular dependency
    from ..rotated_model_config import RotatedModelConfig

    return RotatedModelConfig(
        path=save_directory,
        dtype=model_config.dtype,
        device=model_config.device,
    )
