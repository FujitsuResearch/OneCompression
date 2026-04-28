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
import time
from logging import getLogger
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM

from ..calibration import CalibrationConfig
from ..model_config import ModelConfig
from .rotation_utils import cleanup_memory

if TYPE_CHECKING:
    from ..rotated_model_config import RotatedModelConfig


logger = getLogger(__name__)

_VALID_ROTATION_MODES = ("random_hadamard", "hadamard", "random", "identity")
_VALID_SCALING_MODES = ("identity", "random_ones", "random")
_VALID_CALIBRATION_STRATEGIES = ("concat_chunk", "concat_chunk_align", "drop_head", "drop_rand")


def _validate_prepare_rotated_model_params(
    *,
    rotation,
    scaling,
    rotation_mode,
    scaling_mode,
    seed,
    enable_training,
    calibration_config: CalibrationConfig,
    wbits,
    sym,
    groupsize,
    mse,
    norm,
    grid,
    fp32_had,
    use_sdpa,
    training_args_override,
):
    """Validate parameters for :func:`prepare_rotated_model`.

    Accepts all keyword arguments of :func:`prepare_rotated_model`.
    Boolean and complex-type parameters are received for completeness
    but only enum/numeric parameters are range-checked.

    Raises:
        ValueError: If any parameter is out of its valid range.
    """
    bad = []

    if rotation_mode not in _VALID_ROTATION_MODES:
        bad.append(
            f"Invalid rotation_mode: {rotation_mode!r} "
            f"(expected one of {_VALID_ROTATION_MODES})."
        )

    if scaling_mode not in _VALID_SCALING_MODES:
        bad.append(
            f"Invalid scaling_mode: {scaling_mode!r} " f"(expected one of {_VALID_SCALING_MODES})."
        )

    calibration_strategy = calibration_config.strategy
    if calibration_strategy not in _VALID_CALIBRATION_STRATEGIES:
        bad.append(
            f"Invalid calibration_strategy: {calibration_strategy!r} "
            f"(expected one of {_VALID_CALIBRATION_STRATEGIES})."
        )

    if not (isinstance(wbits, int) and 1 <= wbits <= 64):
        bad.append(f"Invalid wbits: {wbits!r} (expected int in 1..64).")

    if not (isinstance(groupsize, int) and (groupsize == -1 or groupsize >= 1)):
        bad.append(f"Invalid groupsize: {groupsize!r} (expected int: -1 or >= 1).")

    num_calibration_samples = calibration_config.num_calibration_samples
    if not (isinstance(num_calibration_samples, int) and num_calibration_samples >= 1):
        bad.append(
            f"Invalid num_calibration_samples: {num_calibration_samples!r} "
            f"(expected int >= 1)."
        )

    max_length = calibration_config.max_length
    if not (isinstance(max_length, int) and max_length >= 1):
        bad.append(f"Invalid max_length: {max_length!r} (expected int >= 1).")

    if not (isinstance(seed, int) and seed >= 0):
        bad.append(f"Invalid seed: {seed!r} (expected int >= 0).")

    if mse:
        if not (isinstance(grid, int) and grid >= 1):
            bad.append(f"Invalid grid: {grid!r} (expected int >= 1 when mse=True).")

        if not (isinstance(norm, (int, float)) and norm > 0):
            bad.append(f"Invalid norm: {norm!r} (expected numeric > 0 when mse=True).")

    if bad:
        raise ValueError("; ".join(bad))


def prepare_rotated_model(
    model_config: ModelConfig,
    save_directory: str,
    *,
    rotation: bool = True,
    scaling: bool = False,
    rotation_mode: str = "random_hadamard",
    scaling_mode: str = "identity",
    seed: int = 0,
    enable_training: bool = True,
    calibration_config: CalibrationConfig | None = None,
    wbits: int = 4,
    sym: bool = False,
    groupsize: int = -1,
    mse: bool = False,
    norm: float = 2.4,
    grid: int = 100,
    fp32_had: bool = False,
    use_sdpa: bool = False,
    training_args_override: dict | None = None,
) -> RotatedModelConfig:
    """Optionally train rotation/scaling matrices, apply them to model weights, and save.

    Args:
        model_config: Original model configuration (``model_id`` or ``path``).
        save_directory: Directory to save the rotated model.
        rotation: Whether to apply rotation matrices (R1, R2).
        scaling: Whether to apply scaling diagonals (S_*).
        rotation_mode: ``"random_hadamard"`` | ``"hadamard"`` | ``"random"`` | ``"identity"``.
        scaling_mode: ``"identity"`` | ``"random_ones"`` | ``"random"``.
        seed: Random seed for rotation matrix initialisation and
            calibration data preparation.  Note that the Trainer uses a
            separate seed (``TrainingArguments.seed``, default ``42``) for
            data shuffling and training reproducibility.
        enable_training: If ``True``, train the rotation/scaling matrices;
            otherwise use the randomly initialised matrices directly.
        calibration_config: Calibration data configuration.  When ``None``
            (default), a :class:`CalibrationConfig` with default values
            is created automatically (``calibration_dataset="c4"``,
            ``max_length=2048``, ``num_calibration_samples=512``,
            ``strategy="drop_rand"``).
            See :class:`~onecomp.calibration.CalibrationConfig`.
        wbits: Weight quantisation bit-width for the RTN proxy during
            training (default: 4).  Should match the quantizer's ``wbits``.
        sym: Symmetric quantisation for the RTN proxy.
        groupsize: Group size for the RTN proxy (``-1`` = per-channel,
            default: -1).  Should match the quantizer's ``groupsize``.
            When positive, the value must evenly divide the
            ``out_features`` of every ``nn.Linear`` layer in the model.
        mse: Enable MSE grid search for optimal clipping in the RTN
            proxy during training.
        norm: Lp norm exponent for the MSE grid search (default: 2.4).
        grid: Number of candidate shrink levels for the MSE grid search
            (default: 100).
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
    if calibration_config is None:
        calibration_config = CalibrationConfig()
    from .train_rotation import (
        PreprocessManager,
        apply_preprocess_eval,
        apply_preprocess_train,
    )
    from ..calibration import prepare_calibration_dataset

    _validate_prepare_rotated_model_params(
        rotation=rotation,
        scaling=scaling,
        rotation_mode=rotation_mode,
        scaling_mode=scaling_mode,
        seed=seed,
        enable_training=enable_training,
        calibration_config=calibration_config,
        wbits=wbits,
        sym=sym,
        groupsize=groupsize,
        mse=mse,
        norm=norm,
        grid=grid,
        fp32_had=fp32_had,
        use_sdpa=use_sdpa,
        training_args_override=training_args_override,
    )

    total_start_time = time.time()

    model_path = model_config.get_model_id_or_path()

    # 1. Load the original model and tokenizer
    logger.info("Loading original model from %s ...", model_path)
    t0 = time.time()
    model = model_config.load_model()
    tokenizer = model_config.load_tokenizer()
    logger.info("Model loading done (%.2f s)", time.time() - t0)

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
        t0 = time.time()
        calibration_inputs = prepare_calibration_dataset(
            tokenizer=tokenizer,
            device="cpu",
            calibration_config=calibration_config,
            model=model,
        )
        logger.info("Calibration data preparation done (%.2f s)", time.time() - t0)

        # Release the original model to halve peak CPU memory during training.
        # Training only needs config and model_type (already extracted above);
        # model weights are re-loaded from model_path inside the training pipeline.
        logger.info("Releasing original model to free memory for training ...")
        del model
        cleanup_memory()

        logger.info("Starting rotation training ...")
        t0 = time.time()
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
            mse=mse,
            norm=norm,
            grid=grid,
            use_sdpa=use_sdpa,
            fp32_had=fp32_had,
            training_args_override=training_args_override,
        )
        logger.info("Rotation training done (%.2f s)", time.time() - t0)

        # Re-load the original model on CPU for rotation application.
        logger.info("Re-loading original model on CPU ...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        logger.info("Model re-loading done (%.2f s)", time.time() - t0)

    # 4. Apply rotation to model weights (eval path)
    rotation_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    apply_preprocess_eval(
        model,
        manager,
        fp32_had=fp32_had,
        dev=rotation_dev,
    )
    logger.info("Rotation application done (%.2f s)", time.time() - t0)

    # 5. Save the rotated model and tokenizer
    os.makedirs(save_directory, exist_ok=True)
    logger.info("Saving rotated model to %s ...", save_directory)
    t0 = time.time()
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    logger.info("Model saving done (%.2f s)", time.time() - t0)

    # Embed rotation metadata into config.json
    import json

    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["fp32_had"] = fp32_had
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    total_elapsed = time.time() - total_start_time
    logger.info(
        "Rotation preprocessing complete (total: %.2f s). Saved to %s",
        total_elapsed,
        save_directory,
    )

    del model
    cleanup_memory()

    # Lazy import to avoid circular dependency
    from ..rotated_model_config import RotatedModelConfig

    return RotatedModelConfig(
        path=save_directory,
        dtype=model_config.dtype,
        device=model_config.device,
    )
