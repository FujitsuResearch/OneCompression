"""Runtime helpers shared by post-quantization processes.

Copyright 2025-2026 Fujitsu Ltd.
"""

from typing import Iterable

from ..rotated_model_config import RotatedModelConfig
from ..utils.quant_config import validate_quantized_model_config

# Key under ``model.config.quantization_config`` that accumulates one audit
# entry per applied post-process, preserved across save/load cycles.
POST_PROCESS_HISTORY_KEY = "onecomp_post_processes"


def validate_rotated_checkpoint_consistency(quant_config: dict, model_config) -> None:
    """Guard against a rotated/non-rotated mismatch between checkpoint and config.

    A rotated checkpoint can only be post-processed (or re-saved) with a
    :class:`RotatedModelConfig`, and a non-rotated checkpoint only with a plain
    config.  Mixing the two would silently apply the wrong forward path
    (rotation matrices present/absent, or ``fp32_had`` on/off) and produce a
    subtly wrong model, so the mismatch is rejected up-front instead.

    Args:
        quant_config (dict):
            The checkpoint's ``quantization_config``; its ``rotated`` /
            ``fp32_had`` flags describe how it was produced.
        model_config:
            The ``ModelConfig`` supplied for the current run.

    Raises:
        RuntimeError: If the checkpoint's ``rotated`` flag disagrees with
            whether ``model_config`` is a ``RotatedModelConfig``, or if both
            are rotated but their ``fp32_had`` settings differ.
    """
    checkpoint_rotated = bool(quant_config.get("rotated", False))
    model_config_rotated = isinstance(model_config, RotatedModelConfig)

    if checkpoint_rotated and not model_config_rotated:
        raise RuntimeError(
            "Loaded checkpoint is marked as rotated, but model_config is not "
            "a RotatedModelConfig."
        )
    if not checkpoint_rotated and model_config_rotated:
        raise RuntimeError(
            "Loaded checkpoint is not marked as rotated, but model_config is "
            "a RotatedModelConfig."
        )
    if checkpoint_rotated:
        checkpoint_fp32_had = bool(quant_config.get("fp32_had", False))
        model_config_fp32_had = bool(getattr(model_config, "fp32_had", False))
        if checkpoint_fp32_had != model_config_fp32_had:
            raise RuntimeError(
                "Loaded checkpoint fp32_had does not match model_config.fp32_had "
                f"({checkpoint_fp32_had} != {model_config_fp32_had})."
            )


def prepare_quantized_model_for_post_process(model, model_config, context: str):
    """Validate a quantized model and return it on CPU for post-process input.

    Shared entry guard used by :meth:`PostQuantizationProcess.run` before the
    subclass body runs.  Post-processes assume their input lives on CPU with a
    well-formed ``quantization_config``, so this:

    1. requires a ``model_config`` (post-processes need tokenizer/device info),
    2. moves the model to CPU,
    3. validates ``model.config.quantization_config`` via
       :func:`validate_quantized_model_config`, and
    4. checks rotated-checkpoint consistency against ``model_config``.

    Args:
        model:
            The quantized model to feed into a post-process.
        model_config:
            The ``ModelConfig`` for the current run.
        context (str):
            Caller label surfaced in error messages.

    Returns:
        The same model, moved to CPU, ready for post-processing.

    Raises:
        RuntimeError: If ``model_config`` is ``None``, or if the
            rotated-checkpoint consistency check fails.
        ValueError: If ``model.config.quantization_config`` fails validation.
    """
    if model_config is None:
        raise RuntimeError(f"{context} requires model_config.")

    model = model.cpu()
    quant_config = validate_quantized_model_config(model, context)
    validate_rotated_checkpoint_consistency(quant_config, model_config)
    return model


def append_post_process_metadata(
    quant_config: dict,
    metadata_entries: Iterable[dict],
) -> bool:
    """Append post-process audit entries to ``quant_config`` in place.

    Entries accumulate under ``quant_config[POST_PROCESS_HISTORY_KEY]`` so the
    record of which post-processes were applied travels with the model and is
    persisted to ``config.json`` by the save path.  Appending (rather than
    overwriting) preserves history across repeated load → post-process →
    re-save cycles.

    Args:
        quant_config (dict):
            The model's ``quantization_config`` to update.
        metadata_entries (Iterable[dict]):
            Audit entries to append (typically the result of
            :meth:`PostQuantizationProcess.build_metadata`).

    Returns:
        bool: ``True`` if at least one entry was appended; ``False`` if
        ``metadata_entries`` was empty (``quant_config`` left untouched).

    Raises:
        ValueError: If an existing history value is present but is not a list
            (e.g. a hand-edited ``config.json``), rather than silently
            discarding it.
    """
    pending_metadata = list(metadata_entries)
    if not pending_metadata:
        return False

    existing_metadata = quant_config.get(POST_PROCESS_HISTORY_KEY, [])
    if existing_metadata is None:
        existing_metadata = []
    if not isinstance(existing_metadata, list):
        raise ValueError(f"quantization_config['{POST_PROCESS_HISTORY_KEY}'] must be a list.")

    quant_config[POST_PROCESS_HISTORY_KEY] = existing_metadata + pending_metadata
    return True
