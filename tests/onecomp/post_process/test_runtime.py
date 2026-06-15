"""Unit tests for post-process runtime helpers.

Covers the shared runtime helpers in :mod:`onecomp.post_process._runtime`:

- :func:`validate_rotated_checkpoint_consistency` — rotated/non-rotated and
  ``fp32_had`` consistency between checkpoint and ``model_config``.
- :func:`prepare_quantized_model_for_post_process` — input guard that
  requires ``model_config``, moves the model to CPU, and runs the schema and
  rotated-consistency checks.
- :func:`append_post_process_metadata` — ordered, non-destructive
  accumulation of audit metadata.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from onecomp.post_process._runtime import (
    POST_PROCESS_HISTORY_KEY,
    append_post_process_metadata,
    prepare_quantized_model_for_post_process,
    validate_rotated_checkpoint_consistency,
)
from tests.onecomp.fixtures.quant_config import valid_quant_config
from tests.onecomp.post_process._doubles import (
    FakeModel,
    PlainModelConfig,
    make_rotated_model_config,
)


# ===========================================================================
# validate_rotated_checkpoint_consistency
# ===========================================================================


def test_rotated_checkpoint_rejects_plain_model_config():
    """A rotated checkpoint paired with a plain model_config is rejected."""
    quant_config = {"rotated": True}
    with pytest.raises(RuntimeError):
        validate_rotated_checkpoint_consistency(quant_config, PlainModelConfig())


def test_plain_checkpoint_rejects_rotated_model_config():
    """A plain checkpoint paired with a RotatedModelConfig is rejected."""
    quant_config = {"rotated": False}
    with pytest.raises(RuntimeError):
        validate_rotated_checkpoint_consistency(
            quant_config, make_rotated_model_config()
        )


def test_rotated_checkpoint_rejects_fp32_had_mismatch():
    """Two rotated sides whose ``fp32_had`` differ are rejected."""
    quant_config = {"rotated": True, "fp32_had": True}
    with pytest.raises(RuntimeError):
        validate_rotated_checkpoint_consistency(
            quant_config, make_rotated_model_config(fp32_had=False)
        )


@pytest.mark.parametrize("fp32_had", [True, False], ids=["fp32", "fp16"])
def test_rotated_checkpoint_accepts_matching_fp32_had(fp32_had):
    """Matching ``fp32_had`` (True/True or False/False) passes."""
    quant_config = {"rotated": True, "fp32_had": fp32_had}
    # Should not raise.
    validate_rotated_checkpoint_consistency(
        quant_config, make_rotated_model_config(fp32_had=fp32_had)
    )


def test_plain_checkpoint_accepts_plain_model_config():
    """Both checkpoint and model_config non-rotated passes."""
    quant_config = {"rotated": False}
    # Should not raise.
    validate_rotated_checkpoint_consistency(quant_config, PlainModelConfig())


def test_missing_rotated_keys_default_to_plain():
    """An absent ``rotated`` key is treated as non-rotated."""
    quant_config = {}  # no 'rotated' key
    # Should not raise: absent 'rotated' is treated as non-rotated.
    validate_rotated_checkpoint_consistency(quant_config, PlainModelConfig())


def test_missing_rotated_keys_reject_rotated_model_config():
    """A key-less checkpoint paired with a RotatedModelConfig is rejected."""
    quant_config = {}  # no 'rotated' key -> treated as plain
    with pytest.raises(RuntimeError):
        validate_rotated_checkpoint_consistency(
            quant_config, make_rotated_model_config()
        )


def test_missing_fp32_had_defaults_false_for_rotated_checkpoint():
    """An absent ``fp32_had`` defaults to False for a rotated checkpoint."""
    quant_config = {"rotated": True}  # no 'fp32_had' key -> defaults to False
    # Matches a rotated model_config with fp32_had=False, so no error.
    validate_rotated_checkpoint_consistency(
        quant_config, make_rotated_model_config(fp32_had=False)
    )


# ===========================================================================
# prepare_quantized_model_for_post_process
# ===========================================================================


def test_prepare_requires_model_config():
    """A None model_config raises before any side effect on the model."""
    model = FakeModel(valid_quant_config())
    with pytest.raises(RuntimeError):
        prepare_quantized_model_for_post_process(model, None, context="ctx")
    # The guard fires before any side effect on the model.
    assert model.cpu_called is False


def test_prepare_moves_model_to_cpu():
    """The model is moved to CPU and returned unchanged."""
    model = FakeModel(valid_quant_config())
    result = prepare_quantized_model_for_post_process(
        model, PlainModelConfig(), context="ctx"
    )
    assert model.cpu_called is True
    assert result is model


def test_prepare_rejects_invalid_quant_config():
    """An invalid quant_config is rejected (delegates to schema validation)."""
    # Missing 'quant_method' -> fails schema validation.
    model = FakeModel({"modules_in_block_to_quantize": []})
    with pytest.raises(ValueError):
        prepare_quantized_model_for_post_process(
            model, PlainModelConfig(), context="ctx"
        )


def test_prepare_rejects_rotated_mismatch():
    """A rotated mismatch is rejected (delegates to the consistency check)."""
    # Schema-valid but rotated checkpoint paired with a plain model_config.
    model = FakeModel(valid_quant_config(rotated=True))
    with pytest.raises(RuntimeError):
        prepare_quantized_model_for_post_process(
            model, PlainModelConfig(), context="ctx"
        )


def test_prepare_returns_model_for_valid_input():
    """A valid input returns the same model."""
    model = FakeModel(valid_quant_config(rotated=False))
    result = prepare_quantized_model_for_post_process(
        model, PlainModelConfig(), context="ctx"
    )
    assert result is model


# ===========================================================================
# append_post_process_metadata
# ===========================================================================


def test_append_no_entries_returns_false_without_mutation():
    """Empty input returns False and creates no history key."""
    quant_config = {}
    result = append_post_process_metadata(quant_config, [])
    assert result is False
    assert POST_PROCESS_HISTORY_KEY not in quant_config


def test_append_creates_history_when_absent():
    """A new history list is created when the key is absent."""
    quant_config = {}
    entry = {"name": "A"}
    result = append_post_process_metadata(quant_config, [entry])
    assert result is True
    assert quant_config[POST_PROCESS_HISTORY_KEY] == [entry]


def test_append_treats_existing_none_as_empty_history():
    """An existing None history is treated as empty."""
    quant_config = {POST_PROCESS_HISTORY_KEY: None}
    entry = {"name": "A"}
    result = append_post_process_metadata(quant_config, [entry])
    assert result is True
    assert quant_config[POST_PROCESS_HISTORY_KEY] == [entry]


def test_append_rejects_non_list_history():
    """A non-list existing history is rejected with ValueError."""
    quant_config = {POST_PROCESS_HISTORY_KEY: "corrupted"}
    with pytest.raises(ValueError):
        append_post_process_metadata(quant_config, [{"name": "A"}])


def test_append_accumulates_entries_in_order():
    """Entries are appended after existing history, in order."""
    quant_config = {POST_PROCESS_HISTORY_KEY: [{"name": "A"}]}
    append_post_process_metadata(quant_config, [{"name": "B"}, {"name": "C"}])
    assert quant_config[POST_PROCESS_HISTORY_KEY] == [
        {"name": "A"},
        {"name": "B"},
        {"name": "C"},
    ]


def test_append_accumulates_across_multiple_calls():
    """Repeated calls accumulate history in order across save/load cycles."""
    quant_config = {}
    append_post_process_metadata(quant_config, [{"name": "blockwise1"}])
    append_post_process_metadata(quant_config, [{"name": "blockwise2"}])
    assert quant_config[POST_PROCESS_HISTORY_KEY] == [
        {"name": "blockwise1"},
        {"name": "blockwise2"},
    ]


def test_append_records_supplied_metadata_verbatim():
    """Supplied metadata is stored verbatim (the exact object, unmodified)."""
    quant_config = {}
    entry = {"name": "A", "config": {"lr": 1e-4}}
    append_post_process_metadata(quant_config, [entry])
    # The stored object is the exact one supplied, unmodified.
    assert quant_config[POST_PROCESS_HISTORY_KEY][0] is entry


def test_append_accepts_metadata_iterable():
    """A non-list iterable (generator) is accepted."""
    quant_config = {}
    entries = (e for e in [{"name": "A"}, {"name": "B"}])  # generator
    result = append_post_process_metadata(quant_config, entries)
    assert result is True
    assert quant_config[POST_PROCESS_HISTORY_KEY] == [{"name": "A"}, {"name": "B"}]


def test_append_returns_true_when_entries_added():
    """Appending at least one entry returns True."""
    quant_config = {}
    result = append_post_process_metadata(quant_config, [{"name": "A"}])
    assert result is True


def test_append_does_not_mutate_existing_list_in_place():
    """A new list is reassigned, leaving the previously held reference intact."""
    original_history = [{"name": "A"}]
    quant_config = {POST_PROCESS_HISTORY_KEY: original_history}
    append_post_process_metadata(quant_config, [{"name": "B"}])
    # A new list is reassigned; the previously held reference is untouched.
    assert quant_config[POST_PROCESS_HISTORY_KEY] is not original_history
    assert original_history == [{"name": "A"}]
