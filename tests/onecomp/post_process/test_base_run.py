"""Unit tests for the ``PostQuantizationProcess.run`` template method.

Covers the orchestration contract of
:meth:`onecomp.post_process._base.PostQuantizationProcess.run` — not the input
guard, metadata accumulation or metadata generation tested elsewhere, but the
way ``run`` composes them around the subclass body ``_run``:

- audit metadata is appended only after ``_run`` succeeds,
- a failing input guard does not call ``_run`` or append metadata,
- a failing ``_run`` propagates and leaves no metadata behind,
- a post-``_run`` validation failure leaves no metadata behind,
- after ``_run`` starts, the model is restored to ``eval()`` on CPU whether
  ``_run`` succeeds or raises,
- repeated ``run`` calls accumulate history in order.

The subclass algorithm body (``_run``) itself is out of scope; the doubles here
implement only the minimal ``_run`` needed to exercise ``run``'s control flow.

Copyright 2025-2026 Fujitsu Ltd.
"""

from dataclasses import dataclass

import pytest

from onecomp.post_process._base import PostQuantizationProcess
from onecomp.post_process._runtime import POST_PROCESS_HISTORY_KEY
from tests.onecomp.fixtures.quant_config import valid_quant_config
from tests.onecomp.post_process._doubles import FakeModel, PlainModelConfig

# ---------------------------------------------------------------------------
# Process doubles
# ---------------------------------------------------------------------------


@dataclass
class _RecordingProcess(PostQuantizationProcess):
    """A post-process whose ``_run`` succeeds and records that it ran."""

    ran: bool = False

    def _run(self, quantized_model, model_config) -> None:
        self.ran = True


@dataclass
class _FailingProcess(PostQuantizationProcess):
    """A post-process whose ``_run`` always raises."""

    def _run(self, quantized_model, model_config) -> None:
        raise RuntimeError("boom")


@dataclass
class _InvalidatingProcess(PostQuantizationProcess):
    """A post-process that succeeds but corrupts quantization_config."""

    ran: bool = False

    def _run(self, quantized_model, model_config) -> None:
        self.ran = True
        quantized_model.config.quantization_config.pop("quant_method")


@dataclass
class _MovesToCudaProcess(PostQuantizationProcess):
    """A post-process that leaves the model on a GPU-like device."""

    fail: bool = False

    def _run(self, quantized_model, model_config) -> None:
        quantized_model.cuda()
        if self.fail:
            raise RuntimeError("boom")


# ===========================================================================
# PostQuantizationProcess.run
# ===========================================================================


def test_run_invokes_subclass_run():
    """``run`` invokes the subclass ``_run`` after the input guard."""
    process = _RecordingProcess()
    process.run(FakeModel(valid_quant_config()), PlainModelConfig())
    assert process.ran is True


def test_run_does_not_invoke_subclass_run_when_guard_fails():
    """A failing input guard prevents ``_run`` and metadata append."""
    model = FakeModel(valid_quant_config())
    process = _RecordingProcess()
    with pytest.raises(RuntimeError):
        process.run(model, None)
    assert process.ran is False
    assert POST_PROCESS_HISTORY_KEY not in model.config.quantization_config


def test_run_appends_metadata_on_success():
    """Exactly one audit entry is appended after ``_run`` succeeds."""
    model = FakeModel(valid_quant_config())
    _RecordingProcess(name="step-A").run(model, PlainModelConfig())
    history = model.config.quantization_config[POST_PROCESS_HISTORY_KEY]
    assert [entry["name"] for entry in history] == ["step-A"]


def test_run_does_not_append_metadata_on_failure():
    """A failing ``_run`` propagates and appends no metadata."""
    model = FakeModel(valid_quant_config())
    with pytest.raises(RuntimeError):
        _FailingProcess().run(model, PlainModelConfig())
    # The failed run must leave no audit entry behind.
    assert POST_PROCESS_HISTORY_KEY not in model.config.quantization_config


def test_run_does_not_append_metadata_when_post_run_validation_fails():
    """Post-``_run`` validation failure prevents metadata append."""
    model = FakeModel(valid_quant_config())
    process = _InvalidatingProcess()
    with pytest.raises(ValueError):
        process.run(model, PlainModelConfig())
    assert process.ran is True
    assert POST_PROCESS_HISTORY_KEY not in model.config.quantization_config


def test_run_restores_model_to_cpu_and_eval_on_success():
    """On success the model is restored to ``eval()`` on CPU."""
    model = FakeModel(valid_quant_config())
    _MovesToCudaProcess().run(model, PlainModelConfig())
    assert model.device == "cpu"
    assert model.eval_calls >= 1


def test_run_restores_model_to_cpu_and_eval_on_failure():
    """After ``_run`` starts, failure still restores ``eval()`` on CPU."""
    model = FakeModel(valid_quant_config())
    with pytest.raises(RuntimeError):
        _MovesToCudaProcess(fail=True).run(model, PlainModelConfig())
    # The finally block restores the model even though _run raised.
    assert model.device == "cpu"
    assert model.eval_calls >= 1


def test_run_accumulates_metadata_across_multiple_processes():
    """Multiple ``run`` calls accumulate history in order (regression)."""
    model = FakeModel(valid_quant_config())
    model_config = PlainModelConfig()
    _RecordingProcess(name="first").run(model, model_config)
    _RecordingProcess(name="second").run(model, model_config)
    history = model.config.quantization_config[POST_PROCESS_HISTORY_KEY]
    assert [entry["name"] for entry in history] == ["first", "second"]
