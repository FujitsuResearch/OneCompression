"""

Copyright 2025-2026 Fujitsu Ltd.

"""

import importlib.util
import logging
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "quantization_progress",
    _REPO_ROOT / "onecomp" / "utils" / "quantization_progress.py",
)
_quant_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_quant_mod)
QuantizationProgressTracker = _quant_mod.QuantizationProgressTracker


def test_step_complete_logs_fraction_and_eta(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_eta")
    tracker = QuantizationProgressTracker(logger, total_steps=2, label="Test layers")

    with patch.object(_quant_mod.time, "monotonic", side_effect=[0.0, 10.0, 30.0]):
        tracker.step_complete("layer_a")
        tracker.step_complete("layer_b")

    joined = " ".join(r.message for r in caplog.records)
    assert "1/2" in joined
    assert "2/2" in joined
    assert "[progress]" in joined
    assert "ETA" in joined


def test_thread_safe_reaches_total():
    logger = logging.getLogger("test_progress_thread")
    logger.addHandler(logging.NullHandler())
    tracker = QuantizationProgressTracker(
        logger, total_steps=100, label="Parallel", thread_safe=True
    )

    def worker():
        for _ in range(10):
            tracker.step_complete()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.done == 100


def test_zero_total_no_crash():
    logger = logging.getLogger("test_progress_zero")
    logger.addHandler(logging.NullHandler())
    tracker = QuantizationProgressTracker(logger, total_steps=0, label="Empty")
    tracker.step_complete()  # should not raise


def test_eta_unknown_until_first_step(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_first")
    tracker = QuantizationProgressTracker(logger, total_steps=3, label="Layers")

    with patch.object(_quant_mod.time, "monotonic", side_effect=[0.0, 5.0]):
        tracker.step_complete("first")

    first_line = caplog.records[0].message
    assert "1/3" in first_line
    assert "ETA" in first_line
