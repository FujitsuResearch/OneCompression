"""

Copyright 2025-2026 Fujitsu Ltd.

"""

import importlib.util
import logging
import threading
from pathlib import Path
from unittest.mock import patch

# Load the module directly from its file path to avoid importing the rest of
# ``onecomp.utils`` (which pulls in torch and other heavy dependencies that
# are not required for these unit tests).
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


def test_step_complete_with_detail_appends_suffix(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_detail")
    tracker = QuantizationProgressTracker(logger, total_steps=1, label="Layers")
    tracker.step_complete("layer_xyz")

    assert caplog.records[-1].message.endswith("(layer_xyz)")


def test_step_complete_without_detail_has_no_suffix(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_no_detail")
    tracker = QuantizationProgressTracker(logger, total_steps=1, label="Layers")
    tracker.step_complete()

    msg = caplog.records[-1].message
    assert not msg.endswith(")")
    assert "()" not in msg


def test_elapsed_and_eta_use_expected_format(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_format")

    # Patch monotonic to control both __init__ (_start=0.0) and step_complete (now=10.0, 30.0).
    with patch.object(_quant_mod.time, "monotonic", side_effect=[0.0, 10.0, 30.0]):
        tracker = QuantizationProgressTracker(logger, total_steps=2, label="Layers")
        tracker.step_complete()
        tracker.step_complete()

    msgs = [r.message for r in caplog.records]
    assert msgs[0] == "[progress] Layers: 1/2 (50.0%) elapsed=10s ETA=10s"
    assert msgs[1] == "[progress] Layers: 2/2 (100.0%) elapsed=30s ETA=0s"


def test_reaches_total_logs_zero_eta(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_complete")
    tracker = QuantizationProgressTracker(logger, total_steps=3, label="Layers")

    for _ in range(3):
        tracker.step_complete()

    final = caplog.records[-1].message
    assert "3/3" in final
    assert "(100.0%)" in final
    assert "ETA=0s" in final


def test_overflow_calls_are_suppressed(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_overflow")
    tracker = QuantizationProgressTracker(logger, total_steps=2, label="Layers")

    for _ in range(4):
        tracker.step_complete()

    progress_lines = [r for r in caplog.records if "[progress]" in r.message]
    # Only the first 2 calls (== _total) emit log lines; the rest are silently dropped.
    assert len(progress_lines) == 2
    # Internal counter still reflects every call for observability.
    assert tracker.done == 4


def test_thread_safe_log_count_matches_total(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_thread_log")
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

    progress_lines = [r for r in caplog.records if "[progress]" in r.message]
    assert len(progress_lines) == 100
    assert tracker.done == 100


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


def test_first_step_logs_concrete_eta(caplog):
    """First completed step yields a concrete ETA value (never the historical "unknown")."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_progress_first")

    with patch.object(_quant_mod.time, "monotonic", side_effect=[0.0, 5.0]):
        tracker = QuantizationProgressTracker(logger, total_steps=3, label="Layers")
        tracker.step_complete("first")

    first_line = caplog.records[0].message
    assert "1/3" in first_line
    assert "elapsed=5s" in first_line
    assert "ETA=10s" in first_line  # (elapsed / done) * (total - done) = (5 / 1) * 2 = 10
    assert "unknown" not in first_line
