"""The public compaction benchmark stays reproducible and body-safe."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parents[3] / "benchmark" / "compaction" / "run_benchmark.py"


def _module():
    spec = importlib.util.spec_from_file_location("onecomp_compaction_benchmark", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_benchmark_proves_exact_recovery_and_writes_visuals(tmp_path):
    result = _module().run_benchmark(
        tmp_path,
        observations=8,
        body_chars=700,
        context_chars=3000,
        seed=17,
    )
    metrics = result["metrics"]
    assert metrics["exact_reconstruction_rate"] == 1.0
    assert metrics["checksum_verification_rate"] == 1.0
    assert metrics["needle_recovery_rate"] == 1.0
    assert metrics["catalog_completeness_rate"] == 1.0
    assert metrics["retry_idempotent"] is True
    assert metrics["provider_guard_coverage"] == 1.0
    assert metrics["indexed_body_search_exact"] is True
    assert metrics["chunk_recall_latency_ms_p95"] >= 0
    assert metrics["sequential_chunk_reconstruction_exact"] is True
    assert metrics["sequential_chunk_reconstruction_ms"] >= 0
    assert metrics["compression_ratio"] < 1.0
    assert result["corpus"]["keep_recent"] == 0
    report_text = (tmp_path / "run-report.json").read_text(encoding="utf-8")
    assert "VERIFY-17" not in report_text
    assert all(Path(path).is_file() for path in result["artifacts"]["dashboards"])
    json.dumps(result)
