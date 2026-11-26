#!/usr/bin/env python3
"""Reproducible effectiveness benchmark for formal context compaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

from onecomp.__version__ import __version__
from onecomp.analyzer.compaction import build_run_report, save_run_report, visualize_run
from onecomp.compaction import CompactionConfig, CompactionRunner
from onecomp.compaction.adapters.messages import message_text
from onecomp.compaction.adapters.provider import GuardedProviderAdapter


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/compaction/formal-v1"))
    parser.add_argument("--observations", type=int, default=48)
    parser.add_argument("--body-chars", type=int, default=8000)
    parser.add_argument("--context-chars", type=int, default=24_000)
    parser.add_argument("--keep-recent", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args(argv)
    for name in ("observations", "body_chars", "context_chars"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.keep_recent < 0:
        parser.error("--keep-recent must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(
        output_dir,
        observations=args.observations,
        body_chars=args.body_chars,
        context_chars=args.context_chars,
        keep_recent=args.keep_recent,
        seed=args.seed,
    )
    result_path = output_dir / "benchmark-results.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": str(result_path), "metrics": result["metrics"]}, indent=2))


def run_benchmark(
    output_dir: Path,
    *,
    observations: int,
    body_chars: int,
    context_chars: int,
    seed: int,
    keep_recent: int = 0,
) -> dict[str, Any]:
    archive_path = output_dir / "archive.sqlite3"
    if archive_path.exists():
        raise ValueError(
            f"benchmark archive already exists: {archive_path}; select a new output directory"
        )
    config = CompactionConfig(
        profile="formal-v1",
        archive_path=archive_path,
        keep_recent=keep_recent,
        result_max_chars=500,
        summary_max_chars=min(12_000, context_chars // 2),
        max_recall_units=1024,
        recalls_per_step=max(8, observations),
        context_window=context_chars + 512,
        completion_reserve=256,
        serialization_margin=128,
    )
    rng = random.Random(seed)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Run the deterministic repository audit."},
        {"role": "user", "content": "Find every VERIFY marker and report exact values."},
    ]
    originals: dict[str, str] = {}
    write_latencies_ms: list[float] = []

    with CompactionRunner(config) as runner:
        for index in range(observations):
            runner.begin_step()
            command, body = _observation(rng, index, body_chars, seed)
            started = time.perf_counter_ns()
            record = runner.record_observation(
                command=command,
                content=body,
                source_key=f"benchmark-call-{index}",
                returncode=1 if index % 17 == 16 else 0,
                tags=["benchmark", _scenario(index)],
            )
            write_latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
            originals[record.obs_id] = body
            messages.extend(
                [
                    {"role": "assistant", "content": f"Running `{command}`"},
                    runner.make_observation_message(record),
                ]
            )

        # Retry the first host event. It must resolve to its original occurrence
        # without adding an archive row.
        first_id, first_body = next(iter(originals.items()))
        retry = runner.record_observation(
            command=_observation_command(0),
            content=first_body,
            source_key="benchmark-call-0",
            tags=["benchmark", _scenario(0)],
        )
        retry_idempotent = retry.obs_id == first_id

        baseline_chars = _request_chars({"messages": messages})
        provider_calls: list[int] = []
        adapter = GuardedProviderAdapter(runner, _request_chars)
        compact_started = time.perf_counter_ns()
        adapter.dispatch(
            {"model": "benchmark-provider", "messages": messages},
            lambda request: provider_calls.append(_request_chars(request)),
            compact_to_chars=context_chars,
        )
        compaction_ms = (time.perf_counter_ns() - compact_started) / 1e6
        compacted_chars = provider_calls[0]

        assert runner.archive is not None and runner.service is not None
        catalog_ids: list[str] = []
        cursor = None
        while True:
            page = runner.service.catalog(after_sequence=cursor, limit=13)
            catalog_ids.extend(item.obs_id for item in page.items)
            if not page.has_more:
                break
            cursor = page.next_cursor

        reconstruction_latencies_ms: list[float] = []
        exact = 0
        checksum_exact = 0
        needles_exact = 0
        for obs_id, original in originals.items():
            started = time.perf_counter_ns()
            reconstructed = "".join(
                chunk.text for chunk in runner.service.text_chunks(obs_id, max_units=1024)
            )
            reconstruction_latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
            exact += reconstructed == original
            checksum_exact += (
                hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
                == runner.archive.get_metadata(obs_id).content_sha256
            )
            marker = f"VERIFY-{seed}-{runner.archive.get_metadata(obs_id).sequence:04d}"
            needles_exact += marker in reconstructed

        first_record_id = next(iter(originals))
        first_metadata = runner.archive.get_metadata(first_record_id)
        total_service_chunks = max(1, (first_metadata.char_count + 1023) // 1024)
        chunk_recall_latencies_ms: list[float] = []
        service_pieces: list[str] = []
        sequential_started = time.perf_counter_ns()
        for index in range(total_service_chunks):
            started = time.perf_counter_ns()
            chunk = runner.service.recall_text_chunk(first_record_id, index, max_units=1024)
            chunk_recall_latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
            service_pieces.append(chunk.text)
        sequential_chunk_ms = (time.perf_counter_ns() - sequential_started) / 1e6

        search_marker = f"VERIFY-{seed}-{observations - 1:04d}"
        search_started = time.perf_counter_ns()
        search_exact = runner.archive.contains_value(search_marker)
        search_latency_ms = (time.perf_counter_ns() - search_started) / 1e6

        for obs_id in list(originals)[: min(4, observations)]:
            runner.handle_action(f"_recall_chunk §{obs_id} 0", [])

        report = build_run_report(runner, run_name="Formal compaction benchmark")
        save_run_report(report, output_dir / "run-report.json")
        dashboard_paths = visualize_run(report, output_dir / "run-dashboard")
        guard_audit = adapter.audit()

    return {
        "schema": "onecomp-compaction-benchmark-v1",
        "onecomp_version": __version__,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "corpus": {
            "seed": seed,
            "observations": observations,
            "body_chars_target": body_chars,
            "context_chars": context_chars,
            "keep_recent": keep_recent,
            "public_generated_data": True,
        },
        "metrics": {
            "baseline_chars": baseline_chars,
            "compacted_chars": compacted_chars,
            "compression_ratio": compacted_chars / baseline_chars,
            "saved_fraction": 1 - compacted_chars / baseline_chars,
            "exact_reconstruction_rate": exact / observations,
            "checksum_verification_rate": checksum_exact / observations,
            "needle_recovery_rate": needles_exact / observations,
            "catalog_completeness_rate": len(set(catalog_ids) & originals.keys()) / observations,
            "occurrence_count": len(set(originals)),
            "retry_idempotent": retry_idempotent,
            "provider_guard_coverage": (
                guard_audit.dispatched_requests / guard_audit.checked_requests
            ),
            "write_latency_ms_p50": statistics.median(write_latencies_ms),
            "write_latency_ms_p95": _percentile(write_latencies_ms, 0.95),
            "compaction_and_dispatch_ms": compaction_ms,
            "reconstruction_latency_ms_p50": statistics.median(reconstruction_latencies_ms),
            "reconstruction_latency_ms_p95": _percentile(reconstruction_latencies_ms, 0.95),
            "chunk_recall_latency_ms_p50": statistics.median(chunk_recall_latencies_ms),
            "chunk_recall_latency_ms_p95": _percentile(chunk_recall_latencies_ms, 0.95),
            "sequential_chunk_reconstruction_ms": sequential_chunk_ms,
            "sequential_chunk_reconstruction_exact": (
                "".join(service_pieces) == originals[first_record_id]
            ),
            "indexed_body_search_exact": search_exact,
            "indexed_body_search_latency_ms": search_latency_ms,
            "body_search_index_enabled": runner.archive.body_search_index_enabled,
        },
        "artifacts": {
            "run_report": str(output_dir / "run-report.json"),
            "dashboards": [str(path) for path in dashboard_paths],
            "archive": str(archive_path),
            "archive_contains_private_bodies": True,
        },
    }


def _observation(rng: random.Random, index: int, target_chars: int, seed: int) -> tuple[str, str]:
    command = _observation_command(index)
    marker = f"VERIFY-{seed}-{index:04d}"
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 日本語 λ→✓\n"  # jp-check: ignore
    prefix_length = max(0, target_chars // 2 - len(marker) // 2)
    prefix = "".join(rng.choice(alphabet) for _ in range(prefix_length))
    suffix = "".join(
        rng.choice(alphabet) for _ in range(max(0, target_chars - len(prefix) - len(marker)))
    )
    return command, prefix + marker + suffix


def _observation_command(index: int) -> str:
    scenario = _scenario(index)
    return f"{scenario} workspace/component_{index % 11}.txt --attempt {index}"


def _scenario(index: int) -> str:
    return ("build", "test", "inspect", "search", "trace")[index % 5]


def _request_chars(request: dict[str, Any]) -> int:
    return sum(len(message_text(message)) for message in request["messages"])


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


if __name__ == "__main__":
    main()
