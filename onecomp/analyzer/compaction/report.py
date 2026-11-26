"""
Stable, body-free report schema for one compaction run.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from ...compaction import CompactionRunner

REPORT_SCHEMA = "onecomp-compaction-run-v1"
_MAX_REPORT_ROWS = 100_000
_REPORT_FIELDS = {
    "schema",
    "run_name",
    "generated_ns",
    "profile",
    "config",
    "status",
    "metrics",
    "observations",
    "events",
    "privacy",
}
_CONFIG_FIELDS = {
    "profile",
    "keep_recent",
    "result_max_chars",
    "summary_max_chars",
    "max_recall_units",
    "recalls_per_step",
    "max_archive_bytes",
    "verify_archive_reads",
    "durability",
    "context_window",
    "completion_reserve",
    "serialization_margin",
}
_LEGACY_CONFIG_FIELDS = _CONFIG_FIELDS - {"durability"}
_STATUS_FIELDS = {"profile", "step", "archive", "telemetry_events"}
_ARCHIVE_STATUS_FIELDS = {
    "path",
    "observations",
    "body_bytes",
    "recalled_observations",
    "recall_count",
}
_PRIVACY_FIELDS = {"observation_bodies_included", "free_form_extra_included"}
_EVENT_FIELDS = {
    "kind",
    "step",
    "timestamp_ns",
    "obs_id",
    "chars",
    "returncode",
    "before_messages",
    "after_messages",
    "before_chars",
    "after_chars",
    "enforced_limit",
    "before_tokens",
    "after_tokens",
    "enforced_token_limit",
    "action_kind",
    "exact",
}
_OBSERVATION_FIELDS = {
    "sequence",
    "obs_id",
    "source_key_present",
    "cmd_signature",
    "returncode",
    "created_step",
    "created_ns",
    "content_sha256",
    "char_count",
    "line_count",
    "byte_count",
    "last_recalled",
    "n_recalls",
    "tags",
    "body_redacted",
}
_METRIC_FIELDS = {
    "observations",
    "archived_chars",
    "archived_bytes",
    "compactions",
    "recall_actions",
    "exact_recall_actions",
    "recalled_observations",
    "cumulative_before_chars",
    "cumulative_after_chars",
    "cumulative_saved_chars",
    "successful_observations",
    "checksum_coverage",
}


def build_run_report(
    runner: CompactionRunner,
    *,
    run_name: str = "compaction-run",
) -> dict[str, Any]:
    """Capture one runner without observation bodies or free-form extras."""
    if not isinstance(runner, CompactionRunner):
        raise TypeError("runner must be a CompactionRunner")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string")
    if len(run_name.strip()) > 120:
        raise ValueError("run_name must not exceed 120 characters")
    status = _safe_status(runner.status())
    observations: list[dict[str, Any]] = []
    if runner.archive is not None:
        if runner.archive.n_records > _MAX_REPORT_ROWS:
            raise ValueError(f"run report observations must not exceed {_MAX_REPORT_ROWS}")
        for row in runner.archive.iter_serializable(include_bodies=False):
            observations.append(
                {
                    "sequence": row["sequence"],
                    "obs_id": row["obs_id"],
                    "source_key_present": row["source_key"] is not None,
                    "cmd_signature": str(row["cmd_signature"])[:200],
                    "returncode": row["returncode"],
                    "created_step": row["created_step"],
                    "created_ns": row["created_ns"],
                    "content_sha256": row["content_sha256"],
                    "char_count": row["char_count"],
                    "line_count": row["line_count"],
                    "byte_count": row["byte_count"],
                    "last_recalled": row["last_recalled"],
                    "n_recalls": row["n_recalls"],
                    "tags": list(row["tags"]),
                    "body_redacted": True,
                }
            )
    event_snapshot = runner.events
    if len(event_snapshot) > _MAX_REPORT_ROWS:
        raise ValueError(f"run report events must not exceed {_MAX_REPORT_ROWS}")
    events = [
        {key: value for key, value in event.items() if key in _EVENT_FIELDS}
        for event in event_snapshot
    ]
    metrics = _metrics(observations, events)
    return {
        "schema": REPORT_SCHEMA,
        "run_name": run_name.strip(),
        "generated_ns": time.time_ns(),
        "profile": runner.profile,
        "config": _safe_config(runner),
        "status": status,
        "metrics": metrics,
        "observations": observations,
        "events": events,
        "privacy": {
            "observation_bodies_included": False,
            "free_form_extra_included": False,
        },
    }


def save_run_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and atomically save a private deterministic UTF-8 report."""
    normalized = validate_run_report(report)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def load_run_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned, body-free report."""
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return validate_run_report(value)


def validate_run_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the body-free fields consumed by the single-run renderer."""
    if not isinstance(value, Mapping):
        raise TypeError("run report must be a mapping")
    normalized = dict(value)
    _reject_unknown(normalized, _REPORT_FIELDS, "run report")
    missing_report_fields = _REPORT_FIELDS - set(normalized)
    if missing_report_fields:
        raise ValueError(f"run report is missing fields: {sorted(missing_report_fields)}")
    if normalized.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"unsupported run report schema: {normalized.get('schema')!r}")
    _require_string("run_name", normalized.get("run_name"), minimum=1, maximum=120)
    _require_integer("generated_ns", normalized.get("generated_ns"), minimum=0)
    profile = normalized.get("profile")
    if profile not in {"formal-v1", "experimental-v1"}:
        raise ValueError("run report profile must be 'formal-v1' or 'experimental-v1'")
    config = normalized.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("run report config must be an object")
    # ``durability`` was added additively to the v1 report. Keep reports made
    # by earlier OneComp builds readable and normalize them to the safest mode.
    if set(config) == _LEGACY_CONFIG_FIELDS:
        config = dict(config)
        config["durability"] = "full"
        normalized["config"] = config
    _reject_unknown(config, _CONFIG_FIELDS, "run report config")
    if set(config) != _CONFIG_FIELDS:
        raise ValueError(
            f"run report config is missing fields: {sorted(_CONFIG_FIELDS - set(config))}"
        )
    if config.get("profile") != profile:
        raise ValueError("run report config.profile must match profile")
    for name in (
        "keep_recent",
        "result_max_chars",
        "summary_max_chars",
        "recalls_per_step",
        "max_archive_bytes",
        "completion_reserve",
        "serialization_margin",
    ):
        _require_integer(f"config.{name}", config.get(name), minimum=0)
    _require_integer("config.max_recall_units", config.get("max_recall_units"), minimum=1)
    if not isinstance(config.get("verify_archive_reads"), bool):
        raise ValueError("run report config.verify_archive_reads must be a boolean")
    if config.get("durability") not in {"full", "normal"}:
        raise ValueError("run report config.durability must be 'full' or 'normal'")
    context_window = config.get("context_window")
    if context_window is not None:
        _require_integer("config.context_window", context_window, minimum=1)
        if config["completion_reserve"] >= context_window:
            raise ValueError("run report completion reserve must be below context window")
    elif config["completion_reserve"] or config["serialization_margin"]:
        raise ValueError("run report context window is required when reserves are configured")
    status = normalized.get("status")
    if not isinstance(status, Mapping):
        raise ValueError("run report status must be an object")
    _reject_unknown(status, _STATUS_FIELDS, "run report status")
    if set(status) != _STATUS_FIELDS:
        raise ValueError(
            f"run report status is missing fields: {sorted(_STATUS_FIELDS - set(status))}"
        )
    if status.get("profile") != profile:
        raise ValueError("run report status.profile must match profile")
    _require_integer("status.step", status.get("step"), minimum=0)
    _require_integer("status.telemetry_events", status.get("telemetry_events"), minimum=0)
    archive_status = status.get("archive")
    if not isinstance(archive_status, Mapping):
        raise ValueError("run report status.archive must be an object")
    _reject_unknown(archive_status, _ARCHIVE_STATUS_FIELDS, "run report status.archive")
    required_archive = {"path", "observations"}
    missing_archive = required_archive - set(archive_status)
    if missing_archive:
        raise ValueError(f"run report status.archive is missing fields: {sorted(missing_archive)}")
    archive_path = archive_status.get("path")
    if archive_path is not None:
        _require_string("status.archive.path", archive_path, minimum=1, maximum=4096)
    for name in set(archive_status) - {"path"}:
        _require_integer(f"status.archive.{name}", archive_status[name], minimum=0)
    if not isinstance(normalized.get("metrics"), Mapping):
        raise ValueError("run report metrics must be an object")
    _reject_unknown(normalized["metrics"], _METRIC_FIELDS, "run report metrics")
    missing_metrics = _METRIC_FIELDS - set(normalized["metrics"])
    if missing_metrics:
        raise ValueError(f"run report metrics are missing: {sorted(missing_metrics)}")
    for name in _METRIC_FIELDS:
        value = normalized["metrics"][name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"run report metric {name!r} must be a non-negative integer")
    if not isinstance(normalized.get("observations"), list):
        raise ValueError("run report observations must be a list")
    if not isinstance(normalized.get("events"), list):
        raise ValueError("run report events must be a list")
    if len(normalized["observations"]) > _MAX_REPORT_ROWS:
        raise ValueError(f"run report observations must not exceed {_MAX_REPORT_ROWS}")
    if len(normalized["events"]) > _MAX_REPORT_ROWS:
        raise ValueError(f"run report events must not exceed {_MAX_REPORT_ROWS}")
    privacy = normalized.get("privacy")
    if not isinstance(privacy, Mapping):
        raise ValueError("run report privacy must be an object")
    _reject_unknown(privacy, _PRIVACY_FIELDS, "run report privacy")
    if set(privacy) != _PRIVACY_FIELDS:
        raise ValueError(
            f"run report privacy is missing fields: {sorted(_PRIVACY_FIELDS - set(privacy))}"
        )
    if privacy.get("observation_bodies_included") is not False:
        raise ValueError("visualization reports must not contain observation bodies")
    if privacy.get("free_form_extra_included") is not False:
        raise ValueError("visualization reports must not contain free-form extras")
    for index, row in enumerate(normalized["observations"]):
        if not isinstance(row, Mapping):
            raise ValueError(f"observations[{index}] must be an object")
        unknown = set(row) - _OBSERVATION_FIELDS
        if unknown:
            raise ValueError(
                f"observations[{index}] contains non-report fields: {sorted(unknown)}"
            )
        missing = _OBSERVATION_FIELDS - set(row)
        if missing:
            raise ValueError(f"observations[{index}] is missing fields: {sorted(missing)}")
        if row.get("body_redacted") is not True:
            raise ValueError(f"observations[{index}] must declare body_redacted=true")
        _validate_observation(row, index)
    for index, event in enumerate(normalized["events"]):
        if not isinstance(event, Mapping):
            raise ValueError(f"events[{index}] must be an object")
        unknown = set(event) - _EVENT_FIELDS
        if unknown:
            raise ValueError(f"events[{index}] contains unsupported fields: {sorted(unknown)}")
        _validate_event(event, index)
    metrics = normalized["metrics"]
    if metrics["observations"] != len(normalized["observations"]):
        raise ValueError("run report observations metric does not match observation rows")
    for name in ("recalled_observations", "successful_observations", "checksum_coverage"):
        if metrics[name] > metrics["observations"]:
            raise ValueError(f"run report metric {name!r} exceeds observations")
    if metrics["exact_recall_actions"] > metrics["recall_actions"]:
        raise ValueError("run report exact recall actions exceed all recall actions")
    normalized["config"] = dict(config)
    normalized["status"] = {**dict(status), "archive": dict(archive_status)}
    normalized["metrics"] = dict(metrics)
    normalized["observations"] = [dict(row) for row in normalized["observations"]]
    normalized["events"] = [dict(event) for event in normalized["events"]]
    normalized["privacy"] = dict(privacy)
    return normalized


def _validate_observation(row: Mapping[str, Any], index: int) -> None:
    label = f"observations[{index}]"
    for name in (
        "sequence",
        "created_step",
        "created_ns",
        "char_count",
        "line_count",
        "byte_count",
        "last_recalled",
        "n_recalls",
    ):
        _require_integer(f"{label}.{name}", row.get(name), minimum=0)
    _require_integer(f"{label}.returncode", row.get("returncode"))
    if not isinstance(row.get("source_key_present"), bool):
        raise ValueError(f"{label}.source_key_present must be a boolean")
    obs_id = _require_string(f"{label}.obs_id", row.get("obs_id"), minimum=4, maximum=128)
    if any(character not in "0123456789abcdef" for character in obs_id):
        raise ValueError(f"{label}.obs_id must be lowercase hexadecimal")
    digest = _require_string(
        f"{label}.content_sha256", row.get("content_sha256"), minimum=64, maximum=64
    )
    if any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label}.content_sha256 must be lowercase hexadecimal")
    _require_string(f"{label}.cmd_signature", row.get("cmd_signature"), maximum=200)
    tags = row.get("tags")
    if not isinstance(tags, list) or len(tags) > 64:
        raise ValueError(f"{label}.tags must be a list with at most 64 entries")
    for tag_index, tag in enumerate(tags):
        _require_string(f"{label}.tags[{tag_index}]", tag, minimum=1, maximum=64)


def _validate_event(event: Mapping[str, Any], index: int) -> None:
    label = f"events[{index}]"
    kind = event.get("kind")
    if kind not in {"observation", "compaction", "action"}:
        raise ValueError(f"{label}.kind must be observation, compaction, or action")
    _require_integer(f"{label}.step", event.get("step"), minimum=0)
    _require_integer(f"{label}.timestamp_ns", event.get("timestamp_ns"), minimum=0)
    for name in (
        "chars",
        "before_messages",
        "after_messages",
        "before_chars",
        "after_chars",
        "enforced_limit",
        "before_tokens",
        "after_tokens",
        "enforced_token_limit",
    ):
        if name in event:
            _require_integer(f"{label}.{name}", event[name], minimum=0)
    if "returncode" in event:
        _require_integer(f"{label}.returncode", event["returncode"])
    if "exact" in event and not isinstance(event["exact"], bool):
        raise ValueError(f"{label}.exact must be a boolean")
    if "obs_id" in event and event["obs_id"] is not None:
        _require_string(f"{label}.obs_id", event["obs_id"], minimum=4, maximum=128)
    if "action_kind" in event:
        _require_string(f"{label}.action_kind", event["action_kind"], minimum=1, maximum=64)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        rendered = sorted((repr(item) for item in unknown))
        raise ValueError(f"{label} contains unsupported fields: {rendered}")


def _require_integer(name: str, value: Any, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"run report {name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"run report {name} must be >= {minimum}")
    return value


def _require_string(
    name: str,
    value: Any,
    *,
    minimum: int = 0,
    maximum: int,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ValueError(f"run report {name} must be a string with {minimum}-{maximum} characters")
    return value


def _metrics(observations: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    compactions = [event for event in events if event.get("kind") == "compaction"]
    actions = [event for event in events if event.get("kind") == "action"]
    before_chars = sum(int(event.get("before_chars", 0)) for event in compactions)
    after_chars = sum(int(event.get("after_chars", 0)) for event in compactions)
    return {
        "observations": len(observations),
        "archived_chars": sum(int(row["char_count"]) for row in observations),
        "archived_bytes": sum(int(row["byte_count"]) for row in observations),
        "compactions": len(compactions),
        "recall_actions": len(actions),
        "exact_recall_actions": sum(bool(event.get("exact")) for event in actions),
        "recalled_observations": sum(int(row["n_recalls"]) > 0 for row in observations),
        "cumulative_before_chars": before_chars,
        "cumulative_after_chars": after_chars,
        "cumulative_saved_chars": max(0, before_chars - after_chars),
        "successful_observations": sum(int(row["returncode"]) == 0 for row in observations),
        "checksum_coverage": sum(bool(row["content_sha256"]) for row in observations),
    }


def _safe_config(runner: CompactionRunner) -> dict[str, Any]:
    config = runner.config.to_dict()
    return {
        key: config[key]
        for key in (
            "profile",
            "keep_recent",
            "result_max_chars",
            "summary_max_chars",
            "max_recall_units",
            "recalls_per_step",
            "max_archive_bytes",
            "verify_archive_reads",
            "durability",
            "context_window",
            "completion_reserve",
            "serialization_margin",
        )
    }


def _safe_status(status: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(status)
    archive = dict(normalized.get("archive", {}))
    # Reports are routinely shared as benchmark evidence. The archive path can
    # disclose usernames, mount layouts, and project names without helping the
    # renderer, so status keeps the field shape but redacts its value.
    archive["path"] = None
    normalized["archive"] = archive
    return normalized


__all__ = [
    "REPORT_SCHEMA",
    "build_run_report",
    "load_run_report",
    "save_run_report",
    "validate_run_report",
]
