"""Single-run report and visualization checks."""

import json

import pytest

from onecomp.analyzer.compaction import (
    build_run_report,
    load_run_report,
    save_run_report,
    validate_run_report,
    visualize_run,
)
from onecomp.compaction import CompactionConfig, CompactionRunner


def _run():
    runner = CompactionRunner(
        CompactionConfig(
            keep_recent=1,
            result_max_chars=60,
            summary_max_chars=600,
            max_recall_units=128,
            recalls_per_step=10,
        )
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    records = []
    for index in range(6):
        runner.begin_step()
        record = runner.record_observation(
            command=f"inspect module-{index}.py",
            content=f"PRIVATE-BODY-{index}\n" + chr(65 + index) * (200 + index * 100),
            source_key=f"call-{index}",
            returncode=1 if index == 4 else 0,
            tags=["test"],
        )
        records.append(record)
        messages.extend(
            [
                {"role": "assistant", "content": f"inspect module-{index}.py"},
                runner.make_observation_message(record),
            ]
        )
    runner.enforce_char_budget(messages, 1600)
    runner.handle_action(f"_recall_chunk §{records[-1].obs_id} 0", [])
    return runner


def test_report_is_body_free_and_roundtrips(tmp_path):
    runner = _run()
    try:
        report = build_run_report(runner, run_name="formal evidence")
        serialized = json.dumps(report)
        assert "PRIVATE-BODY" not in serialized
        assert report["privacy"]["observation_bodies_included"] is False
        assert report["metrics"]["observations"] == 6
        assert report["metrics"]["checksum_coverage"] == 6
        assert report["status"]["archive"]["path"] is None
        path = save_run_report(report, tmp_path / "run.json")
        assert load_run_report(path)["run_name"] == "formal evidence"
        assert path.stat().st_mode & 0o077 == 0
        assert not list(tmp_path.glob(".run.json.*.tmp"))
    finally:
        runner.close()


def test_report_validator_accepts_legacy_v1_config_without_durability():
    runner = _run()
    try:
        report = build_run_report(runner)
    finally:
        runner.close()
    report["config"].pop("durability")

    normalized = validate_run_report(report)

    assert normalized["config"]["durability"] == "full"


def test_report_validator_rejects_body_fields_and_malformed_privacy():
    runner = _run()
    try:
        report = build_run_report(runner)
    finally:
        runner.close()
    report["observations"][0]["full_content"] = "secret"
    with pytest.raises(ValueError, match="non-report fields"):
        validate_run_report(report)
    report["observations"][0].pop("full_content")
    report["privacy"] = []
    with pytest.raises(ValueError, match="privacy"):
        validate_run_report(report)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ((), "full_content"),
        (("metrics",), "body"),
        (("config",), "secret_prompt"),
        (("status",), "tool_output"),
        (("status", "archive"), "body_preview"),
        (("privacy",), "free_form_payload"),
    ],
)
def test_report_validator_rejects_unknown_fields_at_every_level(path, field):
    runner = _run()
    try:
        report = build_run_report(runner)
    finally:
        runner.close()
    target = report
    for part in path:
        target = target[part]
    target[field] = "PRIVATE-BODY"

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_run_report(report)


def test_dashboard_renders_png_and_svg(tmp_path):
    runner = _run()
    try:
        paths = visualize_run(
            runner,
            tmp_path / "compaction-dashboard",
            run_name="Professional ARC run",
            dpi=120,
        )
        assert {path.suffix for path in paths} == {".png", ".svg"}
        assert all(path.stat().st_size > 10_000 for path in paths)
        import matplotlib.image as image  # pylint: disable=import-outside-toplevel

        pixels = image.imread(next(path for path in paths if path.suffix == ".png"))
        assert (pixels[0, 0, :3] > 0.97).all(), "dashboard must use a white base"
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("metrics", "archived_chars"), -1, "archived_chars"),
        (("observations", 0, "char_count"), -1, "char_count"),
        (("observations", 0, "source_key_present"), 1, "source_key_present"),
        (("events", 0, "kind"), "payload", "kind"),
        (("events", 0, "chars"), -1, "chars"),
    ],
)
def test_report_validator_rejects_malformed_render_values(path, value, message):
    runner = _run()
    try:
        report = build_run_report(runner)
    finally:
        runner.close()
    target = report
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_run_report(report)
