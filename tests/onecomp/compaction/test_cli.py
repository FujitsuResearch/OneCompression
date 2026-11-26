"""End-to-end standalone CLI checks."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _run(config: Path, *arguments: str, input_text: str | None = None):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "onecomp.compaction.cli",
            "--config",
            str(config),
            *arguments,
        ],
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
    )


def test_init_record_catalog_recall_export_and_doctor(tmp_path):
    config = tmp_path / "arc" / "config.json"
    initialized = _run(config, "init", "--max-recall-units", "64")
    assert initialized.returncode == 0, initialized.stderr
    assert config.stat().st_mode & 0o777 == 0o600

    recorded = _run(
        config,
        "record",
        "--command",
        "cat result.txt",
        "--content",
        "EXACT-NEEDLE\nresult",
        "--source-key",
        "call-1",
        "--tag",
        "test",
    )
    assert recorded.returncode == 0, recorded.stderr
    obs_id = json.loads(recorded.stdout)["obs_id"]

    catalog = _run(config, "catalog", "--query", "result.txt")
    assert json.loads(catalog.stdout)["observations"][0]["obs_id"] == obs_id

    metadata = _run(config, "recall", obs_id)
    assert "EXACT-NEEDLE" not in metadata.stdout
    exact = _run(config, "recall", obs_id, "--kind", "full")
    assert exact.stdout == "EXACT-NEEDLE\nresult"

    exported = _run(config, "export")
    payload = json.loads(exported.stdout)
    assert payload["bodies_included"] is False
    assert "full_content" not in payload["observations"][0]
    assert payload["observations"][0]["body_redacted"] is True
    assert "extra" not in payload["observations"][0]
    assert payload["observations"][0]["extra_redacted"] is True
    export_path = tmp_path / "private-export.json"
    written = _run(config, "export", "--output", str(export_path))
    assert written.returncode == 0, written.stderr
    assert export_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".private-export.json.*.tmp"))
    bodies_path = tmp_path / "private-bodies.json"
    bodies = _run(
        config,
        "export",
        "--include-bodies",
        "--output",
        str(bodies_path),
    )
    assert bodies.returncode == 0, bodies.stderr
    body_payload = json.loads(bodies_path.read_text(encoding="utf-8"))
    assert body_payload["observations"][0]["full_content"] == "EXACT-NEEDLE\nresult"
    archive_path = config.parent / "archive.sqlite3"
    refused = _run(config, "export", "--output", str(archive_path))
    assert refused.returncode == 2
    assert "must not overwrite" in refused.stderr
    refused_sidecar = _run(config, "export", "--output", str(archive_path) + "-wal")
    assert refused_sidecar.returncode == 2
    assert "must not overwrite" in refused_sidecar.stderr

    doctor = _run(config, "doctor")
    report = json.loads(doctor.stdout)
    assert report["ok"] is True
    assert report["verified_observations"] == 1


def test_export_refuses_existing_output_unless_forced(tmp_path):
    config = tmp_path / "config.json"
    assert _run(config, "init").returncode == 0
    output = tmp_path / "existing-export.json"
    original = b"existing content must survive\n"
    output.write_bytes(original)

    refused = _run(config, "export", "--output", str(output))

    assert refused.returncode == 2, refused.stderr
    assert "already exists" in refused.stderr
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".existing-export.json.*.tmp"))

    replaced = _run(config, "export", "--output", str(output), "--force")

    assert replaced.returncode == 0, replaced.stderr
    assert (
        json.loads(output.read_text(encoding="utf-8"))["schema"] == "onecomp-compaction-export-v1"
    )
    assert output.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".existing-export.json.*.tmp"))


@pytest.mark.parametrize(
    "protected_name",
    [
        "config.json",
        "archive.sqlite3",
        "archive.sqlite3-wal",
        "archive.sqlite3-shm",
        "archive.sqlite3-journal",
    ],
)
def test_forced_export_still_protects_config_and_archive(tmp_path, protected_name):
    config = tmp_path / "config.json"
    assert _run(config, "init").returncode == 0
    assert _run(config, "record", "--command", "test", "--content", "preserved").returncode == 0

    refused = _run(config, "export", "--output", str(tmp_path / protected_name), "--force")

    assert refused.returncode == 2, refused.stderr
    assert "must not overwrite" in refused.stderr
    doctor = _run(config, "doctor")
    assert doctor.returncode == 0, doctor.stderr
    assert json.loads(doctor.stdout)["verified_observations"] == 1


def test_export_preserves_destination_created_while_streaming(tmp_path):
    from onecomp.compaction.cli import _atomic_write_export

    output = tmp_path / "racing-export.json"
    original = b"created by another writer\n"

    def rows():
        output.write_bytes(original)
        yield {"sequence": 0}

    with pytest.raises(ValueError, match="already exists"):
        _atomic_write_export(output, rows(), include_bodies=False)

    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".racing-export.json.*.tmp"))


def test_forced_export_preserves_existing_output_when_serialization_fails(tmp_path):
    from onecomp.compaction.cli import _atomic_write_export

    output = tmp_path / "failed-export.json"
    original = b"existing content must survive\n"
    output.write_bytes(original)

    with pytest.raises(TypeError):
        _atomic_write_export(
            output,
            [{"unsupported": object()}],
            include_bodies=False,
            overwrite=True,
        )

    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".failed-export.json.*.tmp"))


def test_init_refuses_overwrite_and_plugin_generation(tmp_path):
    config = tmp_path / "config.json"
    assert _run(config, "init").returncode == 0
    assert _run(config, "init").returncode == 2
    plugin = tmp_path / "plugin.ts"
    generated = _run(
        config,
        "plugin",
        "--host",
        "opencode",
        "--output",
        str(plugin),
    )
    assert generated.returncode == 0, generated.stderr
    assert '"tool.execute.after"' in plugin.read_text(encoding="utf-8")
    assert (
        _run(
            config,
            "plugin",
            "--host",
            "opencode",
            "--output",
            str(plugin),
        ).returncode
        == 2
    )
    replaced = _run(
        config,
        "plugin",
        "--host",
        "opencode",
        "--output",
        str(plugin),
        "--force",
    )
    assert replaced.returncode == 0, replaced.stderr

    guide = _run(config, "guide", "--host", "codex")
    assert guide.returncode == 0
    assert "can replace completed output" in guide.stdout
    assert "onecomp-arc --config" in guide.stdout


def test_init_does_not_change_existing_parent_permissions(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)

    initialized = _run(shared / "config.json", "init")

    assert initialized.returncode == 0, initialized.stderr
    assert shared.stat().st_mode & 0o777 == 0o755


def test_run_passes_short_output_and_archives_verbose_output(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert (
        _run(
            config,
            "init",
            "--result-max-chars",
            "32",
            "--max-recall-units",
            "10000",
        ).returncode
        == 0
    )

    short = _run(config, "run", "--", sys.executable, "-c", "print('short')")
    assert short.returncode == 0
    assert short.stdout == "short\n"
    assert json.loads(_run(config, "status").stdout)["archive"]["observations"] == 0

    verbose = _run(
        config,
        "run",
        "--tag",
        "coding-test",
        "--",
        sys.executable,
        "-c",
        "print('NEEDLE-' + 'x' * 5000)",
    )
    assert verbose.returncode == 0, verbose.stderr
    assert "<observation_ref" in verbose.stdout
    assert "NEEDLE" in verbose.stdout
    assert len(verbose.stdout) < 2000
    match = re.search(r'id="§([0-9a-f]+)"', verbose.stdout)
    assert match is not None
    obs_id = match.group(1)
    recalled = _run(config, "recall", obs_id, "--kind", "full")
    assert recalled.stdout == "NEEDLE-" + "x" * 5000 + "\n"


def test_run_preserves_failure_status_and_fails_open_on_archive_quota(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert (
        _run(
            config,
            "init",
            "--result-max-chars",
            "8",
            "--max-archive-bytes",
            "16",
        ).returncode
        == 0
    )
    command = "import sys; print('FAIL-' + 'z' * 100); raise SystemExit(7)"
    result = _run(config, "run", "--", sys.executable, "-c", command)

    assert result.returncode == 7
    assert result.stdout == "FAIL-" + "z" * 100 + "\n"
    assert "archival failed, preserving full output" in result.stderr
    assert json.loads(_run(config, "status").stdout)["archive"]["observations"] == 0


def test_run_tee_streams_progress_to_stderr_but_keeps_citation_on_stdout(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert _run(config, "init", "--result-max-chars", "8").returncode == 0

    result = _run(
        config,
        "run",
        "--tee",
        "--",
        sys.executable,
        "-c",
        "print('LIVE-' + 'x' * 200)",
    )

    assert result.returncode == 0
    assert "LIVE-" in result.stderr
    assert "<observation_ref" in result.stdout


def test_run_output_ceiling_streams_exact_fail_open_bytes(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert _run(config, "init", "--result-max-chars", "8").returncode == 0

    result = _run(
        config,
        "run",
        "--max-output-bytes",
        "32",
        "--",
        sys.executable,
        "-c",
        "print('0123456789' * 20)",
    )

    assert result.returncode == 0
    assert result.stdout == "0123456789" * 20 + "\n"
    assert "exceeded --max-output-bytes" in result.stderr
    assert json.loads(_run(config, "status").stdout)["archive"]["observations"] == 0


def test_run_timeout_interrupts_a_silent_child(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert _run(config, "init").returncode == 0

    result = _run(
        config,
        "run",
        "--timeout",
        "0.2",
        "--",
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    )

    assert result.returncode == 124
    assert "timed out after 0.2s" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="process-group isolation is POSIX-specific")
def test_run_timeout_terminates_a_grandchild_holding_the_output_pipe(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert _run(config, "init").returncode == 0
    spawn = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)'])"
    )

    started = time.monotonic()
    result = _run(
        config,
        "run",
        "--timeout",
        "0.2",
        "--",
        sys.executable,
        "-c",
        spawn,
    )

    assert result.returncode == 124
    assert time.monotonic() - started < 5
    assert "timed out after 0.2s" in result.stderr


def test_config_option_is_accepted_after_subcommand_but_not_stolen_from_child(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert _run(config, "init").returncode == 0
    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "onecomp.compaction.cli",
            "status",
            "--config",
            str(config),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    child = _run(
        config,
        "run",
        "--",
        sys.executable,
        "-c",
        "import sys; print(sys.argv[1:])",
        "--config",
        "child-value",
    )
    assert child.returncode == 0, child.stderr
    assert "child-value" in child.stdout


def test_migrate_uses_stable_identity_for_a_growing_wal(tmp_path):
    config = tmp_path / "arc" / "config.json"
    wal = tmp_path / "growing.jsonl"
    wal.write_text(json.dumps({"cmd_signature": "first", "full_content": "one"}) + "\n")
    assert _run(config, "init").returncode == 0

    first = _run(config, "migrate", wal, "--migration-id", "stable-run")
    assert first.returncode == 0, first.stderr
    with wal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"cmd_signature": "second", "full_content": "two"}) + "\n")
    second = _run(config, "migrate", wal, "--migration-id", "stable-run")

    assert second.returncode == 0, second.stderr
    assert json.loads(_run(config, "status").stdout)["archive"]["observations"] == 2
