"""Host wire contracts and formal ingestion tests."""

from pathlib import Path

import pytest

from onecomp.compaction.adapters.claude_code import post_tool_use_output as claude_output
from onecomp.compaction.adapters.claude_code import (
    replace_tool_output,
)
from onecomp.compaction.adapters.codex import post_tool_use_output as codex_output
from onecomp.compaction.adapters.hooks import _archive_path, process_post_tool_use
from onecomp.compaction.formal import (
    ArcCoreService,
    ArchiveCapacityExceeded,
    RecallRequiresChunking,
    SQLiteObservationArchive,
)


def test_claude_preserves_known_tool_shape_and_fails_open_for_unknown_shape():
    original = {
        "stdout": "large output",
        "stderr": "warning",
        "interrupted": False,
        "isImage": False,
    }
    replacement = replace_tool_output(original, "citation")
    assert replacement == {
        "stdout": "citation",
        "stderr": "",
        "interrupted": False,
        "isImage": False,
    }
    assert original["stdout"] == "large output"
    assert claude_output({"opaque": 1}, "citation") == {}
    assert (
        replace_tool_output([{"type": "image", "source": {"data": "opaque"}}], "citation") is None
    )
    assert replace_tool_output(
        [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        "citation",
    ) == [{"type": "text", "text": "citation"}]


def test_codex_preferred_and_compatibility_contracts():
    assert codex_output("citation") == {"decision": "block", "reason": "citation"}
    assert codex_output("citation", mode="context") == {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "citation",
        }
    }
    assert codex_output("citation", mode="stop") == {
        "continue": False,
        "stopReason": "citation",
    }
    assert codex_output("citation", mode="block") == {
        "decision": "block",
        "reason": "citation",
    }
    assert codex_output("citation", mode="replace") == {
        "decision": "block",
        "reason": "citation",
    }
    with pytest.raises(ValueError, match="mode"):
        codex_output("citation", mode="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("platform", ["claude", "codex", "opencode"])
def test_hook_commits_before_replacement_and_retries_idempotently(tmp_path, platform):
    payload = {
        "session_id": f"{platform}-session",
        "turn_id": "turn-1",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "cat large.log"},
        "tool_response": "NEEDLE\n" + "x" * 5000,
        "tool_use_id": "call-1",
    }
    first = process_post_tool_use(payload, platform=platform, hook_root=tmp_path)
    second = process_post_tool_use(payload, platform=platform, hook_root=tmp_path)
    rendered = str(first)
    assert "§" in rendered and "NEEDLE" in rendered
    assert second == first

    archive_path = _archive_path(payload["session_id"], tmp_path)
    with SQLiteObservationArchive(archive_path) as archive:
        assert archive.n_records == 1
        record = archive.get(archive.known_ids()[0])
        assert record.full_content == payload["tool_response"]
        assert record.source_key.startswith(f"hook:{platform}:")
        assert len(record.source_key) < 128


def test_hook_keeps_identical_bodies_as_distinct_host_occurrences(tmp_path):
    payload = {
        "session_id": "occurrences",
        "tool_name": "Read",
        "tool_input": {"path": "same.txt"},
        "tool_response": "same" * 1000,
    }
    process_post_tool_use(
        {**payload, "tool_use_id": "call-a"}, platform="codex", hook_root=tmp_path
    )
    process_post_tool_use(
        {**payload, "tool_use_id": "call-b"}, platform="codex", hook_root=tmp_path
    )
    with SQLiteObservationArchive(_archive_path("occurrences", tmp_path)) as archive:
        assert len(set(archive.known_ids())) == 2


def test_hook_without_host_occurrence_id_never_content_deduplicates(tmp_path):
    payload = {
        "session_id": "no-host-id",
        "tool_name": "Read",
        "tool_input": {"path": "same.txt"},
        "tool_response": "same" * 1000,
    }
    process_post_tool_use(payload, platform="codex", hook_root=tmp_path)
    process_post_tool_use(payload, platform="codex", hook_root=tmp_path)

    with SQLiteObservationArchive(_archive_path("no-host-id", tmp_path)) as archive:
        assert archive.n_records == 2
        assert all(archive.get(obs_id).source_key is None for obs_id in archive.known_ids())


def test_hook_full_recall_is_bounded_but_exact_chunks_reconstruct(tmp_path):
    payload = {
        "session_id": "bounded",
        "tool_name": "Read",
        "tool_input": {"path": "huge.txt"},
        "tool_response": "0123456789" * 100,
        "tool_use_id": "call-huge",
    }
    process_post_tool_use(
        payload,
        platform="codex",
        hook_root=tmp_path,
        max_recall_units=32,
    )
    with SQLiteObservationArchive(_archive_path("bounded", tmp_path)) as archive:
        service = ArcCoreService(archive, max_recall_units=32)
        obs_id = archive.known_ids()[0]
        with pytest.raises(RecallRequiresChunking, match="chunk or range"):
            service.recall_full(obs_id)
        assert (
            "".join(chunk.text for chunk in service.text_chunks(obs_id))
            == payload["tool_response"]
        )


def test_hook_paths_do_not_expose_raw_session_ids(tmp_path):
    path = _archive_path("secret/session", tmp_path)
    assert path.parent == Path(tmp_path).resolve()
    assert "secret" not in path.name
    assert path.suffix == ".sqlite3"


def test_hook_archive_quota_fails_before_emitting_a_false_citation(tmp_path):
    payload = {
        "session_id": "quota-session",
        "tool_name": "Read",
        "tool_input": {"path": "large.txt"},
        "tool_response": "x" * 1000,
        "tool_use_id": "call-quota",
    }
    with pytest.raises(ArchiveCapacityExceeded, match="quota exceeded"):
        process_post_tool_use(
            payload,
            platform="codex",
            hook_root=tmp_path,
            max_archive_bytes=100,
        )


def test_unknown_claude_output_shape_is_not_archived(tmp_path):
    payload = {
        "session_id": "unsupported-shape",
        "tool_name": "ThirdPartyTool",
        "tool_input": {"value": 1},
        "tool_response": {"opaque": "x" * 1000},
        "tool_use_id": "call-unsupported",
    }

    assert process_post_tool_use(payload, platform="claude", hook_root=tmp_path) == {}
    assert not _archive_path("unsupported-shape", tmp_path).exists()
