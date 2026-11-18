"""Tests for onecomp.mcp.compaction_tools (ARC over MCP)."""

import fcntl
import io
import json
import os

import pytest

from onecomp.mcp import compaction_tools as ct
from onecomp.mcp.compaction_tools import build_compaction_tools
from onecomp.mcp.protocol import MCPServer
from onecomp.mcp.tools import build_tools


@pytest.fixture(autouse=True)
def _clean_sessions():
    ct.close_all_sessions()
    yield
    ct.close_all_sessions()


def _start() -> str:
    return ct.compaction_start_session()["session_id"]


def test_start_session_returns_id_and_snippet():
    out = ct.compaction_start_session(recall_per_step=3)
    assert len(out["session_id"]) == 12
    assert "_recall \u00a7<obs_id>" in out["system_prompt_snippet"]
    assert "At most 3 recall(s)" in out["system_prompt_snippet"]


def test_sessions_are_isolated():
    sid_a, sid_b = _start(), _start()
    ct.compaction_record_observation(sid_a, command="ls", content="x" * 1000)
    assert ct.compaction_session_stats(sid_a)["n_observations"] == 1
    assert ct.compaction_session_stats(sid_b)["n_observations"] == 0


def test_authenticated_session_rejects_missing_token_and_cross_tenant_access():
    started = ct.compaction_start_session(
        tenant_id="team-a",
        require_session_token=True,
    )
    session_id = started["session_id"]
    token = started["session_token"]
    assert started["auth_required"] is True
    with pytest.raises(ValueError, match="Unknown session_id"):
        ct.compaction_session_stats(session_id, tenant_id="team-a")
    with pytest.raises(ValueError, match="Unknown session_id"):
        ct.compaction_session_stats(
            session_id,
            session_token=token,
            tenant_id="team-b",
        )
    assert (
        ct.compaction_session_stats(
            session_id,
            session_token=token,
            tenant_id="team-a",
        )["n_observations"]
        == 0
    )


def test_close_session_frees_and_unknown_id_raises():
    sid = _start()
    assert ct.compaction_close_session(sid) == {"closed": sid}
    with pytest.raises(ValueError, match="Unknown session_id"):
        ct.compaction_session_stats(sid)


def test_session_limit():
    for _ in range(ct._MAX_SESSIONS):
        _start()
    with pytest.raises(ValueError, match="Session limit"):
        _start()


def test_wal_path_is_confined_to_configured_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    sid = ct.compaction_start_session(wal_path="sessions/agent.wal")["session_id"]
    ct.compaction_record_observation(sid, command="echo ok", content="ok")
    ct.compaction_close_session(sid)
    assert (tmp_path / "sessions" / "agent.wal").is_file()

    with pytest.raises(ValueError, match="ONECOMP_MCP_WAL_ROOT"):
        ct.compaction_start_session(wal_path="../escape.wal")


def test_close_all_sessions_flushes_wals(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    sid = ct.compaction_start_session(wal_path="shutdown.wal")["session_id"]
    ct.compaction_record_observation(sid, command="echo bye", content="bye")
    ct.close_all_sessions()
    assert not ct._SESSIONS
    assert "echo bye" in (tmp_path / "shutdown.wal").read_text()


def test_resume_wal_restores_observations_without_duplicate_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    first = ct.compaction_start_session(wal_path="resume.wal")
    rec = ct.compaction_record_observation(
        first["session_id"], command="cat result", content="persisted result"
    )
    ct.compaction_close_session(first["session_id"])

    resumed = ct.compaction_start_session(wal_path="resume.wal", resume_wal=True)
    assert resumed["restored_observations"] == 1
    recalled = ct.compaction_recall(resumed["session_id"], rec["obs_id"])
    assert recalled["found"] is True
    assert "persisted result" in recalled["content"]
    ct.compaction_close_session(resumed["session_id"])
    assert len((tmp_path / "resume.wal").read_text().splitlines()) == 1


def test_resume_wal_requires_a_path():
    with pytest.raises(ValueError, match="requires wal_path"):
        ct.compaction_start_session(resume_wal=True)


def test_formal_resume_requires_an_existing_archive_without_creating_one(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        ct.compaction_start_session(
            profile="formal-v1",
            wal_path="missing.sqlite3",
            resume_wal=True,
        )

    assert not (tmp_path / "missing.sqlite3").exists()


def test_session_identity_is_validated_before_archive_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="tenant_id"):
        ct.compaction_start_session(
            profile="formal-v1",
            wal_path="must-not-exist.sqlite3",
            tenant_id="x" * 257,
        )

    assert not (tmp_path / "must-not-exist.sqlite3").exists()


def test_formal_profile_is_occurrence_addressed_bounded_and_resumable(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    started = ct.compaction_start_session(
        profile="formal-v1",
        wal_path="formal.sqlite3",
        max_recall_units=16,
        recall_per_step=10,
        require_session_token=True,
    )
    session_id = started["session_id"]
    token = started["session_token"]
    auth = {"session_token": token}
    first = ct.compaction_record_observation(
        session_id,
        command="same",
        content="same body",
        source_key="call-1",
        **auth,
    )
    second = ct.compaction_record_observation(
        session_id,
        command="same",
        content="same body",
        source_key="call-2",
        **auth,
    )
    retry = ct.compaction_record_observation(
        session_id,
        command="same",
        content="same body",
        source_key="call-1",
        **auth,
    )
    assert first["obs_id"] != second["obs_id"]
    assert retry["obs_id"] == first["obs_id"]

    huge = ct.compaction_record_observation(
        session_id,
        command="huge",
        content="0123456789abcdef-MIDDLE-0123456789abcdef",
        source_key="call-huge",
        **auth,
    )
    full = ct.compaction_recall(session_id, huge["obs_id"], **auth)
    assert full["found"] is False
    assert "chunk or range" in full["error"]
    chunk = ct.compaction_handle_action(
        session_id,
        f"_recall_chunk §{huge['obs_id']} 1",
        **auth,
    )
    assert chunk["handled"] is True
    assert chunk["exact"] is True
    assert chunk["content"] == "-MIDDLE-01234567"
    assert ct.compaction_session_stats(session_id, **auth)["profile"] == "formal-v1"
    ct.compaction_close_session(session_id, **auth)

    resumed = ct.compaction_start_session(
        profile="formal-v1",
        wal_path="formal.sqlite3",
        resume_wal=True,
        max_recall_units=64,
        require_session_token=True,
    )
    resumed_auth = {"session_token": resumed["session_token"]}
    recalled = ct.compaction_recall(
        resumed["session_id"],
        first["obs_id"],
        **resumed_auth,
    )
    assert recalled["found"] is True
    assert recalled["content"] == "same body"


def test_formal_wal_has_one_active_session_and_persistent_tenant_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    first = ct.compaction_start_session(
        profile="formal-v1",
        wal_path="owned.sqlite3",
        tenant_id="team-a",
        require_session_token=True,
    )
    with pytest.raises(ValueError, match="already active"):
        ct.compaction_start_session(
            profile="formal-v1",
            wal_path="owned.sqlite3",
            resume_wal=True,
            tenant_id="team-a",
        )
    ct.compaction_close_session(
        first["session_id"],
        session_token=first["session_token"],
        tenant_id="team-a",
    )

    with pytest.raises(ct.FormalArcError, match="different owner"):
        ct.compaction_start_session(
            profile="formal-v1",
            wal_path="owned.sqlite3",
            resume_wal=True,
            tenant_id="team-b",
        )

    resumed = ct.compaction_start_session(
        profile="formal-v1",
        wal_path="owned.sqlite3",
        resume_wal=True,
        tenant_id="team-a",
        require_session_token=True,
    )
    ct.compaction_close_session(
        resumed["session_id"],
        session_token=resumed["session_token"],
        tenant_id="team-a",
    )


def test_persistent_session_rejects_archive_locked_by_another_process(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    lock_path = tmp_path / "external.sqlite3.onecomp.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another compaction process"):
            ct.compaction_start_session(
                profile="formal-v1",
                wal_path="external.sqlite3",
            )
        assert not (tmp_path / "external.sqlite3").exists()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_formal_tool_builder_forces_profile_and_session_auth():
    tools = {
        tool.name: tool
        for tool in build_compaction_tools(profile="formal-v1", require_session_token=True)
    }
    started = tools["compaction_start_session"].handler()
    assert started["profile"] == "formal-v1"
    assert started["auth_required"] is True
    assert "session_token" in tools["compaction_recall"].input_schema["required"]
    assert all(tool.output_schema is not None for tool in tools.values())


def test_default_mcp_builder_is_formal_authenticated_and_bounded():
    tools = {tool.name: tool for tool in build_compaction_tools()}
    started = tools["compaction_start_session"].handler()
    try:
        assert started["profile"] == "formal-v1"
        assert started["auth_required"] is True
        recall_schema = tools["compaction_recall"].input_schema
        assert "session_token" in recall_schema["required"]
        assert recall_schema["properties"]["obs_id"]["maxLength"] == 128
        list_schema = tools["compaction_list_observations"].input_schema
        assert list_schema["properties"]["query"]["maxLength"] == 200
    finally:
        ct.compaction_close_session(
            started["session_id"],
            session_token=started["session_token"],
        )


def test_formal_mcp_native_chunk_and_range_recall():
    started = ct.compaction_start_session(
        profile="formal-v1",
        max_recall_units=8,
        recall_per_step=10,
    )
    obs_id = ct.compaction_record_observation(
        started["session_id"],
        command="read",
        content="0123456789abcdef",
        source_key="chunk-call",
    )["obs_id"]
    chunk = ct.compaction_recall(
        started["session_id"],
        obs_id,
        recall_kind="chunk",
        chunk_index=1,
    )
    exact_range = ct.compaction_recall(
        started["session_id"],
        obs_id,
        recall_kind="range",
        range_start=3,
        range_end=7,
    )
    assert chunk["found"] is True and chunk["content"] == "89abcdef"
    assert exact_range["content"] == "3456"


def test_formal_mcp_exposes_and_enforces_archive_quota():
    started = ct.compaction_start_session(
        profile="formal-v1",
        max_archive_bytes=5,
    )
    ct.compaction_record_observation(
        started["session_id"],
        command="first",
        content="12345",
    )
    with pytest.raises(ct.FormalArcError, match="quota exceeded"):
        ct.compaction_record_observation(
            started["session_id"],
            command="second",
            content="6",
        )


def test_formal_observation_list_uses_stable_body_free_cursor():
    started = ct.compaction_start_session(profile="formal-v1")
    ids = [
        ct.compaction_record_observation(
            started["session_id"],
            command=f"formal-{index}",
            content="private-" + str(index),
            source_key=f"formal-list-{index}",
        )["obs_id"]
        for index in range(5)
    ]
    first = ct.compaction_list_observations(started["session_id"], limit=2)
    second = ct.compaction_list_observations(
        started["session_id"], cursor=first["next_cursor"], limit=2
    )
    assert [row["obs_id"] for row in first["observations"]] == ids[:2]
    assert [row["obs_id"] for row in second["observations"]] == ids[2:4]
    assert "private" not in repr(first)
    assert first["total"] == 5


def test_record_flags_long_observations():
    sid = _start()
    short = ct.compaction_record_observation(sid, command="echo hi", content="hi")
    long = ct.compaction_record_observation(sid, command="cat big", content="y" * 5000)
    assert short["will_be_cited"] is False
    assert long["will_be_cited"] is True
    assert long["char_count"] == 5000


def test_parallel_observations_can_share_one_explicit_step():
    sid = _start()
    assert ct.compaction_begin_step(sid)["step"] == 1
    first = ct.compaction_record_observation(sid, "tool a", "a", advance_step=False)
    second = ct.compaction_record_observation(sid, "tool b", "b", advance_step=False)
    assert first["obs_id"] != second["obs_id"]
    assert ct.compaction_session_stats(sid)["step"] == 1


def test_record_rejects_non_boolean_advance_step():
    sid = _start()
    with pytest.raises(ValueError, match="advance_step"):
        ct.compaction_record_observation(sid, "tool", "output", advance_step=1)


def test_compact_messages_produces_citations_and_recall_roundtrip():
    sid = _start()
    big = "SECRET=abc123\n" + "noise\n" * 800
    rec = ct.compaction_record_observation(sid, command="cat data.log", content=big)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "```bash\ncat data.log\n```"},
        {
            "role": "user",
            "content": f"<output>\n{big}\n</output>",
            "_obs_id": rec["obs_id"],
        },
        {"role": "assistant", "content": "```bash\necho done\n```"},
        {"role": "user", "content": "<output>\ndone\n</output>"},
    ]
    out = ct.compaction_compact_messages(sid, messages)
    summary = out["messages"][2]["content"]
    assert big not in summary
    assert f"\u00a7{rec['obs_id']}" in summary
    assert out["total_chars"] < len(big)

    # MCP-native recall brings back the verbatim body.
    recalled = ct.compaction_recall(sid, rec["obs_id"])
    assert recalled["found"] is True
    assert "SECRET=abc123" in recalled["content"]

    meta = ct.compaction_recall(sid, rec["obs_id"], recall_kind="meta")
    assert meta["found"] is True
    assert len(meta["content"]) < 1200


def test_compact_messages_with_char_budget():
    sid = _start()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(6):
        rec = ct.compaction_record_observation(sid, command=f"cat p{i}", content=f"c{i} " * 800)
        messages.append({"role": "assistant", "content": f"```bash\ncat p{i}\n```"})
        messages.append({"role": "user", "content": f"c{i} " * 800, "_obs_id": rec["obs_id"]})
    out = ct.compaction_compact_messages(sid, messages, max_chars=6000)
    assert out["total_chars"] <= 6000


def test_compact_messages_accepts_provider_content_blocks():
    sid = _start()
    messages = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    out = ct.compaction_compact_messages(sid, messages)
    assert out["total_chars"] == 1


def test_compact_messages_accepts_openai_responses_items_without_roles():
    sid = _start()
    rec = ct.compaction_record_observation(
        sid, command="shell({})", content="RESPONSES-MCP\n" + "r" * 4000
    )
    messages = [
        {"role": "developer", "content": "instructions"},
        {"role": "user", "content": "task"},
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "RESPONSES-MCP\n" + "r" * 4000,
            "_obs_id": rec["obs_id"],
        },
        {"type": "message", "role": "assistant", "content": "recent"},
        {"role": "user", "content": "recent result"},
    ]
    out = ct.compaction_compact_messages(sid, messages)
    assert f"§{rec['obs_id']}" in out["messages"][2]["content"]


def test_handle_action_passthrough_and_errors():
    sid = _start()
    assert ct.compaction_handle_action(sid, "ls -la") == {"handled": False}

    malformed = ct.compaction_handle_action(sid, "_recall deadbeef")
    assert malformed["handled"] is True and "Malformed" in malformed["error"]

    unknown = ct.compaction_recall(sid, "deadbeef")
    assert unknown["found"] is False and "Unknown observation id" in unknown["error"]


def test_recall_kind_validation():
    sid = _start()
    with pytest.raises(ValueError, match="recall_kind must be"):
        ct.compaction_recall(sid, "deadbeef", recall_kind="huge")


def test_verify_value_guard():
    sid = _start()
    record = ct.compaction_record_observation(
        sid, command="cat log2", content="noise\ntrace-key: TR-REAL42\nnoise"
    )
    assert ct.compaction_verify_value(sid, "TR-REAL42")["seen"] is True
    assert ct.compaction_verify_value(sid, "TR-REAL42", prefix="trace-key: ")["seen"] is True
    missing = ct.compaction_verify_value(sid, "TR-GUESSED", prefix="trace-key: ")
    assert missing["seen"] is False
    assert missing["candidate_obs_ids"]
    assert "Do not guess" in missing["hint"]
    assert (
        ct.compaction_verify_value(
            sid,
            "TR-REAL42",
            prefix="trace-key: ",
            match_mode="exact_line",
            obs_id=record["obs_id"],
        )["seen"]
        is True
    )
    assert ct.compaction_verify_value(sid, "REAL", match_mode="exact_line")["seen"] is False
    with pytest.raises(ValueError, match="match_mode"):
        ct.compaction_verify_value(sid, "TR-REAL42", match_mode="regex")


def test_recall_budget_enforced_across_calls_and_reset_by_step_boundary():
    sid = ct.compaction_start_session(recall_per_step=1)["session_id"]
    obs_id = ct.compaction_record_observation(sid, command="cat x", content="z" * 900)["obs_id"]

    # 1st recall within budget; 2nd (no step boundary in between) exceeds it.
    assert ct.compaction_recall(sid, obs_id)["found"] is True
    over = ct.compaction_recall(sid, obs_id)
    assert over["found"] is False and "Recall budget exceeded" in over["error"]

    # An explicit step boundary resets the budget...
    ct.compaction_begin_step(sid)
    assert ct.compaction_recall(sid, obs_id)["found"] is True

    # ...and so does recording the next execution.
    ct.compaction_recall(sid, obs_id)  # exhaust again
    ct.compaction_record_observation(sid, command="echo hi", content="hi")
    assert ct.compaction_recall(sid, obs_id)["found"] is True


def test_mcp_recall_reports_external_eviction_instructions():
    sid = ct.compaction_start_session(recall_per_step=10, recall_budget_chars=1500)["session_id"]
    first = ct.compaction_record_observation(sid, command="one", content="A" * 1000)["obs_id"]
    second = ct.compaction_record_observation(sid, command="two", content="B" * 1000)["obs_id"]
    assert ct.compaction_recall(sid, first)["evicted_obs_ids"] == []
    recalled = ct.compaction_recall(sid, second)
    assert recalled["evicted_obs_ids"] == [first]


def test_bounded_observation_index_pages_newest_first():
    sid = _start()
    ids = [
        ct.compaction_record_observation(sid, command=f"c{i}", content=str(i))["obs_id"]
        for i in range(5)
    ]
    page = ct.compaction_list_observations(sid, limit=2)
    assert [entry["obs_id"] for entry in page["observations"]] == ids[-2:][::-1]
    assert page["next_cursor"] == 2


def test_observation_index_filters_by_command_path():
    sid = _start()
    wanted = ct.compaction_record_observation(
        sid, command="inspect src/special_target.py", content="wanted"
    )
    ct.compaction_record_observation(sid, command="inspect logs/other.log", content="other")
    page = ct.compaction_list_observations(sid, query="SPECIAL_target")
    assert page["total"] == 1
    assert page["observations"][0]["obs_id"] == wanted["obs_id"]


def test_observation_index_rejects_non_string_query():
    sid = _start()
    with pytest.raises(ValueError, match="query"):
        ct.compaction_list_observations(sid, query=1)


def test_tools_registered_in_build_tools():
    names = {t.name for t in build_tools()}
    expected = {t.name for t in build_compaction_tools()}
    assert expected <= names
    assert len(expected) == 10


def _rpc(server_input: str) -> list[dict]:
    reader, writer = io.StringIO(server_input), io.StringIO()
    MCPServer(reader, writer, tools=build_compaction_tools()).serve_forever()
    return [json.loads(line) for line in writer.getvalue().splitlines()]


def test_end_to_end_over_jsonrpc_stdio():
    """Full MCP wire-level flow, exactly as an MCP host would drive it."""
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "compaction_start_session", "arguments": {}},
        },
    ]
    responses = _rpc("\n".join(json.dumps(l) for l in lines) + "\n")
    assert responses[0]["result"]["serverInfo"]["name"] == "onecomp"
    tool_names = {t["name"] for t in responses[1]["result"]["tools"]}
    assert "compaction_recall" in tool_names

    body = json.loads(responses[2]["result"]["content"][0]["text"])
    sid = body["session_id"]
    token = body["session_token"]

    lines2 = [
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "compaction_record_observation",
                "arguments": {
                    "session_id": sid,
                    "session_token": token,
                    "command": "cat x",
                    "content": "NEEDLE=42\n" + "n\n" * 600,
                },
            },
        },
    ]
    responses2 = _rpc("\n".join(json.dumps(l) for l in lines2) + "\n")
    rec = json.loads(responses2[0]["result"]["content"][0]["text"])
    assert rec["will_be_cited"] is True

    lines3 = [
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "compaction_recall",
                "arguments": {
                    "session_id": sid,
                    "session_token": token,
                    "obs_id": rec["obs_id"],
                },
            },
        },
    ]
    responses3 = _rpc("\n".join(json.dumps(l) for l in lines3) + "\n")
    recalled = json.loads(responses3[0]["result"]["content"][0]["text"])
    assert recalled["found"] is True and "NEEDLE=42" in recalled["content"]
