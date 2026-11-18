"""Concurrent multi-agent coding trajectory contract for formal MCP sessions."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from onecomp.mcp.compaction_tools import (
    close_all_sessions,
    compaction_close_session,
    compaction_compact_messages,
    compaction_recall,
    compaction_record_observation,
    compaction_session_stats,
    compaction_start_session,
)


def _start_agent(tmp_path, name):
    return compaction_start_session(
        profile="formal-v1",
        wal_path=str(tmp_path / f"{name}.sqlite3"),
        tenant_id=f"coding-team:{name}",
        require_session_token=True,
        result_max_chars=80,
        summary_max_chars=4000,
        max_recall_units=512,
        recall_per_step=32,
    )


def test_concurrent_coding_agents_compact_recall_and_remain_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    agents = {name: _start_agent(tmp_path, name) for name in ("scout", "coder", "reviewer")}

    def record(name, index):
        handle = agents[name]
        content = (
            f"AGENT={name} FILE=module_{index}.py ASSERTION=case_{index}\n"
            + f"{name}-{index}-" * 300
        )
        result = compaction_record_observation(
            handle["session_id"],
            f"pytest -vv tests/test_module_{index}.py",
            content,
            source_key=f"{name}:tool:{index}",
            advance_step=False,
            session_token=handle["session_token"],
            tenant_id=handle["tenant_id"],
        )
        return name, index, content, result

    try:
        with ThreadPoolExecutor(max_workers=9) as pool:
            recorded = list(
                pool.map(
                    lambda item: record(*item),
                    ((name, index) for name in agents for index in range(8)),
                )
            )

        by_agent = {name: [] for name in agents}
        for name, index, content, result in recorded:
            by_agent[name].append((index, content, result))

        for name, handle in agents.items():
            messages = [
                {"role": "system", "content": "Solve the repository task exactly."},
                {"role": "user", "content": f"Role: {name}"},
            ]
            for index, content, result in sorted(by_agent[name]):
                messages.extend(
                    [
                        {"role": "assistant", "content": f"Inspecting test shard {index}"},
                        {
                            "role": "user",
                            "content": content,
                            "_obs_id": result["obs_id"],
                        },
                    ]
                )
            compacted = compaction_compact_messages(
                handle["session_id"],
                messages,
                max_chars=4000,
                session_token=handle["session_token"],
                tenant_id=handle["tenant_id"],
            )
            assert compacted["total_chars"] <= 4000
            rendered = str(compacted["messages"])
            assert "§" in rendered
            assert "ASSERTION=case_0" not in rendered

            index, original, result = sorted(by_agent[name])[3]
            chunk_count = (len(original) + 511) // 512
            chunks = [
                compaction_recall(
                    handle["session_id"],
                    result["obs_id"],
                    recall_kind="chunk",
                    chunk_index=chunk,
                    session_token=handle["session_token"],
                    tenant_id=handle["tenant_id"],
                )["content"]
                for chunk in range(chunk_count)
            ]
            assert "".join(chunks) == original
            assert (
                compaction_session_stats(
                    handle["session_id"],
                    session_token=handle["session_token"],
                    tenant_id=handle["tenant_id"],
                )["n_observations"]
                == 8
            )

        scout = agents["scout"]
        coder = agents["coder"]
        foreign_id = by_agent["coder"][0][2]["obs_id"]
        with pytest.raises(ValueError, match="credentials"):
            compaction_recall(
                scout["session_id"],
                foreign_id,
                session_token=coder["session_token"],
                tenant_id=coder["tenant_id"],
            )
    finally:
        for handle in agents.values():
            try:
                compaction_close_session(
                    handle["session_id"],
                    session_token=handle["session_token"],
                    tenant_id=handle["tenant_id"],
                )
            except ValueError:
                pass
        close_all_sessions()
