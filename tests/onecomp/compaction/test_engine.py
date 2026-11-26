"""Tests for onecomp.compaction.engine (framework-agnostic ARC facade)."""

import pytest

from onecomp.compaction.engine import ArcEngine, ArcEngineConfig, RecallError, sanitize_messages
from onecomp.compaction.store import ObsStore


def _base_messages() -> list[dict]:
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "the task"},
    ]


def _run_turn(engine: ArcEngine, messages: list[dict], cmd: str, output: str) -> str:
    """Simulate one executed agent turn; returns the obs_id."""
    messages.append({"role": "assistant", "content": f"do it\n\n```bash\n{cmd}\n```"})
    rec = engine.record_observation(command=cmd, content=output)
    messages.append(
        engine.make_observation_message(rec, rendered=f"<output>\n{output}\n</output>")
    )
    return rec.obs_id


def test_full_loop_compact_then_recall():
    engine = ArcEngine()
    messages = _base_messages()
    big = "needle-in-line-42\n" + "noise\n" * 1000

    engine.begin_step()
    obs_id = _run_turn(engine, messages, "cat haystack.txt", big)
    engine.begin_step()
    _run_turn(engine, messages, "echo recent", "recent-ok")

    # Context overflow -> compact: the big observation becomes a citation.
    messages = engine.compact(messages)
    summary = messages[2]["content"]
    assert big not in summary
    assert f"\u00a7{obs_id}" in summary

    # The LLM decides to materialise it.
    engine.begin_step()
    outcome = engine.handle_action(f"_recall \u00a7{obs_id}", messages)
    assert outcome is not None and outcome.kind == "recall"
    assert "needle-in-line-42" in messages[-1]["content"]
    assert messages[-1]["_obs_id"] == obs_id
    assert engine.recall_log[-1]["obs_id"] == obs_id


def test_non_recall_actions_pass_through():
    engine = ArcEngine()
    messages = _base_messages()
    assert engine.handle_action("ls -la", messages) is None
    assert messages == _base_messages()  # untouched


def test_malformed_recall_raises_helpful_error():
    engine = ArcEngine()
    with pytest.raises(RecallError, match="Malformed recall"):
        engine.handle_action("_recall deadbeef", _base_messages())  # missing sigil
    with pytest.raises(RecallError, match="Malformed recall"):
        engine.handle_action("_recall \u00a7deadbeef | tee out", _base_messages())


def test_unknown_id_suggests_close_matches_without_charging_budget():
    engine = ArcEngine(ArcEngineConfig(recall_per_step=1))
    messages = _base_messages()
    engine.begin_step()
    obs_id = _run_turn(engine, messages, "cat file", "content " * 200)

    typo = obs_id[:-1] + ("0" if obs_id[-1] != "0" else "1")
    with pytest.raises(RecallError, match="Unknown observation id"):
        engine.handle_action(f"_recall \u00a7{typo}", messages)

    # The failed attempt must not have consumed the (single) recall slot.
    outcome = engine.handle_action(f"_recall \u00a7{obs_id}", messages)
    assert outcome is not None


def test_recall_budget_per_step():
    engine = ArcEngine(ArcEngineConfig(recall_per_step=1))
    messages = _base_messages()
    engine.begin_step()
    oid1 = _run_turn(engine, messages, "cmd one", "one " * 300)
    oid2 = _run_turn(engine, messages, "cmd two", "two " * 300)

    engine.begin_step()
    assert engine.handle_action(f"_recall \u00a7{oid1}", messages) is not None
    with pytest.raises(RecallError, match="Recall budget exceeded"):
        engine.handle_action(f"_recall \u00a7{oid2}", messages)

    # A new step resets the budget.
    engine.begin_step()
    assert engine.handle_action(f"_recall \u00a7{oid2}", messages) is not None


def test_recall_meta_is_bounded_and_skips_lru():
    engine = ArcEngine()
    messages = _base_messages()
    engine.begin_step()
    obs_id = _run_turn(engine, messages, "cat big", "Z" * 50_000)

    outcome = engine.handle_action(f"_recall_meta \u00a7{obs_id}", messages)
    assert outcome.kind == "recall_meta"
    assert len(messages[-1]["content"]) < 1200  # bounded preview
    assert obs_id not in engine.in_context_obs  # meta never enters the LRU


def test_lru_eviction_replaces_old_recall_with_placeholder():
    engine = ArcEngine(ArcEngineConfig(recall_budget_chars=1500, recall_per_step=10))
    messages = _base_messages()
    engine.begin_step()
    oid1 = _run_turn(engine, messages, "cmd one", "A" * 1000)
    oid2 = _run_turn(engine, messages, "cmd two", "B" * 1000)

    engine.begin_step()
    engine.handle_action(f"_recall \u00a7{oid1}", messages)
    engine.handle_action(f"_recall \u00a7{oid2}", messages)  # pushes oid1 over budget

    assert oid1 not in engine.in_context_obs
    assert oid2 in engine.in_context_obs
    evicted = [m for m in messages if m.get("_is_evicted")]
    assert len(evicted) == 1 and evicted[0]["_obs_id"] == oid1
    assert "A" * 1000 not in evicted[0]["content"]
    # The body remains recall-able from the store.
    assert engine.store.get(oid1).full_content == "A" * 1000
    assert engine.eviction_log[-1]["reason"] == "budget"


def test_recall_windowing_caps_reinjection():
    engine = ArcEngine(ArcEngineConfig(recall_max_chars=400))
    messages = _base_messages()
    engine.begin_step()
    obs_id = _run_turn(engine, messages, "cat huge", "H" * 200 + "M" * 5000 + "T" * 200)

    engine.handle_action(f"_recall \u00a7{obs_id}", messages)
    body = messages[-1]["content"]
    assert "recall windowed" in body
    assert len(body) < 1200
    # Store keeps the verbatim body.
    assert len(engine.store.get(obs_id).full_content) == 5400


def test_enforce_char_budget_shrinks_until_fit():
    engine = ArcEngine(ArcEngineConfig(recall_per_step=10))
    messages = _base_messages()
    for i in range(8):
        engine.begin_step()
        _run_turn(engine, messages, f"cat part-{i}", f"chunk-{i} " * 500)

    assert sum(len(m["content"]) for m in messages) > 8_000
    trimmed = engine.enforce_char_budget(messages, max_chars=8_000)
    assert sum(len(m["content"]) for m in trimmed) <= 8_000
    # System + task survive intact.
    assert trimmed[0] == messages[0]
    assert trimmed[1]["content"] == "the task"
    assert engine.compaction_log  # at least one compaction happened


def test_enforce_char_budget_noop_when_already_fitting():
    engine = ArcEngine()
    messages = _base_messages()
    out = engine.enforce_char_budget(messages, max_chars=10_000)
    assert out == messages
    assert not engine.compaction_log


def test_enforce_token_budget_uses_host_counter_and_logs_tokens():
    engine = ArcEngine(ArcEngineConfig(summary_max_chars=10_000))
    messages = _base_messages()
    for i in range(8):
        engine.begin_step()
        _run_turn(engine, messages, f"cat token-{i}", f"token-{i} " * 700)

    def count_tokens(payload):
        # Deterministic stand-in for a provider's exact tokenizer.
        chars = sum(len(str(message.get("content", ""))) for message in payload)
        return (chars + 3) // 4

    before = count_tokens(sanitize_messages(messages))
    trimmed = engine.enforce_token_budget(messages, 1_500, count_tokens)
    after = count_tokens(sanitize_messages(trimmed))
    assert before > 1_500
    assert after <= 1_500
    assert engine.compaction_log[-1]["before_tokens"] == before
    assert engine.compaction_log[-1]["after_tokens"] == after


def test_token_budget_candidate_search_commits_only_selected_state():
    engine = ArcEngine(ArcEngineConfig(summary_max_chars=20_000))
    messages = _base_messages()
    for i in range(12):
        engine.begin_step()
        _run_turn(engine, messages, f"cat candidate-{i}", f"payload-{i} " * 700)

    def count_tokens(payload):
        return (sum(len(str(message.get("content", ""))) for message in payload) + 3) // 4

    trimmed = engine.enforce_token_budget(messages, 2_100, count_tokens)
    assert count_tokens(sanitize_messages(trimmed)) <= 2_100
    # Candidate caps are exploratory.  Only the history returned to the host
    # may update LRU bookkeeping and produce a compaction telemetry record.
    assert len(engine.compaction_log) == 1
    assert engine.compaction_log[0]["after_chars"] == sum(
        len(str(message.get("content", ""))) for message in trimmed
    )


def test_token_budget_search_reaches_minimum_summary_for_extreme_ratio():
    engine = ArcEngine(ArcEngineConfig(summary_max_chars=131_072))
    messages = _base_messages()
    for i in range(3):
        engine.begin_step()
        _run_turn(engine, messages, f"cat extreme-{i}", (f"payload-{i}-" * 20_000))

    def count_tokens(payload):
        return sum(len(str(message.get("content", ""))) for message in payload)

    minimum = engine._compact_once(messages, 1)
    minimum_tokens = count_tokens(sanitize_messages(minimum))
    trimmed = engine.enforce_token_budget(messages, minimum_tokens, count_tokens)
    assert count_tokens(sanitize_messages(trimmed)) <= minimum_tokens
    # Tiny caps produce an empty summary rather than a malformed header
    # prefix. Several adjacent cap values therefore have the same minimum.
    assert engine.compaction_log[-1]["summary_max_chars"] <= 256
    assert minimum[2]["content"] == ""


def test_recall_window_body_honours_configured_body_cap():
    engine = ArcEngine(ArcEngineConfig(recall_max_chars=120))
    body, elided = engine._window_recall_body("x" * 5_000)
    assert len(body) <= 120
    assert elided > 0


def test_compact_syncs_lru_and_logs():
    engine = ArcEngine(ArcEngineConfig(recall_per_step=10, keep_recent=1))
    messages = _base_messages()
    engine.begin_step()
    oid = _run_turn(engine, messages, "cat data", "D" * 3000)
    engine.handle_action(f"_recall \u00a7{oid}", messages)
    assert oid in engine.in_context_obs

    # Push the recall out of the kept-recent window, then compact.
    _run_turn(engine, messages, "echo recent", "fresh")
    compacted = engine.compact(messages)
    # The recall message is gone from the buffer -> LRU must not lie.
    assert oid not in engine.in_context_obs
    assert engine.compaction_log[-1]["evicted_by_compaction"] == [oid]
    assert f"\u00a7{oid}" in compacted[2]["content"]  # still cited, recall-able


def test_system_prompt_snippet_mentions_budget():
    engine = ArcEngine(ArcEngineConfig(recall_per_step=3))
    snippet = engine.system_prompt_snippet()
    assert "_recall \u00a7<obs_id>" in snippet
    assert "At most 3 recall(s)" in snippet


def test_sanitize_messages_strips_private_keys():
    engine = ArcEngine()
    messages = _base_messages()
    engine.begin_step()
    obs_id = _run_turn(engine, messages, "cat x", "y" * 900)
    engine.handle_action(f"_recall \u00a7{obs_id}", messages)

    clean = sanitize_messages(messages)
    assert all(set(m) <= {"role", "content"} for m in clean)
    # Originals keep their tags.
    assert any(m.get("_obs_id") for m in messages)


def test_observation_message_rejects_mismatched_rendering_without_explicit_trust():
    engine = ArcEngine()
    record = engine.record_observation(command="cat result", content="archived body")

    raw = engine.make_observation_message(record)
    wrapped = engine.make_observation_message(
        record,
        rendered="<output>archived body</output>",
    )
    with pytest.raises(ValueError, match="complete archived body"):
        engine.make_observation_message(record, rendered="different visible body")
    custom = engine.make_observation_message(
        record,
        rendered="structured host result",
        trusted_rendering=True,
    )

    assert raw["_arc_rendered_kind"] == "raw"
    assert wrapped["_arc_rendered_kind"] == "verbatim_wrapper"
    assert custom["_arc_rendered_kind"] == "trusted_custom"
    assert "_arc_rendered_kind" not in sanitize_messages([custom])[0]


def test_snapshot_and_wal_resume(tmp_path):
    wal = tmp_path / "engine.wal"
    engine = ArcEngine(ArcEngineConfig(wal_path=str(wal)))
    engine.begin_step()
    messages = _base_messages()
    obs_id = _run_turn(engine, messages, "cat file", "persist me " * 100)
    engine.close()

    snap = engine.snapshot()
    assert snap["obs_store"][0]["obs_id"] == obs_id

    restored = ObsStore.from_wal(wal)
    assert restored.get(obs_id).full_content == "persist me " * 100


def test_engine_from_wal_restores_records_and_continues_steps(tmp_path):
    wal = tmp_path / "engine-resume.wal"
    engine = ArcEngine(ArcEngineConfig(wal_path=str(wal)))
    engine.begin_step()
    first = engine.record_observation(command="echo one", content="one")
    engine.close()

    resumed = ArcEngine.from_wal(str(wal))
    assert resumed.store.get(first.obs_id).full_content == "one"
    assert resumed.step == 1
    assert resumed.begin_step() == 2
    resumed.record_observation(command="echo two", content="two")
    resumed.close()

    restored = ObsStore.from_wal(wal)
    assert restored.n_records == 2


def test_engine_from_wal_rejects_a_different_append_target(tmp_path):
    source = tmp_path / "source.wal"
    ArcEngine(ArcEngineConfig(wal_path=str(source))).close()
    config = ArcEngineConfig(wal_path=str(tmp_path / "different.wal"))
    with pytest.raises(ValueError, match="WAL being resumed"):
        ArcEngine.from_wal(str(source), config)


def test_engine_rejects_invalid_runtime_budgets():
    engine = ArcEngine()
    with pytest.raises(ValueError, match="max_chars"):
        engine.enforce_char_budget(_base_messages(), -1)
    with pytest.raises(ValueError, match="summary_max_chars"):
        engine.compact(_base_messages(), summary_max_chars=-1)


def test_find_prior_executions():
    engine = ArcEngine()
    messages = _base_messages()
    engine.begin_step()
    _run_turn(engine, messages, "grep -rn foo .", "hit1")
    assert engine.find_prior_executions("grep   -rn   foo .")  # normalised match
    assert not engine.find_prior_executions("ls")


@pytest.mark.parametrize(
    "name,value",
    [
        ("keep_recent", -1),
        ("summary_max_chars", -1),
        ("recall_per_step", -1),
        ("recall_budget_chars", -1),
        ("recall_max_chars", -1),
        ("result_max_chars", True),
    ],
)
def test_engine_config_rejects_invalid_limits(name, value):
    with pytest.raises(ValueError, match=name):
        ArcEngineConfig(**{name: value})


def test_engine_config_rejects_invalid_regex():
    with pytest.raises(ValueError, match="action_regex"):
        ArcEngineConfig(action_regex="[")
