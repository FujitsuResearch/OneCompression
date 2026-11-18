"""OneComp-style compaction configuration and runner tests."""

import threading
from pathlib import Path

import pytest

from onecomp.compaction import (
    BudgetExceeded,
    CompactionConfig,
    CompactionRunner,
    InvalidAction,
    RecallRequiresChunking,
    sanitize_messages,
)
from onecomp.compaction.adapters.messages import total_text_chars
from onecomp.compaction.formal import UnknownObservation
from onecomp.compaction.runner import _message_chars


@pytest.mark.parametrize(
    "field,pattern",
    [
        ("action_regex", r"```bash.*```"),
        ("observation_regex", r"(<output>)(.*)</output>"),
    ],
)
def test_config_requires_one_extractor_capture_group(field, pattern):
    with pytest.raises(ValueError, match="exactly one capture group"):
        CompactionConfig(**{field: pattern})


def test_config_is_side_effect_free_serializable_and_has_explicit_profiles(tmp_path):
    root = tmp_path / "run"
    config = CompactionConfig.persistent(root, max_recall_units=64)
    assert not root.exists()
    assert config.profile == "formal-v1"
    assert config.to_dict()["archive_path"] == str(root / "archive.sqlite3")

    experimental = CompactionConfig.experimental(tmp_path / "legacy.jsonl")
    assert experimental.profile == "experimental-v1"
    assert experimental.archive_path == tmp_path / "legacy.jsonl"


def test_formal_runner_records_occurrences_compacts_and_recovers_exactly(tmp_path):
    config = CompactionConfig.persistent(
        tmp_path / "run",
        keep_recent=1,
        result_max_chars=10,
        max_recall_units=4096,
    )
    with CompactionRunner(config) as runner:
        runner.begin_step()
        first = runner.record_observation(
            command="cat output.log",
            content="identical long output",
            source_key="tool-call-1",
        )
        second = runner.record_observation(
            command="cat output.log",
            content="identical long output",
            source_key="tool-call-2",
        )
        assert first.obs_id != second.obs_id

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "```bash\ncat output.log\n```"},
            runner.make_observation_message(first),
            {"role": "assistant", "content": "```bash\necho next\n```"},
            runner.make_observation_message(second),
            {"role": "assistant", "content": "recent thought"},
            {"role": "user", "content": "recent result"},
        ]
        compacted = runner.compact(messages)
        summary = compacted[2]["content"]
        assert f"§{first.obs_id}" in summary
        assert f"§{second.obs_id}" in summary
        assert first.content_sha256 in summary
        assert "Use _catalog" in summary

        action_messages = []
        response = runner.handle_action(f"_recall §{first.obs_id}", action_messages)
        assert response is not None
        assert response.body == first.full_content
        assert action_messages[-1]["content"] == first.full_content
        sanitized = sanitize_messages(action_messages)
        assert sanitized == [{"role": "user", "content": first.full_content}]

        assert runner.handle_action("pytest -q", action_messages) is None
        with pytest.raises(InvalidAction):
            runner.handle_action(f"_recall §{first.obs_id} # unsafe suffix")
        status = runner.status()
        assert status["profile"] == "formal-v1"
        assert status["archive"]["observations"] == 2
        assert status["archive"]["recall_count"] == 1

    archive_path = Path(config.archive_path)
    assert archive_path.is_file()


def test_formal_runner_requires_explicit_trust_for_non_verbatim_rendering(tmp_path):
    with CompactionRunner(CompactionConfig.persistent(tmp_path / "run")) as runner:
        record = runner.record_observation(command="tool", content="canonical")
        with pytest.raises(ValueError, match="complete archived body"):
            runner.make_observation_message(record, rendered="display-only")
        message = runner.make_observation_message(
            record,
            rendered="display-only",
            trusted_rendering=True,
        )

    assert message["_obs_id"] == record.obs_id
    assert message["_arc_rendered_kind"] == "trusted_custom"


def test_runner_close_waits_for_an_active_operation(tmp_path, monkeypatch):
    runner = CompactionRunner(CompactionConfig.persistent(tmp_path / "run"))
    assert runner.service is not None
    entered = threading.Event()
    release = threading.Event()
    original = runner.service.record

    def delayed_record(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner.service, "record", delayed_record)
    recorded = []
    worker = threading.Thread(
        target=lambda: recorded.append(runner.record_observation(command="tool", content="body"))
    )
    closer = threading.Thread(target=runner.close)
    worker.start()
    assert entered.wait(timeout=5)
    closer.start()
    assert closer.is_alive()
    release.set()
    worker.join(timeout=5)
    closer.join(timeout=5)

    assert recorded[0].full_content == "body"
    with pytest.raises(RuntimeError, match="closed"):
        runner.status()


def test_runner_hard_recall_limit_requires_chunking(tmp_path):
    config = CompactionConfig.persistent(tmp_path / "run", max_recall_units=8)
    with CompactionRunner(config) as runner:
        record = runner.record_observation(
            command="cmd",
            content="0123456789abcdef",
            source_key="call",
        )
        with pytest.raises(RecallRequiresChunking):
            runner.handle_action(f"_recall §{record.obs_id}")
        chunk = runner.handle_action(f"_recall_chunk §{record.obs_id} 1")
        assert chunk is not None
        assert chunk.body == "89abcdef"


def test_formal_citation_respects_asymmetric_preview_limits_without_overlap(tmp_path):
    config = CompactionConfig.persistent(
        tmp_path / "run",
        keep_recent=1,
        result_max_chars=4,
        citation_head_chars=0,
        citation_tail_chars=5,
    )
    with CompactionRunner(config) as runner:
        record = runner.record_observation(command="cmd", content="0123456789")
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "run"},
            runner.make_observation_message(record),
            {"role": "assistant", "content": "recent"},
            {"role": "user", "content": "tail"},
        ]
        summary = runner.compact(messages)[2]["content"]

    assert "…\n56789" in summary
    assert "01234" not in summary
    assert summary.count("56789") == 1


def test_runner_provider_guard_uses_configured_final_envelope():
    config = CompactionConfig(
        context_window=20,
        completion_reserve=5,
        serialization_margin=2,
    )
    with CompactionRunner(config) as runner:
        guard = runner.create_budget_guard(lambda request: len(request["serialized_request"]))
        assert (
            guard.dispatch(
                {"serialized_request": "x" * 13},
                lambda request: request["serialized_request"],
            )
            == "x" * 13
        )


def test_experimental_profile_remains_available_without_formal_semantics(tmp_path):
    config = CompactionConfig.experimental(
        tmp_path / "legacy.jsonl",
        max_recall_units=100,
    )
    with CompactionRunner(config) as runner:
        runner.begin_step()
        first = runner.record_observation(command="cmd", content="same")
        second = runner.record_observation(command="cmd", content="same")
        assert first.obs_id == second.obs_id
        assert runner.profile == "experimental-v1"


def test_formal_action_allowance_charges_failed_attempts_and_catalog_has_no_null_tag(tmp_path):
    config = CompactionConfig.persistent(tmp_path / "run", recalls_per_step=2)
    with CompactionRunner(config) as runner:
        runner.begin_step()
        runner.record_observation(command="cmd", content="body")
        with pytest.raises(UnknownObservation):
            runner.handle_action(f"_recall §{'0' * 32}")
        with pytest.raises(InvalidAction):
            runner.handle_action("_recall missing-section-sign")
        with pytest.raises(InvalidAction, match="allowance exceeded"):
            runner.handle_action(f"_recall §{'1' * 32}")

        runner.begin_step()
        messages = []
        runner.handle_action("_catalog", messages)
        assert "_obs_id" not in messages[0]
        assert messages[0]["_is_recall"] is False


def test_runner_budget_uses_the_same_provider_text_projection_for_content_blocks():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "x" * 5000}],
        },
    ]
    assert _message_chars(messages) == total_text_chars(messages) == 5005


def test_formal_token_budget_uses_host_counter_and_fails_closed(tmp_path):
    config = CompactionConfig.persistent(
        tmp_path / "run",
        keep_recent=1,
        result_max_chars=40,
        summary_max_chars=1200,
    )
    with CompactionRunner(config) as runner:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        for index in range(8):
            record = runner.record_observation(
                command=f"tool-{index}",
                content="x" * 1000,
                source_key=f"call-{index}",
            )
            messages.extend(
                [
                    {"role": "assistant", "content": f"tool-{index}"},
                    runner.make_observation_message(record),
                ]
            )

        def count_tokens(candidate):
            return sum((len(str(message.get("content", ""))) + 3) // 4 for message in candidate)

        compacted = runner.enforce_token_budget(messages, 500, count_tokens)
        assert count_tokens(sanitize_messages(compacted)) <= 500
        assert runner.events[-1]["enforced_token_limit"] == 500
        with pytest.raises(BudgetExceeded, match="token envelope"):
            runner.enforce_token_budget(messages, 1, count_tokens)


def test_runner_token_budget_fails_closed_for_experimental_compatibility_profile():
    with CompactionRunner(CompactionConfig(profile="experimental-v1")) as runner:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]

        with pytest.raises(BudgetExceeded, match="experimental compaction"):
            runner.enforce_token_budget(messages, 1, lambda _messages: 2)
