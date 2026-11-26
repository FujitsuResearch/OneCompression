"""Optional contract test against the installed OpenHands SDK."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import json

import pytest

pytest.importorskip("openhands.sdk", reason="optional OpenHands SDK")

from openhands.sdk.context.view import View  # noqa: E402
from openhands.sdk.event.llm_convertible import (  # noqa: E402
    ActionEvent,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.llm import Message, MessageToolCall, TextContent  # noqa: E402
from openhands.sdk.tool import Action, Observation  # noqa: E402

from onecomp.compaction.adapters.openhands import (  # noqa: E402
    OpenHandsArcCondenser,
    OpenHandsArcListAction,
    OpenHandsArcListTool,
    OpenHandsArcRecallAction,
    OpenHandsArcRecallObservation,
    OpenHandsArcRecallTool,
)
from onecomp.compaction.formal import RecallRequiresChunking  # noqa: E402


class _ShellAction(Action):
    command: str


class _ShellObservation(Observation):
    pass


def _trajectory() -> list:
    events = [
        MessageEvent(
            source="user",
            llm_message=Message(
                role="user",
                content=[TextContent(text="Inspect every shard exactly")],
            ),
        )
    ]
    for index in range(4):
        call_id = f"call-{index}"
        action = ActionEvent(
            source="agent",
            thought=[TextContent(text=f"Inspecting shard {index}")],
            action=_ShellAction(command=f"cat shard-{index}.log"),
            tool_name="terminal",
            tool_call_id=call_id,
            tool_call=MessageToolCall(
                id=call_id,
                name="terminal",
                arguments=json.dumps({"command": f"cat shard-{index}.log"}),
                origin="completion",
            ),
            llm_response_id=f"response-{index}",
        )
        observation = ObservationEvent(
            source="environment",
            observation=_ShellObservation.from_text(
                f"NEEDLE_{index}=value-{index}\n" + chr(65 + index) * 3000
            ),
            tool_name="terminal",
            tool_call_id=call_id,
            action_id=action.id,
        )
        events.extend((action, observation))
    return events


def test_native_condensation_archive_and_tools(tmp_path):
    condenser = OpenHandsArcCondenser(
        runtime_id="onecomp-contract-test",
        max_size=6,
        keep_first=1,
        result_max_chars=128,
        summary_max_chars=4000,
        recall_per_step=8,
        recall_max_chars=1024,
        wal_path=str(tmp_path / "archive.sqlite3"),
    )
    try:
        view = View.from_events(_trajectory())
        input_ids = [event.id for event in view.events]
        condensation = condenser.condense(view)

        assert [event.id for event in view.events] == input_ids
        assert condensation.__class__.__name__ == "Condensation"
        assert len(condensation.apply(view.events)) < len(view.events)
        assert condenser.engine.archive is not None
        obs_id = condenser.engine.archive.known_ids()[0]
        record = condenser.engine.archive.get(obs_id)

        with pytest.raises(RecallRequiresChunking):
            condenser.recall(obs_id, "full")
        condenser.engine.begin_step()
        chunk_count = (record.char_count + 1023) // 1024
        chunks = [
            condenser.recall(obs_id, "chunk", chunk_index=index) for index in range(chunk_count)
        ]
        assert "".join(chunks) == record.full_content

        recall_tool = OpenHandsArcRecallTool.create(runtime_id=condenser.runtime_id)[0]
        recalled = recall_tool.executor(
            OpenHandsArcRecallAction(
                obs_id=obs_id,
                recall_kind="chunk",
                chunk_index=0,
            )
        )
        assert not recalled.is_error
        assert recalled.text == chunks[0]

        list_tool = OpenHandsArcListTool.create(runtime_id=condenser.runtime_id)[0]
        listed = list_tool.executor(OpenHandsArcListAction(query="shard", limit=10))
        assert json.loads(listed.text)["total"] == 4
    finally:
        condenser.close()


def test_native_recall_result_reuses_original_archive_occurrence(tmp_path):
    condenser = OpenHandsArcCondenser(
        runtime_id="onecomp-recall-dedup-test",
        max_size=6,
        keep_first=1,
        wal_path=str(tmp_path / "archive.sqlite3"),
    )
    try:
        record = condenser.engine.record_observation(
            command="cat result.log",
            content="already archived body",
            source_key="original-call",
        )
        call_id = "recall-call"
        action = ActionEvent(
            source="agent",
            thought=[TextContent(text="Recall exact evidence")],
            action=OpenHandsArcRecallAction(obs_id=record.obs_id),
            tool_name="arc_recall",
            tool_call_id=call_id,
            tool_call=MessageToolCall(
                id=call_id,
                name="arc_recall",
                arguments=json.dumps({"obs_id": record.obs_id}),
                origin="completion",
            ),
            llm_response_id="recall-response",
        )
        observation = ObservationEvent(
            source="environment",
            observation=OpenHandsArcRecallObservation.from_text(record.full_content),
            tool_name="arc_recall",
            tool_call_id=call_id,
            action_id=action.id,
        )
        converted = condenser._event_to_arc_message(observation, {action.id: action})

        assert converted is not None
        assert converted["_obs_id"] == record.obs_id
        assert condenser.engine.archive is not None
        assert condenser.engine.archive.n_records == 1
    finally:
        condenser.close()
