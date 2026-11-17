"""Result data at the provider boundary, during a turn and in later history."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger
from roomkit.channels._compaction import _STORED_OVER_CHARS, with_results_stored
from roomkit.channels._tool_eviction import ToolEviction
from roomkit.channels.ai import AIChannel
from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.base import (
    AIMessage,
    AIResponse,
    AIToolResultPart,
    response_stream_events,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import current_tool_call
from tests.conftest import make_event
from tests.test_channels.test_tool_history import _answer, _call_rows, _round, _rounds, _turns
from tests.test_tool_context import _binding, _tool_round_responses
from tests.tool_loop_modes import respond


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("copy", [None, {}, {"page": 3}])
async def test_next_round_reads_the_structured_copy_and_typed_outcome(
    streaming: bool, copy: dict[str, Any] | None
) -> None:
    async def handler(name: str, arguments: dict[str, Any]) -> str:
        call = current_tool_call()
        assert call is not None
        call.structured_content = copy
        return "model text"

    provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=streaming)
    channel = AIChannel("ai1", provider=provider, tool_handler=handler)
    try:
        await respond(
            channel,
            make_event(room_id="r1", body="read", channel_id="sms1"),
            _binding("r1"),
            RoomContext(room=Room(id="r1")),
        )
        assert len(provider.calls) == 2
        parts = [
            part
            for message in provider.calls[1].messages
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, AIToolResultPart)
        ]
        assert len(parts) == 1
        part = parts[0]
        assert part.structured_content == copy
        assert part.outcome == "served"
        assert part.as_text() == "model text"
        assert part.split_for_message() == ("model text", [])
    finally:
        await channel.close()


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_later_turn_replays_typed_outcomes(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _round(("c1", "lookup", {}), ("c2", "fail", {})),
            _answer("First done."),
            _answer("Second done."),
        ],
        streaming=streaming,
    )
    contexts = await _turns(provider, "first", "second")
    assert len(contexts) == 3
    for context in contexts[1:]:
        parts = [
            part
            for message in context.messages
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, AIToolResultPart)
        ]
        assert [(p.tool_call_id, p.outcome, p.is_error) for p in parts] == [
            ("c1", "served", False),
            ("c2", "failed", True),
        ]


@pytest.mark.parametrize("copy", [None, {"page": 3}])
def test_compaction_stores_text_without_erasing_the_result_carriers(copy: Any) -> None:
    text = "A" * (_STORED_OVER_CHARS + 100)
    original = AIToolResultPart(
        tool_call_id="c1", name="lookup", result=text, structured_content=copy, outcome="served"
    )
    [message] = with_results_stored([AIMessage(role="tool", content=[original])], ToolEviction())
    assert isinstance(message.content, list)
    [stored] = message.content
    assert isinstance(stored, AIToolResultPart)
    assert stored.as_text().startswith("Result too large (")
    assert original.result == text
    assert stored.structured_content == copy
    assert stored.outcome == "served"


@pytest.mark.parametrize("streaming", [False, True])
async def test_history_replays_the_outcome_but_not_the_structured_copy(streaming: bool) -> None:
    copy = {"page": 7}

    async def enrich(event: Any, context: Any) -> HookResult:
        return HookResult.modify(replace(event, structured_content=copy))

    provider = MockAIProvider(
        ai_responses=[
            _round(("c1", "lookup", {})),
            _answer("First done."),
            _answer("Second done."),
        ],
        streaming=streaming,
    )
    contexts = await _turns(
        provider, "first", "second", hooks=[(HookTrigger.ON_TOOL_CALL, enrich)]
    )
    assert len(contexts) == 3
    for context, expected in [(contexts[1], copy), (contexts[2], None)]:
        parts = [
            part
            for message in context.messages
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, AIToolResultPart)
        ]
        assert len(parts) == 1
        assert parts[0].structured_content == expected
        assert parts[0].outcome == "served"


def test_history_preserves_outcome_independently_of_legacy_status() -> None:
    kinds = ["served", "failed", "blocked", "refused", "unserved", None]
    rows = [
        row
        for n, kind in enumerate(kinds)
        for row in _call_rows(1 + n * 2, f"c{n}", "body", outcome=kind)
    ]
    parts = [part for round_ in _rounds(rows) for part in round_.results.content]
    assert [p.outcome for p in parts] == kinds
    assert [p.is_error for p in parts] == [False, True, False, False, False, False]


def test_response_lists_are_kept_by_the_model_but_not_added_to_the_stream() -> None:
    response = AIResponse(content="answer", finish_reason="stop")
    assert response.tasks == [] and response.observations == []
    expected_stream = list(response_stream_events(response))
    task = Task(id="t1", room_id="r1", title="Task")
    observation = Observation(id="o1", room_id="r1", channel_id="ai1", content="Observation")
    response.tasks.append(task)
    response.observations.append(observation)
    assert response.tasks == [task]
    assert response.observations == [observation]
    assert list(response_stream_events(response)) == expected_stream
    other = AIResponse(content="other")
    assert other.tasks == [] and other.observations == []
