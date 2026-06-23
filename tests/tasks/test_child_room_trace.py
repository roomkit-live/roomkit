"""run_agent_in_child_room persists the worker's FULL trace.

A delegated agent's child room must record what it actually did — each tool
call (with arguments and result) plus its text — not just the final answer, so
the room is a complete, linkable transcript. The parent link lives in the child
room's ``metadata.parent_room_id`` (asserted in test_integration), so the
parent↔child relationship is rebuildable from persistence alone.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from roomkit.core.mixins.delegation import _persist_child_stream, run_agent_in_child_room
from roomkit.models.enums import ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.models.streaming import ThinkingDeltaMarker, ToolCallEndMarker, ToolCallStartMarker


def _recording_store() -> MagicMock:
    store = MagicMock()
    store.added = []

    async def _add(room_id: str, event: RoomEvent) -> RoomEvent:
        store.added.append(event)
        return event

    store.add_event_auto_index = AsyncMock(side_effect=_add)
    return store


def _sr(stream: Any) -> SimpleNamespace:
    return SimpleNamespace(
        stream=stream,
        source_channel_id="agent:w1",
        source_channel_type=ChannelType.AI,
    )


class TestPersistChildStream:
    async def test_persists_tool_calls_and_text_segments_in_order(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()

        async def _stream() -> Any:
            yield "Let me search. "
            yield ToolCallStartMarker(
                tool_name="WebSearch", tool_id="t1", arguments={"q": "world cup"}
            )
            yield ToolCallEndMarker(
                tool_name="WebSearch",
                tool_id="t1",
                arguments={"q": "world cup"},
                result="standings...",
                status="completed",
                duration_ms=42,
            )
            yield "Here is the answer."

        text = await _persist_child_stream(kit, "parent::task-1", _sr(_stream()), chain_depth=1)

        # Return value is the full concatenated text (the worker's output).
        assert text == "Let me search. Here is the answer."

        # Order: text segment, tool start, tool end, final text segment.
        seq = [(e.type, getattr(e.content, "tool_name", None)) for e in kit.store.added]
        assert seq == [
            (EventType.MESSAGE, None),
            (EventType.TOOL_CALL_START, "WebSearch"),
            (EventType.TOOL_CALL_END, "WebSearch"),
            (EventType.MESSAGE, None),
        ]

        # The tool-end event carries the arguments + result + timing.
        end = next(e for e in kit.store.added if e.type == EventType.TOOL_CALL_END)
        assert end.content.arguments == {"q": "world cup"}
        assert end.content.result == "standings..."
        assert end.content.duration_ms == 42
        assert end.content.status == "completed"

    async def test_thinking_markers_are_not_persisted(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()

        async def _stream() -> Any:
            yield ThinkingDeltaMarker(thinking="hmm")
            yield "final answer"

        text = await _persist_child_stream(kit, "parent::task-2", _sr(_stream()), chain_depth=1)
        assert text == "final answer"
        # Only the text segment is persisted — thinking is transient.
        assert [e.type for e in kit.store.added] == [EventType.MESSAGE]

    async def test_text_only_stream_persists_single_message(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()

        async def _stream() -> Any:
            yield "just "
            yield "text"

        text = await _persist_child_stream(kit, "parent::task-3", _sr(_stream()), chain_depth=1)
        assert text == "just text"
        assert len(kit.store.added) == 1
        assert kit.store.added[0].content.body == "just text"


class TestRunAgentNonStreaming:
    async def test_persists_all_response_events_not_just_final_text(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()
        kit.get_room = AsyncMock(
            return_value=Room(id="parent::task-1", metadata={"parent_room_id": "parent"})
        )
        kit.store.list_bindings = AsyncMock(return_value=[])
        kit.store.list_events = AsyncMock(return_value=[])

        source = EventSource(channel_id="agent:w1", channel_type=ChannelType.AI)
        tool_event = RoomEvent(
            room_id="parent::task-1",
            source=source,
            type=EventType.TOOL_CALL_END,
            content=ToolCallContent(tool_name="WebSearch", tool_id="t1", status="completed"),
        )
        msg_event = RoomEvent(
            room_id="parent::task-1",
            source=source,
            type=EventType.MESSAGE,
            content=TextContent(body="the answer"),
        )
        output = SimpleNamespace(responded=True, response_events=[tool_event, msg_event])
        broadcast_result = SimpleNamespace(outputs={"w1": output}, streaming_responses=[])

        router = MagicMock()
        router.broadcast = AsyncMock(return_value=broadcast_result)
        kit._get_router = MagicMock(return_value=router)

        text = await run_agent_in_child_room(kit, "parent::task-1", "do the task")

        assert text == "the answer"
        # task message + tool-call event + final message all persisted.
        persisted_types = [e.type for e in kit.store.added]
        assert EventType.TOOL_CALL_END in persisted_types
        assert persisted_types.count(EventType.MESSAGE) == 2  # task + answer
