"""run_agent_in_child_room persists the worker's FULL trace.

A delegated agent's child room must record what it actually did — each tool
call (with arguments and result) plus its text — not just the final answer, so
the room is a complete, linkable transcript. The parent link lives in the child
room's ``metadata.parent_room_id`` (asserted in test_integration), so the
parent↔child relationship is rebuildable from persistence alone.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.event_router import BroadcastResult
from roomkit.core.exceptions import TaskCutShortError
from roomkit.core.framework import RoomKit
from roomkit.core.mixins._child_execution import _broadcast_and_collect, _collect_answer
from roomkit.core.mixins.delegation import _persist_child_stream, run_agent_in_child_room
from roomkit.models.channel import ChannelOutput
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.models.store_filter import EventFilter
from roomkit.models.streaming import ThinkingDeltaMarker, ToolCallEndMarker, ToolCallStartMarker
from roomkit.providers.ai.base import AIImagePart, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_tool_call
from tests.buffered_agent import BufferedAgent


def _recording_store() -> MagicMock:
    store = MagicMock()
    store.added = []

    async def _add(room_id: str, event: RoomEvent) -> RoomEvent:
        store.added.append(event)
        return event

    store.add_event_auto_index = AsyncMock(side_effect=_add)
    store.commit_event = AsyncMock(side_effect=_add)
    return store


# The task's message in its child room: what the delegated turn answers.
_TASK_MESSAGE = RoomEvent(
    id="task-message",
    room_id="parent::task-3",
    source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
    content=TextContent(body="Do the task."),
)


def _sr(stream: Any) -> SimpleNamespace:
    return SimpleNamespace(
        stream=stream,
        source_channel_id="agent:w1",
        source_channel_type=ChannelType.AI,
        response_metadata={},
        turn_record=None,
        trigger_event=_TASK_MESSAGE,
    )


def _recording_kit() -> MagicMock:
    kit = MagicMock()
    kit.store = _recording_store()
    kit._commit_indexed = kit.store.commit_event
    # No ON_ERROR hook to fire a failed turn's error to.
    kit._hook_context = AsyncMock(return_value=None)
    return kit


class TestPersistChildStream:
    async def test_persists_tool_calls_and_text_segments_in_order(self) -> None:
        kit = _recording_kit()

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

        # Return value is the last segment, the worker's answer, as a
        # non-streaming worker's last message is (RMK-289).
        assert text == "Here is the answer."

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

    async def test_a_tool_end_keeps_a_bounded_share_of_its_images(self) -> None:
        """The child room is persisted like any room: its TOOL_CALL_END events
        keep at most 512 KB of a result's images (RMK-260)."""
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()
        kit._report_intelligence_errors = AsyncMock()
        header = "data:image/png;base64,"
        shot = AIImagePart(url=header + "A" * (300 * 1024 - len(header)), mime_type="image/png")

        async def _stream() -> Any:
            yield ToolCallStartMarker(tool_name="shoot", tool_id="t1", arguments={})
            yield ToolCallEndMarker(
                tool_name="shoot", tool_id="t1", result=[shot, shot, shot], status="completed"
            )

        await _persist_child_stream(kit, "parent::task-9", _sr(_stream()), chain_depth=1)

        end = next(e for e in kit.store.added if e.type == EventType.TOOL_CALL_END)
        assert sum(isinstance(p, AIImagePart) for p in end.content.result) == 1

    async def test_thinking_markers_are_not_persisted(self) -> None:
        kit = _recording_kit()

        async def _stream() -> Any:
            yield ThinkingDeltaMarker(thinking="hmm")
            yield "final answer"

        text = await _persist_child_stream(kit, "parent::task-2", _sr(_stream()), chain_depth=1)
        assert text == "final answer"
        # Only the text segment is persisted — thinking is transient.
        assert [e.type for e in kit.store.added] == [EventType.MESSAGE]

    async def test_text_only_stream_persists_single_message(self) -> None:
        kit = _recording_kit()

        async def _stream() -> Any:
            yield "just "
            yield "text"

        text = await _persist_child_stream(kit, "parent::task-3", _sr(_stream()), chain_depth=1)
        assert text == "just text"
        assert len(kit.store.added) == 1
        assert kit.store.added[0].content.body == "just text"
        # The child room's rows name the task message they answer (RFC §8.5).
        assert kit.store.added[0].responds_to == "task-message"


class TestAChildTraceCutShort:
    """RMK-291: a delegated turn cut short leaves no call open in its child room."""

    async def test_a_failed_stream_closes_its_open_call(self) -> None:
        kit = _recording_kit()

        async def _stream() -> Any:
            yield "Looking. "
            yield ToolCallStartMarker(tool_name="search", tool_id="t1", arguments={})
            raise RuntimeError("upstream 500")

        with pytest.raises(RuntimeError, match="upstream 500"):
            await _persist_child_stream(kit, "parent::task-4", _sr(_stream()), chain_depth=1)

        rows = [(e.type, e.content) for e in kit.store.added]
        assert [t for t, _ in rows] == [
            EventType.MESSAGE,
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_END,
        ]
        end = rows[-1][1]
        assert (end.tool_id, end.status, end.error) == ("t1", "failed", "turn failed")

    async def test_a_cancelled_stream_keeps_its_text_marked_cancelled(self) -> None:
        kit = _recording_kit()
        produced = asyncio.Event()

        async def _stream() -> Any:
            yield "Half an ans"
            produced.set()
            await asyncio.sleep(3600)
            yield "wer."

        task = asyncio.create_task(
            _persist_child_stream(kit, "parent::task-5", _sr(_stream()), chain_depth=1)
        )
        await asyncio.wait_for(produced.wait(), 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        (message,) = kit.store.added
        assert message.content.body == "Half an ans"
        assert message.metadata.get("cancelled") is True

    async def test_a_delegation_cancelled_mid_write_closes_its_response(self) -> None:
        """Nothing else reads a delegated response: a cancelled one is closed,
        so its generation ends with the delegation (RFC §12.2 step 13s)."""
        kit = _recording_kit()
        writing, release = asyncio.Event(), asyncio.Event()
        closed: list[bool] = []

        async def _slow_commit(room_id: str, event: RoomEvent) -> RoomEvent:
            writing.set()
            await release.wait()
            kit.store.added.append(event)
            return event

        kit._commit_indexed = AsyncMock(side_effect=_slow_commit)

        async def _stream() -> Any:
            try:
                yield "Looking. "
                yield ToolCallStartMarker(tool_name="search", tool_id="t1", arguments={})
                yield "never read"
            finally:
                closed.append(True)

        task = asyncio.create_task(
            _persist_child_stream(kit, "parent::task-6", _sr(_stream()), chain_depth=1)
        )
        await asyncio.wait_for(writing.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(task, return_exceptions=True)

        assert closed == [True]
        assert [e.type for e in kit.store.added] == [
            EventType.MESSAGE,
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_END,
        ]

    async def test_a_delegation_cancelled_while_its_tool_runs_leaves_no_call_open(
        self, streaming: bool
    ) -> None:
        started = asyncio.Event()

        async def _slow(name: str, args: dict[str, Any]) -> str:
            started.set()
            await asyncio.sleep(3600)
            return "never"

        kit = _delegating_kit(
            MockAIProvider(
                streaming=streaming,
                ai_responses=[
                    AIResponse(
                        content="Working.",
                        finish_reason="tool_calls",
                        tool_calls=[AIToolCall(id="tc1", name="slow", arguments={})],
                    ),
                    AIResponse(content="Done."),
                ],
            ),
            _slow,
            "slow",
        )
        await kit.create_room(room_id="parent")
        task = asyncio.create_task(kit.delegate("parent", "worker", "go", wait=True))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        rows = await _child_rows(kit)
        # Every start has its end: a start row never stays pending, and the
        # text already produced is kept, marked cancelled (RFC §23.3).
        assert [(t, getattr(c, "status", None)) for t, c in rows] == [
            (EventType.MESSAGE, None),
            (EventType.TOOL_CALL_START, "pending"),
            (EventType.TOOL_CALL_END, "failed"),
        ]
        assert rows[2][1].error == "cancelled"
        assert rows[0][1].body == "Working."
        # The response is closed, so its generation ends with the delegation.
        assert not kit.channels["worker"]._active_loops
        await kit.close()

    async def test_a_tool_end_keeps_its_structured_copy(self, streaming: bool) -> None:
        async def _query(name: str, args: dict[str, Any]) -> str:
            current_tool_call().structured_content = {"rows": [1, 2, 3]}
            return "3 rows"

        kit = _delegating_kit(
            MockAIProvider(
                streaming=streaming,
                ai_responses=[
                    AIResponse(
                        content="",
                        finish_reason="tool_calls",
                        tool_calls=[AIToolCall(id="tc1", name="query", arguments={})],
                    ),
                    AIResponse(content="Done."),
                ],
            ),
            _query,
            "query",
        )
        await kit.create_room(room_id="parent")

        await kit.delegate("parent", "worker", "go", wait=True)

        end = next(c for t, c in await _child_rows(kit) if t == EventType.TOOL_CALL_END)
        assert (end.result, end.structured_content) == ("3 rows", {"rows": [1, 2, 3]})
        await kit.close()


def _delegating_kit(provider: MockAIProvider, handler: Any, tool: str) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(
        AIChannel(
            "worker",
            provider=provider,
            tool_handler=handler,
            tools=[AITool(name=tool, description="d")],
            tool_search=False,
        )
    )
    return kit


async def _child_events(kit: RoomKit) -> list[RoomEvent]:
    (child,) = [r for r in await kit.store.list_rooms() if r.id != "parent"]
    return [e for e in await kit.store.list_events(child.id) if e.source.channel_id == "worker"]


async def _child_rows(kit: RoomKit) -> list[tuple[EventType, Any]]:
    return [(e.type, e.content) for e in await _child_events(kit)]


class TestEveryResponseIsRead:
    """RMK-291 review: a delegated broadcast's responses are all read (RFC §8.3)."""

    async def test_a_second_response_is_read_and_the_first_answer_wins(self) -> None:
        kit = _recording_kit()
        read: list[str] = []

        def _stream(name: str) -> Any:
            async def _gen() -> Any:
                read.append(name)
                yield f"{name}'s answer"

            return _sr(_gen())

        result = BroadcastResult(streaming_responses=[_stream("w1"), _stream("w2")])

        text = await _collect_answer(kit, "parent::task-7", result, 1)

        assert text == "w1's answer"
        assert read == ["w1", "w2"]
        assert [e.content.body for e in kit.store.added] == ["w1's answer", "w2's answer"]

    async def test_a_failed_response_fails_the_turn_once_all_are_read(self) -> None:
        kit = _recording_kit()

        async def _failing() -> Any:
            yield "Looking."
            raise RuntimeError("upstream 500")

        async def _answering() -> Any:
            yield "Done."

        result = BroadcastResult(streaming_responses=[_sr(_failing()), _sr(_answering())])

        with pytest.raises(RuntimeError, match="upstream 500"):
            await _collect_answer(kit, "parent::task-8", result, 1)
        assert [e.content.body for e in kit.store.added] == ["Looking.", "Done."]


class TestRunAgentNonStreaming:
    async def test_persists_all_response_events_not_just_final_text(self) -> None:
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()
        kit._report_intelligence_errors = AsyncMock()
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
        output = ChannelOutput(responded=True, response_events=[tool_event, msg_event])
        broadcast_result = BroadcastResult(outputs={"w1": output})

        router = MagicMock()
        router.broadcast = AsyncMock(return_value=broadcast_result)
        kit._get_router = MagicMock(return_value=router)

        text = await run_agent_in_child_room(kit, "parent::task-1", "do the task")

        assert text == "the answer"
        # task message + tool-call event + final message all persisted.
        persisted_types = [e.type for e in kit.store.added]
        assert EventType.TOOL_CALL_END in persisted_types
        assert persisted_types.count(EventType.MESSAGE) == 2  # task + answer

    async def test_a_buffered_turn_cut_short_fails_with_its_narration(self) -> None:
        """RMK-414, RFC §23.3: a buffered agent's turn its round cap cut has no
        answer; its narration rides the failure."""
        kit = MagicMock()
        kit.store = _recording_store()
        kit._commit_indexed = kit.store.commit_event
        kit._commit_blocked_events = AsyncMock()
        kit._persist_side_effects = AsyncMock()
        kit._report_intelligence_errors = AsyncMock()
        kit.get_room = AsyncMock(
            return_value=Room(id="parent::task-1", metadata={"parent_room_id": "parent"})
        )
        kit.store.list_bindings = AsyncMock(return_value=[])
        kit.store.list_events = AsyncMock(return_value=[])
        narration = RoomEvent(
            room_id="parent::task-1",
            source=EventSource(channel_id="agent:w1", channel_type=ChannelType.AI),
            type=EventType.MESSAGE,
            content=TextContent(body="Still checking."),
            metadata={"loop_end_reason": "max_rounds"},
        )
        output = ChannelOutput(responded=True, response_events=[narration])
        router = MagicMock()
        router.broadcast = AsyncMock(return_value=BroadcastResult(outputs={"w1": output}))
        kit._get_router = MagicMock(return_value=router)

        with pytest.raises(TaskCutShortError) as cut:
            await run_agent_in_child_room(kit, "parent::task-1", "do the task")

        assert (cut.value.reason, cut.value.narration) == ("max_rounds", "Still checking.")


async def test_a_muted_worker_answer_is_committed_once() -> None:
    """A delegated room's trace commits each answer once: a muted worker's
    buffered answer used to be stored twice under one id, BLOCKED by the
    router and DELIVERED by the trace (found reviewing RMK-344). A muted
    AIChannel's stream is closed unread, so the case is a buffered agent's."""
    kit = RoomKit()
    kit.register_channel(BufferedAgent("ai1", "W1"))
    await kit.create_room(room_id="child")
    await kit.attach_channel("child", "ai1", category=ChannelCategory.INTELLIGENCE, muted=True)

    await _broadcast_and_collect(kit, "child", "task")

    events = await kit.store.list_events("child", event_filter=EventFilter(include_blocked=True))
    assert len([e for e in events if e.source.channel_id == "ai1"]) == 1
    await kit.close()
