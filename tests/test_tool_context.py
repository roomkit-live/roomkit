"""Tests for per-call tool execution context (roomkit.tools.context).

The channel object is registered once per channel_id and shared by every
room it serves, so every accessor here must reflect the turn being
processed rather than any state stored on the channel: the room
(``current_tool_room_id()`` and the ``Room`` object itself,
``current_tool_room()``), its author (``current_tool_actor_id()``)
and the toolset it resolved (``current_tool_allowed_names()``).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, IdentificationStatus
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import (
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_call,
    current_tool_room,
    current_tool_room_id,
)
from roomkit.tools.context import _current_loop_ctx, _current_tool_call, _ToolLoopContext
from roomkit.tools.external import BeforeToolDecision
from tests.conftest import make_event
from tests.tool_loop_modes import respond


def _binding(room_id: str) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [{"name": "search", "description": "Search"}]},
    )


def _binding_without_tools(room_id: str) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


def _tool_round_responses() -> list[AIResponse]:
    return [
        AIResponse(
            content="Let me search.",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
        ),
        AIResponse(
            content="Done.",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        ),
    ]


class TestCurrentToolActorId:
    """The person a tool acts for is the author of the turn, never the identity
    a handler captured when it was built: one channel object serves every room
    and every speaker."""

    async def test_handler_sees_the_turn_author(self) -> None:
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_actor_id())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        await respond(
            ch,
            make_event(room_id="room-a", body="go", channel_id="sms1", participant_id="alice"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )

        assert seen == ["alice"]

    async def test_shared_channel_tracks_each_speaker(self, streaming: bool) -> None:
        """Two people, one channel object: each turn reports its own author."""
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_actor_id())
            return "ok"

        provider = MockAIProvider(
            ai_responses=_tool_round_responses() + _tool_round_responses(),
            streaming=streaming,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        for who in ("alice", "bob"):
            await respond(
                ch,
                make_event(room_id="room-a", body="go", channel_id="sms1", participant_id=who),
                _binding("room-a"),
                RoomContext(room=Room(id="room-a")),
            )

        assert seen == ["alice", "bob"]

    async def test_no_participant_reports_none(self, streaming: bool) -> None:
        """An event that names no participant has no author. ``None`` says
        so, so the caller decides rather than inheriting whoever spoke last."""
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_actor_id())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        await respond(
            ch,
            make_event(room_id="room-a", body="go", channel_id="sms1"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )

        assert seen == [None]

    async def test_streaming_handler_sees_the_turn_author(self) -> None:
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_actor_id())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        output = await ch.on_event(
            make_event(room_id="room-a", body="go", channel_id="sms1", participant_id="alice"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )
        assert output.response_stream is not None
        _ = [chunk async for chunk in output.response_stream]

        assert seen == ["alice"]

    async def test_an_unidentified_speaker_reads_back_the_same(self, streaming: bool) -> None:
        """The accessor names the turn, it does not vet it: a participant the
        room has not identified reports its id like any other, so a handler
        that treats the value as a principal has to check the roster itself."""
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_actor_id())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        unidentified = Participant(
            id="pending-9f3c2a1b",
            room_id="room-a",
            channel_id="sms1",
            identification=IdentificationStatus.PENDING,
        )
        await respond(
            ch,
            make_event(
                room_id="room-a", body="go", channel_id="sms1", participant_id=unidentified.id
            ),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a"), participants=[unidentified]),
        )

        assert seen == ["pending-9f3c2a1b"]

    def test_none_outside_tool_loop(self) -> None:
        assert current_tool_actor_id() is None


class TestCurrentToolRoomId:
    async def test_streaming_handler_sees_turn_room(self) -> None:
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room_id())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        output = await ch.on_event(
            make_event(room_id="room-a", body="go", channel_id="sms1"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )
        assert output.response_stream is not None
        _ = [chunk async for chunk in output.response_stream]

        assert seen == ["room-a"]

    async def test_non_streaming_handler_sees_turn_room(self) -> None:
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room_id())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        await respond(
            ch,
            make_event(room_id="room-b", body="go", channel_id="sms1"),
            _binding("room-b"),
            RoomContext(room=Room(id="room-b")),
        )

        assert seen == ["room-b"]

    async def test_shared_channel_tracks_each_turn(self, streaming: bool) -> None:
        """One channel object serving two rooms reports each turn's room."""
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room_id())
            return "ok"

        provider = MockAIProvider(
            ai_responses=_tool_round_responses() + _tool_round_responses(),
            streaming=streaming,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        for room_id in ("room-a", "room-b"):
            await respond(
                ch,
                make_event(room_id=room_id, body="go", channel_id="sms1"),
                _binding(room_id),
                RoomContext(room=Room(id=room_id)),
            )

        assert seen == ["room-a", "room-b"]

    async def test_interleaved_streams_keep_tool_and_hook_room(self) -> None:
        """Starting another room must not overwrite a suspended stream's room."""
        handler_rooms: list[tuple[str | None, str | None]] = []
        hook_rooms: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            tool_call = _current_tool_call.get()
            handler_rooms.append(
                (current_tool_room_id(), tool_call.room_id if tool_call is not None else None)
            )
            return "ok"

        async def before_tool_call(event: Any) -> BeforeToolDecision:
            hook_rooms.append(event.room_id)
            return BeforeToolDecision(allowed=True)

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
        ch._before_tool_call_hook = before_tool_call

        output_a = await ch.on_event(
            make_event(room_id="room-a", body="go", channel_id="sms1"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )
        output_b = await ch.on_event(
            make_event(room_id="room-b", body="go", channel_id="sms1"),
            _binding("room-b"),
            RoomContext(room=Room(id="room-b")),
        )

        assert output_a.response_stream is not None
        _ = [chunk async for chunk in output_a.response_stream]
        if output_b.response_stream is not None and hasattr(output_b.response_stream, "aclose"):
            await output_b.response_stream.aclose()

        assert handler_rooms == [("room-a", "room-a")]
        assert hook_rooms == ["room-a"]

    def test_none_outside_tool_loop(self) -> None:
        assert current_tool_room_id() is None


class TestCurrentToolRoom:
    """The turn's ``Room`` object, not only its id: the one ``RoomContext.room``
    holds, so a handler reads the room's metadata or organization without a
    store round trip, and reads the same snapshot the turn's providers hold."""

    async def test_non_streaming_handler_sees_the_turn_room_object(self) -> None:
        seen: list[Room | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
        context = RoomContext(room=Room(id="room-a", metadata={"tenant": "acme"}))

        await respond(
            ch,
            make_event(room_id="room-a", body="go", channel_id="sms1"),
            _binding("room-a"),
            context,
        )

        assert len(seen) == 1
        assert seen[0] is context.room
        assert seen[0].metadata["tenant"] == "acme"

    async def test_streaming_handler_sees_the_turn_room_object(self) -> None:
        seen: list[Room | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
        context = RoomContext(room=Room(id="room-a"))

        output = await ch.on_event(
            make_event(room_id="room-a", body="go", channel_id="sms1"), _binding("room-a"), context
        )
        assert output.response_stream is not None
        _ = [chunk async for chunk in output.response_stream]

        assert len(seen) == 1
        assert seen[0] is context.room

    async def test_second_round_still_sees_the_room(self, streaming: bool) -> None:
        """The loop context ``for_loop`` builds inherits the room: a handler
        called on the second tool round reads the same object as on the first."""
        seen: list[Room | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room())
            return "ok"

        two_rounds = [
            AIResponse(
                content="First.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[AIToolCall(id="tc1", name="search", arguments={"q": "x"})],
            ),
            AIResponse(
                content="Again.",
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                tool_calls=[AIToolCall(id="tc2", name="search", arguments={"q": "y"})],
            ),
            AIResponse(
                content="Done.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10},
            ),
        ]
        provider = MockAIProvider(ai_responses=two_rounds, streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
        context = RoomContext(room=Room(id="room-a"))

        await respond(
            ch,
            make_event(room_id="room-a", body="go", channel_id="sms1"),
            _binding("room-a"),
            context,
        )

        assert len(seen) == 2
        assert all(room is context.room for room in seen)

    async def test_shared_channel_tracks_each_turn_room(self, streaming: bool) -> None:
        """One channel object serving two rooms hands each turn its own Room."""
        seen: list[Room | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room())
            return "ok"

        provider = MockAIProvider(
            ai_responses=_tool_round_responses() + _tool_round_responses(),
            streaming=streaming,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)
        contexts = [RoomContext(room=Room(id=room_id)) for room_id in ("room-a", "room-b")]

        for context in contexts:
            await respond(
                ch,
                make_event(room_id=context.room.id, body="go", channel_id="sms1"),
                _binding(context.room.id),
                context,
            )

        assert [room is context.room for room, context in zip(seen, contexts, strict=True)] == [
            True,
            True,
        ]

    def test_none_outside_tool_loop(self) -> None:
        assert current_tool_room() is None

    def test_for_loop_takes_the_room_of_a_loop_started_without_a_turn(self) -> None:
        """No parent: the loop's Room is the one it is given, and the id
        follows the Room rather than the separately passed ``room_id``."""
        room = Room(id="r-solo")
        ctx = _ToolLoopContext.for_loop(None, "stale-id", room=room)
        assert ctx.room is room
        assert ctx.room_id == "r-solo"

    def test_for_loop_inherits_the_parent_room_by_reference(self) -> None:
        parent = _ToolLoopContext()
        parent.room = Room(id="r-parent")
        parent.room_id = "r-parent"
        ctx = _ToolLoopContext.for_loop(parent, None)
        assert ctx.room is parent.room
        assert ctx.room_id == "r-parent"


class TestCurrentToolAllowedNames:
    async def test_handler_sees_turn_toolset_streaming(self) -> None:
        seen: list[set[str] | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_allowed_names())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        output = await ch.on_event(
            make_event(room_id="room-a", body="go", channel_id="sms1"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )
        assert output.response_stream is not None
        _ = [chunk async for chunk in output.response_stream]

        assert len(seen) == 1
        assert seen[0] is not None
        assert "search" in seen[0]

    async def test_handler_sees_turn_toolset_non_streaming(self) -> None:
        seen: list[set[str] | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_allowed_names())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        await respond(
            ch,
            make_event(room_id="room-b", body="go", channel_id="sms1"),
            _binding("room-b"),
            RoomContext(room=Room(id="room-b")),
        )

        assert len(seen) == 1
        assert seen[0] is not None
        assert "search" in seen[0]

    def test_none_outside_tool_loop(self) -> None:
        assert current_tool_allowed_names() is None

    async def test_unoffered_tool_is_not_forwarded_non_streaming(self) -> None:
        handler = AsyncMock(return_value="must not run")
        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=handler)

        await respond(
            ch,
            make_event(room_id="room-empty", body="go", channel_id="sms1"),
            _binding_without_tools("room-empty"),
            RoomContext(room=Room(id="room-empty")),
        )

        handler.assert_not_awaited()

    async def test_unoffered_tool_is_not_forwarded_streaming(self) -> None:
        handler = AsyncMock(return_value="must not run")
        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=handler)

        output = await ch.on_event(
            make_event(room_id="room-empty", body="go", channel_id="sms1"),
            _binding_without_tools("room-empty"),
            RoomContext(room=Room(id="room-empty")),
        )
        assert output.response_stream is not None
        _ = [chunk async for chunk in output.response_stream]

        handler.assert_not_awaited()


class TestCurrentToolCall:
    def test_none_outside_a_tool_call(self) -> None:
        assert current_tool_call() is None

    async def test_handler_sees_its_call_and_may_fill_the_structured_copy(
        self, streaming: bool
    ) -> None:
        from roomkit.providers.ai.base import AIResponse, AIToolCall

        seen: list[Any] = []

        async def handler(name: str, args: dict[str, Any]) -> str:
            call = current_tool_call()
            assert call is not None
            seen.append((call.tool_call_id, call.room_id, call.channel_id))
            call.structured_content = {"rewritten": True}
            return "ok"

        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[AIToolCall(id="tc-9", name="search", arguments={"q": "x"})],
                ),
                AIResponse(content="done", finish_reason="stop"),
            ],
            streaming=streaming,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=handler)
        run = await respond(
            ch,
            make_event(room_id="r1", body="go", channel_id="sms1"),
            _binding("r1"),
            RoomContext(room=Room(id="r1")),
        )

        assert seen == [("tc-9", "r1", "ai1")]
        assert run.calls and run.calls[0].structured_content == {"rewritten": True}
        assert current_tool_call() is None


class TestLoopContextRestoration:
    """A tool loop puts back the loop context it replaced: a handler that runs
    a child channel's turn inside its own turn (delegation) reads its own room
    again once the child's answer, streamed or not, is consumed."""

    async def _child_turn(self, streaming: bool) -> None:
        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(return_value="ok"))
        output = await ch.on_event(
            make_event(room_id="child-room", body="go", channel_id="sms1"),
            _binding("child-room"),
            RoomContext(room=Room(id="child-room")),
        )
        if output.response_stream is not None:
            _ = [chunk async for chunk in output.response_stream]

    @pytest.mark.parametrize("streaming", [False, True])
    async def test_the_enclosing_context_survives_a_nested_turn(self, streaming: bool) -> None:
        outer = _ToolLoopContext()
        outer.room = Room(id="parent-room")
        outer.room_id = "parent-room"
        token = _current_loop_ctx.set(outer)
        try:
            await self._child_turn(streaming)
            assert current_tool_room_id() == "parent-room"
            assert current_tool_room() is outer.room
        finally:
            _current_loop_ctx.reset(token)

    async def test_the_turn_context_outlives_the_loop_within_the_turn(
        self, streaming: bool
    ) -> None:
        """The loop ends before the turn does, and what runs after it (the
        after-response hook) reads the turn's context, not None."""
        seen: list[str | None] = []

        async def after_response(event: Any) -> None:
            seen.append(current_tool_room_id())

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=streaming)
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(return_value="ok"))
        ch._after_response_hook = after_response
        await respond(
            ch,
            make_event(room_id="room-a", body="go", channel_id="sms1"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )

        assert seen == ["room-a"]

    async def test_a_stream_drained_in_another_task_restores_that_task_context(self) -> None:
        """The streamed turn runs in the consumer's context: ending it must not
        fail on a token from another context, and must put the consumer's own
        value back."""
        outer = _ToolLoopContext()
        outer.room_id = "consumer-room"
        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=True)
        ch = AIChannel("ai1", provider=provider, tool_handler=AsyncMock(return_value="ok"))
        output = await ch.on_event(
            make_event(room_id="child-room", body="go", channel_id="sms1"),
            _binding("child-room"),
            RoomContext(room=Room(id="child-room")),
        )
        assert output.response_stream is not None
        stream = output.response_stream

        async def drain() -> str | None:
            _current_loop_ctx.set(outer)
            _ = [chunk async for chunk in stream]
            return current_tool_room_id()

        assert await asyncio.create_task(drain()) == "consumer-room"
