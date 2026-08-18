"""Tests for per-call tool execution context (roomkit.tools.context).

The channel object is registered once per channel_id and shared by every
room it serves — ``current_tool_room_id()`` must reflect the room of the
turn being processed, not any state stored on the channel.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from roomkit.channels.ai import AIChannel, _current_loop_ctx, _ToolLoopContext
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import (
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_room_id,
)
from roomkit.tools.context import _current_tool_call
from roomkit.tools.external import BeforeToolDecision
from tests.conftest import make_event


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

        await ch.on_event(
            make_event(room_id="room-a", body="go", channel_id="sms1", participant_id="alice"),
            _binding("room-a"),
            RoomContext(room=Room(id="room-a")),
        )

        assert seen == ["alice"]

    async def test_shared_channel_tracks_each_speaker(self) -> None:
        """Two people, one channel object: each turn reports its own author."""
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_actor_id())
            return "ok"

        provider = MockAIProvider(
            ai_responses=_tool_round_responses() + _tool_round_responses(),
            streaming=False,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        for who in ("alice", "bob"):
            await ch.on_event(
                make_event(room_id="room-a", body="go", channel_id="sms1", participant_id=who),
                _binding("room-a"),
                RoomContext(room=Room(id="room-a")),
            )

        assert seen == ["alice", "bob"]

    async def test_no_participant_reports_none(self) -> None:
        """A system injection has no author. ``None`` says so, so the caller
        decides rather than inheriting whoever spoke last."""
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_actor_id())
            return "ok"

        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        await ch.on_event(
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

        await ch.on_event(
            make_event(room_id="room-b", body="go", channel_id="sms1"),
            _binding("room-b"),
            RoomContext(room=Room(id="room-b")),
        )

        assert seen == ["room-b"]

    async def test_shared_channel_tracks_each_turn(self) -> None:
        """One channel object serving two rooms reports each turn's room."""
        seen: list[str | None] = []

        async def tool_handler(name: str, args: dict[str, Any]) -> str:
            seen.append(current_tool_room_id())
            return "ok"

        provider = MockAIProvider(
            ai_responses=_tool_round_responses() + _tool_round_responses(),
            streaming=False,
        )
        ch = AIChannel("ai1", provider=provider, tool_handler=tool_handler)

        for room_id in ("room-a", "room-b"):
            await ch.on_event(
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

        await ch.on_event(
            make_event(room_id="room-b", body="go", channel_id="sms1"),
            _binding("room-b"),
            RoomContext(room=Room(id="room-b")),
        )

        assert len(seen) == 1
        assert seen[0] is not None
        assert "search" in seen[0]

    def test_none_outside_tool_loop(self) -> None:
        assert current_tool_allowed_names() is None

    def test_empty_set_for_a_resolved_turn_without_tools(self) -> None:
        ctx = _ToolLoopContext(room_id="room-empty", all_context_tools=[])
        token = _current_loop_ctx.set(ctx)
        try:
            assert current_tool_allowed_names() == set()
        finally:
            _current_loop_ctx.reset(token)

    async def test_unoffered_tool_is_not_forwarded_non_streaming(self) -> None:
        handler = AsyncMock(return_value="must not run")
        provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=False)
        ch = AIChannel("ai1", provider=provider, tool_handler=handler)

        await ch.on_event(
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
