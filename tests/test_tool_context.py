"""Tests for per-call tool execution context (roomkit.tools.context).

The channel object is registered once per channel_id and shared by every
room it serves — ``current_tool_room_id()`` must reflect the room of the
turn being processed, not any state stored on the channel.
"""

from __future__ import annotations

from typing import Any

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools import current_tool_allowed_names, current_tool_room_id
from tests.conftest import make_event


def _binding(room_id: str) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [{"name": "search", "description": "Search"}]},
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
