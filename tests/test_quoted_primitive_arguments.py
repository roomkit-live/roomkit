"""A primitive the model's server quoted reaches the tool as its declared type,
on every door (RMK-605).

Some servers turn a model's tool call into JSON by the schema the request
declared, so every value of a tool the request did not declare (a catalogue
tool recovered at call time) arrives as a string. The gate reads a string that
spells the declared primitive's literal as that primitive before validating,
as it folds a hub tool's hoisted arguments: the call is served instead of being
refused round after round for a value the model cannot send otherwise.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider
from tests.conference.test_conference_realtime import ROOM, realtime_kit
from tests.conftest import make_event
from tests.test_realtime_tool_recovery import _session
from tests.tool_loop_modes import respond

MOVE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"card_id": {"type": "string"}, "position": {"type": "integer"}},
    "required": ["card_id", "position"],
    "additionalProperties": False,
}
FILLER = [AITool(name=f"x{i}", description=f"x{i} tool", parameters={}) for i in range(30)]


class _Handler:
    """A host tool handler recording the arguments each call reached it with."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append(arguments)
        return "moved"


async def _text_turn(*, tool_search: bool) -> _Handler:
    """One turn whose model calls ``move_card`` with a quoted position."""
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.enums import ChannelType

    handler = _Handler()
    call = AIToolCall(id="call-1", name="move_card", arguments={"card_id": "c1", "position": "0"})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[
            AITool(name="move_card", description="Move a card", parameters=MOVE_PARAMS),
            *FILLER,
        ],
        tool_handler=handler,
        tool_search=tool_search,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")
    binding = ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI)
    event = make_event(room_id="r1", body="move it", channel_id="sms1")
    await respond(channel, event, binding, await kit._build_context("r1"))
    if tool_search:
        # The incident's shape: the tool was not declared, and was recovered.
        assert "move_card" not in {
            t.name for t in provider.calls[0].tools or [] if not t.defer_loading
        }
    await kit.close()
    return handler


async def test_a_declared_tool_reads_a_quoted_integer() -> None:
    handler = await _text_turn(tool_search=False)
    assert handler.calls == [{"card_id": "c1", "position": 0}]


async def test_a_tool_recovered_at_call_time_reads_a_quoted_integer() -> None:
    handler = await _text_turn(tool_search=True)
    assert handler.calls == [{"card_id": "c1", "position": 0}]


async def test_a_realtime_call_reads_a_quoted_integer() -> None:
    seen: list[dict[str, Any]] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        seen.append(arguments)
        return "sunny"

    provider = MockRealtimeProvider()
    kit, _channel, session = await _session(provider, "rt-quoted", handler=handler)

    await provider.simulate_tool_call(session, "call-1", "lookup", {"city": "Paris", "limit": "3"})
    await asyncio.sleep(0.1)

    assert seen == [{"city": "Paris", "limit": 3}]
    await kit.close()


async def test_a_conference_call_reads_a_quoted_integer() -> None:
    seen: list[dict[str, Any]] = []

    async def handler(room_id: str, tool: str, arguments: dict[str, Any]) -> str:
        seen.append(arguments)
        return "found"

    provider = MockRealtimeProvider()
    tool = {"name": "move_card", "description": "Move a card", "parameters": MOVE_PARAMS}
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(provider=provider, tools=[tool], tool_handler=handler),
    )
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(
        session, "call-1", "move_card", {"card_id": "c1", "position": "0"}
    )
    for _ in range(50):
        if seen:
            break
        await asyncio.sleep(0.02)

    assert seen == [{"card_id": "c1", "position": 0}]
    await kit.close()
