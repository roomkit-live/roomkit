"""Claims obtained through the public tool context follow their turn's calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from tests.conftest import make_event
from tests.tool_loop_modes import respond

from roomkit import AIChannel, ChannelBinding, ChannelCategory, ChannelType, Room, RoomContext
from roomkit.providers.ai import AIResponse, AITool, AIToolCall, MockAIProvider
from roomkit.tools import (
    ToolCallContext,
    current_tool_room_id,
    tool_turn_context,
    turn_report_claim,
)

Claim = Callable[[], bool]
Handler = Callable[[str, dict[str, Any]], Awaitable[str]]


def _call(q: str) -> AIResponse:
    return AIResponse(
        content="",
        tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"q": q})],
    )


def _channel(streaming: bool, handler: Handler, responses: list[AIResponse]) -> AIChannel:
    return AIChannel(
        "ai",
        provider=MockAIProvider(ai_responses=responses, streaming=streaming),
        tools=[AITool(name="lookup", description="Look up", parameters={"type": "object"})],
        tool_handler=handler,
    )


async def _respond(channel: AIChannel, room_id: str = "r1") -> None:
    binding = ChannelBinding(
        channel_id="ai",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    result = await respond(
        channel,
        make_event(body="go", room_id=room_id),
        binding,
        RoomContext(room=Room(id=room_id)),
    )
    assert result.text == "done"


def test_no_claim_exists_without_an_announced_turn_call() -> None:
    assert turn_report_claim("c1", "ai") is None
    call = ToolCallContext(room_id="r1", channel_id="ai", tool_call_id="c1")
    with tool_turn_context(room_id="r1", call=call):
        assert turn_report_claim("c1", "ai") is None


async def test_a_claim_matches_its_channel_and_announced_id(streaming: bool) -> None:
    claimed: list[bool] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        assert turn_report_claim("missing", "ai") is None
        assert turn_report_claim("c1", "other") is None
        first = turn_report_claim("c1", "ai")
        second = turn_report_claim("c1", "ai")
        assert first is not None and second is not None
        claimed.extend((first(), second(), first()))
        return "found"

    channel = _channel(streaming, handler, [_call("once"), AIResponse(content="done")])
    try:
        await _respond(channel)
    finally:
        await channel.close()
    assert claimed == [True, False, False]
    assert turn_report_claim("c1", "ai") is None


async def test_a_retained_callback_resolves_a_reused_id_in_its_turn(streaming: bool) -> None:
    retained: list[Claim] = []
    claimed: list[bool] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None
        if not retained:
            retained.append(claim)
        claimed.extend((retained[0](), claim()))
        return str(arguments["q"])

    channel = _channel(
        streaming, handler, [_call("first"), _call("second"), AIResponse(content="done")]
    )
    try:
        await _respond(channel)
    finally:
        await channel.close()
    assert claimed == [True, False, True, False]
    assert retained[0]() is False


async def test_a_completed_calls_callback_keeps_its_claim_after_the_turn(streaming: bool) -> None:
    retained: list[Claim] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None
        retained.append(claim)
        return "found"

    channel = _channel(streaming, handler, [_call("once"), AIResponse(content="done")])
    try:
        await _respond(channel)
    finally:
        await channel.close()
    assert turn_report_claim("c1", "ai") is None
    assert [claim() for claim in retained] == [False]


async def test_a_callback_keeps_its_turn_inside_an_explicit_context(streaming: bool) -> None:
    claimed: list[bool] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None
        with tool_turn_context(room_id="another"):
            assert turn_report_claim("c1", "ai") is None
            claimed.append(claim())
        restored = turn_report_claim("c1", "ai")
        assert restored is not None
        claimed.append(restored())
        return "found"

    channel = _channel(streaming, handler, [_call("once"), AIResponse(content="done")])
    try:
        await _respond(channel)
    finally:
        await channel.close()
    assert claimed == [True, False]


async def test_two_turns_on_one_channel_keep_separate_claims(streaming: bool) -> None:
    entered: set[str] = set()
    ready = asyncio.Event()
    claimed: dict[str, tuple[bool, bool]] = {}

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        room = current_tool_room_id()
        claim = turn_report_claim("c1", "ai")
        assert room is not None and claim is not None
        entered.add(room)
        if len(entered) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 5)
        claimed[room] = (claim(), claim())
        return "found"

    responses = [
        _call("first"),
        _call("second"),
        AIResponse(content="done"),
        AIResponse(content="done"),
    ]
    channel = _channel(streaming, handler, responses)
    try:
        await asyncio.gather(_respond(channel, "r1"), _respond(channel, "r2"))
    finally:
        await channel.close()
    assert claimed == {"r1": (True, False), "r2": (True, False)}


async def test_concurrent_turns_can_invoke_each_others_captured_claims(streaming: bool) -> None:
    callbacks: dict[str, Claim] = {}
    ready, first_finished = asyncio.Event(), asyncio.Event()
    claimed: dict[str, tuple[bool, bool]] = {}

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        room = current_tool_room_id()
        claim = turn_report_claim("c1", "ai")
        assert room is not None and claim is not None
        callbacks[room] = claim
        if len(callbacks) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 3)
        if room == "r1":
            claimed[room] = (callbacks["r2"](), callbacks["r1"]())
            first_finished.set()
        else:
            await asyncio.wait_for(first_finished.wait(), 3)
            claimed[room] = (callbacks["r2"](), callbacks["r1"]())
        return "found"

    channel = _channel(
        streaming,
        handler,
        [
            _call("first"),
            _call("second"),
            AIResponse(content="done"),
            AIResponse(content="done"),
        ],
    )
    try:
        await asyncio.gather(_respond(channel, "r1"), _respond(channel, "r2"))
    finally:
        await channel.close()
    assert claimed == {"r1": (True, True), "r2": (False, False)}


async def test_an_earlier_turns_callback_cannot_claim_a_later_turn(streaming: bool) -> None:
    retained: list[Claim] = []
    claimed: list[bool] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None
        if retained:
            claimed.append(retained[0]())
        retained.append(claim)
        claimed.append(claim())
        return "found"

    responses = [
        _call("first"),
        AIResponse(content="done"),
        _call("second"),
        AIResponse(content="done"),
    ]
    channel = _channel(streaming, handler, responses)
    try:
        await _respond(channel)
        assert retained[0]() is False
        await _respond(channel)
    finally:
        await channel.close()
    assert claimed == [True, False, True]


async def test_an_unreported_failure_remains_claimable_after_its_turn(streaming: bool) -> None:
    retained: list[Claim] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None
        retained.append(claim)
        raise RuntimeError("failed before reporting")

    channel = _channel(streaming, handler, [_call("once"), AIResponse(content="done")])
    try:
        await _respond(channel)
    finally:
        await channel.close()
    assert turn_report_claim("c1", "ai") is None
    assert len(retained) == 1
    assert retained[0]() is True
    assert retained[0]() is False


async def test_an_unreported_earlier_call_keeps_its_claim_during_a_later_turn(
    streaming: bool,
) -> None:
    retained: list[Claim] = []
    claimed: list[bool] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        claim = turn_report_claim("c1", "ai")
        assert claim is not None
        if not retained:
            retained.append(claim)
            raise RuntimeError("failed before reporting")
        claimed.extend((retained[0](), claim(), retained[0](), claim()))
        return "found"

    responses = [
        _call("first"),
        AIResponse(content="done"),
        _call("second"),
        AIResponse(content="done"),
    ]
    channel = _channel(streaming, handler, responses)
    try:
        await _respond(channel)
        await _respond(channel)
    finally:
        await channel.close()
    assert claimed == [True, True, False, False]
    assert retained[0]() is False
