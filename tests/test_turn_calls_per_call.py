"""Each call of a text turn is held as a call of its own (RMK-506, RFC §9.3, §12.4).

A provider's id names a call within its round. A call under an id an earlier
round used is a new call: reported, and cancelled when the turn cuts it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai.base import AIResponse, AIToolCall, ServedCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel
from tests.tool_doors import DOORS, TOOL

TEXT_DOORS = [door for door in DOORS if door.startswith("text-")]


def _call(q: str, *, name: str = "lookup", served: str | None = None) -> AIToolCall:
    ran = ServedCall(result=served) if served is not None else None
    return AIToolCall(id="c1", name=name, arguments={"q": q}, served=ran)


def _provider(rounds: list[list[AIToolCall]], door: str) -> MockAIProvider:
    responses = [AIResponse(content="", finish_reason="tool_calls", tool_calls=c) for c in rounds]
    return MockAIProvider(
        ai_responses=[*responses, AIResponse(content="done")], streaming=door == "text-stream"
    )


async def _room(channel: AIChannel) -> tuple[RoomKit, list[ToolCallEvent]]:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", channel.channel_id, category=ChannelCategory.INTELLIGENCE)
    reports: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, context: Any) -> None:
        reports.append(event)

    return kit, reports


def _message() -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Go."))


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(300):
        if condition():
            return
        await asyncio.sleep(0.01)


async def _cut(kit: RoomKit, ready: Callable[[], bool]) -> None:
    """Run a turn until *ready*, then cancel it from outside."""
    turn = asyncio.create_task(kit.process_inbound(_message()))
    await _until(ready)
    turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn


async def _rows(kit: RoomKit) -> tuple[int, list[tuple[str | None, str]]]:
    """How many start rows the turn stored, and each end row's outcome and query."""
    events = await kit.store.list_events("r")
    starts = sum(event.type == EventType.TOOL_CALL_START for event in events)
    ends = [
        (event.content.outcome, event.content.arguments["q"])
        for event in events
        if event.type == EventType.TOOL_CALL_END and isinstance(event.content, ToolCallContent)
    ]
    return starts, ends


def _outcomes(reports: list[ToolCallEvent]) -> list[tuple[str, bool, bool, bool]]:
    return sorted((r.arguments["q"], r.refused, r.cancelled, r.is_error) for r in reports)


@pytest.mark.parametrize("door", TEXT_DOORS)
async def test_a_call_under_an_id_an_earlier_round_used_is_a_new_call(door: str) -> None:
    ran: list[str] = []

    async def serves(name: str, arguments: dict[str, Any]) -> str:
        ran.append(arguments["q"])
        return f"fine {arguments['q']}"

    rounds = [[_call("first")], [_call("second")]]
    channel = AIChannel("ai", provider=_provider(rounds, door), tools=[TOOL], tool_handler=serves)
    kit, reports = await _room(channel)

    await kit.process_inbound(_message())
    await _until(lambda: len(reports) == 2)
    _, ends = await _rows(kit)
    await kit.close()

    assert ran == ["first", "second"]
    assert _outcomes(reports) == [("first", False, False, False), ("second", False, False, False)]
    assert ends == [("served", "first"), ("served", "second")]


@pytest.mark.parametrize("door", TEXT_DOORS)
async def test_a_reused_id_call_the_turn_cuts_is_reported_cancelled(door: str) -> None:
    """The earlier round's call under the id was reported: the cut call is
    still owed its own report, cancelled, and its end row."""
    ran: list[str] = []

    async def hangs_second(name: str, arguments: dict[str, Any]) -> str:
        ran.append(arguments["q"])
        if arguments["q"] == "second":
            await asyncio.sleep(30)
        return "fine"

    rounds = [[_call("first")], [_call("second")]]
    channel = AIChannel(
        "ai", provider=_provider(rounds, door), tools=[TOOL], tool_handler=hangs_second
    )
    kit, reports = await _room(channel)

    await _cut(kit, lambda: "second" in ran)
    await _until(lambda: len(reports) == 2)
    starts, ends = await _rows(kit)
    await kit.close()

    assert _outcomes(reports) == [("first", False, False, False), ("second", False, True, True)]
    assert (starts, ends) == (2, [("served", "first"), ("cancelled", "second")])
