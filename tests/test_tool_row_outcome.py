"""A stored tool row states how its call ended; the history a later turn
replays from the rows, and the memory rebuilt from them, keep the calls the
live one keeps (RFC §6.4, RMK-308).

Every case runs against a provider that streams and one read through its
``generate()``.
"""

from __future__ import annotations

import asyncio
from typing import Any, get_args

from roomkit import ToolCallOutcome, ToolFailedError, ToolRefusedError
from roomkit.channels._tool_history import tool_rounds
from roomkit.channels._tool_usage import ToolUsageMemory
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent, ToolCallContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools._outcome import OutcomeKind
from tests.test_framework import SimpleChannel

_TOOLS = [
    AITool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    for name in ("ok", "boom", "refuse", "fails")
]


def _call(call_id: str, name: str) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments={"q": call_id})],
    )


async def _handler(name: str, arguments: dict[str, Any]) -> str:
    if name == "boom":
        raise RuntimeError("postgres://user:secret@db")
    if name == "refuse":
        raise ToolRefusedError("not today")
    if name == "fails":
        raise ToolFailedError("the disk is full")
    return f"answer-{arguments['q']}"


async def _turn(streaming: bool, *calls: AIResponse) -> tuple[RoomKit, AIChannel]:
    provider = MockAIProvider(
        ai_responses=[*calls, AIResponse(content="Done.")], streaming=streaming
    )
    ai = AIChannel(
        "ai1",
        provider=provider,
        tools=_TOOLS,
        tool_handler=_handler,
        tool_search=False,
        evict_threshold_tokens=200,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)
    return kit, ai


async def _outcomes(kit: RoomKit) -> dict[str, str | None]:
    events = await kit.store.list_events("r1")
    return {
        e.content.tool_name: e.content.outcome
        for e in events
        if e.type == EventType.TOOL_CALL_END and isinstance(e.content, ToolCallContent)
    }


async def _replayed(kit: RoomKit, ai: AIChannel) -> list[str]:
    """The calls a later turn's history replays, read from the stored rows."""
    rows = await kit.turn_executor.wiring.build_tool_history_loader(ai.channel_id)("r1", -1)
    rounds = tool_rounds(rows, resolves=lambda _: False)
    return [part.name for r in rounds for part in r.calls.content]  # type: ignore[union-attr]


def test_the_stored_vocabulary_is_the_outcome_kinds() -> None:
    assert set(get_args(ToolCallOutcome)) == {kind.value for kind in OutcomeKind}


async def test_each_end_row_states_how_its_call_ended(streaming: bool) -> None:
    kit, _ = await _turn(
        streaming,
        _call("c1", "ok"),
        _call("c2", "boom"),
        _call("c3", "refuse"),
        _call("c4", "ghost"),
        _call("c5", "fails"),
    )

    assert await _outcomes(kit) == {
        "ok": "served",
        "boom": "failed",
        "refuse": "refused",
        "ghost": "refused",
        "fails": "failed",
    }


async def test_the_rebuilt_memory_keeps_what_the_live_one_keeps(streaming: bool) -> None:
    """Refusals stay out of the tool memory, served and failed calls are in;
    the history replays every call the turn ended, as it was sent (RMK-1370)."""
    kit, ai = await _turn(
        streaming,
        _call("c1", "ok"),
        _call("c2", "boom"),
        _call("c3", "refuse"),
        _call("c4", "ghost"),
        _call("c5", "fails"),
    )

    rebuilt = ToolUsageMemory(recorded=ai._in_tool_memory)
    rebuilt.seed("r1", await kit._build_tool_usage_loader(ai.channel_id)("r1"))

    assert await _replayed(kit, ai) == ["ok", "boom", "refuse", "ghost", "fails"]
    assert rebuilt.tool_names("r1") == ai._tool_usage.tool_names("r1") == {"ok", "boom", "fails"}
