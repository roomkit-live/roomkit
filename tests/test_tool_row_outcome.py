"""A stored tool row states how its call ended, and the memory rebuilt from the
rows keeps the calls the live one keeps (RFC §6.4, RMK-308).

The live memory keeps an oversized answer's data where the row holds its
eviction placeholder (RMK-217); a memory rebuilt after a restart, when the
evicted copy is gone, cannot.

Every case runs against a provider that streams and one read through its
``generate()``.
"""

from __future__ import annotations

import asyncio
from typing import Any, get_args

from roomkit import ToolCallOutcome, ToolFailedError, ToolRefusedError
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


async def _digests(kit: RoomKit, ai: AIChannel) -> tuple[str, str]:
    """The live digest, and the one rebuilt from the stored rows."""
    reseeded = ToolUsageMemory(result_keep_chars=800, recorded=ai._in_usage_digest)
    reseeded.seed("r1", await kit._build_tool_usage_loader(ai.channel_id)("r1"))
    return ai._tool_usage.render_digest("r1") or "", reseeded.render_digest("r1") or ""


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


async def test_the_rebuilt_memory_reads_like_the_live_one(streaming: bool) -> None:
    """Refusals stay out of both, served and failed calls are in both."""
    kit, ai = await _turn(
        streaming,
        _call("c1", "ok"),
        _call("c2", "boom"),
        _call("c3", "refuse"),
        _call("c4", "ghost"),
        _call("c5", "fails"),
    )

    live, rebuilt = await _digests(kit, ai)

    assert live == rebuilt
    assert "ok(q=“c1”)" in live and "boom(q=“c2”)" in live and "fails(q=“c5”)" in live
    assert "refuse" not in live and "ghost" not in live
