"""A channel's tool memory and skill activations, rebuilt from the room's
stored tool rows, are the channel's own (RFC §7.5 rule 8, RMK-393).

The rebuild runs at a channel's first turn in a room. It reads the channel's
rows only, whatever another agent of the room did or let it see, and pairs
each call's end with its own start, a call id repeated across turns included.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventStatus, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.providers.ai.base import (
    AIResponse,
    AITool,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.turns._wiring import remembered_calls
from tests.test_framework import SimpleChannel

LOOKUP = AITool(
    name="lookup",
    description="lookup",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


def _registry(tmp: Path) -> SkillRegistry:
    skill_dir = tmp / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: A test skill\n---\nRULES-FOR-ALPHA.", encoding="utf-8"
    )
    registry = SkillRegistry()
    registry.discover(tmp)
    return registry


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments)],
    )


async def _lookup(name: str, arguments: dict[str, Any]) -> str:
    return f"secret-of-ai1:{arguments.get('q')}"


async def _say(kit: RoomKit, body: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body=body))
    )


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("visibility", ["sms1", "all"])
async def test_a_joining_agent_does_not_rebuild_another_agents_calls(
    tmp_path: Path, streaming: bool, visibility: str
) -> None:
    registry = _registry(tmp_path)
    p1 = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            _call("s1", "activate_skill", {"name": "alpha"}),
            _call("l1", "lookup", {"q": "balance"}),
            AIResponse(content="ai1 done."),
            *[AIResponse(content="ai1 again.")] * 5,
        ],
    )
    p2 = MockAIProvider(streaming=streaming, ai_responses=[AIResponse(content="ai2 here.")] * 5)
    common: dict[str, Any] = {
        "tools": [LOOKUP],
        "tool_handler": _lookup,
        "tool_search": False,
        "skills": registry,
    }
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(AIChannel("ai1", provider=p1, **common))
    ai2 = AIChannel("ai2", provider=p2, **common)
    kit.register_channel(ai2)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel(
        "r1", "ai1", category=ChannelCategory.INTELLIGENCE, visibility=visibility
    )
    await _say(kit, "one")
    await kit.attach_channel("r1", "ai2", category=ChannelCategory.INTELLIGENCE)
    await _say(kit, "two")

    first = p2.calls[0]
    context = " ".join(str(m.content) for m in first.messages)
    assert "secret-of-ai1" not in context
    assert "lookup(q=“balance”)" not in context
    assert "RULES-FOR-ALPHA" not in (first.system_prompt or "")
    assert ai2._tool_usage.tool_names("r1") == set()
    assert ai2._skill_activation.active_names("r1") == set()
    await kit.close()


@pytest.mark.parametrize("streaming", [True, False])
async def test_a_call_id_repeated_across_turns_replays_both_calls(streaming: bool) -> None:
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            _call("c1", "lookup", {"q": "paris"}),
            AIResponse(content="Paris done."),
            _call("c1", "lookup", {"q": "rome"}),
            AIResponse(content="Rome done."),
            AIResponse(content="Both done."),
        ],
    )
    ai = AIChannel(
        "ai1", provider=provider, tools=[LOOKUP], tool_handler=_lookup, tool_search=False
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await _say(kit, "paris")
    await _say(kit, "rome")
    await _say(kit, "both?")

    replayed = [
        part
        for message in provider.calls[-1].messages
        if message.role in ("assistant", "tool") and isinstance(message.content, list)
        for part in message.content
    ]
    calls = [(p.id, p.arguments) for p in replayed if isinstance(p, AIToolCallPart)]
    results = [(p.tool_call_id, p.result) for p in replayed if isinstance(p, AIToolResultPart)]
    # Each under the id the model gave it, as its turn sent it (RMK-1370).
    assert calls == [("c1", {"q": "paris"}), ("c1", {"q": "rome"})]
    assert results == [("c1", "secret-of-ai1:paris"), ("c1", "secret-of-ai1:rome")]
    await kit.close()


@pytest.mark.parametrize("streaming", [True, False])
async def test_two_turns_at_once_reusing_a_call_id_rebuild_each_call(streaming: bool) -> None:
    """Two turns asked at once reusing a call id: each end pairs with its own
    turn's start, read by the response's correlation id. The room runs the
    two turns one after the other, its rows start, end, start, end."""

    async def slow_lookup(name: str, arguments: dict[str, Any]) -> str:
        return f"result:{arguments.get('q')}"

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            _call("c1", "lookup", {"q": "paris"}),
            _call("c1", "lookup", {"q": "rome"}),
            AIResponse(content="done A."),
            AIResponse(content="done B."),
        ],
    )
    ai = AIChannel(
        "ai1", provider=provider, tools=[LOOKUP], tool_handler=slow_lookup, tool_search=False
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    await asyncio.gather(
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u", content=TextContent(body="paris"))
        ),
        kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u2", content=TextContent(body="rome"))
        ),
    )

    calls = await kit._build_tool_usage_loader("ai1")("r1")

    assert sorted((c["arguments"].get("q"), c["result"]) for c in calls) == [
        ("paris", "result:paris"),
        ("rome", "result:rome"),
    ]
    await kit.close()


@pytest.mark.parametrize(("deleted", "held"), [("start", "secret-q"), ("end", "secret")])
def test_a_deleted_row_rebuilds_nothing_of_what_it_held(deleted: str, held: str) -> None:
    """A tool row a delete removed is withheld from the channel's rebuilt
    memory as from its history (RFC §7.5 rule 8, §10.3)."""
    source = EventSource(channel_id="ai1", channel_type=ChannelType.AI)
    call = ToolCallContent(tool_name="lookup", tool_id="l1", arguments={"q": "secret-q"})
    rows = [
        RoomEvent(
            room_id="r1",
            type=kind,
            source=source,
            content=content,
            status=EventStatus.DELIVERED,
            correlation_id="turn-1",
            metadata={"deleted": True} if deleted == name else {},
        )
        for name, kind, content in [
            ("start", EventType.TOOL_CALL_START, call),
            (
                "end",
                EventType.TOOL_CALL_END,
                call.model_copy(update={"result": "secret-result", "outcome": "served"}),
            ),
        ]
    ]

    remembered = remembered_calls(rows)

    assert held not in str(remembered)


def _row(kind: EventType, tool_id: str, name: str, **fields: Any) -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        type=kind,
        source=EventSource(channel_id="ai1", channel_type=ChannelType.AI),
        content=ToolCallContent(tool_name=name, tool_id=tool_id, **fields),
        status=EventStatus.DELIVERED,
        correlation_id="turn-1",
    )


def test_two_calls_under_one_id_rebuild_each_with_its_own_arguments() -> None:
    rows = [
        _row(EventType.TOOL_CALL_START, "c1", "lookup", arguments={"q": "first"}),
        _row(EventType.TOOL_CALL_START, "c1", "other", arguments={"q": "second"}),
        _row(EventType.TOOL_CALL_END, "c1", "lookup", result="fine", outcome="served"),
        _row(EventType.TOOL_CALL_END, "c1", "other", result="refused", outcome="refused"),
    ]

    assert [(c["name"], c["arguments"]) for c in remembered_calls(rows)] == [
        ("lookup", {"q": "first"})
    ]


def test_a_withheld_end_leaves_its_call_out_of_a_later_call_s_rebuild() -> None:
    left = _row(EventType.TOOL_CALL_START, "c1", "lookup", arguments={"q": "left"})
    ended = _row(EventType.TOOL_CALL_END, "c1", "lookup", result="gone", outcome="served")
    rows = [
        left,
        ended.model_copy(update={"metadata": {"deleted": True}}),
        _row(EventType.TOOL_CALL_START, "c1", "lookup", arguments={"q": "new"}),
        _row(EventType.TOOL_CALL_END, "c1", "lookup", result="its own", outcome="served"),
    ]

    assert [(c["arguments"], c["result"]) for c in remembered_calls(rows)] == [
        ({"q": "new"}, "its own")
    ]


def test_a_later_round_s_call_under_an_id_takes_its_own_end_in_the_rebuild() -> None:
    rows = [
        _row(EventType.TOOL_CALL_START, "c1", "lookup", arguments={"q": "left"}),
        _row(EventType.TOOL_CALL_START, "c2", "lookup", arguments={"q": "other"}),
        _row(EventType.TOOL_CALL_END, "c2", "lookup", result="its", outcome="served"),
        _row(EventType.TOOL_CALL_START, "c1", "lookup", arguments={"q": "new"}),
        _row(EventType.TOOL_CALL_END, "c1", "lookup", result="its own", outcome="served"),
    ]

    assert [(c["arguments"], c["result"]) for c in remembered_calls(rows)] == [
        ({"q": "other"}, "its"),
        ({"q": "new"}, "its own"),
    ]


def test_a_rebuilt_activation_keeps_what_the_model_wrote_and_what_ran() -> None:
    name = "activate_skill"
    rows = [
        _row(EventType.TOOL_CALL_START, "c1", name, arguments={"name": "alpha"}),
        _row(
            EventType.TOOL_CALL_END,
            "c1",
            name,
            arguments={"name": "beta"},
            result="{}",
            outcome="served",
        ),
    ]

    [call] = remembered_calls(rows)

    assert (call["arguments"], call["ran_with"]) == ({"name": "alpha"}, {"name": "beta"})


def test_a_deleted_start_leaves_its_activation_what_ran() -> None:
    name = "activate_skill"
    start = _row(EventType.TOOL_CALL_START, "c1", name, arguments={"name": "alpha"})
    rows = [
        start.model_copy(update={"metadata": {"deleted": True}}),
        _row(
            EventType.TOOL_CALL_END,
            "c1",
            name,
            arguments={"name": "alpha"},
            result="{}",
            outcome="served",
        ),
    ]

    [call] = remembered_calls(rows)

    assert (call["arguments"], call["ran_with"]) == ({}, {"name": "alpha"})
