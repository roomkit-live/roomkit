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

from roomkit.channels._tool_usage import ToolUsageMemory
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
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
    await asyncio.sleep(0.05)


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
async def test_a_call_id_repeated_across_turns_rebuilds_both_calls(streaming: bool) -> None:
    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            _call("c1", "lookup", {"q": "paris"}),
            AIResponse(content="Paris done."),
            _call("c1", "lookup", {"q": "rome"}),
            AIResponse(content="Rome done."),
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

    rebuilt = ToolUsageMemory(result_keep_chars=800, recorded=ai._in_usage_digest)
    rebuilt.seed("r1", await kit._build_tool_usage_loader("ai1")("r1"))

    digest = rebuilt.render_digest("r1") or ""
    assert "lookup(q=“paris”)" in digest
    assert "lookup(q=“rome”)" in digest
    assert digest == (ai._tool_usage.render_digest("r1") or "")
    await kit.close()


@pytest.mark.parametrize("streaming", [True, False])
async def test_two_turns_at_once_reusing_a_call_id_rebuild_each_call(streaming: bool) -> None:
    """The ends of two concurrent turns interleave: each pairs with its own
    turn's start, read by the response's correlation id."""

    async def slow_lookup(name: str, arguments: dict[str, Any]) -> str:
        await asyncio.sleep(0.2)
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
    await asyncio.sleep(0.3)

    calls = await kit._build_tool_usage_loader("ai1")("r1")

    assert sorted((c["arguments"].get("q"), c["result"]) for c in calls) == [
        ("paris", "result:paris"),
        ("rome", "result:rome"),
    ]
    await kit.close()
