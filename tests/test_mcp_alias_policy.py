"""A call under an MCP alias is judged under the tool it runs (RMK-483, RFC §21.1).

``MCPToolProvider.as_tool_handler()`` serves ``mcp__<server>__<tool>`` as
``<tool>``. The tool policy and a skill's gate judge both names, on every
door, so a tool they refuse never runs under its alias; the alias itself is
still served for a tool they admit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mcp.types as mt
import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import PolicyExternalToolHandler
from roomkit.tools.mcp import MCPToolProvider
from roomkit.tools.policy import ToolPolicy, judged_names, policy_refusal, served_tool_name
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.conftest import make_event
from tests.test_realtime_skills import _registry_with_skill
from tests.tool_loop_modes import respond

ALIAS = "mcp__crm__delete_records"


class _Server:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def send_request(self, request: Any, result_type: Any) -> Any:
        name = request.params.name
        self.ran.append(name)
        return mt.CallToolResult(
            content=[mt.TextContent(type="text", text=f"{name} done")], isError=False
        )


def _mcp_handler(server: _Server) -> Any:
    provider = MCPToolProvider("http://crm.invalid/mcp")
    provider._connected = True
    provider._session = server  # type: ignore[assignment]
    provider._tools = [
        AITool(name=n, description=n, parameters={}) for n in ("search_records", "delete_records")
    ]
    provider._tool_set = {"search_records", "delete_records"}
    return provider.as_tool_handler()


def test_an_alias_is_judged_under_both_names() -> None:
    assert served_tool_name(ALIAS) == "delete_records"
    assert served_tool_name("delete_records") == "delete_records"
    assert judged_names(ALIAS) == (ALIAS, "delete_records")
    assert judged_names("search_records") == ("search_records",)


async def _on_session(policy: ToolPolicy, server: _Server, name: str) -> str:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=_mcp_handler(server),
        tool_policy=policy,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c1", name, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()
    return provider.tool_results[0][2]


async def _on_conference(policy: ToolPolicy, server: _Server, name: str) -> str:
    provider = MockRealtimeProvider()
    handler = _mcp_handler(server)

    async def room_handler(room_id: str, tool: str, arguments: dict[str, Any]) -> Any:
        return await handler(tool, arguments)

    config = ConferenceRealtimeConfig(
        provider=provider, tool_handler=room_handler, tool_policy=policy
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    await provider.simulate_tool_call(session, "c1", name, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()
    return provider.tool_results[0][2]


DOORS = {"session": _on_session, "conference": _on_conference}


@pytest.mark.parametrize("door", list(DOORS))
@pytest.mark.parametrize(
    "policy",
    [ToolPolicy(deny=["delete_*"]), ToolPolicy(allow=["search_*"])],
    ids=["deny", "allow"],
)
async def test_a_tool_the_policy_refuses_never_runs_under_its_alias(
    door: str, policy: ToolPolicy
) -> None:
    server = _Server()

    answer = await DOORS[door](policy, server, ALIAS)

    assert "not permitted by the agent's tool policy" in answer
    assert server.ran == []


@pytest.mark.parametrize("door", list(DOORS))
async def test_an_alias_of_an_admitted_tool_is_still_served(door: str) -> None:
    server = _Server()

    answer = await DOORS[door](ToolPolicy(deny=["delete_*"]), server, "mcp__crm__search_records")

    assert "search_records done" in answer
    assert server.ran == ["search_records"]


async def test_a_tool_a_skill_gates_never_runs_under_its_alias(tmp_path: Path) -> None:
    server = _Server()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=_mcp_handler(server),
        skills=_registry_with_skill(tmp_path, allowed_tools="delete_*"),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")

    await provider.simulate_tool_call(session, "c1", ALIAS, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()

    assert "gated by a skill" in provider.tool_results[0][2]
    assert server.ran == []


def _text_channel(server: _Server, **kwargs: Any) -> tuple[AIChannel, MockAIProvider]:
    """A text channel declaring the tool under its alias, as a host that names
    MCP tools that way does; its model calls the alias once."""
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name=ALIAS, arguments={})],
            ),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name=ALIAS, description="delete", parameters={})],
        tool_handler=_mcp_handler(server),
        tool_search=False,
        **kwargs,
    )
    return channel, provider


async def _text_answer(channel: AIChannel, provider: MockAIProvider) -> str:
    binding = ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI)
    await respond(
        channel, make_event(room_id="r1", body="go"), binding, RoomContext(room=Room(id="r1"))
    )
    [answer] = [
        str(part.result)
        for message in provider.calls[1].messages
        if message.role == "tool"
        for part in message.content
    ]
    return answer


async def test_a_text_turn_judges_an_alias_under_both_names() -> None:
    server = _Server()
    channel, provider = _text_channel(server, tool_policy=ToolPolicy(deny=["delete_*"]))

    answer = await _text_answer(channel, provider)

    assert "not permitted by the agent's tool policy" in answer
    assert server.ran == []


async def test_a_text_turn_gates_an_alias_as_its_tool(tmp_path: Path) -> None:
    server = _Server()
    skills = _registry_with_skill(tmp_path, allowed_tools="delete_*")
    channel, provider = _text_channel(server, skills=skills)

    answer = await _text_answer(channel, provider)

    assert "gated by a skill" in answer
    assert server.ran == []


async def test_an_external_handler_judges_an_alias_under_both_names() -> None:
    handler = PolicyExternalToolHandler(policy=ToolPolicy(deny=["delete_*"]))

    decision = await handler.process_tool_call(ALIAS, {})

    assert decision.approved is False
    assert decision.reason == policy_refusal(ALIAS)
