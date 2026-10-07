"""A realtime voice channel refuses a session tool its reasoning backend's
agent serves itself, and the agent's own tool policy composes with the
channel's (RMK-527, RFC §12.4.1)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.policy import ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend


def _tool(name: str) -> dict[str, Any]:
    return {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}}


async def _host(name: str, arguments: dict[str, Any]) -> str:
    return f"host served {name}"


def _backend(provider: MockAIProvider | None = None, **agent: Any) -> AgentReasoningBackend:
    return AgentReasoningBackend(Agent("reasoner", provider=provider or MockAIProvider(), **agent))


def _channel(
    names: list[str], backend: AgentReasoningBackend | None, **options: Any
) -> RealtimeVoiceChannel:
    return RealtimeVoiceChannel(
        "rt",
        provider=MockRealtimeProvider(full_duplex=True),
        transport=MockRealtimeTransport(),
        tools=[_tool(name) for name in names],
        tool_handler=_host,
        reasoning_backend=backend,
        **options,
    )


@pytest.mark.parametrize(
    ("name", "agent"),
    [
        ("read_stored_result", {}),
        ("list_tools", {"tool_search": True}),
        ("find_tools", {"tool_search": True}),
    ],
)
def test_a_session_tool_under_a_name_the_agent_serves_is_refused(
    name: str, agent: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="reasoning backend's agent"):
        _channel([name, "weather"], _backend(**agent), tool_search=False)


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        ("list_tools", lambda: _backend(tool_search=False)),
        ("read_stored_result", lambda: None),
    ],
    ids=["agent-without-tool-search", "no-backend"],
)
def test_a_name_no_backend_serves_is_accepted(name: str, backend: Any) -> None:
    channel = _channel([name], backend(), tool_search=False)

    assert [tool["name"] for tool in channel._tools or []] == [name]


async def test_the_agents_own_policy_composes_with_the_channels() -> None:
    """A call the agent's policy denies is refused before the channel's gate
    and seen by the channel's ON_TOOL_CALL as any refusal."""
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="b1", name="lookup", arguments={})],
            ),
            AIResponse(content="done"),
        ]
    )
    backend = _backend(provider, tool_policy=ToolPolicy(deny=["lookup"]))
    channel = _channel(["lookup", "weather"], backend, tool_search=False)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def observe(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    session = await channel.start_session("r1", "u1", "ws")
    await channel._provider.simulate_delegation(session, "d1", "integrator")
    for _ in range(300):
        if seen:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert [(event.name, event.is_error, event.refused) for event in seen] == [
        ("lookup", True, True)
    ]
