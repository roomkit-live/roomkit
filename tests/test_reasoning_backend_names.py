"""A realtime voice channel refuses a session tool its reasoning backend's
agent serves itself, on every door it is given through, and does not declare
one that arrives later; the agent's own tool policy composes with the
channel's, resolved for the session participant's role (RMK-527, RFC
§12.4.1)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.participant import Participant
from roomkit.models.tool_call import ToolCallEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.tools.policy import RoleOverride, ToolPolicy
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
        ("find_tools", {}),
    ],
    ids=["re-read", "list-tools", "find-tools", "find-tools-default-agent"],
)
def test_a_session_tool_under_a_name_the_agent_serves_is_refused(
    name: str, agent: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="reasoning backend"):
        _channel([name, "weather"], _backend(**agent), tool_search=False)


def test_configure_refuses_a_name_the_agent_serves() -> None:
    channel = _channel(["weather"], _backend(), tool_search=False)

    with pytest.raises(ValueError, match="reasoning backend"):
        channel.configure(tools=[_tool("read_stored_result")])


def test_a_human_input_tool_under_a_name_the_agent_serves_is_refused() -> None:
    human = HumanInputToolHandler(
        tool_names={"read_stored_result"},
        tool_definitions=[AITool(name="read_stored_result", description="Ask the person.")],
    )

    with pytest.raises(ValueError, match="reasoning backend"):
        _channel(["weather"], _backend(), tool_search=False, human_input_handler=human)


@pytest.mark.parametrize("door", ["session-metadata", "reconfigure"])
async def test_a_tool_arriving_later_under_a_name_the_agent_serves_is_not_declared(
    door: str, caplog: pytest.LogCaptureFixture
) -> None:
    channel = _channel(["weather"], _backend(), tool_search=False)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    late = [_tool("read_stored_result"), _tool("forecast")]

    with caplog.at_level(logging.WARNING, logger="roomkit"):
        if door == "session-metadata":
            session = await channel.start_session("r1", "u1", "ws", metadata={"tools": late})
        else:
            session = await channel.start_session("r1", "u1", "ws")
            await channel.reconfigure_session(session, tools=late)
    declared = [tool["name"] for tool in channel._session_base_tools(session.id)]
    await kit.close()

    assert declared == ["forecast"]
    assert any("read_stored_result" in record.getMessage() for record in caplog.records)


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


_OBSERVER_DENIED = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["lookup"])})


@pytest.mark.parametrize(
    ("policy", "role", "refused"),
    [
        (ToolPolicy(deny=["lookup"]), None, True),
        (_OBSERVER_DENIED, "observer", True),
        (_OBSERVER_DENIED, "member", False),
    ],
    ids=["base-deny", "role-override", "other-role"],
)
async def test_the_agents_own_policy_composes_with_the_channels(
    policy: ToolPolicy, role: str | None, refused: bool
) -> None:
    """A call the agent's policy denies, for the session participant's role,
    is refused before the channel's gate and seen by the channel's
    ON_TOOL_CALL as any refusal."""
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
    backend = _backend(provider, tool_policy=policy)
    channel = _channel(["lookup", "weather"], backend, tool_search=False)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    if role is not None:
        await kit.store.add_participant(
            Participant(id="u1", room_id="r1", channel_id="rt", role=role)
        )
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
        ("lookup", refused, refused)
    ]
