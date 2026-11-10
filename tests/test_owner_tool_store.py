"""The tools orchestration sets up are kept by the room's owner, per channel and
room, beside the channel's declaration, never on a live agent (RFC §10.6.2,
§10.6.5)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from roomkit import (
    Agent,
    ChannelDeclaration,
    RoomKit,
    Swarm,
    ToolNameCollisionError,
)
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore
from roomkit.tasks import DelegateHandler, setup_delegation
from roomkit.turns import InMemoryTurnBackend, TurnExecutor
from tests.test_framework import SimpleChannel
from tests.tool_room import room_tool_names

SUBMITTED = {"status": "completed", "summary": "The weather: sunny.", "data": {"c": 21}}


def _agent(channel_id: str, *responses: AIResponse, **kwargs: Any) -> Agent:
    provider = MockAIProvider(ai_responses=[*responses, *(AIResponse(content="ok"),) * 4])
    return Agent(channel_id, provider=provider, **kwargs)


def _calls(name: str, **arguments: Any) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"call-{name}", name=name, arguments=arguments)],
    )


class _Host:
    """A host whose executor builds the agents it hosts, served on the kit's
    backend, the store shared."""

    def __init__(self, **agents: Agent) -> None:
        self.store = InMemoryStore()
        self.backend = InMemoryTurnBackend()
        self.executor = TurnExecutor(
            agents.get,
            hosts=lambda channel_id: channel_id in agents,
        )

    async def kit(self) -> RoomKit:
        await self.backend.serve(self.executor)
        kit = RoomKit(store=self.store, turn_backend=self.backend)
        kit.register_channel(SimpleChannel("sms"))
        return kit

    async def close(self, kit: RoomKit) -> None:
        await kit.close()
        await self.executor.close()


async def test_a_strategy_s_tools_are_the_owner_s_never_the_agent_s() -> None:
    kit = RoomKit()
    ana, bob = _agent("ana"), _agent("bob")
    await kit.create_room(room_id="r", orchestration=Swarm(agents=[ana, bob]))

    on_agent = ana._registry.owner_entries("r")
    kept = room_tool_names(kit, "ana", "r")
    await kit.uninstall_strategy("r")
    after = room_tool_names(kit, "ana", "r")
    await kit.close()

    assert on_agent == []
    assert kept == ["handoff_conversation"]
    assert after == []


async def test_a_declared_worker_hands_its_structured_result_back() -> None:
    """The result tool is set up for the worker the kit holds by declaration:
    its turn on the host's executor is given it, and its call is the result,
    where it used to fall back to the worker's text."""
    worker = _agent("ana", _calls("submit_result", **SUBMITTED))
    host = _Host(ana=worker)
    kit = await host.kit()
    kit.register_channel(ChannelDeclaration(channel_id="ana"))
    await kit.create_room(room_id="parent")

    task = await kit.delegate(
        "parent", "ana", "The weather.", wait=True, require_structured_result=True
    )
    await host.close(kit)

    assert task.result is not None
    assert json.loads(task.result.output)["data"] == {"c": 21}
    declared = {tool.name for tool in worker.provider.calls[0].tools or []}
    assert "submit_result" in declared


async def test_an_executor_refuses_a_turn_whose_owner_tool_its_channel_declares() -> None:
    """The kit holds the agent by its declaration, which names no tool as the
    owner's, so the setup passes; the host's agent does: its turn is refused,
    rather than a call reaching one handler or the other by executor."""
    own = AITool(name="delegate_task", description="The host's own.", parameters={})
    hosted = _agent("ana", _calls("delegate_task", agent="x", task="y"), tools=[own])
    host = _Host(ana=hosted)
    kit = await host.kit()
    kit.register_channel(ChannelDeclaration(channel_id="ana"))
    setup_delegation("ana", DelegateHandler(kit))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    await kit.attach_channel("r", "ana", category=ChannelCategory.INTELLIGENCE)
    await kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="marie", content=TextContent(body="Go.")),
        room_id="r",
    )
    said = [e for e in await kit.store.list_events("r") if e.source.channel_id == "ana"]
    await host.close(kit)

    assert said == []
    assert hosted.provider.calls == []


def test_the_owner_refuses_a_tool_a_live_agent_already_has() -> None:
    kit = RoomKit()
    own = AITool(name="delegate_task", description="The host's own.", parameters={})
    agent = _agent("ana", tools=[own])
    kit.register_channel(agent)

    with pytest.raises(ToolNameCollisionError):
        setup_delegation(agent, DelegateHandler(kit))

    assert room_tool_names(kit, "ana", "r") == []
