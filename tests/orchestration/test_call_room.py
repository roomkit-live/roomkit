"""A handoff or a delegation acts on the room of its call (RFC §19.6, §23.4; RMK-275).

The orchestration tools read the room from the tool call context, on both
tool loops: not from a value a routing hook left behind, and not from the
room that happened to install them. A child room's parent is the room whose
tool call asked for it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.orchestration import ConversationRouter, HandoffHandler, setup_handoff
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.orchestration.strategies.swarm import Swarm
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import DelegateHandler, setup_delegation
from tests.test_framework import SimpleChannel
from tests.tool_room import owner_tools_in


async def _until(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not await predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def _calling(name: str, arguments: dict[str, Any], *, streaming: bool) -> MockAIProvider:
    """A model that calls one tool, then answers.

    The mock replays its script in a loop, and a delegated result wakes the
    delegating agent again: enough answers follow the call that it is never
    replayed within a test.
    """
    call = AIResponse(content="", tool_calls=[AIToolCall(id="c1", name=name, arguments=arguments)])
    return MockAIProvider(
        ai_responses=[call, *(AIResponse(content="done") for _ in range(20))],
        streaming=streaming,
    )


def _answering(text: str, *, streaming: bool) -> MockAIProvider:
    return MockAIProvider(responses=[text], streaming=streaming)


async def _say(kit: RoomKit, channel_id: str, body: str) -> None:
    await kit.process_inbound(
        InboundMessage(channel_id=channel_id, sender_id="user", content=TextContent(body=body))
    )


async def _children(kit: RoomKit) -> dict[str, str | None]:
    """Every delegated room, with the parent it records."""
    rooms = await kit.store.list_rooms()
    return {r.id: r.metadata.get("parent_room_id") for r in rooms if "::task-" in r.id}


def _tool_results(provider: MockAIProvider) -> list[str]:
    return [
        part.result
        for call in provider.calls
        for message in call.messages
        if message.role == "tool"
        for part in message.content
    ]


async def test_a_swarm_handoff_acts_on_the_room_of_its_call(streaming: bool) -> None:
    arguments = {"target": "billing", "reason": "billing question", "summary": "invoice"}
    triage_model = _calling("handoff_conversation", arguments, streaming=streaming)
    triage = Agent("triage", provider=triage_model, description="Front desk", tool_search=False)
    billing = Agent(
        "billing",
        provider=_answering("billing here", streaming=streaming),
        description="Billing",
        tool_search=False,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(triage)
    kit.register_channel(billing)
    await kit.create_room(room_id="r1", orchestration=Swarm(agents=[triage, billing]))
    await kit.attach_channel("r1", "sms1")

    await _say(kit, "sms1", "my invoice?")

    async def handed_off() -> bool:
        return get_conversation_state(await kit.get_room("r1")).active_agent_id == "billing"

    await _until(handed_off)
    assert all("error" not in json.loads(result) for result in _tool_results(triage_model))
    await kit.close()


async def test_setup_delegation_without_a_router_delegates_from_the_call_room(
    streaming: bool,
) -> None:
    front_model = _calling(
        "delegate_task", {"agent": "worker", "task": "research X"}, streaming=streaming
    )
    front = AIChannel("front", provider=front_model, tool_search=False)
    worker = AIChannel("worker", provider=_answering("worker result", streaming=streaming))
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(front)
    kit.register_channel(worker)
    setup_delegation(front, DelegateHandler(kit))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "front", category=ChannelCategory.INTELLIGENCE)

    await _say(kit, "sms1", "please research X")

    async def delegated() -> bool:
        return bool(await _children(kit))

    await _until(delegated)
    assert set((await _children(kit)).values()) == {"r1"}
    await kit.close()


@pytest.mark.parametrize(
    ("tool", "strategy"),
    [("delegate_to_researcher", None), ("delegate_workers", "sequential")],
    ids=["per-worker", "strategy"],
)
async def test_a_supervisor_shared_by_two_rooms_delegates_from_the_room_that_asked(
    tool: str, strategy: str | None, streaming: bool
) -> None:
    supervisor_model = _calling(tool, {"task": "look into B's account"}, streaming=streaming)
    supervisor = Agent("sup", provider=supervisor_model, tool_search=False)
    researcher = Agent(
        "researcher", provider=_answering("findings", streaming=streaming), tool_search=False
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms-a"))
    kit.register_channel(SimpleChannel("sms-b"))
    kit.register_channel(supervisor)
    kit.register_channel(researcher)
    for tenant, channel_id in (("tenant-A", "sms-a"), ("tenant-B", "sms-b")):
        orchestration = Supervisor(supervisor, [researcher], strategy=strategy)
        await kit.create_room(room_id=tenant, orchestration=orchestration)
        await kit.attach_channel(tenant, channel_id)

    # Only tenant B's user speaks; tenant A installed the supervisor first.
    await _say(kit, "sms-b", "check my account")

    async def delegated() -> bool:
        return bool(await _children(kit))

    await _until(delegated)
    children = await _children(kit)
    assert set(children.values()) == {"tenant-B"}
    assert all(room_id.startswith("tenant-B::task-") for room_id in children)
    await kit.close()


async def test_a_worker_delegating_in_turn_hangs_its_task_off_its_own_room(
    streaming: bool,
) -> None:
    front = AIChannel(
        "front",
        provider=_calling("delegate_task", {"agent": "worker", "task": "t0"}, streaming=streaming),
        tool_search=False,
    )
    worker = AIChannel(
        "worker",
        provider=_calling("delegate_task", {"agent": "leaf", "task": "t1"}, streaming=streaming),
        tool_search=False,
    )
    leaf = AIChannel("leaf", provider=_answering("leaf", streaming=streaming))
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    for channel in (front, worker, leaf):
        kit.register_channel(channel)
    setup_delegation(front, DelegateHandler(kit))
    setup_delegation(worker, DelegateHandler(kit))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "front", category=ChannelCategory.INTELLIGENCE)

    await _say(kit, "sms1", "go")

    async def both_delegated() -> bool:
        return len(await _children(kit)) == 2

    await _until(both_delegated)
    children = await _children(kit)
    worker_room = next(room_id for room_id, parent in children.items() if parent == "r1")
    leaf_room = next(room_id for room_id in children if room_id != worker_room)
    assert children[leaf_room] == worker_room
    await kit.close()


@pytest.mark.parametrize(
    ("tool", "strategy"),
    [("delegate_to_researcher", None), ("delegate_workers", "sequential")],
    ids=["per-worker", "strategy"],
)
async def test_a_supervisor_tool_called_outside_a_tool_call_is_not_served(
    tool: str, strategy: str | None
) -> None:
    """The tool is the installed room's (RFC §19.7): a call that names no room
    reaches nothing, and delegates nothing."""
    supervisor = Agent("sup", provider=_answering("hi", streaming=False), tool_search=False)
    researcher = Agent("researcher", provider=_answering("findings", streaming=False))
    kit = RoomKit()
    kit.register_channel(supervisor)
    kit.register_channel(researcher)
    orchestration = Supervisor(supervisor, [researcher], strategy=strategy)
    await kit.create_room(room_id="r1", orchestration=orchestration)

    # Given r1's tools, as a turn of r1 is: a call naming no room is not r1's.
    with owner_tools_in(kit, "sup", "r1"), pytest.raises(UnservedToolCallError):
        await supervisor._channel_tool_handler(tool, {"task": "x"})

    assert await _children(kit) == {}
    await kit.close()


async def test_a_handoff_tool_called_outside_a_tool_call_refuses_without_a_room() -> None:
    """A call made outside a tool call has no room to act on, and the handoff
    tool, set up for every room the agent serves, refuses it rather than guess
    (RFC §19.6): no room hands anything off."""
    triage = Agent("triage", provider=_answering("hi", streaming=False), tool_search=False)
    billing = Agent("billing", provider=_answering("billing here", streaming=False))
    kit = RoomKit()
    kit.register_channel(triage)
    kit.register_channel(billing)
    setup_handoff(triage, HandoffHandler(kit, ConversationRouter(default_agent_id="triage")))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "triage", category=ChannelCategory.INTELLIGENCE)

    # Given r1's tools, as a turn of r1 is, outside any of its tool calls.
    arguments = {"target": "billing", "reason": "billing question", "summary": "invoice"}
    with owner_tools_in(kit, "triage", "r1"), pytest.raises(UnservedToolCallError):
        await triage._channel_tool_handler("handoff_conversation", arguments)

    assert get_conversation_state(await kit.get_room("r1")).active_agent_id is None
    await kit.close()
