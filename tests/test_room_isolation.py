"""A channel shared by two rooms keeps no state for either (RFC §19.7; RMK-307).

One agent object and one realtime channel serve every room they are attached
to. Whatever a room installs or a turn does is kept in the channel's registry,
set up before the rooms run, or in the room and its sessions: no attribute of a
shared channel is assigned on a room's behalf, nor any binding it shares, while
two rooms with different configurations run at once.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_framework import SimpleChannel

ROOMS = ("bank-A", "clinic-B")


class _Delegating(MockAIProvider):
    """A supervisor's model: it calls the delegation tool its turn offers, then answers."""

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        if any(message.role == "tool" for message in context.messages):
            return AIResponse(content="done")
        name = next(t.name for t in context.tools or [] if t.name.startswith("delegate_to_"))
        call = AIToolCall(id=f"call-{name}", name=name, arguments={"task": "look into it"})
        return AIResponse(content="", tool_calls=[call])


def _attributes(channel: Any) -> dict[str, int]:
    """Each attribute of *channel*, by the identity of what it holds."""
    return {name: id(value) for name, value in vars(channel).items()}


def _registered(channel: Any) -> list[tuple[str, str, int]]:
    """What *channel*'s registry serves, scope by scope."""
    registry = channel._registry
    scopes = [("", registry._channel), *registry._rooms.items()]
    return sorted(
        (room_id, name, id(held.entry))
        for room_id, scope in scopes
        for name, held in scope.items()
    )


async def _bindings(kit: RoomKit) -> dict[str, list[dict[str, Any]]]:
    return {
        room_id: [b.model_dump(mode="json") for b in await kit.store.list_bindings(room_id)]
        for room_id in ROOMS
    }


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def test_two_rooms_run_at_once_and_leave_the_shared_agent_as_it_was(
    streaming: bool,
) -> None:
    sup_model = _Delegating(streaming=streaming)
    sup = Agent("sup", provider=sup_model, tool_search=False)
    kit = RoomKit()
    workers = {}
    for room_id in ROOMS:
        workers[room_id] = Agent(
            f"worker-{room_id}", provider=MockAIProvider(responses=[f"findings of {room_id}"])
        )
        kit.register_channel(SimpleChannel(f"sms-{room_id}"))
        await kit.create_room(
            room_id=room_id,
            orchestration=Supervisor(sup, [workers[room_id]], wait_for_result=True),
        )
        await kit.attach_channel(room_id, f"sms-{room_id}")
    attributes, registered, bindings = _attributes(sup), _registered(sup), await _bindings(kit)

    await asyncio.gather(
        *(
            kit.process_inbound(
                InboundMessage(
                    channel_id=f"sms-{room_id}", sender_id="u", content=TextContent(body="go")
                )
            )
            for room_id in ROOMS
        )
    )

    # Each room delegated to its own worker, with its own install.
    results = {
        call.messages[-1].content[0].result
        for call in sup_model.calls
        if call.messages and call.messages[-1].role == "tool"
    }
    assert {json.loads(result)["worker"] for result in results} == {
        f"worker-{room_id}" for room_id in ROOMS
    }
    # And the shared agent is as the installs left it. The one binding that
    # changed is each room's SMS binding, which now names its sender: the
    # router retains the first sender it routes to a room (RFC §10.4).
    assert _attributes(sup) == attributes
    assert _registered(sup) == registered
    assert await _bindings(kit) == {
        room_id: [
            {**b, "participant_id": "u"} if b["channel_id"] == f"sms-{room_id}" else b
            for b in room_bindings
        ]
        for room_id, room_bindings in bindings.items()
    }
    await kit.close()


async def test_two_rooms_talk_at_once_and_leave_the_shared_voice_channel_as_it_was() -> None:
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt="the channel's",
    )
    lookup = AITool(name="lookup", description="Look up", parameters={"type": "object"})
    refund = AITool(name="refund", description="Refund", parameters={"type": "object"})

    async def served(name: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"served": name})

    triage = Agent("triage", system_prompt="I am TRIAGE", tools=[lookup], tool_handler=served)
    billing = Agent("billing", system_prompt="I am BILLING", tools=[refund], tool_handler=served)
    kit = RoomKit()
    for channel in (voice, triage, billing):
        kit.register_channel(channel)
    ConversationPipeline(
        stages=[
            PipelineStage(phase="triage", agent_id="triage", next="billing"),
            PipelineStage(phase="billing", agent_id="billing", next=None),
        ]
    ).install(kit, [triage, billing], voice_channel_id="voice")
    sessions = {}
    for room_id in ROOMS:
        room = await kit.create_room(room_id=room_id)
        triage_state = ConversationState(phase="triage", active_agent_id="triage")
        await kit.store.update_room(set_conversation_state(room, triage_state))
        await kit.attach_channel(room_id, "voice")
        sessions[room_id] = await voice.start_session(room_id, f"caller-{room_id}", "ws")
    attributes, registered, bindings = _attributes(voice), _registered(voice), await _bindings(kit)

    handoff = {"target": "billing", "reason": "refund", "summary": "wants a refund"}
    await asyncio.gather(
        provider.simulate_tool_call(sessions["bank-A"], "h1", "handoff_conversation", handoff),
        provider.simulate_tool_call(sessions["clinic-B"], "l1", "lookup", {}),
    )
    await _until(lambda: len(provider.tool_results) == 2)
    late = await voice.start_session("clinic-B", "late-caller", "ws")

    results = {call_id: json.loads(result) for _s, call_id, result in provider.tool_results}
    assert results["h1"]["accepted"] is True
    assert results["l1"] == {"served": "lookup"}
    # A caller joining the room that did not hand off meets its own agent.
    connected = [c.args for c in provider.calls if c.method == "connect"][-1]
    assert connected["session_id"] == late.id
    assert connected["system_prompt"].startswith("I am TRIAGE")
    # And the shared channel is as the installs left it.
    assert _attributes(voice) == attributes
    assert _registered(voice) == registered
    assert await _bindings(kit) == bindings
    await kit.close()


class _Submitting(MockAIProvider):
    """A delegated agent's model: it submits the task it was given as its result,
    and records whether its channel was as the rooms found it while it ran."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.channel: Any = None
        self.before: dict[str, int] = {}
        self.untouched: list[bool] = []

    async def generate(self, context: AIContext) -> AIResponse:
        self.calls.append(context)
        self.untouched.append(_attributes(self.channel) == self.before)
        if any(message.role == "tool" for message in context.messages):
            return AIResponse(content="submitted")
        task = next(m.content for m in reversed(context.messages) if m.role == "user")
        text = task if isinstance(task, str) else str(task)
        summary = next(room_id for room_id in ROOMS if room_id in text)
        call = AIToolCall(
            id="submit",
            name="submit_result",
            arguments={"status": "completed", "summary": summary},
        )
        return AIResponse(content="", tool_calls=[call])


async def test_two_rooms_capture_one_shared_agent_s_results_at_once(streaming: bool) -> None:
    """Each delegation's result tool is set up for its child room: the shared
    agent is not swapped a handler while the other room's delegation runs."""
    model = _Submitting(streaming=streaming)
    worker = Agent("worker", provider=model, tool_search=False)
    model.channel = worker
    kit = RoomKit()
    kit.register_channel(worker)
    for room_id in ROOMS:
        await kit.create_room(room_id=room_id)
    model.before = _attributes(worker)
    registered = _registered(worker)

    delegated = await asyncio.gather(
        *(
            kit.delegate(
                room_id, "worker", f"task of {room_id}", wait=True, require_structured_result=True
            )
            for room_id in ROOMS
        )
    )

    summaries = [json.loads(task.result.output)["summary"] for task in delegated]
    assert summaries == list(ROOMS)
    assert model.untouched and all(model.untouched)
    assert _registered(worker) == registered
    await kit.close()
