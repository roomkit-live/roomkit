"""A strategy installed in several rooms keeps what it adds per room (RFC §19.7; RMK-276).

Agents and channels are shared by every room they serve. Installing a strategy
in a second room declares nothing twice and wraps nothing twice; a tool it adds
for one room is declared in that room only; and each call runs for the room it
came from.
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
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.mixins._result_capture import capture_result
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.orchestration.result import SUBMIT_RESULT
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.orchestration.strategies import loop as loop_module
from roomkit.orchestration.strategies.loop import Loop
from roomkit.orchestration.strategies.supervisor import Supervisor, _install_auto
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import current_tool_chain_depth
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.test_framework import SimpleChannel
from tests.tool_loop_modes import respond
from tests.tool_room import owner_tools_in, room_tool_names

TENANTS = ("tenant-A", "tenant-B")


def _agent(channel_id: str, *tools: AITool, **kwargs: Any) -> Agent:
    return Agent(
        channel_id,
        provider=MockAIProvider(responses=["ok"]),
        tools=list(tools),
        tool_search=False,
        **kwargs,
    )


def _tool(name: str) -> AITool:
    return AITool(name=name, description=name, parameters={"type": "object", "properties": {}})


async def _voice_rooms(
    orchestration: Callable[[], Any],
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, dict[str, Any]]:
    """A voice channel attached to two rooms, one session in each."""
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel("voice", provider=provider, transport=MockRealtimeTransport())
    kit = RoomKit()
    kit.register_channel(voice)
    sessions = {}
    for tenant in TENANTS:
        await kit.create_room(room_id=tenant, orchestration=orchestration())
        await kit.attach_channel(tenant, "voice")
        sessions[tenant] = await voice.start_session(tenant, f"user-{tenant}", "ws")
    return kit, voice, provider, sessions


def _declared(provider: MockRealtimeProvider, session: Any) -> list[str]:
    """The tools *session* was declared when it connected."""
    connects = [
        c.args
        for c in provider.calls
        if c.method == "connect" and c.args["session_id"] == session.id
    ]
    return [tool["name"] for tool in connects[-1]["tools"] or []]


async def test_a_voice_supervisor_in_two_rooms_is_declared_in_each_and_runs_for_the_calling_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran_for: list[str] = []

    async def run_and_deliver(**kwargs: Any) -> None:
        ran_for.append(kwargs["room_id"])
        kwargs["on_done"]()

    monkeypatch.setattr(_install_auto, "_async_run_and_deliver", run_and_deliver)
    supervisor, worker = _agent("sup"), _agent("worker")
    kit, voice, provider, sessions = await _voice_rooms(
        lambda: Supervisor(
            supervisor, [worker], strategy="sequential", auto_delegate=True, async_delivery=True
        )
    )

    await provider.simulate_tool_call(
        sessions["tenant-A"], "c1", "delegate_workers", {"task": "A"}
    )
    await _until(lambda: len(provider.tool_results) == 1)

    assert all(_declared(provider, sessions[t]) == ["delegate_workers"] for t in TENANTS)
    assert ran_for == ["tenant-A"]
    assert json.loads(provider.tool_results[0][2])["status"] == "dispatched"
    await kit.close()


async def test_a_voice_loop_in_two_rooms_is_declared_in_each_and_runs_for_the_calling_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran_for: list[str] = []

    async def loop_and_deliver(**kwargs: Any) -> None:
        ran_for.append(kwargs["room_id"])
        kwargs["on_done"]()

    monkeypatch.setattr(loop_module, "_async_loop_and_deliver", loop_and_deliver)
    producer, reviewer = _agent("writer"), _agent("editor")
    kit, voice, provider, sessions = await _voice_rooms(
        lambda: Loop(agent=producer, reviewer=reviewer, async_delivery=True)
    )

    await provider.simulate_tool_call(sessions["tenant-A"], "c1", "delegate_loop", {"task": "A"})
    await _until(lambda: len(provider.tool_results) == 1)

    assert all(_declared(provider, sessions[t]) == ["delegate_loop"] for t in TENANTS)
    assert ran_for == ["tenant-A"]
    await kit.close()


async def test_voice_results_continue_the_chain_of_the_models_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RFC §12.4, §23.3: a realtime tool call belongs to the model's answer, one
    deeper than what the model heard, and the results it dispatches continue it."""
    dispatched: list[tuple[str, int]] = []

    async def run_and_deliver(**kwargs: Any) -> None:
        dispatched.append(("supervisor", current_tool_chain_depth()))
        kwargs["on_done"]()

    async def loop_and_deliver(**kwargs: Any) -> None:
        dispatched.append(("loop", current_tool_chain_depth()))
        kwargs["on_done"]()

    monkeypatch.setattr(_install_auto, "_async_run_and_deliver", run_and_deliver)
    monkeypatch.setattr(loop_module, "_async_loop_and_deliver", loop_and_deliver)
    for orchestration, tool in (
        (
            lambda: Supervisor(
                _agent("sup"),
                [_agent("worker")],
                strategy="sequential",
                auto_delegate=True,
                async_delivery=True,
            ),
            "delegate_workers",
        ),
        (
            lambda: Loop(agent=_agent("writer"), reviewer=_agent("editor"), async_delivery=True),
            "delegate_loop",
        ),
    ):
        kit, voice, provider, sessions = await _voice_rooms(orchestration)
        session = sessions["tenant-A"]
        await voice.inject_text(session, "an agent's answer", chain_depth=3)
        await provider.simulate_tool_call(session, "c1", tool, {"task": "A"})
        await _until(lambda p=provider: len(p.tool_results) == 1)
        await kit.close()

    assert dispatched == [("supervisor", 4), ("loop", 4)]


async def test_a_voice_supervisor_is_not_served_in_a_room_it_was_not_installed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran_for: list[str] = []

    async def run_and_deliver(**kwargs: Any) -> None:
        ran_for.append(kwargs["room_id"])
        kwargs["on_done"]()

    monkeypatch.setattr(_install_auto, "_async_run_and_deliver", run_and_deliver)
    supervisor, worker = _agent("sup"), _agent("worker")
    kit, voice, provider, _ = await _voice_rooms(
        lambda: Supervisor(
            supervisor, [worker], strategy="sequential", auto_delegate=True, async_delivery=True
        )
    )
    await kit.create_room(room_id="plain")  # no orchestration
    await kit.attach_channel("plain", "voice")
    session = await voice.start_session("plain", "user-plain", "ws")

    await provider.simulate_tool_call(session, "c1", "delegate_workers", {"task": "C"})
    await _until(lambda: len(provider.tool_results) == 1)

    # Not declared in that room's session, and nothing serves it there.
    assert _declared(provider, session) == []
    assert "error" in json.loads(provider.tool_results[0][2])
    assert ran_for == []
    await kit.close()


async def test_a_sync_loop_in_two_rooms_takes_their_turns_and_wraps_nothing() -> None:
    """Each room's install takes that room's turns (RMK-307): the producer's
    own handling is left as it is, and another room gets the producer's."""
    producer, reviewer = _agent("writer"), _agent("editor")
    kit = RoomKit()
    for tenant in TENANTS:
        await kit.create_room(
            room_id=tenant, orchestration=Loop(agent=producer, reviewer=reviewer)
        )

    assert "on_event" not in vars(producer)
    assert all(kit._turn_runners.get(tenant, producer.channel_id) for tenant in TENANTS)
    assert kit._turn_runners.get("tenant-C", producer.channel_id) is None
    await kit.close()


async def test_a_voice_supervisor_busy_in_one_room_is_free_in_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()

    async def run_and_deliver(**kwargs: Any) -> None:
        await release.wait()
        kwargs["on_done"]()

    monkeypatch.setattr(_install_auto, "_async_run_and_deliver", run_and_deliver)
    supervisor, worker = _agent("sup"), _agent("worker")
    kit, _, provider, sessions = await _voice_rooms(
        lambda: Supervisor(
            supervisor, [worker], strategy="sequential", auto_delegate=True, async_delivery=True
        )
    )

    for call_id, tenant in (("c1", "tenant-A"), ("c2", "tenant-B"), ("c3", "tenant-A")):
        arguments = {"task": f"analyze for {tenant}"}
        await provider.simulate_tool_call(sessions[tenant], call_id, "delegate_workers", arguments)
    await _until(lambda: len(provider.tool_results) == 3)
    release.set()

    statuses = {
        call_id: json.loads(result)["status"] for _s, call_id, result in provider.tool_results
    }
    assert statuses == {"c1": "dispatched", "c2": "dispatched", "c3": "already_running"}
    await kit.close()


async def test_a_sync_auto_supervisor_in_two_rooms_runs_once_per_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran_for: list[str] = []

    async def one_pass(
        kit: Any,
        rid: str,
        supervisor: Any,
        original_on_event: Any,
        event: Any,
        binding: Any,
        context: Any,
        *args: Any,
        **kwargs: Any,
    ) -> ChannelOutput:
        # Like the real pass, it ends by letting the supervisor answer through
        # the on_event it wrapped: a second wrapper there would delegate again.
        ran_for.append(rid)
        return await original_on_event(event, binding, context)

    monkeypatch.setattr(_install_auto, "_one_pass_delegate", one_pass)
    monkeypatch.setattr(_install_auto, "_two_pass_delegate", one_pass)
    supervisor, worker = _agent("sup"), _agent("worker")
    kit = RoomKit()
    for tenant, channel_id in zip(TENANTS, ("sms-a", "sms-b"), strict=True):
        kit.register_channel(SimpleChannel(channel_id))
        orchestration = Supervisor(supervisor, [worker], strategy="sequential", auto_delegate=True)
        await kit.create_room(room_id=tenant, orchestration=orchestration)
        await kit.attach_channel(tenant, channel_id)

    await kit.process_inbound(
        InboundMessage(channel_id="sms-b", sender_id="u", content=TextContent(body="analyze X"))
    )

    assert ran_for == ["tenant-B"]
    await kit.close()


async def test_a_result_tool_is_declared_in_the_child_room_only() -> None:
    seen: list[str] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        seen.append(name)
        return "ok"

    submitting = AIToolCall(
        id="c1", name="submit_result", arguments={"status": "completed", "summary": "x"}
    )
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", tool_calls=[submitting]),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel(
        "ai1", provider=provider, tool_handler=handler, tools=[_tool("lookup")], tool_search=False
    )
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="customer-room",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )

    kit = RoomKit()

    # A delegation to this same agent runs in another room meanwhile.
    with (
        capture_result(kit._owner_tools.of("ai1"), "parent::task-abc", SUBMIT_RESULT),
        owner_tools_in(kit, "ai1", "customer-room"),
    ):
        await respond(
            channel,
            make_event(room_id="customer-room", body="hi", channel_id="sms1"),
            binding,
            RoomContext(room=Room(id="customer-room")),
        )

    assert [tool.name for tool in provider.calls[0].tools or []] == [
        "lookup",
        "read_stored_result",
    ]
    assert seen == []
    assert not room_tool_names(kit, "ai1", "parent::task-abc")


async def test_the_supervisor_runs_its_sub_runs_without_its_strategy_tool() -> None:
    """``delegate_workers`` is declared where the supervisor was installed; its
    dispatch and review turns, in ``::task-`` rooms, never see it."""
    call = AIToolCall(id="d1", name="delegate_workers", arguments={"task": "research X"})
    sup_model = MockAIProvider(
        ai_responses=[
            AIResponse(content="", tool_calls=[call]),
            *(AIResponse(content="noted") for _ in range(20)),
        ]
    )
    supervisor = Agent("sup", provider=sup_model, tool_search=False)
    worker = Agent("worker", provider=MockAIProvider(responses=["findings"]), tool_search=False)
    kit = RoomKit()
    for tenant, channel_id in zip(TENANTS, ("sms-a", "sms-b"), strict=True):
        kit.register_channel(SimpleChannel(channel_id))
        orchestration = Supervisor(supervisor, [worker], strategy="sequential")
        await kit.create_room(room_id=tenant, orchestration=orchestration)
        await kit.attach_channel(tenant, channel_id)

    await kit.process_inbound(
        InboundMessage(channel_id="sms-a", sender_id="u", content=TextContent(body="go"))
    )
    await _until(lambda: len(sup_model.calls) >= 3)

    declared = [[tool.name for tool in call.tools or []] for call in sup_model.calls]
    assert "delegate_workers" in declared[0]
    assert all("delegate_workers" not in names for names in declared[1:-1])
    assert room_tool_names(kit, "sup", "tenant-B") == ["delegate_workers"]
    await kit.close()


class _Pipeline:
    """A voice channel with its own tool, driven by a two-agent pipeline."""

    def __init__(self) -> None:
        self.agent_calls: list[str] = []
        self.channel_calls: list[str] = []
        self.provider = MockRealtimeProvider()
        webcam = {"name": "describe_webcam", "description": "see", "parameters": {}}
        self.voice = RealtimeVoiceChannel(
            "voice",
            provider=self.provider,
            transport=MockRealtimeTransport(),
            tools=[webcam],
            tool_handler=self._channel_handler,
        )
        # triage serves lookup_order itself; billing declares refund and a
        # describe_webcam of its own without a handler for either.
        self.triage = _agent("triage", _tool("lookup_order"), tool_handler=self._agent_handler)
        self.billing = _agent("billing", _tool("refund"), _tool("describe_webcam"))

    async def _agent_handler(self, name: str, arguments: dict[str, Any]) -> str:
        self.agent_calls.append(name)
        return '{"served_by": "agent"}'

    async def _channel_handler(self, name: str, arguments: dict[str, Any]) -> str:
        self.channel_calls.append(name)
        return '{"served_by": "channel"}'

    async def start(self) -> tuple[RoomKit, Any]:
        kit = RoomKit()
        for channel in (self.voice, self.triage, self.billing):
            kit.register_channel(channel)
        ConversationPipeline(
            stages=[
                PipelineStage(phase="triage", agent_id="triage", next="billing"),
                PipelineStage(phase="billing", agent_id="billing", next=None),
            ]
        ).install(kit, [self.triage, self.billing], voice_channel_id="voice")
        room = await kit.create_room(room_id="r1")
        triage_state = ConversationState(phase="triage", active_agent_id="triage")
        await kit.store.update_room(set_conversation_state(room, triage_state))
        await kit.attach_channel("r1", "voice")
        return kit, await self.voice.start_session("r1", "caller", "ws")

    async def call(self, session: Any, call_id: str, name: str, **arguments: Any) -> Any:
        await self.provider.simulate_tool_call(session, call_id, name, arguments)
        await _until(lambda: any(c == call_id for _s, c, _r in self.provider.tool_results))
        return json.loads(self.provider.tool_results[-1][2])


async def test_a_realtime_pipeline_declares_the_channels_and_the_active_agents_tools() -> None:
    pipeline = _Pipeline()
    kit, session = await pipeline.start()

    agent_tool = await pipeline.call(session, "c1", "lookup_order")
    channel_tool = await pipeline.call(session, "c2", "describe_webcam")

    assert _declared(pipeline.provider, session) == [
        "describe_webcam",
        "lookup_order",
        "handoff_conversation",
    ]
    assert agent_tool == {"served_by": "agent"}
    assert channel_tool == {"served_by": "channel"}
    await kit.close()


async def test_a_handoff_declares_the_next_agents_tools_and_the_channel_serves_the_rest() -> None:
    pipeline = _Pipeline()
    kit, session = await pipeline.start()

    await pipeline.call(
        session, "c1", "handoff_conversation", target="billing", reason="r", summary="s"
    )
    refund = await pipeline.call(session, "c2", "refund")
    webcam = await pipeline.call(session, "c3", "describe_webcam")

    connected = [c.args for c in pipeline.provider.calls if c.method == "connect"][-1]
    assert [t["name"] for t in connected["tools"]] == [
        "describe_webcam",
        "refund",
        "handoff_conversation",
    ]
    # billing has no handler of its own: its tools, the specialised
    # describe_webcam included, fall to the channel's.
    assert (refund, webcam) == ({"served_by": "channel"}, {"served_by": "channel"})
    assert pipeline.channel_calls == ["refund", "describe_webcam"]
    await kit.close()


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)
