"""A proactive delivery can be the application's instruction (RFC §22.1, RMK-310).

``deliver(..., instruction=True)`` goes through the strategy, the delivery
hooks and the backend like any delivery, and arrives as an INSTRUCTION
through the text pipeline (§10.1.1) or with the system intent in a realtime
session (§12.4): never stored or read as a participant's words.
"""

from __future__ import annotations

import asyncio

from roomkit import ChannelCategory, HookResult, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.delivery import DeliveryContext, _QueuedRequest
from roomkit.delivery.base import DeliveryItem
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.worker import execute_delivery
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_framework import SimpleChannel


async def _room(kit: RoomKit | None = None) -> tuple[RoomKit, AIChannel]:
    kit = kit or RoomKit()
    agent = AIChannel("assistant", provider=MockAIProvider(responses=["ok"]))
    kit.register_channel(agent)
    kit.register_channel(SimpleChannel("phone"))
    await kit.create_room(room_id="call")
    await kit.attach_channel("call", "phone")
    await kit.attach_channel("call", "assistant", category=ChannelCategory.INTELLIGENCE)
    return kit, agent


async def _stored_text(kit: RoomKit, kind: EventType = EventType.MESSAGE) -> str:
    events = await kit.store.list_events("call")
    return " ".join(getattr(e.content, "body", "") for e in events if e.type == kind)


async def test_an_instruction_opens_the_agent_s_turn_and_is_not_stored_as_the_room_s() -> None:
    """Only the agent keeps it, as its own row (RFC §10.1.1)."""
    kit, agent = await _room()

    outcome = await kit.deliver(
        "call",
        "Tell the caller their order shipped.",
        addressed_to=["assistant"],
        instruction=True,
    )

    assert outcome.status == "sent"
    told = str(agent._provider.calls[0].messages[-1].content)
    assert told.startswith("[Instruction from the application")
    assert "order shipped" in told
    assert "order shipped" not in await _stored_text(kit)
    assert "order shipped" in await _stored_text(kit, EventType.INSTRUCTION_READ)
    await kit.close()


async def test_an_unaddressed_instruction_is_refused() -> None:
    kit, agent = await _room()

    outcome = await kit.deliver("call", "Say hello.", instruction=True)

    assert (outcome.status, outcome.reason) == ("blocked", "instruction_unaddressed")
    assert agent._provider.calls == []
    await kit.close()


async def test_the_delivery_hooks_see_and_gate_an_instruction() -> None:
    kit, agent = await _room()
    seen: list[EventType] = []

    @kit.hook(HookTrigger.BEFORE_DELIVER)
    async def quiet_hours(event: RoomEvent, ctx: RoomContext) -> HookResult:
        seen.append(event.type)
        return HookResult.block("quiet hours")

    outcome = await kit.deliver("call", "Nudge.", addressed_to=["assistant"], instruction=True)

    assert outcome.status == "blocked"
    assert seen == [EventType.INSTRUCTION]
    assert agent._provider.calls == []
    await kit.close()


async def test_a_realtime_session_takes_an_instruction_as_system() -> None:
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel("voice", provider=provider, transport=MockRealtimeTransport())
    async with RoomKit() as kit:
        kit.register_channel(voice)
        await kit.create_room(room_id="call")
        await kit.attach_channel("call", "voice")
        await voice.start_session("call", "caller", object())

        await kit.deliver("call", "Direction.", channel_id="voice", instruction=True)
        await kit.deliver("call", "Content.", channel_id="voice")

    assert [(text, role) for _, text, role in provider.injected_texts] == [
        ("Direction.", "system"),
        ("Content.", "user"),
    ]


async def test_queued_never_merges_an_instruction_with_a_message() -> None:
    kit = RoomKit()
    loop = asyncio.get_running_loop()

    def request(*, instruction: bool) -> _QueuedRequest:
        ctx = DeliveryContext(kit=kit, room_id="r", content="x", instruction=instruction)
        return _QueuedRequest(ctx, "phone", loop.create_future())

    assert not request(instruction=True).compatible(request(instruction=False))
    assert request(instruction=True).compatible(request(instruction=True))


async def test_a_queued_instruction_stays_one_through_the_backend() -> None:
    backend = InMemoryDeliveryBackend()
    kit, agent = await _room(RoomKit(delivery_backend=backend))
    async with kit:
        queued = await kit.deliver("call", "Nudge.", addressed_to=["assistant"], instruction=True)
        [item] = await backend.dequeue("worker", timeout=0.1)
        restored = DeliveryItem.model_validate_json(item.model_dump_json())

        outcome = await execute_delivery(kit, restored)

    assert queued.status == "queued" and restored.instruction
    assert outcome.status == "sent"
    assert str(agent._provider.calls[0].messages[-1].content).startswith("[Instruction")
