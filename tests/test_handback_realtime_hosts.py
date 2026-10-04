"""A background result is handed back into the realtime model that asked for
it, on every channel that hosts one (RMK-501, RFC §23.3 step 8).

A realtime voice channel, an audio-video one and a conference with a realtime
model plugged in: the result is injected into the model's session with the
``system`` intent, and nothing is published to the room's other channels. The
injection is announced to ON_REALTIME_TEXT_INJECTED, ``WaitForIdle`` waits for
the model's answer to end and for nobody to be heard, and a delivery whose
sessions were pinned is refused, not published, when the model is gone by the
time it goes out. An agent attached without a category takes part as the
intelligence channel it is, and ``deliver()`` to one never re-enters itself.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.conference.mock import MockConferenceBackend
from roomkit.core.delivery import WaitForIdle
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel


async def _host(kit: RoomKit, kind: str) -> tuple[Any, MockRealtimeProvider]:
    """A channel of *kind* hosting a realtime model, attached to room ``r``
    with its session open."""
    provider = MockRealtimeProvider()
    if kind == "conference":
        config = ConferenceRealtimeConfig(provider=provider)
        channel: Any = ConferenceChannel("host", backend=MockConferenceBackend(), realtime=config)
    else:
        cls = RealtimeVoiceChannel if kind == "voice" else RealtimeAudioVideoChannel
        channel = cls("host", provider=provider, transport=MockRealtimeTransport())
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "host")
    if kind == "conference":
        await channel._realtime.ensure_session("r")
    else:
        await channel.start_session("r", "u1", "ws")
    return channel, provider


HOSTS = ["voice", "audio-video", "conference"]


def _injected(provider: MockRealtimeProvider) -> list[tuple[Any, str]]:
    return [
        (call.args.get("role"), str(call.args.get("text")))
        for call in provider.calls
        if call.method == "inject_text"
    ]


@pytest.mark.parametrize("kind", HOSTS)
async def test_a_background_result_reaches_the_model_that_asked_for_it(kind: str) -> None:
    kit = RoomKit()
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["The draft."])))
    sms = SimpleChannel("sms")
    kit.register_channel(sms)
    _, provider = await _host(kit, kind)
    await kit.attach_channel("r", "sms")

    await kit.delegate("r", "worker", "Write it.", notify="host")
    await until(lambda: bool(_injected(provider)), timeout=5)
    await kit.close()

    [(role, text)] = _injected(provider)
    assert role == "system"
    assert "The draft." in text
    assert sms.delivered == []


@pytest.mark.parametrize("kind", HOSTS)
async def test_an_injection_is_announced_on_every_host(kind: str) -> None:
    kit = RoomKit()
    announced: list[tuple[str, Any]] = []

    @kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
    async def audit(event: Any, context: Any) -> None:
        announced.append((event.source.channel_id, event.metadata.get("injected_role")))

    await _host(kit, kind)

    outcome = await kit.deliver("r", "[result]", channel_id="host", instruction=True)
    await until(lambda: bool(announced))
    await kit.close()

    assert outcome.status == "sent"
    assert announced == [("host", "system")]


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.parametrize("busy", ["answering", "hearing"])
@pytest.mark.parametrize("kind", HOSTS)
async def test_wait_for_idle_waits_on_every_host(kind: str, busy: str) -> None:
    """Nothing is injected while the model answers or hears someone speak."""
    kit = RoomKit()
    channel, provider = await _host(kit, kind)
    [session] = channel.get_room_sessions("r")
    start, end = (
        (provider.simulate_response_start, provider.simulate_response_end)
        if busy == "answering"
        else (provider.simulate_speech_start, provider.simulate_speech_end)
    )
    await start(session)

    strategy = WaitForIdle(buffer=0, playback_timeout=5.0)
    delivery = asyncio.create_task(
        kit.deliver("r", "[result]", channel_id="host", instruction=True, strategy=strategy)
    )
    await _settle()
    waited = _injected(provider)
    await end(session)
    outcome = await asyncio.wait_for(delivery, timeout=5)
    await kit.close()

    assert waited == []
    assert outcome.status == "sent"
    assert [role for role, _ in _injected(provider)] == ["system"]


async def test_a_pinned_delivery_is_refused_when_the_model_is_unplugged_meanwhile() -> None:
    """The session was pinned before the wait; the conference's model is
    unplugged during it. The instruction is refused, never published to the
    room as the conference's own words."""
    kit = RoomKit()
    sms = SimpleChannel("sms")
    kit.register_channel(sms)
    channel, provider = await _host(kit, "conference")
    await kit.attach_channel("r", "sms")
    [session] = channel.get_room_sessions("r")
    await provider.simulate_response_start(session)

    strategy = WaitForIdle(buffer=0, playback_timeout=5.0)
    delivery = asyncio.create_task(
        kit.deliver("r", "[result]", channel_id="host", instruction=True, strategy=strategy)
    )
    await _settle()
    await channel.unplug_realtime()
    outcome = await asyncio.wait_for(delivery, timeout=5)
    published = [
        event
        for event in await kit.store.list_events("r")
        if event.type in (EventType.MESSAGE, EventType.INSTRUCTION)
    ]
    await kit.close()

    assert (outcome.status, outcome.reason) == ("unavailable", "voice_session_replaced")
    assert _injected(provider) == []
    assert sms.delivered == []
    assert published == []


@pytest.mark.parametrize("attached_first", [True, False], ids=["agent-first", "agent-second"])
async def test_an_agent_attached_without_a_category_is_an_intelligence_channel(
    attached_first: bool,
) -> None:
    kit = RoomKit()
    boss = Agent("boss", provider=MockAIProvider(responses=["Noted."]))
    kit.register_channel(boss)
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    order = ["boss", "sms"] if attached_first else ["sms", "boss"]
    for channel_id in order:
        await kit.attach_channel("r", channel_id)

    binding = await kit.store.get_binding("r", "boss")
    outcome = await kit.deliver("r", "[result]", addressed_to=["boss"], instruction=True)
    await kit.close()

    assert binding is not None and binding.category == ChannelCategory.INTELLIGENCE
    assert outcome.status == "sent"
    assert len(boss.provider.calls) == 1  # type: ignore[attr-defined]


async def test_deliver_to_an_agent_attached_as_a_transport_never_re_enters() -> None:
    """An agent attached as a transport, explicitly: the instruction has no
    transport to ride, and is refused rather than delivered through itself."""
    kit = RoomKit()
    kit.register_channel(Agent("boss", provider=MockAIProvider(responses=["Noted."])))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "boss", category=ChannelCategory.TRANSPORT)

    outcome = await kit.deliver("r", "[result]", channel_id="boss", instruction=True)
    await kit.close()

    assert (outcome.status, outcome.reason) == ("unavailable", "no_transport")


async def test_an_agent_attached_as_a_transport_is_never_the_room_s_transport() -> None:
    """With no channel named, the room's real transport carries the delivery,
    not an agent attached as one ahead of it."""
    kit = RoomKit()
    kit.register_channel(Agent("boss", provider=MockAIProvider(responses=["Noted."])))
    sms = SimpleChannel("sms")
    kit.register_channel(sms)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "boss", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel("r", "sms")

    outcome = await kit.deliver("r", "Your order shipped.")
    await kit.close()

    assert outcome.status == "sent"
    assert outcome.inbound is not None and outcome.inbound.event is not None
    assert outcome.inbound.event.source.channel_id == "sms"
