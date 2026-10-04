"""A background result is handed back into the realtime model that asked for
it, on every channel that hosts one (RMK-501, RFC §23.3 step 8).

A realtime voice channel, an audio-video one and a conference with a realtime
model plugged in: the result is injected into the model's session with the
``system`` intent, and nothing is published to the room's other channels. An
agent attached without a category takes part as the intelligence channel it
is, and ``deliver()`` to one never re-enters itself.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.conference.mock import MockConferenceBackend
from roomkit.models.enums import ChannelCategory
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


def _injected(provider: MockRealtimeProvider) -> list[tuple[Any, str]]:
    return [
        (call.args.get("role"), str(call.args.get("text")))
        for call in provider.calls
        if call.method == "inject_text"
    ]


@pytest.mark.parametrize("kind", ["voice", "audio-video", "conference"])
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
