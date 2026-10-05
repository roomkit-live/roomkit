"""The events a realtime channel builds name its own channel type: an
audio-video channel's session start, tool hooks and provider errors say
``REALTIME_AUDIO_VIDEO``, so a hook filtered on that type sees them (RMK-501).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_framework import SimpleChannel

KINDS = {
    ChannelType.REALTIME_VOICE: RealtimeVoiceChannel,
    ChannelType.REALTIME_AUDIO_VIDEO: RealtimeAudioVideoChannel,
}


async def _tool(name: str, arguments: dict[str, Any]) -> str:
    return "done"


@pytest.mark.parametrize("kind", list(KINDS), ids=lambda kind: kind.value)
async def test_every_event_names_the_channel_s_own_type(kind: ChannelType) -> None:
    provider = MockRealtimeProvider()
    channel = KINDS[kind](
        "rt", provider=provider, transport=MockRealtimeTransport(), tool_handler=_tool
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "rt")
    seen: dict[str, Any] = {}

    def record(label: str, read: Any = lambda event: event.channel_type) -> Any:
        async def hook(event: Any, context: Any) -> None:
            seen[label] = read(event)

        return hook

    kit.hook(HookTrigger.ON_SESSION_STARTED, HookExecution.ASYNC)(record("started"))
    kit.hook(HookTrigger.BEFORE_TOOL_USE, HookExecution.ASYNC)(record("before"))
    kit.hook(HookTrigger.ON_TOOL_CALL, HookExecution.ASYNC)(record("report"))
    kit.hook(HookTrigger.ON_ERROR, HookExecution.ASYNC)(
        record("error", lambda event: event.source.channel_type)
    )

    session = await channel.start_session("r", "u1", "ws")
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    channel._on_provider_error(session, "rate_limit_exceeded", "slow down")  # noqa: SLF001
    for _ in range(200):
        if len(seen) == 4:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert seen == {"started": kind, "before": kind, "report": kind, "error": kind}


@pytest.mark.parametrize("kind", list(KINDS), ids=lambda kind: kind.value)
async def test_an_auto_greeting_is_the_model_s_own_turn_on_every_realtime_channel(
    kind: ChannelType,
) -> None:
    """Its session start names its own type; the greeting still reaches the
    model as an ``assistant`` turn, never as a broadcast to the room's other
    transports."""
    provider = MockRealtimeProvider()
    channel = KINDS[kind]("rt", provider=provider, transport=MockRealtimeTransport())
    agent = Agent(
        "boss",
        provider=MockAIProvider(responses=["x"]),
        greeting="Welcome aboard!",
        auto_greet=True,
    )
    sms = SimpleChannel("sms")
    kit = RoomKit()
    for registered in (agent, sms, channel):
        kit.register_channel(registered)
    await kit.create_room(room_id="r")
    for channel_id in ("boss", "sms", "rt"):
        await kit.attach_channel("r", channel_id)

    await channel.start_session("r", "u1", "ws")
    for _ in range(200):
        if any(call.method == "inject_text" for call in provider.calls):
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    await kit.close()

    injected = [
        (call.args.get("role"), call.args.get("text"))
        for call in provider.calls
        if call.method == "inject_text"
    ]
    assert injected == [("assistant", "Welcome aboard!")]
    assert sms.delivered == []
