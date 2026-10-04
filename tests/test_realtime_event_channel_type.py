"""The events a realtime channel builds name its own channel type: an
audio-video channel's session start, tool hooks and provider errors say
``REALTIME_AUDIO_VIDEO``, so a hook filtered on that type sees them (RMK-501).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_av import RealtimeAudioVideoChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import ChannelType
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

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
