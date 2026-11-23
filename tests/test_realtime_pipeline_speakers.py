"""A realtime channel's pipeline diarization fires ON_SPEAKER_CHANGE (RFC §12.3.9).

The audio pipeline in speech-to-speech mode lists diarization among its uses
(Section 12.4), and a speaker change it detects MUST fire ON_SPEAKER_CHANGE
— a VoiceChannel's did, a RealtimeVoiceChannel's never reached the hooks.
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.events import SpeakerChangeEvent
from roomkit.voice.pipeline.config import AudioPipelineConfig
from roomkit.voice.pipeline.diarization.base import DiarizationResult
from roomkit.voice.pipeline.diarization.mock import MockDiarizationProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

_FRAME = b"\x10\x00" * 320  # 20 ms at 16 kHz


async def test_pipeline_speaker_changes_fire_the_hook_nobody_excepted() -> None:
    diarization = MockDiarizationProvider(
        results=[
            DiarizationResult(speaker_id="alice", confidence=0.9, is_new_speaker=True),
            DiarizationResult(speaker_id="alice", confidence=0.9, is_new_speaker=False),
            DiarizationResult(speaker_id=None, confidence=0.0, is_new_speaker=False),
            DiarizationResult(speaker_id="bob", confidence=0.8, is_new_speaker=True),
        ]
    )
    transport = MockRealtimeTransport()
    channel = RealtimeVoiceChannel(
        "rt-speakers",
        provider=MockRealtimeProvider(),
        transport=transport,
        pipeline=AudioPipelineConfig(diarization=diarization),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "rt-speakers")
    changes: list[SpeakerChangeEvent] = []

    @kit.hook(HookTrigger.ON_SPEAKER_CHANGE, HookExecution.ASYNC)
    async def on_change(event: SpeakerChangeEvent, _ctx: object) -> None:
        changes.append(event)

    session = await channel.start_session(room.id, "user-1", "conn")
    for _ in range(4):
        await transport.simulate_client_audio(session, _FRAME)
    for _ in range(200):
        if len(changes) >= 2:
            break
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.02)  # nothing further should land

    assert [(c.speaker_id, c.source, c.session.id) for c in changes] == [
        ("alice", "pipeline", session.id),
        ("bob", "pipeline", session.id),
    ]
    await channel.end_session(session)
    await kit.close()
