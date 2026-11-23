"""A voice channel built without one of its components (RFC §12.2).

Each of the backend, the STT and the TTS is optional at construction: the
channel does without the paths that need the missing one. A call that asks it
to speak without a TTS or a backend raises (tests/test_voice_say_play.py).
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.models.enums import EventType
from roomkit.voice import AudioFrame, AudioPipelineConfig, MockVADProvider
from roomkit.voice.backends import MockVoiceBackend
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType
from roomkit.voice.stt import MockSTTProvider
from roomkit.voice.tts import MockTTSProvider
from tests.channel_tasks import settled

_FRAME = AudioFrame(data=b"\x01\x00" * 80)


def _utterance() -> MockVADProvider:
    return MockVADProvider(
        events=[
            VADEvent(type=VADEventType.SPEECH_START),
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"\x01\x00" * 80),
        ]
    )


async def test_without_an_stt_speech_reaches_no_room() -> None:
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice",
        tts=MockTTSProvider(),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=_utterance()),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    ended = asyncio.Event()

    @kit.hook(HookTrigger.ON_SPEECH_END, execution=HookExecution.ASYNC)
    async def on_speech_end(event: Any, ctx: Any) -> None:
        ended.set()

    session = await kit.join(room.id, "voice")
    await backend.simulate_audio_received(session, _FRAME)
    await backend.simulate_audio_received(session, _FRAME)
    await asyncio.wait_for(ended.wait(), 2)
    await settled(channel, timeout=2)

    events = await kit.store.list_events(room.id)
    assert [e for e in events if e.type == EventType.MESSAGE] == []
    await kit.close()


async def test_without_a_tts_a_delivered_message_plays_nothing() -> None:
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice", stt=MockSTTProvider(), backend=backend)
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")
    await kit.join(room.id, "voice")

    await kit.post(room.id, "Bonjour")  # returns once the room's transports had it

    assert backend.sent_audio == []
    await kit.close()
