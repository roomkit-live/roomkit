"""Interruption strategies as wired into VoiceChannel (RFC §12.6).

``tests/test_interruption.py`` covers ``InterruptionHandler`` in isolation.
These cover the wiring, where the duration the handler evaluates comes from:
the first evaluation happens at speech onset with ``speech_duration_ms=0``, so
CONFIRMED (the documented default) only reaches its minimum if something takes
a second look. DISABLED queues the speech until the bot finishes rather than
discarding it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline.vad.base import VADEvent, VADEventType


async def _voice_room(
    interruption: InterruptionConfig,
) -> tuple[RoomKit, VoiceChannel, VoiceSession]:
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice-1", backend=backend, interruption=interruption)
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.connect_voice(room.id, "user-1", "voice-1")
    channel.bind_session(
        session,
        room.id,
        ChannelBinding(room_id=room.id, channel_id="voice-1", channel_type=ChannelType.VOICE),
    )
    return kit, channel, session


def _playing(channel: VoiceChannel, session_id: str) -> None:
    channel._playing_sessions[session_id] = TTSPlaybackState(
        session_id=session_id,
        text="a long answer the user talks over",
        started_at=datetime.now(UTC) - timedelta(seconds=1),
    )


class TestConfirmedStrategyInterrupts:
    async def test_sustained_speech_confirms_the_barge_in(self) -> None:
        """RFC §12.6 — CONFIRMED interrupts once the speech has sustained for
        ``min_speech_ms``. The onset evaluation alone cannot say yes."""
        kit, channel, session = await _voice_room(
            InterruptionConfig(strategy=InterruptionStrategy.CONFIRMED, min_speech_ms=50)
        )
        barge_ins: list[object] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_ins.append(event)

        _playing(channel, session.id)
        channel._on_pipeline_vad_event(session, VADEvent(type=VADEventType.SPEECH_START))

        # Onset: nothing yet, and the speech is held as possible echo.
        await asyncio.sleep(0.01)
        assert barge_ins == []
        assert session.id in channel._suppressed_sessions

        # The speech sustains past min_speech_ms.
        await asyncio.sleep(0.15)
        assert len(barge_ins) == 1
        # Confirmed speech is no longer treated as echo.
        assert session.id not in channel._suppressed_sessions
        assert session.id not in channel._playing_sessions

        await kit.close()

    async def test_speech_shorter_than_minimum_does_not_interrupt(self) -> None:
        """A blip during playback stays a blip — and stays suppressed."""
        kit, channel, session = await _voice_room(
            InterruptionConfig(strategy=InterruptionStrategy.CONFIRMED, min_speech_ms=200)
        )
        barge_ins: list[object] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_ins.append(event)

        _playing(channel, session.id)
        channel._on_pipeline_vad_event(session, VADEvent(type=VADEventType.SPEECH_START))
        await asyncio.sleep(0.02)
        # Speech ends well before the minimum.
        channel._on_pipeline_speech_end(session, b"\x00\x00" * 80)

        await asyncio.sleep(0.3)
        assert barge_ins == []
        assert session.id in channel._playing_sessions

        await kit.close()


class TestDisabledStrategyQueuesSpeech:
    async def test_speech_during_playback_is_queued_not_discarded(self) -> None:
        """RFC §12.6 DISABLED — "user speech is queued until the bot
        finishes", not thrown away."""
        kit, channel, session = await _voice_room(
            InterruptionConfig(strategy=InterruptionStrategy.DISABLED)
        )
        barge_ins: list[object] = []

        @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
        async def on_barge_in(event, context):
            barge_ins.append(event)

        _playing(channel, session.id)
        channel._on_pipeline_vad_event(session, VADEvent(type=VADEventType.SPEECH_START))
        await asyncio.sleep(0.01)
        channel._on_pipeline_speech_end(session, b"\x11\x22" * 160)

        # Playback untouched, and the utterance is held rather than dropped.
        assert barge_ins == []
        assert session.id in channel._playing_sessions
        assert channel._queued_speech[session.id] == [b"\x11\x22" * 160]

        # Once the bot finishes, the held speech gets its turn.
        processed: list[bytes] = []

        async def capture(sess, audio, room_id, stream_state):  # noqa: ANN001
            processed.append(audio)

        channel._process_speech_end = capture  # type: ignore[assignment]
        await channel._flush_queued_speech(session.id)

        assert processed == [b"\x11\x22" * 160]
        assert session.id not in channel._queued_speech

        await kit.close()

    async def test_queue_is_bounded(self) -> None:
        """A stuck playback must not grow the backlog without limit."""
        from roomkit.channels.voice import _QUEUED_SPEECH_MAX_SEGMENTS

        kit, channel, session = await _voice_room(
            InterruptionConfig(strategy=InterruptionStrategy.DISABLED)
        )
        _playing(channel, session.id)

        for _ in range(_QUEUED_SPEECH_MAX_SEGMENTS + 3):
            channel._on_pipeline_vad_event(session, VADEvent(type=VADEventType.SPEECH_START))
            channel._on_pipeline_speech_end(session, b"\x01\x02" * 16)

        assert len(channel._queued_speech[session.id]) == _QUEUED_SPEECH_MAX_SEGMENTS

        await kit.close()
