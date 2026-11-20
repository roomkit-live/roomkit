"""A voice stream observes a partial before consuming its next result (RFC §9.3)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomContext, RoomKit, VoiceChannel
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceSession
from roomkit.voice.events import PartialTranscriptionEvent, TranscriptionEvent
from roomkit.voice.pipeline import AudioPipelineConfig, MockVADProvider, VADEvent, VADEventType
from roomkit.voice.stt.mock import MockSTTProvider


@pytest.mark.parametrize("segmented", [False, True], ids=["continuous", "vad"])
async def test_partial_observer_finishes_before_the_stream_reads_the_final(
    segmented: bool,
) -> None:
    seen: list[str] = []

    class ReadySTT(MockSTTProvider):
        async def transcribe_stream(
            self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
        ) -> AsyncIterator[TranscriptionResult]:
            async for _ in audio_stream:
                yield TranscriptionResult(text="hel", is_final=False)
                seen.append("read final")
                yield TranscriptionResult(text="hello", is_final=True)
                return

    data = b"\x01\x00" * 1600
    vad = (
        MockVADProvider(
            [
                VADEvent(type=VADEventType.SPEECH_START, audio_bytes=data),
                VADEvent(type=VADEventType.SPEECH_END, audio_bytes=data),
            ]
        )
        if segmented
        else None
    )
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice",
        stt=ReadySTT(streaming=True),
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    done = asyncio.Event()

    @kit.hook(HookTrigger.ON_PARTIAL_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def partial(event: PartialTranscriptionEvent, context: RoomContext) -> None:
        # Yield once: scheduling the observer is not observing the result.
        await asyncio.sleep(0)
        seen.append(event.text)

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def final(event: TranscriptionEvent, context: RoomContext) -> HookResult:
        seen.append(event.text)
        done.set()
        return HookResult.allow()

    try:
        room = await kit.create_room()
        await kit.attach_channel(room.id, "voice")
        session = await kit.join(room.id, "voice", participant_id="alice")
        assert isinstance(session, VoiceSession)
        await backend.simulate_audio_received(session, AudioFrame(data=data))
        if segmented:
            await backend.simulate_audio_received(session, AudioFrame(data=data))
        await asyncio.wait_for(done.wait(), 5)
        assert seen == ["hel", "read final", "hello"]
    finally:
        await kit.close()
