"""Only words interrupt (RMK-559): a VoiceChannel in continuous mode, during
playback, with PhraseBackchannelDetector(cut_without_words=False)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

from roomkit import HookExecution, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.voice import InterruptionConfig, InterruptionStrategy, PhraseBackchannelDetector
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.base import STTProvider

_LOUD = AudioFrame(data=(1000).to_bytes(2, "little", signed=True) * 320)  # 20 ms, RMS 1000


class _HearsAfter(STTProvider):
    """A streaming STT that hears *words* 0.2 s into the audio, or nothing at all."""

    def __init__(self, words: str | None) -> None:
        self._words = words

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        async for _ in audio_stream:
            break
        await asyncio.sleep(0.2)
        if self._words:
            yield TranscriptionResult(text=self._words, is_final=False)
        async for _ in audio_stream:
            pass


def _semantic(*, cut_without_words: bool) -> InterruptionConfig:
    return InterruptionConfig(
        strategy=InterruptionStrategy.SEMANTIC,
        backchannel_detector=PhraseBackchannelDetector(cut_without_words=cut_without_words),
        transcript_wait_ms=1000,
    )


async def _cut(words: str | None, *, cut_without_words: bool) -> bool:
    """Whether 2 s of loud audio, with *words* heard in it, cut the agent's reply."""
    backend = MockVoiceBackend()
    channel = VoiceChannel(
        "voice",
        stt=_HearsAfter(words),
        backend=backend,
        pipeline=AudioPipelineConfig(),
        interruption=_semantic(cut_without_words=cut_without_words),
    )
    kit = RoomKit(voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice")
    session = await kit.join("r1", "voice", participant_id="u1")
    channel.bind_session(
        session,
        "r1",
        ChannelBinding(room_id="r1", channel_id="voice", channel_type=ChannelType.VOICE),
    )
    cuts: list[Any] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: Any, ctx: Any) -> None:
        cuts.append(event)

    channel._playing_sessions[session.id] = TTSPlaybackState(
        session_id=session.id,
        text="Voici le récapitulatif complet de la discussion.",
        started_at=datetime.now(UTC) - timedelta(seconds=2),
    )
    try:
        for _ in range(100):  # 2 s of loud audio while the agent speaks
            await backend.simulate_audio_received(session, _LOUD)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
    finally:
        await kit.close()
    return bool(cuts)


async def test_sound_without_words_cuts_by_default() -> None:
    assert await _cut(None, cut_without_words=True) is True


async def test_sound_without_words_does_not_cut_when_only_words_interrupt() -> None:
    assert await _cut(None, cut_without_words=False) is False


async def test_words_still_cut_when_only_words_interrupt() -> None:
    assert await _cut("attends, stop", cut_without_words=False) is True


async def test_an_acknowledgement_still_does_not_cut() -> None:
    assert await _cut("ouais", cut_without_words=False) is False
