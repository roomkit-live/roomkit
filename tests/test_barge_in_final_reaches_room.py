"""The words that cut the agent off are the person's turn, not echo (RFC §12.3.13).

In continuous mode a partial that the interruption strategy lets through cuts
the playback, and the STT's final for the same words follows at once. The cut
runs as its own task (ON_BARGE_IN hooks, then ``interrupt()``), so the final can
be handled while the playback is still registered: the echo guard took it for
the agent's own voice and dropped it, and the person's turn never reached the
room (RMK-543).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, VoiceChannel
from roomkit.channels.voice import TTSPlaybackState
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult, VoiceCapability
from roomkit.voice.interruption import InterruptionConfig, InterruptionStrategy
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.pipeline.backchannel.base import BackchannelDecision
from roomkit.voice.pipeline.backchannel.mock import MockBackchannelDetector
from roomkit.voice.stt.base import STTProvider

_SILENT_CHUNK = AudioFrame(data=b"\x00\x00" * 1600)


class _ScriptSTT(STTProvider):
    """Server-endpointing STT that says its scripted results on the first cycle."""

    def __init__(self, results: list[TranscriptionResult]) -> None:
        self._results = list(results)

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
        while self._results:
            yield self._results.pop(0)


async def _talk_over(results: list[TranscriptionResult], *, hook_delay: float):
    """A playback, then *results* from the STT; ON_BARGE_IN takes *hook_delay*."""
    detector = MockBackchannelDetector(decisions=[BackchannelDecision(is_backchannel=False)])
    stt = _ScriptSTT(results)
    backend = MockVoiceBackend(capabilities=VoiceCapability.INTERRUPTION)
    channel = VoiceChannel(
        "voice-1",
        stt=stt,
        backend=backend,
        pipeline=AudioPipelineConfig(),  # no VAD + streaming STT = continuous
        interruption=InterruptionConfig(
            strategy=InterruptionStrategy.SEMANTIC, backchannel_detector=detector
        ),
    )
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "voice-1")
    session = await kit.connect_voice("r1", "user-1", "voice-1")
    heard: list[str] = []
    cuts: list[object] = []

    @kit.hook(HookTrigger.ON_BARGE_IN, HookExecution.ASYNC)
    async def on_barge_in(event: object, context: object) -> None:
        cuts.append(event)

    @kit.hook(HookTrigger.ON_BARGE_IN)
    async def slow_barge_in_hook(event: object, context: object) -> HookResult:
        await asyncio.sleep(hook_delay)  # the store, a host's hook: the cut takes time
        return HookResult.allow()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION)
    async def on_transcription(event: Any, context: object) -> HookResult:
        heard.append(event.text)
        return HookResult.allow()

    channel._playing_sessions[session.id] = TTSPlaybackState(  # noqa: SLF001
        session_id=session.id, text="Demain à Québec, il fera entre 2,9 et 8,1 degrés."
    )
    await backend.simulate_audio_received(session, _SILENT_CHUNK)
    await asyncio.sleep(0.5)
    await channel.close()
    await kit.close()
    return heard, cuts


async def test_the_final_of_the_words_that_cut_the_agent_reaches_the_room() -> None:
    heard, cuts = await _talk_over(
        [
            TranscriptionResult(text="Attends, et pour", is_final=False),
            TranscriptionResult(text="Attends, et pour Montréal ?", is_final=True),
        ],
        hook_delay=0.1,
    )
    assert len(cuts) == 1
    assert heard == ["Attends, et pour Montréal ?"]


async def test_a_final_over_a_playback_nobody_cut_is_still_echo() -> None:
    # No partial asked to cut: the final heard during playback is the agent's voice.
    heard, cuts = await _talk_over(
        [TranscriptionResult(text="il fera entre 2,9 et 8,1 degrés", is_final=True)],
        hook_delay=0.0,
    )
    assert cuts == []
    assert heard == []
