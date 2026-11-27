"""A provider that cuts its own turns keeps its stream across turns (RFC §12.2).

A VoiceChannel in continuous mode opens a stream per turn, except for a
diarizing STT (RFC §12.2.3) and for a provider reporting
``keeps_stream_across_turns``: its model ends each turn with a final and keeps
listening on the same stream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from roomkit import HookResult, HookTrigger, RoomKit, VoiceChannel, _clock
from roomkit.models.enums import EventType
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig, MockTurnDetector, TurnDecision
from roomkit.voice.stt.base import STTProvider, keeps_stream

_FRAME = AudioFrame(data=b"\x01\x00" * 1600)  # 3200 bytes: one STT buffer flush


class _TurnsSTT(STTProvider):
    """Answers one scripted final per chunk with audio, its turns on one stream."""

    def __init__(self, finals: list[str], *, keeps: bool = True) -> None:
        self._finals = list(finals)
        self._keeps = keeps
        self.streams = 0
        self.languages: list[str | None] = []
        self.chunks: list[AudioChunk] = []

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_language_override(self) -> bool:
        return True

    @property
    def keeps_stream_across_turns(self) -> bool:
        return self._keeps

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="")

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        self.streams += 1
        self.languages.append(language)
        async for chunk in audio_stream:
            self.chunks.append(chunk)
            if any(chunk.data) and self._finals:
                yield TranscriptionResult(text="", is_final=False, is_speech_start=True)
                yield TranscriptionResult(text=self._finals.pop(0), is_final=True)


class _Clock:
    """A media clock (``roomkit._clock``) on real time, with time that can pass at once."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.waited: list[float | None] = []

    def monotonic(self) -> float:
        return time.monotonic() + self.offset

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def wait_for[T](self, awaitable: Awaitable[T], timeout: float | None) -> T:
        self.waited.append(timeout)
        return await asyncio.wait_for(awaitable, timeout)


def _audio_seconds(stt: _TurnsSTT) -> float:
    return sum(len(c.data) for c in stt.chunks) / (2 * 16000)


async def _eventually(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)


class _Room:
    """A kit with one continuous VoiceChannel, and the transcripts its hook saw."""

    def __init__(self, stt: STTProvider, pipeline: AudioPipelineConfig | None = None) -> None:
        self.backend = MockVoiceBackend()
        self.channel = VoiceChannel(
            "voice-1", stt=stt, backend=self.backend, pipeline=pipeline or AudioPipelineConfig()
        )
        self.kit = RoomKit(stt=stt, voice=self.backend)
        self.kit.register_channel(self.channel)
        self.transcripts: list[str] = []
        self.held: dict[str, asyncio.Event] = {}

        @self.kit.hook(HookTrigger.ON_TRANSCRIPTION)
        async def on_transcription(event: Any, ctx: Any) -> HookResult:
            self.transcripts.append(event.text)
            if (hold := self.held.get(event.text)) is not None:
                await asyncio.wait_for(hold.wait(), 3.0)
            return HookResult.allow()

    async def start(self) -> None:
        room = await self.kit.create_room()
        self.room_id = room.id
        await self.kit.attach_channel(room.id, "voice-1")
        self.session = await self.kit.join(room.id, "voice-1", participant_id="owner")
        assert self.channel._continuous_stt

    async def speak(self) -> None:
        await self.backend.simulate_audio_received(self.session, _FRAME)

    async def said(self, count: int) -> list[str]:
        """The channel's messages in the room, once there are ``count`` of them."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 3.0
        while True:
            events = await self.kit.store.list_events(self.room_id, offset=0, limit=50)
            found = [
                e.content.body
                for e in events
                if e.type == EventType.MESSAGE and e.source.channel_id == "voice-1"
            ]
            if len(found) >= count or loop.time() > deadline:
                return found
            await asyncio.sleep(0.02)


class TestTheClaim:
    def test_only_an_explicit_true_counts(self) -> None:
        assert keeps_stream(_TurnsSTT([])) is True
        assert keeps_stream(_TurnsSTT([], keeps=False)) is False
        # A mock's auto-created attribute is no claim.
        assert keeps_stream(MagicMock(spec=["name"])) is False
        assert keeps_stream(MagicMock()) is False

    def test_a_diarizing_provider_keeps_its_stream(self) -> None:
        stt = MagicMock(supports_diarization=True, keeps_stream_across_turns=False)
        assert keeps_stream(stt) is True


class TestVoiceChannelKeepsTheStream:
    async def test_one_stream_carries_turn_after_turn(self) -> None:
        stt = _TurnsSTT(["What is the capital of Italy?", "Answer in one word."])
        room = _Room(stt)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcripts) == 1)
        await room.speak()
        messages = await room.said(2)
        await room.channel.close()

        assert stt.streams == 1
        assert messages == ["What is the capital of Italy?", "Answer in one word."]

    async def test_a_provider_that_does_not_claim_it_gets_a_stream_per_turn(self) -> None:
        stt = _TurnsSTT(["First.", "Second."], keeps=False)
        room = _Room(stt)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcripts) == 1)
        await room.speak()
        messages = await room.said(2)
        await room.channel.close()

        assert stt.streams == 2
        assert messages == ["First.", "Second."]

    async def test_finals_are_routed_in_order_even_when_a_hook_is_slow(self) -> None:
        stt = _TurnsSTT(["First.", "Second."])
        room = _Room(stt)
        first = room.held["First."] = asyncio.Event()
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcripts) == 1)
        await room.speak()
        # The second final arrives while the first is still in its hook.
        await _eventually(lambda: not stt._finals)
        first.set()
        messages = await room.said(2)
        await room.channel.close()

        assert messages == ["First.", "Second."]

    async def test_silence_keeps_the_stream_level_with_the_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A muted mic during playback sends nothing; the model's turn
        # detection runs on audio time, so the pause is sent as silence.
        # The pause passes on the media clock at once, checked every 10 ms.
        clock = _Clock()
        monkeypatch.setattr("roomkit.channels.voice._KEPT_STREAM_PACE_S", 0.01)
        stt = _TurnsSTT(["First.", "Second."])
        room = _Room(stt)
        with _clock.installed(clock):
            await room.start()

            opened = clock.monotonic()
            await room.speak()
            await _eventually(lambda: len(room.transcripts) == 1)
            clock.offset += 0.6
            await _eventually(lambda: _audio_seconds(stt) >= 0.6)
            await room.speak()
            await _eventually(lambda: len(room.transcripts) == 2)
            elapsed = clock.monotonic() - opened
            await room.kit.close()

        assert 0.01 in clock.waited  # the pacing period is waited on the media clock
        assert stt.streams == 1
        assert any(not any(c.data) for c in stt.chunks)
        assert _audio_seconds(stt) >= elapsed - 0.2

    async def test_a_turn_detector_joins_the_turns_it_finds_incomplete(self) -> None:
        # The kept stream's finals are routed under the ordering lock and their
        # delivery is not awaited there: a turn the detector holds waits for
        # the next final, and the two reach the room as one message.
        detector = MockTurnDetector(
            decisions=[TurnDecision(is_complete=False), TurnDecision(is_complete=True)]
        )
        pipeline = AudioPipelineConfig(turn_detector=detector, turn_incomplete_wait_ms=2000)
        stt = _TurnsSTT(["I would like", "a table for two."])
        room = _Room(stt, pipeline)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcripts) == 1)
        await room.speak()
        messages = await room.said(1)
        await room.channel.close()

        assert stt.streams == 1
        assert messages == ["I would like a table for two."]

    async def test_a_language_change_opens_a_new_stream_with_it(self) -> None:
        # A language holds for the life of a stream (RFC §12.2).
        stt = _TurnsSTT(["Hello.", "Bonjour."])
        room = _Room(stt)
        await room.start()

        await room.speak()
        await _eventually(lambda: len(room.transcripts) == 1)
        room.channel.set_stt_language(room.session, "fr")
        await room.speak()  # the next stream opens on this audio
        messages = await room.said(2)
        await room.channel.close()

        assert stt.languages == [None, "fr"]
        assert messages == ["Hello.", "Bonjour."]
