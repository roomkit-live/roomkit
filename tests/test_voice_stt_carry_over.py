"""Audio carried from one continuous STT stream into the next is bounded (RFC
§12.2, RMK-581): a reconnect that waits on a silent service gathers a backlog,
and the next stream gets only the most recent seconds of it."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import RoomKit, VoiceChannel
from roomkit.channels._voice_stt import _CARRY_OVER_MAX_S, _carry_over
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.base import STTProvider

_TENTH = b"\x01\x00" * 1600  # 100 ms of 16 kHz mono


def _queue(*items: AudioChunk | None) -> asyncio.Queue[Any]:
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=500)
    for item in items:
        queue.put_nowait(item)
    return queue


def _drained(queue: asyncio.Queue[Any]) -> list[Any]:
    return [queue.get_nowait() for _ in range(queue.qsize())]


def _tenths(n: int) -> list[AudioChunk]:
    """*n* chunks of 100 ms, each with its index in ``timestamp_ms``."""
    return [AudioChunk(data=_TENTH, timestamp_ms=i) for i in range(n)]


def test_a_short_backlog_is_carried_whole() -> None:
    chunks = _tenths(20)
    new = _queue()
    assert _carry_over(_queue(*chunks, None), new) == 0.0
    assert _drained(new) == [*chunks, None]


def test_a_long_backlog_keeps_its_most_recent_seconds() -> None:
    chunks = _tenths(80)  # 8 s
    new = _queue()

    dropped = _carry_over(_queue(*chunks), new)

    kept = _drained(new)
    assert dropped == pytest.approx(8.0 - _CARRY_OVER_MAX_S)
    assert kept == chunks[-int(_CARRY_OVER_MAX_S * 10) :]


class _HangsThenRecords(STTProvider):
    """A first stream that hangs, as on a handshake the service never answers,
    then fails; the next stream records the audio it is given."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.received: list[bytes] = []
        self.calls = 0

    @property
    def supports_streaming(self) -> bool:
        return True

    async def transcribe(self, audio: Any, *, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="", is_final=True)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        self.calls += 1
        if self.calls == 1:
            await self.release.wait()
            raise ConnectionError("handshake timed out")
        while True:
            try:
                chunk = await asyncio.wait_for(anext(audio_stream), timeout=0.3)
            except (TimeoutError, StopAsyncIteration):
                return
            self.received.append(chunk.data)
        yield TranscriptionResult(text="", is_final=True)  # pragma: no cover


async def test_a_reconnect_after_a_hang_carries_the_last_seconds_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stt = _HangsThenRecords()
    backend = MockVoiceBackend()
    channel = VoiceChannel("voice-1", stt=stt, backend=backend, pipeline=AudioPipelineConfig())
    kit = RoomKit(stt=stt, voice=backend)
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice-1")
    session = await kit.connect_voice(room.id, "user-1", "voice-1")

    # The first chunk opens the stream that hangs; 8 s more queue up behind it.
    await backend.simulate_audio_received(session, AudioFrame(data=_TENTH))
    await asyncio.sleep(0.05)
    for _ in range(80):
        await backend.simulate_audio_received(session, AudioFrame(data=_TENTH))
    await asyncio.sleep(0.05)
    with caplog.at_level(logging.WARNING, logger="roomkit.voice"):
        stt.release.set()
        await asyncio.sleep(0.8)

    assert stt.calls == 2
    assert len(b"".join(stt.received)) == len(_TENTH) * int(_CARRY_OVER_MAX_S * 10)
    assert "dropped 3.0s of buffered audio, kept the last 5s" in caplog.text
    await channel.close()
