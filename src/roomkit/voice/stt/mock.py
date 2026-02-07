"""Mock speech-to-text provider for testing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from roomkit.voice.base import AudioChunk, TranscriptionResult
from roomkit.voice.stt.base import STTProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame


class MockSTTProvider(STTProvider):
    """Mock speech-to-text for testing."""

    def __init__(self, transcripts: list[str] | None = None) -> None:
        self.transcripts = transcripts or ["Hello", "How can I help you?"]
        self.calls: list[AudioContent | AudioChunk | AudioFrame] = []
        self._index = 0

    async def transcribe(
        self, audio: AudioContent | AudioChunk | AudioFrame
    ) -> TranscriptionResult:
        self.calls.append(audio)
        text = self.transcripts[self._index % len(self.transcripts)]
        self._index += 1
        return TranscriptionResult(text=text)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        text = self.transcripts[self._index % len(self.transcripts)]
        self._index += 1
        self.calls.extend(chunks)

        yield TranscriptionResult(text=text, is_final=True)
