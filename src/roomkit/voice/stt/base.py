"""Speech-to-text provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.base import AudioChunk, TranscriptionResult


class STTProvider(ABC):
    """Speech-to-text provider."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'whisper', 'deepgram')."""
        return self.__class__.__name__

    @abstractmethod
    async def transcribe(self, audio: AudioContent | AudioChunk) -> str:
        """Transcribe complete audio to text.

        Args:
            audio: Audio content (URL) or raw audio chunk.

        Returns:
            Transcribed text.
        """
        ...

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription with partial results.

        Override for providers that support streaming.
        Default: buffers all audio and returns single result.
        """
        chunks: list[AudioChunk] = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        # Combine chunks and transcribe
        combined = AudioChunk(
            data=b"".join(c.data for c in chunks),
            sample_rate=chunks[0].sample_rate if chunks else 16000,
        )
        text = await self.transcribe(combined)

        from roomkit.voice.base import TranscriptionResult

        yield TranscriptionResult(text=text, is_final=True)

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses if needed."""
