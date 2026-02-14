"""Speech-to-text provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from roomkit.voice.base import AudioChunk, TranscriptionResult

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.audio_frame import AudioFrame


class STTProvider(ABC):
    """Speech-to-text provider."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'whisper', 'deepgram')."""
        return self.__class__.__name__

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming transcription."""
        return False

    @abstractmethod
    async def transcribe(
        self, audio: AudioContent | AudioChunk | AudioFrame
    ) -> TranscriptionResult:
        """Transcribe complete audio to text.

        Args:
            audio: Audio content (URL), raw audio chunk, or audio frame.

        Returns:
            TranscriptionResult with text and metadata.
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
        result = await self.transcribe(combined)
        yield TranscriptionResult(text=result.text, is_final=True, confidence=result.confidence)

    async def warmup(self) -> None:  # noqa: B027
        """Pre-load models so the first call is fast. Override in subclasses."""

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses if needed."""
