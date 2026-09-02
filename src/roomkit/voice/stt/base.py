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

    @property
    def supports_language_override(self) -> bool:
        """Whether ``transcribe`` and ``transcribe_stream`` honour ``language``.

        A provider that answers ``False`` is never handed a per-call
        language: :class:`~roomkit.channels.VoiceChannel` keeps calling it
        with audio only, so an implementation written against the older
        signature keeps working unchanged.
        """
        return False

    @abstractmethod
    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe complete audio to text.

        Args:
            audio: Audio content (URL), raw audio chunk, or audio frame.
            language: Language for this call only, overriding the provider's
                configuration. Honoured when :attr:`supports_language_override`
                is true; a caller must not pass it otherwise.

        Returns:
            TranscriptionResult with text and metadata. ``language`` on the
            result is what the provider reports, never an echo of the request.
        """
        ...

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream transcription with partial results.

        Override for providers that support streaming.
        Default: buffers all audio and returns single result.

        Args:
            audio_stream: Audio chunks; the stream ends when it is exhausted.
            language: Language for this stream only — see :meth:`transcribe`.
                A language is fixed for the life of a stream: a change takes
                effect on the next stream the caller opens.
        """
        chunks: list[AudioChunk] = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        # Combine chunks and transcribe
        combined = AudioChunk(
            data=b"".join(c.data for c in chunks),
            sample_rate=chunks[0].sample_rate if chunks else 16000,
        )
        # Only a provider that declares the override is ever given one; a
        # subclass on the older ``transcribe`` signature is still called as
        # before.
        if language is None:
            result = await self.transcribe(combined)
        else:
            result = await self.transcribe(combined, language=language)
        yield TranscriptionResult(
            text=result.text,
            is_final=True,
            confidence=result.confidence,
            language=result.language,
        )

    async def warmup(self) -> None:  # noqa: B027
        """Pre-load models so the first call is fast. Override in subclasses."""

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses if needed."""
