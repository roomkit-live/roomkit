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
    """Mock speech-to-text for testing.

    Args:
        transcripts: Texts returned in turn, cycling.
        streaming: Whether to advertise streaming support.
        languages: The language each result reports, in step with
            ``transcripts`` and cycling the same way — what a detecting
            provider would have heard. ``None`` entries report nothing.
    """

    def __init__(
        self,
        transcripts: list[str] | None = None,
        *,
        streaming: bool = False,
        languages: list[str | None] | None = None,
    ) -> None:
        self.transcripts = transcripts or ["Hello", "How can I help you?"]
        self.calls: list[AudioContent | AudioChunk | AudioFrame] = []
        self.languages_requested: list[str | None] = []
        """The ``language`` handed to each call, in order (``None`` = the default)."""
        self._index = 0
        self._streaming = streaming
        self._languages = languages

    @property
    def supports_streaming(self) -> bool:
        return self._streaming

    @property
    def supports_language_override(self) -> bool:
        return True

    def _next(self) -> tuple[str, str | None]:
        i = self._index
        self._index += 1
        text = self.transcripts[i % len(self.transcripts)]
        language = self._languages[i % len(self._languages)] if self._languages else None
        return text, language

    async def transcribe(
        self,
        audio: AudioContent | AudioChunk | AudioFrame,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        self.calls.append(audio)
        self.languages_requested.append(language)
        text, reported = self._next()
        return TranscriptionResult(text=text, language=reported)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionResult]:
        self.languages_requested.append(language)
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        text, reported = self._next()
        self.calls.extend(chunks)

        yield TranscriptionResult(text=text, is_final=True, language=reported)
