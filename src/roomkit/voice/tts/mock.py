"""Mock text-to-speech provider for testing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

from roomkit.voice.base import AudioChunk
from roomkit.voice.tts.base import TTSProvider

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent


class MockTTSProvider(TTSProvider):
    """Mock text-to-speech for testing."""

    def __init__(self, voice: str = "mock-voice") -> None:
        self._default_voice = voice
        self.calls: list[dict[str, str | None]] = []

    @property
    def default_voice(self) -> str:
        return self._default_voice

    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        from roomkit.models.event import AudioContent as AudioContentModel

        self.calls.append({"text": text, "voice": voice or self._default_voice})
        return AudioContentModel(
            url=f"https://mock.test/audio/{uuid4().hex}.mp3",
            mime_type="audio/mpeg",
            transcript=text,
            duration_seconds=len(text) * 0.05,  # ~50ms per char
        )

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        self.calls.append({"text": text, "voice": voice or self._default_voice})
        # Simulate streaming with small chunks
        words = text.split()
        for i, word in enumerate(words):
            yield AudioChunk(
                data=f"mock-audio-{word}".encode(),
                sample_rate=16000,
                is_final=(i == len(words) - 1),
            )
