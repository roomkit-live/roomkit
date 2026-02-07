"""Text-to-speech provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.models.event import AudioContent
    from roomkit.voice.base import AudioChunk


class TTSProvider(ABC):
    """Text-to-speech provider."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'elevenlabs', 'openai')."""
        return self.__class__.__name__

    @property
    def default_voice(self) -> str | None:
        """Default voice ID. Override in subclasses."""
        return None

    @abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> AudioContent:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice ID (uses default_voice if not specified).

        Returns:
            AudioContent with URL to generated audio.
        """
        ...

    async def synthesize_stream(
        self, text: str, *, voice: str | None = None
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio chunks as they're generated.

        Override for providers that support streaming.
        Default: synthesizes full audio and yields single chunk.
        """
        raise NotImplementedError(
            f"{self.name} does not support streaming synthesis. Use synthesize() instead."
        )
        # Make this an async generator (unreachable, but required for type)
        yield  # pragma: no cover

    async def warmup(self) -> None:  # noqa: B027
        """Pre-load models so the first call is fast. Override in subclasses."""

    async def close(self) -> None:  # noqa: B027
        """Release resources. Override in subclasses if needed."""
