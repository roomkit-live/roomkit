"""Acoustic Echo Cancellation provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame


class AECProvider(ABC):
    """Abstract base class for Acoustic Echo Cancellation providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'speex_aec', 'webrtc_aec')."""
        ...

    @abstractmethod
    def process(self, frame: AudioFrame) -> AudioFrame:
        """Remove echo from an audio frame.

        Args:
            frame: The captured audio frame (may contain echo).

        Returns:
            A new or modified AudioFrame with echo removed.
        """
        ...

    @abstractmethod
    def feed_reference(self, frame: AudioFrame) -> None:
        """Feed a reference (playback) frame for echo estimation.

        Called on the outbound path so the AEC can model the echo.

        Args:
            frame: The outbound audio frame being played to speakers.
        """
        ...

    def set_active(self, active: bool) -> None:  # noqa: B027
        """Enable or disable AEC processing (bypass mode).

        When *active* is ``False``, ``process()`` should pass audio
        through without echo cancellation.  Default is no-op (always
        active).
        """

    def reset(self) -> None:  # noqa: B027
        """Reset internal state."""

    def close(self) -> None:  # noqa: B027
        """Release resources."""
