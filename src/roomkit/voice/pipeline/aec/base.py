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
    def process(self, frame: AudioFrame, stream: str) -> AudioFrame:
        """Remove echo from an audio frame.

        Args:
            frame: The captured audio frame (may contain echo).
            stream: Identity of the audio stream this frame belongs to. A
                provider keeps its state per stream: a voice session and a
                conference track are separate speakers, and letting one advance
                the other's detection state makes silence from one close the
                other's utterance.

        Returns:
            A new or modified AudioFrame with echo removed.
        """
        ...

    @abstractmethod
    def feed_reference(self, frame: AudioFrame, stream: str) -> None:
        """Feed a reference (playback) frame for echo estimation.

        Called on the outbound path so the AEC can model the echo.

        Args:
            frame: The outbound audio frame being played to speakers.
            stream: Identity of the audio stream this reference belongs to —
                the same key ``process()`` uses. Each stream owns its echo
                canceller, so an unkeyed reference could not reach the right
                one: in a conference every lane hears a different mix, and
                feeding one lane's output into another's canceller models an
                echo that never happened.
        """
        ...

    def set_active(self, active: bool) -> None:  # noqa: B027
        """Enable or disable AEC processing (bypass mode).

        When *active* is ``False``, ``process()`` should pass audio
        through without echo cancellation.  Default is no-op (always
        active).
        """

    def set_stream_active(self, stream: str, active: bool) -> None:
        """Enable or disable AEC processing for one stream.

        Providers with stream-local bypass state should override this method.
        The default preserves compatibility with providers whose activation is
        global by delegating to :meth:`set_active`.

        Args:
            stream: Identity passed to :meth:`process` and
                :meth:`feed_reference` for this playback stream.
            active: Whether echo cancellation should run for the stream.
        """
        self.set_active(active)

    def reset(self, stream: str) -> None:  # noqa: B027
        """Drop a stream's state.

        Called when the stream ends, so a long-running room does not accumulate
        the state of every speaker that ever joined.
        """

    def close(self) -> None:  # noqa: B027
        """Release resources."""
