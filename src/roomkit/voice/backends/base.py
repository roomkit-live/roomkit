"""VoiceBackend abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
    VoiceCapability,
    VoiceSession,
)

if TYPE_CHECKING:
    from roomkit.voice.audio_frame import AudioFrame

# Callback type for raw audio frames from the transport
AudioReceivedCallback = Callable[["VoiceSession", "AudioFrame"], Any]

# Callback type for audio frames as they are played through the speaker.
# Fired at playback time (time-aligned with actual speaker output) so AEC
# can use the reference to cancel echo accurately.
AudioPlayedCallback = Callable[["VoiceSession", "AudioFrame"], Any]


class VoiceBackend(ABC):
    """Abstract base class for voice transport backends.

    VoiceBackend handles the transport layer for real-time audio:
    - Managing voice session connections
    - Streaming audio to/from clients
    - Delivering raw inbound audio frames via on_audio_received

    The backend is framework-agnostic and a pure transport — all audio
    intelligence (VAD, denoising, diarization) is handled by the
    AudioPipeline.

    Example usage:
        backend = WebRTCVoiceBackend()

        # Register raw audio callback
        backend.on_audio_received(handle_audio_frame)

        # Connect a participant
        session = await backend.connect("room-1", "user-1", "voice-channel")

        # Stream audio to the client
        await backend.send_audio(session, audio_chunks)

        # Disconnect
        await backend.disconnect(session)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'webrtc', 'websocket', 'livekit')."""
        ...

    @abstractmethod
    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        """Create a new voice session for a participant.

        Args:
            room_id: The room to join.
            participant_id: The participant's ID.
            channel_id: The voice channel ID.
            metadata: Optional session metadata.

        Returns:
            A VoiceSession representing the connection.
        """
        ...

    @abstractmethod
    async def disconnect(self, session: VoiceSession) -> None:
        """End a voice session.

        Args:
            session: The session to disconnect.
        """
        ...

    @abstractmethod
    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        """Send audio to a voice session.

        Args:
            session: The target session.
            audio: Raw audio bytes or an async iterator of AudioChunks
                for streaming.
        """
        ...

    def get_session(self, session_id: str) -> VoiceSession | None:
        """Get a session by ID.

        Override for backends that track sessions internally.

        Args:
            session_id: The session ID to look up.

        Returns:
            The VoiceSession if found, None otherwise.
        """
        return None

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        """List all active sessions in a room.

        Override for backends that track sessions internally.

        Args:
            room_id: The room to list sessions for.

        Returns:
            List of active VoiceSessions in the room.
        """
        return []

    async def close(self) -> None:
        """Release backend resources.

        Override in subclasses that need cleanup.
        """

    # -------------------------------------------------------------------------
    # Capabilities
    # -------------------------------------------------------------------------

    @property
    def capabilities(self) -> VoiceCapability:
        """Declare supported capabilities.

        Override to enable features like interruption, barge-in, etc.
        By default, no optional capabilities are supported.

        Returns:
            Flags indicating supported capabilities.
        """
        return VoiceCapability.NONE

    @property
    def feeds_aec_reference(self) -> bool:
        """Whether this backend feeds AEC reference at the transport level.

        When True, the pipeline skips ``aec.feed_reference()`` in the
        outbound path to avoid double-feeding.  Transport-level feeding
        (from the speaker callback) is preferred because it is
        time-aligned with actual speaker output.
        """
        return False

    # -------------------------------------------------------------------------
    # Raw audio delivery (pipeline integration)
    # -------------------------------------------------------------------------

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        """Register a callback for raw inbound audio frames.

        The pipeline or channel calls this to receive every audio frame
        produced by the transport.

        Args:
            callback: Function called with (session, audio_frame).
        """
        pass  # noqa: B027

    # -------------------------------------------------------------------------
    # Barge-in support
    # -------------------------------------------------------------------------

    def on_barge_in(self, callback: BargeInCallback) -> None:  # noqa: B027
        """Register callback for barge-in detection.

        Only called if capabilities includes BARGE_IN.
        Backends should call this when user starts speaking while
        audio is being played (TTS interruption).

        Args:
            callback: Function called with (session).
        """

    async def cancel_audio(self, session: VoiceSession) -> bool:
        """Cancel ongoing audio playback for a session.

        Only works if capabilities includes INTERRUPTION.
        Used for barge-in handling to stop TTS playback.

        Args:
            session: The session to cancel audio for.

        Returns:
            True if audio was cancelled, False if nothing was playing.
        """
        return False  # Default no-op, override if supported

    def end_of_response(self, session: VoiceSession) -> None:  # noqa: B027
        """Signal end of an AI response for outbound pacing."""

    def is_playing(self, session: VoiceSession) -> bool:
        """Check if audio is currently being sent to the session.

        Used for barge-in detection to know if interruption is possible.

        Args:
            session: The session to check.

        Returns:
            True if audio is currently playing, False otherwise.
        """
        return False  # Default: assume not playing

    # -------------------------------------------------------------------------
    # Speaker output notifications (for pipeline AEC)
    # -------------------------------------------------------------------------

    @property
    def supports_playback_callback(self) -> bool:
        """Whether this backend fires :meth:`on_audio_played` callbacks.

        When True, the pipeline can rely on playback-time AEC reference
        instead of generation-time feeding from ``process_outbound``.
        """
        return False

    def on_audio_played(self, callback: AudioPlayedCallback) -> None:  # noqa: B027
        """Register a callback for audio frames as they are played.

        Called with each audio frame at the moment it is output by the
        speaker, providing time-aligned reference for echo cancellation.
        The pipeline uses this to feed AEC reference at the correct time.

        Note:
            Callbacks may be invoked from the audio I/O thread —
            implementations must be thread-safe.

        Args:
            callback: Function called with (session, audio_frame).
        """

    # -------------------------------------------------------------------------
    # Protocol trace
    # -------------------------------------------------------------------------

    def set_trace_emitter(  # noqa: B027
        self,
        emitter: Callable[..., Any] | None,
    ) -> None:
        """Set a callback for emitting protocol traces.

        Called by the owning channel when trace observers are registered.
        Implementations should store the emitter and call it at key
        protocol points (e.g. INVITE, BYE for SIP).

        Args:
            emitter: The channel's :meth:`emit_trace` method, or ``None``
                to disable.
        """

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        """Send transcription text to the client for UI display.

        Optional method for backends that support sending text updates.

        Args:
            session: The voice session to send to.
            text: The transcribed or response text.
            role: Either "user" (transcription) or "assistant" (AI response).
        """
        pass  # noqa: B027
