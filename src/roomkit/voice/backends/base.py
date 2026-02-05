"""VoiceBackend abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
    PartialTranscriptionCallback,
    SpeechEndCallback,
    SpeechStartCallback,
    VADAudioLevelCallback,
    VADSilenceCallback,
    VoiceCapability,
    VoiceSession,
)


class VoiceBackend(ABC):
    """Abstract base class for voice transport backends.

    VoiceBackend handles the transport layer for real-time audio:
    - Managing voice session connections
    - Voice Activity Detection (VAD) callbacks
    - Streaming audio to/from clients

    The backend is framework-agnostic. Web framework integration (FastAPI routes,
    WebSocket endpoints) is the responsibility of the application layer.

    Example usage with a hypothetical WebRTC backend:
        backend = WebRTCVoiceBackend()

        # Register VAD callbacks
        backend.on_speech_start(handle_speech_start)
        backend.on_speech_end(handle_speech_end)

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
    def on_speech_start(self, callback: SpeechStartCallback) -> None:
        """Register a callback for when VAD detects speech starting.

        The callback receives the VoiceSession where speech was detected.

        Args:
            callback: Function called when speech starts.
        """
        ...

    @abstractmethod
    def on_speech_end(self, callback: SpeechEndCallback) -> None:
        """Register a callback for when VAD detects speech ending.

        The callback receives the VoiceSession and the audio bytes
        captured during the speech segment.

        Args:
            callback: Function called when speech ends with audio data.
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
    # Enhanced voice capabilities (RFC §19)
    # -------------------------------------------------------------------------

    @property
    def capabilities(self) -> VoiceCapability:
        """Declare supported capabilities.

        Override to enable features like interruption, partial STT, etc.
        By default, no optional capabilities are supported.

        Returns:
            Flags indicating supported capabilities.
        """
        return VoiceCapability.NONE

    def on_partial_transcription(
        self, callback: PartialTranscriptionCallback
    ) -> None:
        """Register callback for partial transcription results.

        Only called if capabilities includes PARTIAL_STT.
        Backends that support streaming STT should call this callback
        with interim results as they become available.

        Args:
            callback: Function called with (session, text, confidence, is_stable).
        """
        pass  # Default no-op, override if supported

    def on_vad_silence(self, callback: VADSilenceCallback) -> None:
        """Register callback for silence detection.

        Only called if capabilities includes VAD_SILENCE.
        Backends should call this when silence is detected after speech,
        potentially before the full speech_end event.

        Args:
            callback: Function called with (session, silence_duration_ms).
        """
        pass  # Default no-op, override if supported

    def on_vad_audio_level(self, callback: VADAudioLevelCallback) -> None:
        """Register callback for audio level updates.

        Only called if capabilities includes VAD_AUDIO_LEVEL.
        Backends should call this periodically (e.g., 10Hz) with
        current audio level for UI feedback.

        Args:
            callback: Function called with (session, level_db, is_speech).
        """
        pass  # Default no-op, override if supported

    def on_barge_in(self, callback: BargeInCallback) -> None:
        """Register callback for barge-in detection.

        Only called if capabilities includes BARGE_IN.
        Backends should call this when user starts speaking while
        audio is being played (TTS interruption).

        Args:
            callback: Function called with (session).
        """
        pass  # Default no-op, override if supported

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

    def is_playing(self, session: VoiceSession) -> bool:
        """Check if audio is currently being sent to the session.

        Used for barge-in detection to know if interruption is possible.

        Args:
            session: The session to check.

        Returns:
            True if audio is currently playing, False otherwise.
        """
        return False  # Default: assume not playing

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        """Send transcription text to the client for UI display.

        Optional method for backends that support sending text updates.
        Called by VoiceChannel after STT transcription to show the user
        what they said, and after AI response to show what the assistant said.

        Args:
            session: The voice session to send to.
            text: The transcribed or response text.
            role: Either "user" (transcription) or "assistant" (AI response).
        """
        pass  # Default no-op, override if supported
