"""Mock voice backend for testing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from roomkit.voice.backends.base import VoiceBackend
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
    VoiceSessionState,
)


@dataclass
class MockVoiceCall:
    """Record of a call made to MockVoiceBackend."""

    method: str
    args: dict[str, Any] = field(default_factory=dict)


class MockVoiceBackend(VoiceBackend):
    """Mock voice backend for testing.

    Tracks all method calls and provides helpers to simulate VAD events.

    Example:
        backend = MockVoiceBackend()

        # Track calls
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert backend.calls[-1].method == "connect"

        # Simulate speech events
        await backend.simulate_speech_start(session)
        await backend.simulate_speech_end(session, b"audio-data")

        # Simulate enhanced events (RFC §19)
        await backend.simulate_partial_transcription(session, "Hello", 0.8, False)
        await backend.simulate_vad_silence(session, 500)
        await backend.simulate_barge_in(session)
    """

    def __init__(
        self,
        *,
        capabilities: VoiceCapability = VoiceCapability.NONE,
    ) -> None:
        """Initialize MockVoiceBackend.

        Args:
            capabilities: Optional capabilities to enable for testing.
                Defaults to NONE. Set to test capability-dependent behavior.
        """
        self._sessions: dict[str, VoiceSession] = {}
        self._speech_start_callbacks: list[SpeechStartCallback] = []
        self._speech_end_callbacks: list[SpeechEndCallback] = []
        # Enhanced callbacks (RFC §19)
        self._partial_transcription_callbacks: list[PartialTranscriptionCallback] = []
        self._vad_silence_callbacks: list[VADSilenceCallback] = []
        self._vad_audio_level_callbacks: list[VADAudioLevelCallback] = []
        self._barge_in_callbacks: list[BargeInCallback] = []
        # Tracking
        self.calls: list[MockVoiceCall] = []
        self.sent_audio: list[tuple[str, bytes]] = []  # (session_id, audio)
        self.sent_transcriptions: list[tuple[str, str, str]] = []  # (session_id, text, role)
        self._playing_sessions: set[str] = set()  # Sessions currently receiving audio
        self._capabilities = capabilities

    @property
    def name(self) -> str:
        return "MockVoiceBackend"

    @property
    def capabilities(self) -> VoiceCapability:
        return self._capabilities

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        session = VoiceSession(
            id=uuid4().hex,
            room_id=room_id,
            participant_id=participant_id,
            channel_id=channel_id,
            state=VoiceSessionState.ACTIVE,
            metadata=metadata or {},
        )
        self._sessions[session.id] = session
        self.calls.append(
            MockVoiceCall(
                method="connect",
                args={
                    "room_id": room_id,
                    "participant_id": participant_id,
                    "channel_id": channel_id,
                    "metadata": metadata,
                },
            )
        )
        return session

    async def disconnect(self, session: VoiceSession) -> None:
        if session.id in self._sessions:
            self._sessions[session.id] = VoiceSession(
                id=session.id,
                room_id=session.room_id,
                participant_id=session.participant_id,
                channel_id=session.channel_id,
                state=VoiceSessionState.ENDED,
                created_at=session.created_at,
                metadata=session.metadata,
            )
        self.calls.append(
            MockVoiceCall(method="disconnect", args={"session_id": session.id})
        )

    def on_speech_start(self, callback: SpeechStartCallback) -> None:
        self._speech_start_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_speech_start"))

    def on_speech_end(self, callback: SpeechEndCallback) -> None:
        self._speech_end_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_speech_end"))

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        if isinstance(audio, bytes):
            self.sent_audio.append((session.id, audio))
        else:
            # Collect chunks from async iterator
            chunks: list[bytes] = []
            async for chunk in audio:
                chunks.append(chunk.data)
            combined = b"".join(chunks)
            self.sent_audio.append((session.id, combined))

        self.calls.append(
            MockVoiceCall(method="send_audio", args={"session_id": session.id})
        )

    def get_session(self, session_id: str) -> VoiceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, room_id: str) -> list[VoiceSession]:
        return [s for s in self._sessions.values() if s.room_id == room_id]

    async def close(self) -> None:
        self._sessions.clear()
        self._playing_sessions.clear()
        self.calls.append(MockVoiceCall(method="close"))

    async def send_transcription(
        self, session: VoiceSession, text: str, role: str = "user"
    ) -> None:
        self.sent_transcriptions.append((session.id, text, role))
        self.calls.append(
            MockVoiceCall(
                method="send_transcription",
                args={"session_id": session.id, "text": text, "role": role},
            )
        )

    # -------------------------------------------------------------------------
    # Enhanced voice capabilities (RFC §19)
    # -------------------------------------------------------------------------

    def on_partial_transcription(
        self, callback: PartialTranscriptionCallback
    ) -> None:
        self._partial_transcription_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_partial_transcription"))

    def on_vad_silence(self, callback: VADSilenceCallback) -> None:
        self._vad_silence_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_vad_silence"))

    def on_vad_audio_level(self, callback: VADAudioLevelCallback) -> None:
        self._vad_audio_level_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_vad_audio_level"))

    def on_barge_in(self, callback: BargeInCallback) -> None:
        self._barge_in_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_barge_in"))

    async def cancel_audio(self, session: VoiceSession) -> bool:
        was_playing = session.id in self._playing_sessions
        self._playing_sessions.discard(session.id)
        self.calls.append(
            MockVoiceCall(
                method="cancel_audio",
                args={"session_id": session.id, "was_playing": was_playing},
            )
        )
        return was_playing

    def is_playing(self, session: VoiceSession) -> bool:
        return session.id in self._playing_sessions

    # -------------------------------------------------------------------------
    # Test helpers
    # -------------------------------------------------------------------------

    async def simulate_speech_start(self, session: VoiceSession) -> None:
        """Simulate VAD detecting speech start.

        Fires all registered on_speech_start callbacks.
        """
        for callback in self._speech_start_callbacks:
            result = callback(session)
            if hasattr(result, "__await__"):
                await result

    async def simulate_speech_end(self, session: VoiceSession, audio: bytes) -> None:
        """Simulate VAD detecting speech end.

        Fires all registered on_speech_end callbacks with the audio data.
        """
        for callback in self._speech_end_callbacks:
            result = callback(session, audio)
            if hasattr(result, "__await__"):
                await result

    async def simulate_partial_transcription(
        self,
        session: VoiceSession,
        text: str,
        confidence: float = 0.8,
        is_stable: bool = False,
    ) -> None:
        """Simulate streaming STT partial result.

        Fires all registered on_partial_transcription callbacks.
        """
        for callback in self._partial_transcription_callbacks:
            result = callback(session, text, confidence, is_stable)
            if hasattr(result, "__await__"):
                await result

    async def simulate_vad_silence(
        self, session: VoiceSession, silence_duration_ms: int
    ) -> None:
        """Simulate VAD detecting silence.

        Fires all registered on_vad_silence callbacks.
        """
        for callback in self._vad_silence_callbacks:
            result = callback(session, silence_duration_ms)
            if hasattr(result, "__await__"):
                await result

    async def simulate_vad_audio_level(
        self,
        session: VoiceSession,
        level_db: float,
        is_speech: bool = True,
    ) -> None:
        """Simulate audio level update.

        Fires all registered on_vad_audio_level callbacks.
        """
        for callback in self._vad_audio_level_callbacks:
            result = callback(session, level_db, is_speech)
            if hasattr(result, "__await__"):
                await result

    async def simulate_barge_in(self, session: VoiceSession) -> None:
        """Simulate user speaking while TTS is playing (barge-in).

        Fires all registered on_barge_in callbacks.
        """
        for callback in self._barge_in_callbacks:
            result = callback(session)
            if hasattr(result, "__await__"):
                await result

    def start_playing(self, session: VoiceSession) -> None:
        """Mark a session as playing audio (for barge-in testing)."""
        self._playing_sessions.add(session.id)

    def stop_playing(self, session: VoiceSession) -> None:
        """Mark a session as no longer playing audio."""
        self._playing_sessions.discard(session.id)
