"""Mock voice backend for testing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.base import AudioReceivedCallback, SessionReadyCallback, VoiceBackend
from roomkit.voice.base import (
    AudioChunk,
    BargeInCallback,
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

    Tracks all method calls and provides helpers to simulate events.
    The backend is a pure transport — no VAD or audio intelligence.

    Example:
        backend = MockVoiceBackend()

        # Track calls
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert backend.calls[-1].method == "connect"

        # Simulate raw audio received
        frame = AudioFrame(data=b"audio-data")
        await backend.simulate_audio_received(session, frame)

        # Simulate barge-in
        await backend.simulate_barge_in(session)
    """

    def __init__(
        self,
        *,
        capabilities: VoiceCapability = VoiceCapability.NONE,
    ) -> None:
        self._sessions: dict[str, VoiceSession] = {}
        self._audio_received_callbacks: list[AudioReceivedCallback] = []
        self._barge_in_callbacks: list[BargeInCallback] = []
        self._session_ready_callbacks: list[SessionReadyCallback] = []
        # Tracking
        self.calls: list[MockVoiceCall] = []
        self.sent_audio: list[tuple[str, bytes]] = []  # (session_id, audio)
        self.sent_transcriptions: list[tuple[str, str, str]] = []  # (session_id, text, role)
        self._playing_sessions: set[str] = set()
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
        # Fire session ready — audio path is live immediately for mock
        for cb in self._session_ready_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                import asyncio

                asyncio.get_running_loop().create_task(result)
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
        self.calls.append(MockVoiceCall(method="disconnect", args={"session_id": session.id}))

    async def send_audio(
        self,
        session: VoiceSession,
        audio: bytes | AsyncIterator[AudioChunk],
    ) -> None:
        if isinstance(audio, bytes):
            self.sent_audio.append((session.id, audio))
        else:
            chunks: list[bytes] = []
            async for chunk in audio:
                chunks.append(chunk.data)
            combined = b"".join(chunks)
            self.sent_audio.append((session.id, combined))

        self.calls.append(MockVoiceCall(method="send_audio", args={"session_id": session.id}))

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
    # Raw audio delivery
    # -------------------------------------------------------------------------

    def on_audio_received(self, callback: AudioReceivedCallback) -> None:
        self._audio_received_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_audio_received"))

    # -------------------------------------------------------------------------
    # Barge-in support
    # -------------------------------------------------------------------------

    def on_session_ready(self, callback: SessionReadyCallback) -> None:
        self._session_ready_callbacks.append(callback)
        self.calls.append(MockVoiceCall(method="on_session_ready"))

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

    async def simulate_audio_received(self, session: VoiceSession, frame: AudioFrame) -> None:
        """Simulate the backend receiving a raw audio frame.

        Fires all registered on_audio_received callbacks.
        """
        for callback in self._audio_received_callbacks:
            result = callback(session, frame)
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

    async def simulate_session_ready(self, session: VoiceSession) -> None:
        """Simulate the backend signalling that a session's audio path is live.

        Fires all registered on_session_ready callbacks.
        """
        for cb in self._session_ready_callbacks:
            result = cb(session)
            if hasattr(result, "__await__"):
                await result
