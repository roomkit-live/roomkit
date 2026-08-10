"""Base models for voice support."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Flag, StrEnum, auto, unique
from typing import Any

from roomkit.core.exceptions import VoiceSessionEndedError

logger = logging.getLogger("roomkit.voice")


@unique
class VoiceSessionState(StrEnum):
    """State of a voice session."""

    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class VoiceCapability(Flag):
    """Capabilities a VoiceBackend can support.

    Backends declare their capabilities via the `capabilities` property.
    This allows RoomKit to know which features are available and
    enables integrators to choose backends based on their needs.

    Example:
        class MyBackend(VoiceBackend):
            @property
            def capabilities(self) -> VoiceCapability:
                return (
                    VoiceCapability.INTERRUPTION |
                    VoiceCapability.BARGE_IN
                )
    """

    NONE = 0
    """No optional capabilities (default)."""

    INTERRUPTION = auto()
    """Backend can cancel ongoing audio playback (cancel_audio)."""

    BARGE_IN = auto()
    """Backend detects and handles barge-in (user interrupts TTS)."""

    NATIVE_AEC = auto()
    """Backend provides its own Acoustic Echo Cancellation."""

    NATIVE_AGC = auto()
    """Backend provides its own Automatic Gain Control."""

    DTMF_INBAND = auto()
    """Backend can detect DTMF tones from the audio stream."""

    DTMF_SIGNALING = auto()
    """Backend receives DTMF via out-of-band signaling (e.g. SIP INFO)."""

    NATIVE_BRIDGE = auto()
    """Backend can bridge audio at the transport level (RTP relay)."""


@dataclass
class AudioChunk:
    """A chunk of audio data for streaming (used for outbound TTS)."""

    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_s16le"
    timestamp_ms: int | None = None
    is_final: bool = False


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(UTC)


# RFC §12.1 — the state transitions a voice session may make. ENDED is absent
# as a source on purpose: it is terminal, and a participant who reconnects gets
# a new session.
_VOICE_STATE_TRANSITIONS: dict[VoiceSessionState, frozenset[VoiceSessionState]] = {
    VoiceSessionState.CONNECTING: frozenset({VoiceSessionState.ACTIVE, VoiceSessionState.ENDED}),
    VoiceSessionState.ACTIVE: frozenset({VoiceSessionState.PAUSED, VoiceSessionState.ENDED}),
    VoiceSessionState.PAUSED: frozenset({VoiceSessionState.ACTIVE, VoiceSessionState.ENDED}),
    VoiceSessionState.ENDED: frozenset(),
}


@dataclass
class VoiceSession:
    """Active voice connection for a participant.

    ``state`` is guarded (RFC §12.1). Leaving ENDED is refused outright — it is
    the one transition the RFC forbids, and letting a torn-down session go back
    to ACTIVE resurrects audio paths the framework has already released. Any
    other move outside the table is logged and allowed: the table does not
    model every provider's reality (a realtime provider renegotiating goes
    ACTIVE → CONNECTING), and turning an unmodelled transition into a crash
    would trade a documentation gap for an outage.
    """

    id: str
    room_id: str
    participant_id: str
    channel_id: str
    state: VoiceSessionState = VoiceSessionState.CONNECTING
    provider_session_id: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    _last_usage: dict[str, Any] = field(default_factory=dict)

    def renegotiate(self) -> None:
        """Return the session to CONNECTING for a provider renegotiation.

        A reconfigure — swapping an agent's personality, voice or tools during
        a handoff — tears the upstream connection down and builds a new one
        while the participant's session continues. Nobody hung up. The default
        provider implements that as disconnect + connect, which leaves the
        session ENDED in between, and reconnecting from there is the
        resurrection §12.1 forbids.

        This is the only sanctioned way out of ENDED, and it is narrow by
        design: it says "the framework itself just tore this down to rebuild
        it". It does not make ENDED non-terminal for anyone else — a
        participant who really hung up still gets a new session.
        """
        object.__setattr__(self, "state", VoiceSessionState.CONNECTING)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "state":
            current = self.__dict__.get("state")
            if current is not None and current != value:
                if current is VoiceSessionState.ENDED:
                    raise VoiceSessionEndedError(
                        f"Voice session {self.id} has ended; it cannot move to "
                        f"{value}. Create a new session for a reconnecting "
                        f"participant (RFC §12.1)."
                    )
                if value not in _VOICE_STATE_TRANSITIONS.get(current, frozenset()):
                    logger.warning(
                        "Voice session %s made an undocumented transition %s -> %s",
                        self.__dict__.get("id", "?"),
                        current,
                        value,
                    )
        object.__setattr__(self, name, value)


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""

    text: str
    is_final: bool = True
    confidence: float | None = None
    language: str | None = None
    words: list[dict[str, Any]] = field(default_factory=list)
    is_speech_start: bool = False
    """Set by providers with server-side VAD to signal speech detected."""


# Type aliases for voice callbacks
BargeInCallback = Callable[[VoiceSession], Any]
"""Callback for barge-in detection: (session)."""
