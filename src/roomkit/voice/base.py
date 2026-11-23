"""Base models for voice support."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Collection
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


PCM16_FORMATS = frozenset({"pcm", "pcm_s16le"})
"""The ``AudioChunk.format`` values that carry 16-bit signed PCM."""


def require_pcm16(chunk: AudioChunk, consumer: str) -> None:
    """Refuse a chunk *consumer* cannot play: encoded audio, or PCM of another width.

    An encoded chunk is refused because encoding belongs to the backend (RFC
    sections 12.2 and 12.10.3): a caller choosing the wire format would defeat
    the boundary this interface exists to draw. Another PCM width is refused
    because reading it as 16-bit signed would not fail, it would play noise, and
    noise that reaches a call is worse than a chunk that was refused.

    Raises:
        ValueError: *chunk* is not 16-bit signed PCM.
    """
    if not chunk.format.startswith("pcm"):
        raise ValueError(
            f"{consumer} expects decoded PCM, got format {chunk.format!r}. Encoding "
            "belongs to the backend: configure the audio source for PCM output."
        )
    if chunk.format not in PCM16_FORMATS:
        raise ValueError(
            f"{consumer} plays 16-bit signed PCM, and this chunk is {chunk.format!r}. "
            f"Reinterpreting it would play noise rather than fail. Accepted: "
            f"{sorted(PCM16_FORMATS)}."
        )


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

    @property
    def last_usage(self) -> dict[str, Any]:
        """What the provider recorded last for this session (RFC §12.4.2).

        A snapshot, empty until the first report. ``input_tokens`` and
        ``output_tokens`` are the two totals a token-billed provider reports;
        beside them sits whatever breakdown its API sends — per modality, the
        cached share, reasoning or tool use — under that API's own names, so an
        absent key means unreported, not zero. A provider billed by session
        duration reports its seconds here instead, and a hosted backend's
        tokens ride under their own key.

        The next report replaces it and the realtime channel clears it at the
        end of each turn, so a reader that polls can miss a turn. To bill a
        call, register :meth:`RealtimeVoiceProvider.on_usage` instead, which
        fires on every report. Mutating the returned dict changes nothing.
        """
        return dict(self._last_usage)

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


# The word a label may carry before the part that tells voices apart
# ("Speaker 1", "speaker_0", "speaker1"): a whole word, so "Speakerphone"
# stays as it is.
_SPEAKER_PREFIX = re.compile(r"^speaker(?:[\s_:#-]+|(?=\d))", re.IGNORECASE)


def speaker_label(value: Any, *, unattributed: Collection[str] = ()) -> str | None:
    """A provider's speaker label as RoomKit carries it (RFC §12.2.3).

    A string (``0`` becomes ``"0"``); ``None`` for no label, an empty one, or
    one the provider's wire uses for a speaker it could not attribute. Those
    spellings are the provider's own: it passes them in ``unattributed``,
    compared in any case, and RoomKit knows no vendor's. A leading "speaker"
    word is dropped with the separator after it, so ``"Speaker 1"``,
    ``"speaker_1"`` and ``"speaker1"`` all read ``"1"`` and a channel's
    ``"Speaker 1"`` name is not ``"Speaker Speaker 1"``; a longer word that
    starts with it (``"Speakerphone"``) is kept whole. The mapping is fixed,
    so a label stays as stable as the provider's.
    """
    if value is None:
        return None
    label = str(value).strip()
    if not label or label.casefold() in {spelling.casefold() for spelling in unattributed}:
        return None
    return _SPEAKER_PREFIX.sub("", label) or label


@dataclass(frozen=True)
class SpeakerSegment:
    """One speaker's part of a transcription (RFC §12.2.3).

    Attributes:
        speaker: The provider's label for the voice, as a string. Opaque and
            stable within one STT stream only: another stream may give the
            same voice another label. ``None`` when the provider could not
            attribute the words.
        text: The words that speaker said.
        start_ms: Where the segment starts, from the start of the stream, when
            the provider says.
        end_ms: Where it ends, likewise.
    """

    speaker: str | None
    text: str
    start_ms: int | None = None
    end_ms: int | None = None


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
    segments: list[SpeakerSegment] = field(default_factory=list)
    """Who said which part of ``text``, in order (RFC §12.2.3). Empty when the
    provider attributes nothing; always given on a final by a provider that
    reports ``supports_diarization``."""

    @property
    def speaker(self) -> str | None:
        """The label every segment shares, ``None`` when there is not exactly one."""
        labels = {segment.speaker for segment in self.segments}
        return labels.pop() if len(labels) == 1 else None


# Type aliases for voice callbacks
BargeInCallback = Callable[[VoiceSession], Any]
"""Callback for barge-in detection: (session)."""
