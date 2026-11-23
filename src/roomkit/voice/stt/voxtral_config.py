"""Configuration of Mistral's Voxtral speech-to-text provider.

The settings :class:`roomkit.voice.stt.voxtral.VoxtralSTTProvider` takes and
the checks that refuse, before any connection, what the service would refuse:
the streaming delay, the turn thresholds, diarization alongside turn
detection, speaker references. The public import path is
``roomkit.voice.stt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from roomkit.voice.base import speaker_label
from roomkit.voice.stt.voxtral_protocol import (
    REALTIME_PATH,
    STREAMING_DELAYS_MS,
    UNATTRIBUTED_LABELS,
    language_code,
    valid_streaming_delay,
)

DEFAULT_MODEL = "voxtral-mini-transcribe-3"
DEFAULT_REALTIME_MODEL = "voxtral-mini-transcribe-realtime-3"

_LATE_TEXT_MARGIN_MS = 100
"""Added to the streaming delay to give the final wait. A turn's last words
followed its ``turn.end`` by up to 0.19 s at a 480 ms delay, 0.43 s at 480 ms
with Mistral's aggressive preset, 0.16 s at 240 ms and 0.75 s at 1200 ms
(measured 2026-10-21): never more than the delay."""

MAX_SPEAKER_REFERENCES = 8
"""Reference clips the service takes at most, 3 to 15 seconds each."""

_NEXT_TURN_SHARE = 0.95
"""The share of the streaming delay a turn waits after its ``turn.end`` once the
next turn has started. Its last words came up to 0.9 of the delay after the
end (0.43 s at 480 ms, measured 2026-10-21); the next turn's first word comes a
delay after it is spoken, so no sooner than a delay after the end: 0.95 of the
delay parts the two, where the plain final wait (the delay plus 100 ms) joined
the next turn's first word to the last one (measured live)."""


@dataclass(frozen=True)
class VoxtralTurnDetection:
    """When the realtime model says a turn starts and ends.

    The model scores, from 0 to 1, how likely the person is still speaking;
    the thresholds turn that score into turn events. The defaults are the
    service's own (an empty ``turn_detection`` object), and
    :meth:`aggressive` is Mistral's faster preset.

    Attributes:
        start_threshold: Score above which a turn starts or resumes. Raise it
            when noise starts turns.
        eager_end_threshold: Score below which the person has probably
            finished (an eager end, which RoomKit does not act on).
        end_threshold: Score below which the turn ends. Lower it to cut
            people off less often, at the cost of a later answer.
        min_silence_ms: Shortest pause before a turn can end.
        end_timeout_ms: Longest pause before a turn ends anyway.
    """

    start_threshold: float = 0.5
    eager_end_threshold: float = 0.5
    end_threshold: float = 0.33
    min_silence_ms: int = 500
    end_timeout_ms: int = 3000

    def __post_init__(self) -> None:
        if not 0.0 <= self.end_threshold < self.eager_end_threshold <= self.start_threshold <= 1:
            raise ValueError(
                "turn detection thresholds must keep their order, "
                "1 >= start_threshold >= eager_end_threshold > end_threshold >= 0"
            )
        if self.min_silence_ms < 0 or self.end_timeout_ms < self.min_silence_ms:
            raise ValueError("end_timeout_ms must be at least min_silence_ms, itself at least 0")

    @classmethod
    def aggressive(cls) -> VoxtralTurnDetection:
        """Mistral's preset for fast exchanges: turns end sooner, and cut more often."""
        return cls(eager_end_threshold=0.4, min_silence_ms=300, end_timeout_ms=2500)


@dataclass
class VoxtralSTTConfig:
    """Configuration for :class:`VoxtralSTTProvider`.

    Attributes:
        api_key: Mistral API key (sent as ``Bearer``).
        model: File transcription model: ``voxtral-mini-transcribe-3``, or
            ``voxtral-small-transcribe-3`` for the most accurate.
        realtime_model: Realtime transcription model.
        language: Language to transcribe, as a BCP-47 tag or an ISO 639-1
            code (``"fr-CA"`` is sent as ``"fr"``). ``None`` lets the model
            detect it.
        context_bias: Names and terms the model should spell right (a
            product, a person, a place), phrases included.
        streaming_delay_ms: How long the realtime model listens past a word
            before writing it: shorter is sooner, longer is more accurate.
            Any value from 80 to 1200, or 2400.
        turn_detection: The realtime model's turn detection, for a channel
            without a pipeline VAD; ``None`` behind one, where a turn ended
            inside an utterance would otherwise be lost (see the module
            docstring).
        final_wait_ms: How long a turn's final waits after the model ends the
            turn, for its last words, which the text brings up to a streaming
            delay later. ``None`` is the delay plus 100 ms (0.58 s with the
            defaults). Words later than it are dropped, with a warning.
        diarize: Label who spoke: every final names its speaker, in
            ``segments``. Live it excludes ``turn_detection``, which must then
            be ``None``.
        speaker_references: A short recording of each known voice (WAV, MP3,
            FLAC or Opus bytes, 3 to 15 s of mostly speech) by the name its
            segments then carry; at most 8. Live only: the file endpoint took
            them and kept its own labels (measured 2026-10-21). Requires
            ``diarize``.
        diarized_pause_ms: How long without new words ends a diarized turn
            live (no turn events then). One person's words came up to 2 s
            apart, a new speaker's 1.5 s after the last (measured 2026-10-21).
        base_url: API root, without a version.
        realtime_url: Realtime WebSocket endpoint; ``None`` derives it from
            ``base_url``.
        timeout: Read budget of a file transcription, in seconds.
        connect_timeout: TCP connect budget, in seconds, apart from ``timeout``.
        handshake_timeout_s: How long a stream waits, in all, for the
            connection and for the session to accept its settings.
        final_timeout_s: How long a stream waits, once its audio has ended,
            for the end of the transcript.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    realtime_model: str = DEFAULT_REALTIME_MODEL
    language: str | None = None
    context_bias: list[str] = field(default_factory=list)
    streaming_delay_ms: int = 480
    turn_detection: VoxtralTurnDetection | None = field(default_factory=VoxtralTurnDetection)
    final_wait_ms: int | None = None
    diarize: bool = False
    speaker_references: dict[str, bytes] = field(default_factory=dict, repr=False)
    diarized_pause_ms: int = 1500
    base_url: str = "https://api.mistral.ai"
    realtime_url: str | None = None
    timeout: float = 60.0
    connect_timeout: float = 5.0
    handshake_timeout_s: float = 10.0
    final_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("VoxtralSTTConfig.api_key is required")
        if not self.api_key.isascii():
            # Refused here, not as a UnicodeEncodeError from the HTTP client.
            raise ValueError(
                "VoxtralSTTConfig.api_key holds a non-ASCII character (a placeholder such as "
                "'…' copied as is?): an HTTP Authorization header cannot carry it"
            )
        language_code(self.language)
        if not valid_streaming_delay(self.streaming_delay_ms):
            raise ValueError(
                f"streaming_delay_ms must be {STREAMING_DELAYS_MS}, got {self.streaming_delay_ms}"
            )
        if self.final_wait_ms is not None and self.final_wait_ms < 0:
            raise ValueError("final_wait_ms must not be negative")
        if self.handshake_timeout_s <= 0 or self.final_timeout_s <= 0:
            raise ValueError("VoxtralSTTConfig timeouts must be positive")
        self._check_diarization()

    def _check_diarization(self) -> None:
        """The diarization settings the service would refuse, refused here first."""
        if self.diarize and self.turn_detection is not None:
            raise ValueError(
                "Voxtral cannot label speakers and detect turns in one session: "
                "set turn_detection=None with diarize=True"
            )
        if self.speaker_references and not self.diarize:
            raise ValueError("speaker_references require diarize=True")
        if len(self.speaker_references) > MAX_SPEAKER_REFERENCES:
            raise ValueError(
                f"Voxtral takes at most {MAX_SPEAKER_REFERENCES} speaker references, "
                f"got {len(self.speaker_references)}"
            )
        for name, clip in self.speaker_references.items():
            _check_reference(name, clip)
        if self.diarized_pause_ms <= 0:
            raise ValueError("diarized_pause_ms must be positive")

    @property
    def final_wait_s(self) -> float:
        """The wait for a turn's last words after its end, in seconds."""
        if self.final_wait_ms is not None:
            return self.final_wait_ms / 1000
        return (self.streaming_delay_ms + _LATE_TEXT_MARGIN_MS) / 1000

    @property
    def next_turn_wait_s(self) -> float:
        """How long a turn waits after its end once the next turn has started, in seconds."""
        return min(self.final_wait_s, _NEXT_TURN_SHARE * self.streaming_delay_ms / 1000)

    @property
    def stream_url(self) -> str:
        """The realtime endpoint, the model in its query string."""
        base = self.realtime_url or self.base_url.rstrip("/") + REALTIME_PATH
        parts = urlsplit(base)
        scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
        query = urlencode([*parse_qsl(parts.query), ("model", self.realtime_model)])
        return urlunsplit((scheme, parts.netloc, parts.path, query, ""))


def _check_reference(name: str, clip: object) -> None:
    """Refuse a reference whose name the room would not show as given, or no clip.

    A label goes through :func:`~roomkit.voice.base.speaker_label`, with the
    spellings Voxtral may use for nobody: a name it rewrites (``"Speaker
    Bob"``, ``"unknown"``) would not be the one the caller chose, and a number
    could be the label of a voice the references did not match
    (``speaker_3``).
    """
    if not name.strip():
        raise ValueError("speaker_references names must not be blank")
    labelled = speaker_label(name, unattributed=UNATTRIBUTED_LABELS)
    if labelled != name.strip() or name.strip().isdigit():
        raise ValueError(
            f"speaker_references name {name!r} would not label its voice as given: "
            "use a name, not a number or a word such as 'speaker' or 'unknown'"
        )
    if not isinstance(clip, bytes) or not clip:
        raise ValueError(
            f"speaker_references[{name!r}] must be the clip's bytes (WAV, MP3, FLAC or Opus)"
        )
