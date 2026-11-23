"""Wire-level pieces of the Meta Muse Voice Transcribe protocol.

What the service sends and how it fails, kept apart from
:mod:`roomkit.voice.stt.meta`, which drives the connection: the event mapping,
the error shapes of the REST and WebSocket endpoints, and the WAV container the
REST endpoint takes. The public import path is ``roomkit.voice.stt``.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any

from roomkit.providers.ai.base import ProviderError
from roomkit.voice.base import SpeakerSegment, TranscriptionResult, speaker_label

# How this vendor's wire may spell a speaker it could not attribute: the
# provider names them, RoomKit's label rule knows no vendor's (RFC §12.2.3).
_UNATTRIBUTED = ("unknown", "UU", "PENDING")

# 1011 is a server fault ("Max session duration reached" included) and 1013 a
# rate limit: both are worth a new stream. 1008 is a refused request.
_RETRYABLE_CLOSE_CODES = frozenset({1011, 1013})

WAV_MIME_TYPES: frozenset[str] = frozenset(
    {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
)
"""Media types a ``data:`` URI may declare for a clip: the endpoint takes WAV only."""


def label_of(value: Any) -> str | None:
    """Meta's speaker label as RoomKit carries it (RFC §12.2.3)."""
    return speaker_label(value, unattributed=_UNATTRIBUTED)


class MetaSTTError(ProviderError):
    """A request the Meta Model API refused or failed.

    Attributes:
        code: Meta's ``error.code`` / ``errorCode`` (e.g.
            ``"billing_not_configured"``), or the WebSocket close code as a
            string when the service closed without an error frame.
        error_type: Meta's ``error.type`` / ``errorType`` (e.g.
            ``"billing_error"``), when it sent one.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable, provider="MetaSTT", status_code=status_code)
        self.code = code
        self.error_type = error_type


@dataclass
class _TurnMarks:
    speaker: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    ended: bool = False


class Turns:
    """What a ``DIARIZATION`` stream has said about its turns, by ``turnId``.

    A turn's events are ``speechStart``, partials, ``speaker``, ``speechEnd``,
    ``speechComplete`` (observed on every turn on 2026-09-27). The ``speaker``
    event carries no ``turnId``: it labels the oldest turn that has started and
    not ended, because it arrives just before that turn's ``speechEnd`` and
    turns end in order. A change of voice ends a turn without a pause, so the
    next turn may start before the previous one completes; keying by
    ``turnId`` keeps each label and offset on its own turn. The offsets are
    where the service had got to in the stream when it signalled the turn's
    start and end.
    """

    def __init__(self) -> None:
        self._turns: dict[Any, _TurnMarks] = {}

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "speechStart":
            self._turns[event.get("turnId")] = _TurnMarks(start_ms=event.get("audioProcessedMs"))
        elif kind == "speaker":
            open_turn = next((t for t in self._turns.values() if not t.ended), None)
            if open_turn is not None:
                open_turn.speaker = label_of(event.get("label"))
        elif kind == "speechEnd":
            marks = self._turns.get(event.get("turnId"))
            if marks is not None:
                marks.end_ms, marks.ended = event.get("audioProcessedMs"), True

    def close(self, event: dict[str, Any], text: str) -> SpeakerSegment:
        """The turn ``event`` completes, as a segment; its marks are dropped.

        A completion without a known ``turnId`` takes the oldest open turn, and
        a turn nothing was recorded for is unattributed rather than guessed.
        """
        marks = self._turns.pop(event.get("turnId"), None)
        if marks is None and self._turns:
            marks = self._turns.pop(next(iter(self._turns)))
        marks = marks or _TurnMarks()
        return SpeakerSegment(marks.speaker, text, marks.start_ms, marks.end_ms)


def to_result(event: dict[str, Any], turns: Turns | None = None) -> TranscriptionResult | None:
    """Map one server event to a result, ``None`` for the ones RoomKit ignores.

    One mapping serves every mode, as observed on the live service: in
    ``ENDPOINTING`` and ``DIARIZATION`` a turn's final is ``speechComplete``
    and the stream ends on an *empty* ``transcript`` marked final; in
    ``PUSH_TO_TALK`` there is no ``speechComplete`` and the final is that
    ``transcript``, with the text. ``speechEnd`` precedes ``speechComplete``.

    Pass ``turns`` on a ``DIARIZATION`` stream: it follows the turns under way,
    and each final then carries its turn as its one speaker segment.
    """
    kind = event.get("type")
    if kind == "error":
        raise event_error(event)
    if turns is not None:
        turns.observe(event)
    if kind == "speechStart":
        return TranscriptionResult(text="", is_final=False, is_speech_start=True)
    if kind not in ("transcript", "speechComplete"):
        return None
    text = str(event.get("transcript") or "").strip()
    if kind == "speechComplete" and turns is not None:
        # Closed even when empty, so its label cannot drift onto a later turn.
        segment = turns.close(event, text)
        return TranscriptionResult(text=text, segments=[segment]) if text else None
    if not text:
        return None
    is_final = kind == "speechComplete" or bool(event.get("final"))
    if is_final and turns is not None:
        return TranscriptionResult(text=text, segments=[SpeakerSegment(None, text)])
    return TranscriptionResult(text=text, is_final=is_final)


def rest_result(body: dict[str, Any], *, diarized: bool) -> TranscriptionResult:
    """The REST answer ``{transcript, turns: [{speaker, transcript, startMs, endMs}]}``.

    Turns become speaker segments when the request asked for ``DIARIZATION``;
    in the other modes the service answers no speaker and none is invented.
    """
    text = str(body.get("transcript") or "").strip()
    if not diarized:
        return TranscriptionResult(text=text)
    segments = [
        SpeakerSegment(
            label_of(turn.get("speaker")),
            str(turn.get("transcript") or "").strip(),
            turn.get("startMs"),
            turn.get("endMs"),
        )
        for turn in body.get("turns") or []
        if str(turn.get("transcript") or "").strip()
    ]
    if text and not segments:
        # Every diarized final names its segments (RFC §12.2.3): text the
        # service answered with no turn is attributed to nobody, not dropped.
        segments = [SpeakerSegment(None, text)]
    return TranscriptionResult(text=text, segments=segments)


def event_error(event: dict[str, Any]) -> MetaSTTError:
    """The error frame the service sends before closing a refused stream."""
    return MetaSTTError(
        f"Meta STT error: {event.get('message') or 'unknown error'}",
        code=event.get("errorCode"),
        error_type=event.get("errorType"),
    )


def stream_error(exc: Exception) -> MetaSTTError:
    """A failed connection, handshake or close, as an exception.

    Retryable unless the service refused: a server fault (1011), a rate limit
    (1013, or HTTP 429 at the upgrade), a 5xx at the upgrade, a timeout, a
    network error, and a connection dropped without a close frame may all work
    on a new stream. A refusal is a close code (1008) or a 4xx at the upgrade.
    """
    received = getattr(exc, "rcvd", None)
    code = getattr(received, "code", None)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    # Only websockets' ConnectionClosed carries ``rcvd``, and ``None`` there
    # means the peer vanished without a close frame, not that it refused.
    dropped = hasattr(exc, "rcvd") and received is None
    retryable = (
        code in _RETRYABLE_CLOSE_CODES
        or dropped
        or (status is not None and (status == 429 or status >= 500))
        or isinstance(exc, TimeoutError | OSError)
    )
    label = code if code is not None else status if status is not None else type(exc).__name__
    reason = getattr(received, "reason", "") or str(exc) or type(exc).__name__
    return MetaSTTError(
        f"Meta STT stream failed ({label}): {reason}",
        code=str(code) if code is not None else None,
        retryable=retryable,
        status_code=status,
    )


def http_error(response: Any) -> MetaSTTError:
    """A REST error body, ``{"error": {"code", "type", "message"}}``, as an exception."""
    try:
        error = response.json().get("error") or {}
    except ValueError:
        error = {}
    status = response.status_code
    return MetaSTTError(
        f"Meta STT request failed ({status}): {error.get('message') or response.text[:200]}",
        code=error.get("code"),
        error_type=error.get("type"),
        retryable=status == 429 or status >= 500,
        status_code=status,
    )


def read_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """``(pcm, sample_rate, channels, sample_width)`` of a WAV file.

    Raises:
        ValueError: If *data* is not a PCM WAV file.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            pcm = reader.readframes(reader.getnframes())
            return pcm, reader.getframerate(), reader.getnchannels(), reader.getsampwidth()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"not a PCM WAV file: {exc}") from exc


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Mono 16-bit PCM wrapped in a WAV container — the only upload format taken."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()
