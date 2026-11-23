"""Wire-level pieces of Mistral's Voxtral transcription API.

What the service sends and how it fails, kept apart from
:mod:`roomkit.voice.stt.voxtral`, which drives the connection: the language
codes, the file endpoint's answer, and the error shapes of the REST and
WebSocket endpoints. The realtime text is cut into results by
:mod:`roomkit.voice.stt.voxtral_transcript`. The public import path is
``roomkit.voice.stt``.
"""

from __future__ import annotations

from typing import Any

from roomkit.providers.ai.base import ProviderError
from roomkit.voice.base import SpeakerSegment, TranscriptionResult, speaker_label

# How this vendor's wire may spell a speaker it could not attribute: the
# provider names them, RoomKit's label rule knows no vendor's (RFC §12.2.3).
UNATTRIBUTED_LABELS = ("unknown", "UU", "PENDING")

REALTIME_PATH = "/v1/audio/transcriptions/realtime"

STREAMING_DELAYS_MS = "80 to 1200, or 2400"
"""The streaming delays the realtime model was trained for: any other value is
refused when the session is configured (measured 2026-10-21)."""

_DONE = "transcription.done"

# 1011 is a server fault and 1013 a rate limit: both are worth a new stream.
_RETRYABLE_CLOSE_CODES = frozenset({1011, 1013})


class VoxtralSTTError(ProviderError):
    """A request Mistral's transcription API refused or failed.

    Attributes:
        code: Mistral's error ``code`` (``"1500"`` for an unknown model,
            ``3051`` for a refused realtime setting), or the WebSocket close
            code as a string when the service closed without an error event.
        error_type: Mistral's error ``type`` (``"invalid_model"``), when it
            sent one.
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
        super().__init__(
            message, retryable=retryable, provider="VoxtralSTT", status_code=status_code
        )
        self.code = code
        self.error_type = error_type


def valid_streaming_delay(delay_ms: int) -> bool:
    """Whether the realtime model takes *delay_ms* (:data:`STREAMING_DELAYS_MS`)."""
    return 80 <= delay_ms <= 1200 or delay_ms == 2400


def language_code(language: str | None) -> str | None:
    """The code the service takes for a BCP-47 *language*: ``"fr-CA"`` -> ``"fr"``.

    Voxtral takes ISO 639-1 codes only and refuses a region (``"fr-CA"``,
    measured 2026-10-21), so the region is dropped. ``None`` keeps automatic
    detection.

    Raises:
        ValueError: The language is not a two-letter code once its region is
            dropped.
    """
    if language is None:
        return None
    code = language.strip().replace("_", "-").split("-")[0].lower()
    if len(code) != 2 or not code.isascii() or not code.isalpha():
        raise ValueError(
            f"Voxtral takes a two-letter ISO 639-1 language code such as 'fr', got {language!r}"
        )
    return code


def is_done(event: dict[str, Any]) -> bool:
    """Whether *event* ends the session: the service then drops the connection."""
    return event.get("type") == _DONE


def rest_result(body: dict[str, Any], *, diarized: bool = False) -> TranscriptionResult:
    """The file endpoint's answer ``{text, language, segments, usage}``.

    ``language`` is ``null`` in every answer measured, pinned or detected
    (2026-10-21); it is passed on as the service reports it. A diarized
    answer's segments, a sentence or so each with its speaker and times in
    seconds, become one speaker segment per run of consecutive segments of
    one speaker (RFC §12.2.3); text with no segment is nobody's.
    """
    language = body.get("language")
    text = str(body.get("text") or "").strip()
    segments = _speaker_runs(body.get("segments") or []) if diarized else []
    if diarized and text and not segments:
        segments = [SpeakerSegment(None, text)]
    return TranscriptionResult(
        text=text,
        language=language if isinstance(language, str) and language else None,
        segments=segments,
    )


def _speaker_runs(items: list[Any]) -> list[SpeakerSegment]:
    runs: list[SpeakerSegment] = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        speaker = speaker_label(item.get("speaker_id"), unattributed=UNATTRIBUTED_LABELS)
        text = str(item["text"]).strip()
        start, end = _ms(item.get("start")), _ms(item.get("end"))
        if runs and runs[-1].speaker == speaker:
            last = runs[-1]
            joined = f"{last.text} {text}"
            runs[-1] = SpeakerSegment(speaker, joined, last.start_ms, end or last.end_ms)
        else:
            runs.append(SpeakerSegment(speaker, text, start, end))
    return runs


def _ms(seconds: Any) -> int | None:
    return round(seconds * 1000) if isinstance(seconds, int | float) else None


def event_error(event: dict[str, Any]) -> VoxtralSTTError:
    """A realtime ``error`` event, ``{"error": {"message", "code"}}``.

    The service sends one for a setting it refuses, then keeps the session
    open: none of these is worth retrying unchanged.
    """
    raw = event.get("error")
    error: dict[str, Any] = raw if isinstance(raw, dict) else {}
    code = error.get("code")
    return VoxtralSTTError(
        f"VoxtralSTT error: {error.get('message') or code or 'unknown error'}",
        code=str(code) if code is not None else None,
    )


def stream_error(exc: Exception) -> VoxtralSTTError:
    """A failed connection, handshake or close, as an exception.

    Retryable unless the service refused: a server fault (1011), a rate limit
    (1013, or HTTP 429 at the upgrade), a 5xx at the upgrade, a timeout, a
    network error, and a connection dropped without a close frame may all work
    on a new stream. A refusal is any other close code or a 4xx at the upgrade
    (401 for a wrong key, 403 with no body for a model the key cannot use).
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
    if status == 403:
        reason += " (is the realtime model id right, and open to this key?)"
    return VoxtralSTTError(
        f"VoxtralSTT stream failed ({label}): {reason}",
        code=str(code) if code is not None else None,
        retryable=retryable,
        status_code=status,
    )


def http_error(response: Any) -> VoxtralSTTError:
    """A REST error body as an exception.

    Mistral answers ``{"object": "error", "message", "type", "code"}``, a
    rejected key ``{"detail": "Invalid API Key"}``, and a refused field
    ``{"detail": [{"msg", "loc"}]}``.
    """
    try:
        body = response.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    detail = body.get("detail")
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg")) for item in detail if isinstance(item, dict))
    message = body.get("message") or detail or response.text[:200]
    code = body.get("code")
    status = response.status_code
    return VoxtralSTTError(
        f"VoxtralSTT request failed ({status}): {message}",
        code=str(code) if code is not None else None,
        error_type=body.get("type"),
        retryable=status == 429 or status >= 500,
        status_code=status,
    )
