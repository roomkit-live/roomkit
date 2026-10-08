"""Wire-level pieces of Microsoft's MAI-Transcribe realtime protocol.

What the service sends and how it fails, kept apart from
:mod:`roomkit.voice.stt.azure_mai`, which drives the connection: the endpoint
URL, the language codes the service takes, the event mapping and the error
shapes. The public import path stays ``roomkit.voice.stt.azure_mai``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from roomkit.providers.ai.base import ProviderError
from roomkit.voice.base import TranscriptionResult

REALTIME_PATH = "/mai/v1/realtime"

TRANSCRIPTION_EVENT = "conversation.item.input_audio_transcription."

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "af", "ar", "az", "bg", "bn", "bs", "ca", "cs", "da", "de",
        "el", "en", "es", "et", "fa", "fi", "fil", "fr", "gl", "gu",
        "he", "hi", "hu", "hy", "id", "is", "it", "ja", "kk", "kn",
        "ko", "lt", "lv", "mk", "ml", "mr", "ms", "nb", "ne", "nl",
        "pl", "pt", "ro", "ru", "sk", "sl", "sv", "sw", "ta", "te",
        "th", "tr", "uk", "ur", "vi", "yue", "zh",
    }
)  # fmt: skip
"""The language codes MAI-Transcribe-2 takes: Microsoft's list of 60, less the
three the live endpoint refuses (``as``, ``or``, ``pa``; measured 2026-10-08).

Checked here because the endpoint's own check is not the model's: it refuses
a region tag (``"fr-CA"``) and those three with an error naming another list,
and takes codes the model does not list (``no``, ``tl``, ``cy``…), which the
model then reads as no language at all.
"""

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# 1011 is a server fault and 1013 a rate limit: both are worth a new stream.
_RETRYABLE_CLOSE_CODES = frozenset({1011, 1013})

# OpenAI Realtime error types and codes a new stream may get past.
_RETRYABLE_ERRORS = frozenset({"server_error", "rate_limit_exceeded"})


class AzureMAISTTError(ProviderError):
    """A stream the MAI-Transcribe service refused or failed.

    Attributes:
        code: The service's ``error.code``, or the WebSocket close code as a
            string when it closed without an error event.
        error_type: The service's ``error.type`` (e.g.
            ``"invalid_request_error"``), when it sent one.
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
            message, retryable=retryable, provider="AzureMAISTT", status_code=status_code
        )
        self.code = code
        self.error_type = error_type


def realtime_url(endpoint: str) -> str:
    """The transcription WebSocket URL of a Foundry resource.

    Takes the resource URL as the portal shows it
    (``https://<resource>.services.ai.azure.com``), with or without the
    realtime path. Plain ``ws``/``http`` is accepted for a loopback host only:
    anywhere else it would send the key in clear.

    Raises:
        ValueError: *endpoint* is not a resource URL.
    """
    parts = urlsplit(endpoint.strip())
    scheme = {"https": "wss", "wss": "wss", "http": "ws", "ws": "ws"}.get(parts.scheme)
    if scheme is None or not parts.hostname:
        raise ValueError(
            "endpoint must be the Foundry resource URL, such as "
            f"https://<resource>.services.ai.azure.com; got {endpoint!r}"
        )
    if scheme == "ws" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("endpoint must use https: a plain connection would send the key in clear")
    path = parts.path.rstrip("/")
    if path not in ("", REALTIME_PATH):
        raise ValueError(
            f"endpoint must be the resource root or its {REALTIME_PATH} URL; got path {path!r}"
        )
    return urlunsplit((scheme, parts.netloc, REALTIME_PATH, "intent=transcription", ""))


def language_code(language: str | None) -> str | None:
    """The code the service takes for a BCP-47 *language*: ``"fr-CA"`` -> ``"fr"``.

    The service transcribes by language, not by region, so the region is
    dropped. ``None`` keeps automatic detection.

    Raises:
        ValueError: The language is not one the service supports.
    """
    if language is None:
        return None
    code = language.strip().replace("_", "-").split("-")[0].lower()
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"MAI-Transcribe does not support language {language!r}; "
            f"supported codes: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )
    return code


@dataclass
class Transcript:
    """What one stream has heard so far.

    ``delta`` events append text the service has made final; ``intermediate``
    events replace the provisional rest. The service's best guess is the final
    text followed by the provisional rest, appended as sent: the service
    carries the spaces.

    Only an ``intermediate`` gives a partial. The live service firms up a
    guess in two events, a ``delta`` with its first words then an
    ``intermediate`` with the rest (observed 2026-10-08): a partial on the
    ``delta`` would show the guess cut to those first words for an instant.
    """

    final: str = ""
    provisional: str = ""
    last_partial: str = ""

    def partial(self) -> TranscriptionResult | None:
        """The current guess as a partial, ``None`` when empty or unchanged."""
        text = (self.final + self.provisional).strip()
        if not text or text == self.last_partial:
            return None
        self.last_partial = text
        return TranscriptionResult(text=text, is_final=False)


def to_result(event: dict[str, Any], transcript: Transcript) -> TranscriptionResult | None:
    """Map one server event to a result, ``None`` for the ones RoomKit ignores.

    A ``completed`` event is the final, returned even when empty so the caller
    knows the stream is done: the service may finalise to nothing what it had
    guessed (a cough).

    Raises:
        AzureMAISTTError: The event reports an error.
    """
    kind = event.get("type")
    if kind in ("error", TRANSCRIPTION_EVENT + "failed"):
        raise event_error(event)
    if kind == TRANSCRIPTION_EVENT + "delta":
        transcript.final += str(event.get("delta") or "")
        transcript.provisional = ""
        return None
    if kind == TRANSCRIPTION_EVENT + "intermediate":
        transcript.provisional = str(event.get("intermediate") or "")
        return transcript.partial()
    if kind == TRANSCRIPTION_EVENT + "completed":
        return TranscriptionResult(text=str(event.get("transcript") or "").strip())
    return None


def event_error(event: dict[str, Any]) -> AzureMAISTTError:
    """An ``error`` or ``...failed`` event, ``{"error": {type, code, message}}``."""
    raw = event.get("error")
    error: dict[str, Any] = raw if isinstance(raw, dict) else {}
    code = error.get("code")
    error_type = error.get("type")
    return AzureMAISTTError(
        f"MAI-Transcribe error: {error.get('message') or code or 'unknown error'}",
        code=code,
        error_type=error_type,
        retryable=bool(_RETRYABLE_ERRORS & {code, error_type}),
    )


def stream_error(exc: Exception) -> AzureMAISTTError:
    """A failed connection, handshake or close, as an exception.

    Retryable unless the service refused: a server fault (1011), a rate limit
    (1013, or HTTP 429 at the upgrade), a 5xx at the upgrade, a timeout, a
    network error, and a connection dropped without a close frame may all work
    on a new stream. A refusal is any other close code or a 4xx at the upgrade.
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
    return AzureMAISTTError(
        f"MAI-Transcribe stream failed ({label}): {reason}",
        code=str(code) if code is not None else None,
        retryable=retryable,
        status_code=status,
    )
