"""Shared utilities for providers."""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import httpx

    from roomkit.models.event import RoomEvent


class HTTPTimeouts(Protocol):
    """The two timeout knobs every HTTP provider config carries."""

    timeout: float
    connect_timeout: float


def extract_event_text(event: RoomEvent) -> str:
    """The text a transport can send for ``event``: the content's body, or ``""``.

    ``TextContent``, ``RichContent``, ``SystemContent`` and ``TemplateContent``
    carry a body; media, location and tool-call content carry none and yield
    the empty string, which the transports answer with ``empty_message``. A
    template with no body yields the empty string too, never ``"None"``.
    The memory layer reads events through
    :func:`roomkit.memory.token_estimator.extract_event_text`, which builds on
    this one and keeps a rendering for content that has no text.
    """
    body = getattr(event.content, "body", None)
    return body if isinstance(body, str) else ""


def to_data_uri(data: bytes, mime_type: str) -> str:
    """Encode raw bytes as the ``data:<mime>;base64,<payload>`` URI RoomKit carries images in."""
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def parse_data_uri(url: str, *, fallback_mime: str | None = None) -> tuple[str, bytes]:
    """Split a ``data:`` URI into its media type and its decoded bytes.

    The counterpart of :func:`to_data_uri`, and the one place a payload is
    validated: every provider that accepts an image has to reject a corrupt
    one, and each doing it itself is how one of them ends up handing
    malformed bytes to a vendor and reporting the rejection as a provider
    failure rather than a caller error.

    Whitespace — an encoder that wrapped its lines — and missing padding are
    repaired, so the bytes, and anything re-encoded from them, are canonical
    whatever the caller's URI carried. Anything else is refused: a character
    outside the alphabet, a length no padding can complete.

    Args:
        url: The URI to split.
        fallback_mime: Media type to use when the URI declares none. ``None``
            falls back to ``image/png``.

    Returns:
        ``(mime_type, data)``.

    Raises:
        ValueError: If *url* is not a ``data:`` URI, carries no payload, or
            its payload is not base64.
    """
    if not url.startswith("data:"):
        raise ValueError(f"expected a data: URI, got a {url.split(':', 1)[0]} URL")
    header, separator, payload = url.partition(",")
    compact = "".join(payload.split())
    if not separator or not compact:
        raise ValueError("data URI carries no payload")
    mime_type = header[len("data:") :].split(";", 1)[0] or fallback_mime or "image/png"
    compact += "=" * (-len(compact) % 4)
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("data URI payload is not valid base64") from exc
    return mime_type, data


def http_timeout(config: HTTPTimeouts) -> httpx.Timeout:
    """The ``httpx.Timeout`` a provider hands its HTTP client or SDK, from its config."""
    return http_timeout_from(config.timeout, config.connect_timeout)


def http_timeout_from(timeout: float, connect_timeout: float) -> httpx.Timeout:
    """Build the ``httpx.Timeout`` that splits the connect from the read.

    ``timeout`` is the read/write/pool budget, sized for the slowest response
    the caller expects; ``connect_timeout`` bounds the TCP connect alone.
    Handing the client the bare float instead applies the read budget to the
    connect too, so a host that no longer accepts connections is only given up
    on once the kernel exhausts its SYN retries (about two minutes), whatever
    the configured value.

    httpx is an optional dependency, so it is imported here rather than at
    module level; every caller has already imported it, or an SDK that
    depends on it, before reaching this function.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required to build a provider's HTTP timeout. "
            "Install it with: pip install roomkit[httpx]"
        ) from exc
    return httpx.Timeout(timeout, connect=connect_timeout)
