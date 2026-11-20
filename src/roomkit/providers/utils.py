"""Shared utilities for providers."""

from __future__ import annotations

import base64
import binascii
import sys
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import httpx
    import httpx2

    from roomkit.models.event import RoomEvent


class HTTPTimeouts(Protocol):
    """The two timeout knobs every HTTP provider config carries."""

    timeout: float
    connect_timeout: float


async def _aclose_stream(stream: Any) -> None:
    """Join an iterator's finalizer when its contract supports explicit close."""
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


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


def import_openai(owner: str, extra: str) -> Any:
    """The ``openai`` SDK for the client *owner* builds, or an ``ImportError``
    naming the RoomKit *extra* that installs it: when the SDK is missing, and
    when it is older than 3.

    The SDK is an optional dependency, imported where a client is built.
    RoomKit's extras install 3.x, which runs on httpx2; an environment that
    installed RoomKit without one can hold 2.x, which runs on httpx. There a
    client built, a request through an httpx2 transport failed as a lost
    connection, retried, and an httpx transport was refused for a reason
    that is not true of that SDK (measured on openai 2.54.0).
    """
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            f"openai is required for {owner}. Install it with: pip install roomkit[{extra}]"
        ) from exc
    version = getattr(openai, "__version__", None)
    major = version.partition(".")[0] if isinstance(version, str) else ""
    if major.isdigit() and int(major) < 3:
        raise ImportError(
            f"openai {version} is installed, and {owner} needs openai 3 or later. "
            f"Upgrade it with: pip install 'roomkit[{extra}]'"
        )
    return openai


def openai_http_client(
    openai_module: Any, transport: httpx2.AsyncBaseTransport | None
) -> httpx2.AsyncClient | None:
    """The OpenAI SDK's default HTTP client over *transport*; ``None`` without one.

    The AI, image and vision clients RoomKit builds on the ``openai`` SDK for
    an endpoint their config names take their ``http_client`` from here, so a
    host's ``transport=`` reaches each the same way: the seam for an outbound
    policy that judges the address actually dialled. (``OpenAIClassifier``,
    on OpenAI's own endpoint, takes a ``client=`` of the caller's instead.)
    The SDK's own class, not a bare ``httpx2.AsyncClient``, so redirects are
    still followed, each hop through the transport. Its timeout is the one
    the SDK client is built with, applied per request.
    The pool and its limits are the transport's; httpx2 reads no environment
    proxy beside a transport. Closing the SDK client closes it, and the
    transport with it.

    The SDK runs on httpx2 (openai 3.x), so the transport is an httpx2 one;
    one built on ``httpx`` raises ``TypeError`` (:func:`refuse_httpx_transport`).

    *openai_module* is the ``openai`` module the caller imported: the SDK is
    an optional dependency, imported where it is needed.
    """
    if transport is None:
        return None
    refuse_httpx_transport(transport)
    client: httpx2.AsyncClient = openai_module.DefaultAsyncHttpxClient(transport=transport)
    return client


def refuse_httpx_transport(transport: object) -> None:
    """Raise ``TypeError`` for a transport built on ``httpx`` alone, where the
    client it is for runs on httpx2.

    An httpx2 client takes any object as its transport. Handed an httpx one,
    it lets the request out through it, then fails on the response with a
    bare ``AssertionError`` (measured with ``httpx.MockTransport`` on openai
    3.57.0): the caller is told nothing, after the vendor may have run the
    request. Either package is read only if already imported: a transport of
    its cannot exist otherwise.
    """
    legacy, current = sys.modules.get("httpx"), sys.modules.get("httpx2")
    if legacy is None or not isinstance(transport, legacy.AsyncBaseTransport):
        return
    if current is not None and isinstance(transport, current.AsyncBaseTransport):
        return
    raise TypeError(
        f"transport is an httpx transport ({type(transport).__name__}), and the openai SDK "
        "runs on httpx2: build it on httpx2.AsyncBaseTransport instead"
    )


def openai_timeout(openai_module: Any, config: HTTPTimeouts) -> httpx2.Timeout:
    """The OpenAI SDK's own ``Timeout`` for a client, from its config.

    The connect/read split of :func:`http_timeout_from`, on the type the SDK
    runs on: openai 3.x is built on httpx2 and installs no ``httpx``, so an
    ``httpx.Timeout`` is not there to build in a host that installed the SDK
    alone.

    *openai_module* is the ``openai`` module the caller imported.
    """
    timeout: httpx2.Timeout = openai_module.Timeout(config.timeout, connect=config.connect_timeout)
    return timeout


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
