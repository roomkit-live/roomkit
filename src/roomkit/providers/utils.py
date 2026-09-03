"""Shared utilities for providers."""

from __future__ import annotations

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


def http_timeout(config: HTTPTimeouts) -> httpx.Timeout:
    """Build the ``httpx.Timeout`` a provider hands its HTTP client or SDK.

    ``config.timeout`` is the read/write/pool budget, sized for the slowest
    response the provider expects; ``config.connect_timeout`` bounds the TCP
    connect alone. Handing the client the bare float instead applies the read
    budget to the connect too, so a host that no longer accepts connections is
    only given up on once the kernel exhausts its SYN retries (about two
    minutes), whatever the configured value.

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
    return httpx.Timeout(config.timeout, connect=config.connect_timeout)
