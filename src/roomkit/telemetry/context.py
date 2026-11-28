"""Context variable for telemetry span parent propagation."""

from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roomkit.telemetry.base import TelemetryProvider

_parent_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "roomkit_parent_span", default=None
)

# Secondary ContextVar for backend-specific context propagation.  The span_id
# in _parent_span is used by all telemetry providers (Noop, Console, Mock) for
# lightweight parent-child tracking.  Providers that carry richer context
# (e.g. OpenTelemetryProvider) store their native context object here so child
# spans can be created without a dict lookup — making the parent link resilient
# to any key/instance mismatch across async boundaries.
_parent_telemetry_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "roomkit_parent_telemetry_ctx", default=None
)

# The provider of the current span: the kit's, whose spans its telemetry
# names to another process (``inject_context``) for a request the code under
# the span sends elsewhere, an MCP server's (RK-21.2-057).
_parent_telemetry: contextvars.ContextVar[TelemetryProvider | None] = contextvars.ContextVar(
    "roomkit_parent_telemetry", default=None
)


def get_current_span() -> str | None:
    """Get the current parent span ID from context."""
    return _parent_span.get()


def get_current_telemetry_ctx() -> Any:
    """The current span's context as its provider gave it
    (:meth:`~roomkit.telemetry.TelemetryProvider.get_span_context`, or
    :meth:`~roomkit.telemetry.TelemetryProvider.extract_context` for a span
    of another process): a provider written outside RoomKit starts a span
    under it, in the current span's trace. ``None`` under no span, or when
    the provider gives no context."""
    return _parent_telemetry_ctx.get()


def current_telemetry() -> TelemetryProvider | None:
    """The provider the current span was set with, which names it to
    another process (:meth:`~roomkit.telemetry.TelemetryProvider.inject_context`);
    ``None`` when none was given. A span set without its provider keeps
    the one before, so read it with :func:`get_current_span`. Internal, as
    roomkit-rs's ``current_telemetry`` is hidden from its API."""
    return _parent_telemetry.get()


@dataclass(frozen=True, slots=True)
class SpanToken:
    """What :func:`reset_span` puts back: the span before
    :func:`set_current_span`, and its provider when that call set one."""

    span: contextvars.Token[str | None]
    telemetry: contextvars.Token[TelemetryProvider | None] | None = None


def set_current_span(
    span_id: str | None,
    *,
    telemetry_ctx: Any = None,
    telemetry: TelemetryProvider | None = None,
) -> SpanToken:
    """Set the current span ID in context. Returns a token for reset.

    Args:
        span_id: The roomkit span ID string.
        telemetry_ctx: Optional backend-specific context for direct parent
            propagation (e.g. OTel Context, Datadog span context). Kept as
            it was when ``None``, and never reset by :func:`reset_span`.
        telemetry: The provider that started the span, read back by
            :func:`current_telemetry`. Kept as it was when ``None``;
            :func:`reset_span` puts back the one before: a kit called under
            another's span leaves it its provider when it returns.
    """
    if telemetry_ctx is not None:
        _parent_telemetry_ctx.set(telemetry_ctx)
    provider = _parent_telemetry.set(telemetry) if telemetry is not None else None
    return SpanToken(_parent_span.set(span_id), provider)


def reset_span(token: SpanToken) -> None:
    """Put back the span, and its provider, as they were before the
    :func:`set_current_span` that gave *token*."""
    if token.telemetry is not None:
        _parent_telemetry.reset(token.telemetry)
    _parent_span.reset(token.span)


@contextmanager
def restored_span(
    span_id: str | None,
    *,
    telemetry_ctx: Any = None,
    telemetry: TelemetryProvider | None = None,
) -> Generator[None, None, None]:
    """Make a captured span the current one for the duration of the block.

    For work that runs on a fresh :mod:`contextvars` context — a delivery
    lane executor, a detached stream consumer — and must attach the spans it
    opens to the caller that planned it. The span, its backend context and
    its provider are set unconditionally and reset on exit:
    :func:`set_current_span` keeps the previous backend context and provider
    when handed ``None``, and :func:`reset_span` never touches the backend
    context, so restoring through that pair would leave stale ones behind
    for the next span the task opens. ``telemetry_ctx`` is whatever the provider's
    ``get_span_context`` returned for ``span_id`` (``None`` for providers
    that carry none), and ``telemetry`` that provider.
    """
    span_token = _parent_span.set(span_id)
    ctx_token = _parent_telemetry_ctx.set(telemetry_ctx)
    provider_token = _parent_telemetry.set(telemetry)
    try:
        yield
    finally:
        _parent_telemetry.reset(provider_token)
        _parent_telemetry_ctx.reset(ctx_token)
        _parent_span.reset(span_token)
