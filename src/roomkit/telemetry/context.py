"""Context variable for telemetry span parent propagation."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

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


def get_current_span() -> str | None:
    """Get the current parent span ID from context."""
    return _parent_span.get()


def get_current_telemetry_ctx() -> Any:
    """Get the current telemetry backend context (may be None)."""
    return _parent_telemetry_ctx.get()


def set_current_span(
    span_id: str | None, *, telemetry_ctx: Any = None
) -> contextvars.Token[str | None]:
    """Set the current span ID in context. Returns a token for reset.

    Args:
        span_id: The roomkit span ID string.
        telemetry_ctx: Optional backend-specific context for direct parent
            propagation (e.g. OTel Context, Datadog span context).
    """
    if telemetry_ctx is not None:
        _parent_telemetry_ctx.set(telemetry_ctx)
    return _parent_span.set(span_id)


def reset_span(token: contextvars.Token[str | None]) -> None:
    """Reset the span context variable to its previous value."""
    _parent_span.reset(token)


@contextmanager
def restored_span(span_id: str | None, telemetry_ctx: Any = None) -> Iterator[None]:
    """Make a captured span the current one for the duration of the block.

    For work that runs on a fresh :mod:`contextvars` context — a delivery
    lane executor, a detached stream consumer — and must attach the spans it
    opens to the caller that planned it. Both variables are set
    unconditionally and both are reset on exit: :func:`set_current_span`
    keeps the previous backend context when handed ``None`` and
    :func:`reset_span` never touches it, so restoring through that pair
    would leave a stale backend context behind for the next span the task
    opens. ``telemetry_ctx`` is whatever the provider's ``get_span_context``
    returned for ``span_id`` (``None`` for providers that carry none).
    """
    span_token = _parent_span.set(span_id)
    ctx_token = _parent_telemetry_ctx.set(telemetry_ctx)
    try:
        yield
    finally:
        _parent_telemetry_ctx.reset(ctx_token)
        _parent_span.reset(span_token)
