"""Context variable for telemetry span parent propagation."""

from __future__ import annotations

import contextvars
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
