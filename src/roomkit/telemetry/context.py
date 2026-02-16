"""Context variable for telemetry span parent propagation."""

from __future__ import annotations

import contextvars

_parent_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "roomkit_parent_span", default=None
)


def get_current_span() -> str | None:
    """Get the current parent span ID from context."""
    return _parent_span.get()


def set_current_span(span_id: str | None) -> contextvars.Token[str | None]:
    """Set the current span ID in context. Returns a token for reset."""
    return _parent_span.set(span_id)


def reset_span(token: contextvars.Token[str | None]) -> None:
    """Reset the span context variable to its previous value."""
    _parent_span.reset(token)
