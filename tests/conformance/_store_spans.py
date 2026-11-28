"""The run's store tracing its delivery cursor (README, Setup:
``kit.store_spans``).

Each read and each write of a room's delivery cursor (RFC §13.5) records a
``store.query`` span, ``store.<operation>``, to the telemetry provider the kit
handed its store (``_telemetry``), under the span current when the operation is
called, as a store that traces its queries does; the operation itself is the
store's own.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.store.base import ConversationStore
from roomkit.telemetry import SpanKind, get_current_span

CURSOR_OPERATIONS = ("get_delivered_index", "advance_delivered_index")


def trace_cursor(store: ConversationStore) -> None:
    """Has *store* record a span over each of its cursor operations."""
    for name in CURSOR_OPERATIONS:
        setattr(store, name, _spanned(store, name, getattr(store, name)))


def _spanned(
    store: ConversationStore, name: str, operation: Callable[..., Awaitable[Any]]
) -> Callable[..., Awaitable[Any]]:
    async def spanned(*args: Any, **kwargs: Any) -> Any:
        telemetry = getattr(store, "_telemetry", None)
        if telemetry is None:
            return await operation(*args, **kwargs)
        span_id = telemetry.start_span(
            SpanKind.STORE_QUERY, f"store.{name}", parent_id=get_current_span()
        )
        try:
            return await operation(*args, **kwargs)
        finally:
            telemetry.end_span(span_id)

    spanned.__name__ = name
    return spanned
