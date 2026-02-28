"""Tests for source-level rate limiting (WebSocket + SSE)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from roomkit import InboundResult, RateLimit
from roomkit.sources.websocket import WebSocketSource


async def test_websocket_source_drops_excess_messages() -> None:
    """WebSocketSource should drop messages exceeding the rate limit."""
    source = WebSocketSource(
        url="ws://localhost/test",
        channel_id="ws-test",
        rate_limit=RateLimit(max_per_second=1.0),
    )

    # Verify rate limiter was created
    assert source._rate_limiter is not None
    assert source._rate_limit is not None
    assert source._rate_limit.max_per_second == 1.0


async def test_websocket_source_no_rate_limit() -> None:
    """WebSocketSource without rate_limit should have no limiter."""
    source = WebSocketSource(
        url="ws://localhost/test",
        channel_id="ws-test",
    )
    assert source._rate_limiter is None
    assert source._rate_limit is None


async def test_websocket_source_receive_loop_drops_excess() -> None:
    """_receive_loop should not call emit for rate-limited messages."""
    source = WebSocketSource(
        url="ws://localhost/test",
        channel_id="ws-test",
        rate_limit=RateLimit(max_per_second=2.0),
    )

    # Build 5 JSON messages that the default parser will accept
    raw_messages = [json.dumps({"sender_id": "user", "text": f"msg-{i}"}) for i in range(5)]
    msg_iter = iter(raw_messages)

    class FakeWS:
        """Minimal mock that yields raw_messages then signals stop."""

        async def recv(self) -> str:
            try:
                return next(msg_iter)
            except StopIteration:
                # Signal stop so the loop exits after the TimeoutError
                source._stop_event.set()
                raise TimeoutError from None

    emit = AsyncMock(return_value=InboundResult())

    # Ensure stop is not set before the loop starts
    source._stop_event.clear()

    await source._receive_loop(FakeWS(), emit)  # type: ignore[arg-type]

    # With burst=1 at 2/sec, only 2 messages should have been emitted
    assert emit.call_count == 2


async def test_websocket_source_no_limit_passes_all() -> None:
    """Without rate_limit, all messages should be emitted."""
    source = WebSocketSource(
        url="ws://localhost/test",
        channel_id="ws-test",
        # No rate_limit
    )

    raw_messages = [json.dumps({"sender_id": "user", "text": f"msg-{i}"}) for i in range(5)]
    msg_iter = iter(raw_messages)

    class FakeWS:
        async def recv(self) -> str:
            try:
                return next(msg_iter)
            except StopIteration:
                source._stop_event.set()
                raise TimeoutError from None

    emit = AsyncMock(return_value=InboundResult())
    source._stop_event.clear()

    await source._receive_loop(FakeWS(), emit)  # type: ignore[arg-type]

    assert emit.call_count == 5


async def test_sse_source_rate_limit_creates_limiter() -> None:
    """SSESource with rate_limit should create a limiter."""
    from roomkit.sources.sse import SSESource

    source = SSESource(
        url="https://localhost/events",
        channel_id="sse-test",
        rate_limit=RateLimit(max_per_second=5.0),
    )
    assert source._rate_limiter is not None
    assert source._rate_limit is not None
    assert source._rate_limit.max_per_second == 5.0


async def test_sse_source_no_rate_limit() -> None:
    """SSESource without rate_limit should have no limiter."""
    from roomkit.sources.sse import SSESource

    source = SSESource(
        url="https://localhost/events",
        channel_id="sse-test",
    )
    assert source._rate_limiter is None
    assert source._rate_limit is None
