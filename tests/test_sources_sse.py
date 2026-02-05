"""Tests for SSE event source."""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import InboundMessage, RoomKit, TextContent
from roomkit.sources.base import SourceStatus


# =============================================================================
# Test default_json_parser
# =============================================================================


class TestDefaultJsonParser:
    def test_parses_valid_message_event(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("test-channel")

        data = json.dumps(
            {
                "sender_id": "user123",
                "text": "Hello world",
                "external_id": "msg-456",
                "metadata": {"key": "value"},
            }
        )

        msg = parser("message", data, "evt-1")

        assert msg is not None
        assert msg.channel_id == "test-channel"
        assert msg.sender_id == "user123"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello world"
        assert msg.external_id == "msg-456"
        assert msg.metadata == {"key": "value"}

    def test_parses_empty_event_type(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1", "text": "hello"})

        msg = parser("", data, None)

        assert msg is not None
        assert msg.sender_id == "user1"

    def test_parses_msg_event_type(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1"})

        msg = parser("msg", data, None)

        assert msg is not None

    def test_parses_chat_event_type(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1"})

        msg = parser("chat", data, None)

        assert msg is not None

    def test_skips_ping_event(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1"})

        msg = parser("ping", data, None)

        assert msg is None

    def test_skips_heartbeat_event(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1"})

        msg = parser("heartbeat", data, None)

        assert msg is None

    def test_parses_minimal_message(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1"})

        msg = parser("message", data, None)

        assert msg is not None
        assert msg.sender_id == "user1"
        assert msg.content.body == ""
        assert msg.external_id is None
        assert msg.metadata == {}

    def test_uses_event_id_as_fallback_external_id(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1"})

        msg = parser("message", data, "sse-event-123")

        assert msg is not None
        assert msg.external_id == "sse-event-123"

    def test_explicit_external_id_overrides_event_id(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"sender_id": "user1", "external_id": "custom-id"})

        msg = parser("message", data, "sse-event-123")

        assert msg is not None
        assert msg.external_id == "custom-id"

    def test_returns_none_for_missing_sender_id(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")
        data = json.dumps({"text": "no sender"})

        msg = parser("message", data, None)

        assert msg is None

    def test_returns_none_for_invalid_json(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")

        msg = parser("message", "not valid json {", None)

        assert msg is None

    def test_returns_none_for_non_dict(self) -> None:
        from roomkit.sources.sse import default_json_parser

        parser = default_json_parser("ch1")

        assert parser("message", json.dumps([1, 2, 3]), None) is None
        assert parser("message", json.dumps("string"), None) is None
        assert parser("message", json.dumps(123), None) is None


# =============================================================================
# Test SSESource initialization
# =============================================================================


class TestSSESourceInit:
    def test_default_initialization(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        assert source.name == "sse:https://example.com/events"
        assert source.status == SourceStatus.STOPPED
        assert source._url == "https://example.com/events"
        assert source._channel_id == "test"
        assert source._timeout == 30.0
        assert source.last_event_id is None

    def test_custom_parser(self) -> None:
        from roomkit.sources.sse import SSESource

        def custom_parser(
            event: str, data: str, event_id: str | None
        ) -> InboundMessage | None:
            return InboundMessage(
                channel_id="custom",
                sender_id="custom-sender",
                content=TextContent(body=data),
            )

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
            parser=custom_parser,
        )

        # Test the parser is used
        msg = source._parser("message", "hello", None)
        assert msg is not None
        assert msg.sender_id == "custom-sender"
        assert msg.content.body == "hello"

    def test_custom_headers(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
            headers={"Authorization": "Bearer token123"},
        )

        assert source._headers == {"Authorization": "Bearer token123"}

    def test_all_options(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
            headers={"X-Custom": "value"},
            params={"filter": "all"},
            timeout=60.0,
            last_event_id="evt-100",
        )

        assert source._headers == {"X-Custom": "value"}
        assert source._params == {"filter": "all"}
        assert source._timeout == 60.0
        assert source.last_event_id == "evt-100"


# =============================================================================
# Test SSESource with mocked httpx
# =============================================================================


class MockSSEEvent:
    """Mock SSE event for testing."""

    def __init__(self, event: str = "message", data: str = "", id: str | None = None):
        self.event = event
        self.data = data
        self.id = id


class TestSSESourceReceiveLoop:
    """Test the receive loop logic directly without full connection mocking."""

    async def test_receive_loop_processes_events(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        events = [
            MockSSEEvent(
                "message", json.dumps({"sender_id": "user1", "text": "Hello"}), "1"
            ),
            MockSSEEvent(
                "message", json.dumps({"sender_id": "user2", "text": "World"}), "2"
            ),
        ]

        received: list[InboundMessage] = []

        async def emit(msg: InboundMessage):
            received.append(msg)
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        # Create mock event source
        mock_event_source = MagicMock()

        async def mock_aiter():
            for event in events:
                yield event

        mock_event_source.aiter_sse = mock_aiter

        # Call receive loop directly
        await source._receive_loop(mock_event_source, emit)

        assert len(received) == 2
        assert received[0].sender_id == "user1"
        assert received[0].content.body == "Hello"
        assert received[1].sender_id == "user2"
        assert received[1].content.body == "World"

    async def test_receive_loop_skips_unparseable(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        events = [
            MockSSEEvent("message", "not json"),  # Unparseable
            MockSSEEvent("message", json.dumps({"no_sender": True})),  # Missing sender
            MockSSEEvent("ping", json.dumps({"sender_id": "x"})),  # Wrong event type
            MockSSEEvent(
                "message", json.dumps({"sender_id": "user1", "text": "Valid"})
            ),  # Valid
        ]

        received: list[InboundMessage] = []

        async def emit(msg: InboundMessage):
            received.append(msg)
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        mock_event_source = MagicMock()

        async def mock_aiter():
            for event in events:
                yield event

        mock_event_source.aiter_sse = mock_aiter

        await source._receive_loop(mock_event_source, emit)

        assert len(received) == 1
        assert received[0].sender_id == "user1"

    async def test_receive_loop_tracks_event_id(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        events = [
            MockSSEEvent("message", json.dumps({"sender_id": "u1"}), "evt-1"),
            MockSSEEvent("message", json.dumps({"sender_id": "u2"}), "evt-2"),
            MockSSEEvent("message", json.dumps({"sender_id": "u3"}), "evt-3"),
        ]

        async def emit(msg):
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        mock_event_source = MagicMock()

        async def mock_aiter():
            for event in events:
                yield event

        mock_event_source.aiter_sse = mock_aiter

        await source._receive_loop(mock_event_source, emit)

        assert source.last_event_id == "evt-3"

    async def test_receive_loop_stops_when_signaled(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        events_processed = 0

        async def emit(msg):
            nonlocal events_processed
            events_processed += 1
            # Stop after first message
            if events_processed == 1:
                await source.stop()
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        mock_event_source = MagicMock()

        async def mock_aiter():
            for i in range(10):
                yield MockSSEEvent("message", json.dumps({"sender_id": f"u{i}"}))

        mock_event_source.aiter_sse = mock_aiter

        await source._receive_loop(mock_event_source, emit)

        # Should have stopped after first message
        assert events_processed == 1


class TestSSESourceHeaders:
    """Test header handling."""

    def test_builds_last_event_id_header(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
            headers={"Authorization": "Bearer token"},
            last_event_id="resume-id",
        )

        # The headers should be merged when start() is called
        # Test that the initial config is correct
        assert source._headers == {"Authorization": "Bearer token"}
        assert source._last_event_id == "resume-id"


class TestSSESourceErrorHandling:
    """Test error handling scenarios."""

    async def test_status_changes_to_error_on_exception(self) -> None:
        from roomkit.sources.sse import SSESource
        import roomkit.sources.sse as sse_module

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        original_httpx = sse_module.httpx
        original_aconnect = sse_module.aconnect_sse
        original_has_sse = sse_module.HAS_SSE

        # Mock to raise error
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client_ctx

        mock_sse_ctx = MagicMock()
        mock_sse_ctx.__aenter__ = AsyncMock(
            side_effect=ConnectionRefusedError("Connection refused")
        )
        mock_sse_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_aconnect = MagicMock(return_value=mock_sse_ctx)

        sse_module.httpx = mock_httpx
        sse_module.aconnect_sse = mock_aconnect
        sse_module.HAS_SSE = True

        try:

            async def emit(msg):
                from roomkit.models.delivery import InboundResult

                return InboundResult()

            with pytest.raises(ConnectionRefusedError):
                await source.start(emit)

            assert source.status == SourceStatus.ERROR
        finally:
            sse_module.httpx = original_httpx
            sse_module.aconnect_sse = original_aconnect
            sse_module.HAS_SSE = original_has_sse


# =============================================================================
# Test SSESource healthcheck
# =============================================================================


class TestSSESourceHealth:
    async def test_healthcheck_tracks_messages(self) -> None:
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        # Initial health
        health = await source.healthcheck()
        assert health.status == SourceStatus.STOPPED
        assert health.messages_received == 0

        # Simulate receiving messages
        source._set_status(SourceStatus.CONNECTED)
        source._record_message()
        source._record_message()

        health = await source.healthcheck()
        assert health.status == SourceStatus.CONNECTED
        assert health.messages_received == 2
        assert health.connected_at is not None
        assert health.last_message_at is not None


# =============================================================================
# Test SSESource import error handling
# =============================================================================


class TestSSESourceImportError:
    async def test_raises_import_error_when_deps_missing(self) -> None:
        import roomkit.sources.sse as sse_module
        from roomkit.sources.sse import SSESource

        source = SSESource(
            url="https://example.com/events",
            channel_id="test",
        )

        # Temporarily disable SSE deps
        original_has = sse_module.HAS_SSE
        sse_module.HAS_SSE = False

        try:

            async def emit(msg):
                from roomkit.models.delivery import InboundResult

                return InboundResult()

            with pytest.raises(ImportError, match="httpx and httpx-sse are required"):
                await source.start(emit)
        finally:
            sse_module.HAS_SSE = original_has


# =============================================================================
# Integration test with RoomKit
# =============================================================================


class TestSSESourceIntegration:
    async def test_attach_sse_source_to_roomkit(self) -> None:
        import roomkit.sources.sse as sse_module
        from roomkit.sources.sse import SSESource

        kit = RoomKit()
        source = SSESource(
            url="https://example.com/events",
            channel_id="sse-channel",
        )

        # Patch to avoid actual connection
        original_httpx = sse_module.httpx
        original_aconnect = sse_module.aconnect_sse

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client_ctx

        mock_sse_ctx = MagicMock()
        mock_sse_ctx.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError())
        mock_sse_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_aconnect = MagicMock(return_value=mock_sse_ctx)

        sse_module.httpx = mock_httpx
        sse_module.aconnect_sse = mock_aconnect

        try:
            await kit.attach_source("sse-channel", source, auto_restart=False)

            # Verify source is tracked
            assert "sse-channel" in kit._sources
            assert kit._sources["sse-channel"] is source

            # Wait for task to fail
            await asyncio.sleep(0.1)

            # Detach
            await kit.detach_source("sse-channel")

            assert "sse-channel" not in kit._sources
        finally:
            sse_module.httpx = original_httpx
            sse_module.aconnect_sse = original_aconnect

        await kit.close()
