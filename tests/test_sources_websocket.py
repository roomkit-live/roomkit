"""Tests for WebSocket event source."""

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
    def test_parses_valid_json_message(self) -> None:
        from roomkit.sources.websocket import default_json_parser

        parser = default_json_parser("test-channel")

        raw = json.dumps(
            {
                "sender_id": "user123",
                "text": "Hello world",
                "external_id": "msg-456",
                "metadata": {"key": "value"},
            }
        )

        msg = parser(raw)

        assert msg is not None
        assert msg.channel_id == "test-channel"
        assert msg.sender_id == "user123"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello world"
        assert msg.external_id == "msg-456"
        assert msg.metadata == {"key": "value"}

    def test_parses_minimal_message(self) -> None:
        from roomkit.sources.websocket import default_json_parser

        parser = default_json_parser("ch1")
        raw = json.dumps({"sender_id": "user1"})

        msg = parser(raw)

        assert msg is not None
        assert msg.sender_id == "user1"
        assert msg.content.body == ""
        assert msg.external_id is None
        assert msg.metadata == {}

    def test_parses_bytes(self) -> None:
        from roomkit.sources.websocket import default_json_parser

        parser = default_json_parser("ch1")
        raw = json.dumps({"sender_id": "user1", "text": "hello"}).encode("utf-8")

        msg = parser(raw)

        assert msg is not None
        assert msg.content.body == "hello"

    def test_returns_none_for_missing_sender_id(self) -> None:
        from roomkit.sources.websocket import default_json_parser

        parser = default_json_parser("ch1")
        raw = json.dumps({"text": "no sender"})

        msg = parser(raw)

        assert msg is None

    def test_returns_none_for_invalid_json(self) -> None:
        from roomkit.sources.websocket import default_json_parser

        parser = default_json_parser("ch1")

        msg = parser("not valid json {")

        assert msg is None

    def test_returns_none_for_non_dict(self) -> None:
        from roomkit.sources.websocket import default_json_parser

        parser = default_json_parser("ch1")

        assert parser(json.dumps([1, 2, 3])) is None
        assert parser(json.dumps("string")) is None
        assert parser(json.dumps(123)) is None


# =============================================================================
# Test WebSocketSource initialization
# =============================================================================


class TestWebSocketSourceInit:
    def test_default_initialization(self) -> None:
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        assert source.name == "websocket:wss://example.com/ws"
        assert source.status == SourceStatus.STOPPED
        assert source._url == "wss://example.com/ws"
        assert source._channel_id == "test"

    def test_custom_parser(self) -> None:
        from roomkit.sources.websocket import WebSocketSource

        def custom_parser(raw: str) -> InboundMessage | None:
            return InboundMessage(
                channel_id="custom",
                sender_id="custom-sender",
                content=TextContent(body=raw),
            )

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
            parser=custom_parser,
        )

        # Test the parser is used
        msg = source._parser("hello")
        assert msg is not None
        assert msg.sender_id == "custom-sender"
        assert msg.content.body == "hello"

    def test_custom_headers(self) -> None:
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
            headers={"Authorization": "Bearer token123"},
        )

        assert source._headers == {"Authorization": "Bearer token123"}

    def test_all_options(self) -> None:
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
            headers={"X-Custom": "value"},
            subprotocols=["graphql-ws"],
            ping_interval=30.0,
            ping_timeout=10.0,
            close_timeout=5.0,
            max_size=1024 * 1024,
            origin="https://example.com",
        )

        assert source._subprotocols == ["graphql-ws"]
        assert source._ping_interval == 30.0
        assert source._ping_timeout == 10.0
        assert source._close_timeout == 5.0
        assert source._max_size == 1024 * 1024
        assert source._origin == "https://example.com"


# =============================================================================
# Test WebSocketSource with mocked websockets
# =============================================================================


class TestWebSocketSourceWithMock:
    @pytest.fixture
    def mock_websockets(self):
        """Create a mock websockets module."""
        import roomkit.sources.websocket as ws_module

        original = ws_module.websockets
        mock = MagicMock()
        ws_module.websockets = mock
        yield mock
        ws_module.websockets = original

    async def test_connects_and_receives_messages(self, mock_websockets) -> None:
        from roomkit.sources.websocket import WebSocketSource

        # Setup mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"sender_id": "user1", "text": "Hello"}),
                json.dumps({"sender_id": "user2", "text": "World"}),
                TimeoutError(),  # Triggers stop check
                asyncio.CancelledError(),  # Exit
            ]
        )
        mock_ws.close = AsyncMock()

        # Setup connect context manager
        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=None)
        mock_websockets.connect.return_value = mock_connect

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        received: list[InboundMessage] = []

        async def emit(msg: InboundMessage):
            received.append(msg)
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        # Run in background and stop after receiving messages
        task = asyncio.create_task(source.start(emit))

        # Wait for messages to be processed
        await asyncio.sleep(0.1)

        # Stop the source
        await source.stop()

        # Cancel the task if still running
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Verify messages were received
        assert len(received) == 2
        assert received[0].sender_id == "user1"
        assert received[0].content.body == "Hello"
        assert received[1].sender_id == "user2"
        assert received[1].content.body == "World"

    async def test_status_transitions(self, mock_websockets) -> None:
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        statuses: list[SourceStatus] = []
        statuses.append(source.status)  # Initial: STOPPED

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[asyncio.CancelledError()])
        mock_ws.close = AsyncMock()

        mock_connect = MagicMock()

        async def enter_connect(*args, **kwargs):
            statuses.append(source.status)  # Should be CONNECTING
            return mock_ws

        mock_connect.__aenter__ = enter_connect
        mock_connect.__aexit__ = AsyncMock(return_value=None)
        mock_websockets.connect.return_value = mock_connect

        async def emit(msg):
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        with contextlib.suppress(asyncio.CancelledError):
            await source.start(emit)

        # Verify status transitions
        assert statuses[0] == SourceStatus.STOPPED
        assert statuses[1] == SourceStatus.CONNECTING

    async def test_skips_unparseable_messages(self, mock_websockets) -> None:
        from roomkit.sources.websocket import WebSocketSource

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            side_effect=[
                "not json",  # Unparseable
                json.dumps({"no_sender": True}),  # Missing sender_id
                json.dumps({"sender_id": "user1", "text": "Valid"}),  # Valid
                asyncio.CancelledError(),
            ]
        )
        mock_ws.close = AsyncMock()

        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=None)
        mock_websockets.connect.return_value = mock_connect

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        received: list[InboundMessage] = []

        async def emit(msg: InboundMessage):
            received.append(msg)
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        with contextlib.suppress(asyncio.CancelledError):
            await source.start(emit)

        # Only the valid message should be received
        assert len(received) == 1
        assert received[0].sender_id == "user1"

    async def test_handles_connection_error(self, mock_websockets) -> None:
        from roomkit.sources.websocket import WebSocketSource

        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(
            side_effect=ConnectionRefusedError("Connection refused")
        )
        mock_connect.__aexit__ = AsyncMock(return_value=None)
        mock_websockets.connect.return_value = mock_connect

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        async def emit(msg):
            from roomkit.models.delivery import InboundResult

            return InboundResult()

        with pytest.raises(ConnectionRefusedError):
            await source.start(emit)

        assert source.status == SourceStatus.ERROR

    async def test_send_method(self, mock_websockets) -> None:
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        # Manually set up the connected state for this unit test
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        source._ws = mock_ws
        source._set_status(SourceStatus.CONNECTED)

        # Send a message
        await source.send("Hello server")
        mock_ws.send.assert_called_once_with("Hello server")

        # Send bytes
        await source.send(b"Binary data")
        mock_ws.send.assert_called_with(b"Binary data")

    async def test_send_raises_when_not_connected(self) -> None:
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        with pytest.raises(RuntimeError, match="not connected"):
            await source.send("Hello")


# =============================================================================
# Test WebSocketSource healthcheck
# =============================================================================


class TestWebSocketSourceHealth:
    async def test_healthcheck_tracks_messages(self) -> None:
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
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
# Test WebSocketSource import error handling
# =============================================================================


class TestWebSocketSourceImportError:
    async def test_raises_import_error_when_websockets_missing(self) -> None:
        import roomkit.sources.websocket as ws_module
        from roomkit.sources.websocket import WebSocketSource

        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="test",
        )

        # Temporarily disable websockets
        original_has = ws_module.HAS_WEBSOCKETS
        ws_module.HAS_WEBSOCKETS = False

        try:

            async def emit(msg):
                from roomkit.models.delivery import InboundResult

                return InboundResult()

            with pytest.raises(ImportError, match="websockets is required"):
                await source.start(emit)
        finally:
            ws_module.HAS_WEBSOCKETS = original_has


# =============================================================================
# Integration test with RoomKit
# =============================================================================


class TestWebSocketSourceIntegration:
    async def test_attach_websocket_source_to_roomkit(self) -> None:
        import roomkit.sources.websocket as ws_module
        from roomkit.sources.websocket import WebSocketSource

        kit = RoomKit()
        source = WebSocketSource(
            url="wss://example.com/ws",
            channel_id="ws-channel",
        )

        # We can't actually connect without a real server,
        # but we can verify the attach/detach flow works

        # Patch websockets to avoid actual connection
        original = ws_module.websockets
        mock_ws = MagicMock()
        mock_connect = MagicMock()
        mock_connect.__aenter__ = AsyncMock(side_effect=asyncio.CancelledError())
        mock_connect.__aexit__ = AsyncMock(return_value=None)
        mock_ws.connect.return_value = mock_connect
        ws_module.websockets = mock_ws

        try:
            await kit.attach_source("ws-channel", source, auto_restart=False)

            # Verify source is tracked
            assert "ws-channel" in kit._sources
            assert kit._sources["ws-channel"] is source

            # Wait a bit for the task to fail
            await asyncio.sleep(0.1)

            # Detach
            await kit.detach_source("ws-channel")

            assert "ws-channel" not in kit._sources
        finally:
            ws_module.websockets = original

        await kit.close()
