"""Tests for observability: named loggers, structured fields, framework events."""

from __future__ import annotations

import logging

from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult


class TestNamedLoggers:
    def test_framework_logger_exists(self) -> None:
        logger = logging.getLogger("roomkit.framework")
        assert logger.name == "roomkit.framework"

    def test_hooks_logger_exists(self) -> None:
        logger = logging.getLogger("roomkit.hooks")
        assert logger.name == "roomkit.hooks"

    def test_event_router_logger_exists(self) -> None:
        logger = logging.getLogger("roomkit.event_router")
        assert logger.name == "roomkit.event_router"

    def test_channel_loggers_exist(self) -> None:
        for name in [
            "roomkit.channels.sms",
            "roomkit.channels.email",
            "roomkit.channels.ai",
            "roomkit.channels.websocket",
            "roomkit.channels.whatsapp",
        ]:
            logger = logging.getLogger(name)
            assert logger.name == name


class TestFrameworkEventEmission:
    async def test_event_processed_emitted(self) -> None:
        """Framework emits event_processed on successful inbound."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        received: list[FrameworkEvent] = []

        @kit.on("event_processed")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        await kit.process_inbound(msg)
        assert len(received) == 1
        assert received[0].type == "event_processed"
        assert received[0].room_id == "r1"

    async def test_event_blocked_emitted(self) -> None:
        """Framework emits event_blocked when hook blocks."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        received: list[FrameworkEvent] = []

        @kit.on("event_blocked")
        async def handler(fe: FrameworkEvent) -> None:
            received.append(fe)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="blocker")
        async def blocker(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("spam")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="spam"),
        )
        await kit.process_inbound(msg)
        assert len(received) == 1
        assert received[0].type == "event_blocked"
        assert received[0].data.get("reason") == "spam"


class TestStructuredLogging:
    async def test_hook_error_logged(self, caplog: logging.LogRecord) -> None:
        """Hook errors are logged with structured fields."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="failing_hook")
        async def failing(event: RoomEvent, ctx: RoomContext) -> HookResult:
            raise RuntimeError("hook error")

        with caplog.at_level(logging.ERROR, logger="roomkit.hooks"):
            msg = InboundMessage(
                channel_id="ws1",
                sender_id="user1",
                content=TextContent(body="test"),
            )
            await kit.process_inbound(msg)

        assert any("failing_hook" in r.message for r in caplog.records)

    async def test_framework_event_handler_error_logged(self, caplog: logging.LogRecord) -> None:
        """Failed framework event handlers are logged."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        @kit.on("event_processed")
        async def bad_handler(fe: FrameworkEvent) -> None:
            raise RuntimeError("handler error")

        with caplog.at_level(logging.ERROR, logger="roomkit.framework"):
            msg = InboundMessage(
                channel_id="ws1",
                sender_id="user1",
                content=TextContent(body="test"),
            )
            await kit.process_inbound(msg)

        assert any("handler failed" in r.message for r in caplog.records)
