"""Tests for WhatsApp Personal (neonize) event source."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import InboundMessage, TextContent
from roomkit.models.delivery import InboundResult
from roomkit.models.event import AudioContent, LocationContent, MediaContent
from roomkit.sources.base import SourceStatus

# =============================================================================
# Helpers
# =============================================================================


def _make_message_event(
    *,
    sender: str = "1234567890@s.whatsapp.net",
    chat: str = "1234567890@s.whatsapp.net",
    conversation: str | None = "hello",
    is_from_me: bool = False,
    msg_id: str = "msg-001",
    push_name: str = "Alice",
    image: bool = False,
    audio: bool = False,
    video: bool = False,
    document: bool = False,
    location: bool = False,
    sticker: bool = False,
    extended_text: str | None = None,
) -> MagicMock:
    """Build a fake neonize MessageEv."""
    event = MagicMock()
    info = MagicMock()
    info.IsFromMe = is_from_me
    info.Sender = sender
    info.Chat = chat
    info.Timestamp = 1700000000
    info.PushName = push_name
    info.ID = msg_id
    event.Info = info

    msg = MagicMock()
    has_media = image or audio or video or document or location or sticker or extended_text
    msg.conversation = conversation if not has_media else None

    # Extended text
    if extended_text:
        msg.extendedTextMessage = MagicMock()
        msg.extendedTextMessage.text = extended_text
    else:
        msg.extendedTextMessage = None

    # Image
    if image:
        msg.imageMessage = MagicMock()
        msg.imageMessage.mimetype = "image/jpeg"
        msg.imageMessage.caption = "nice pic"
    else:
        msg.imageMessage = None

    # Audio
    if audio:
        msg.audioMessage = MagicMock()
        msg.audioMessage.mimetype = "audio/ogg"
        msg.audioMessage.ptt = True
        msg.audioMessage.seconds = 5
    else:
        msg.audioMessage = None

    # Video
    if video:
        msg.videoMessage = MagicMock()
        msg.videoMessage.mimetype = "video/mp4"
        msg.videoMessage.seconds = 10
    else:
        msg.videoMessage = None

    # Document
    if document:
        msg.documentMessage = MagicMock()
        msg.documentMessage.mimetype = "application/pdf"
        msg.documentMessage.fileName = "report.pdf"
    else:
        msg.documentMessage = None

    # Location
    if location:
        msg.locationMessage = MagicMock()
        msg.locationMessage.degreesLatitude = 45.5
        msg.locationMessage.degreesLongitude = -73.5
        msg.locationMessage.name = "Montreal"
        msg.locationMessage.address = "123 Main St"
    else:
        msg.locationMessage = None

    # Sticker
    if sticker:
        msg.stickerMessage = MagicMock()
    else:
        msg.stickerMessage = None

    event.Message = msg
    return event


def _make_client_mock() -> AsyncMock:
    """Build a fake neonize client with download_any returning bytes."""
    client = AsyncMock()
    client.download_any = AsyncMock(return_value=b"\x00\x01\x02")
    client.send_message = AsyncMock()
    client.disconnect = MagicMock()
    return client


# =============================================================================
# Test default_message_parser
# =============================================================================


class TestDefaultMessageParser:
    async def test_parses_text_conversation(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(conversation="Hello world")

        msg = await parser(client, event)

        assert msg is not None
        assert msg.channel_id == "wa-test"
        assert msg.sender_id == "1234567890"
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Hello world"

    async def test_parses_extended_text(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(extended_text="Extended hello")

        msg = await parser(client, event)

        assert msg is not None
        assert isinstance(msg.content, TextContent)
        assert msg.content.body == "Extended hello"

    async def test_parses_image(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(image=True)

        msg = await parser(client, event)

        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.mime_type == "image/jpeg"
        assert msg.content.url.startswith("data:image/jpeg;base64,")
        assert msg.content.caption == "nice pic"

    async def test_parses_audio_voice_note(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(audio=True)

        msg = await parser(client, event)

        assert msg is not None
        assert isinstance(msg.content, AudioContent)
        assert msg.content.mime_type == "audio/ogg"
        assert msg.content.duration_seconds == 5.0

    async def test_parses_video(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(video=True)

        msg = await parser(client, event)

        assert msg is not None
        # Video stored as MediaContent because VideoContent rejects data: URIs
        assert isinstance(msg.content, MediaContent)
        assert msg.content.mime_type == "video/mp4"

    async def test_parses_document(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(document=True)

        msg = await parser(client, event)

        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.mime_type == "application/pdf"
        assert msg.content.filename == "report.pdf"

    async def test_parses_location(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(location=True)

        msg = await parser(client, event)

        assert msg is not None
        assert isinstance(msg.content, LocationContent)
        assert msg.content.latitude == 45.5
        assert msg.content.longitude == -73.5
        assert msg.content.label == "Montreal"
        assert msg.content.address == "123 Main St"

    async def test_parses_sticker(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(sticker=True)

        msg = await parser(client, event)

        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.mime_type == "image/webp"

    async def test_skips_own_messages(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(is_from_me=True)

        msg = await parser(client, event)

        assert msg is None

    async def test_skips_missing_sender(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(sender="")

        msg = await parser(client, event)

        assert msg is None

    async def test_metadata_extraction(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        event = _make_message_event(
            sender="5551234567@s.whatsapp.net",
            chat="group-id@g.us",
            push_name="Bob",
        )

        msg = await parser(client, event)

        assert msg is not None
        assert msg.metadata["raw_jid"] == "5551234567@s.whatsapp.net"
        assert msg.metadata["is_group"] is True
        assert msg.metadata["push_name"] == "Bob"
        assert msg.external_id == "msg-001"

    async def test_image_download_failure_uses_placeholder(self) -> None:
        from roomkit.sources.neonize import default_message_parser

        parser = default_message_parser("wa-test")
        client = _make_client_mock()
        client.download_any = AsyncMock(side_effect=Exception("download failed"))
        event = _make_message_event(image=True)

        msg = await parser(client, event)

        assert msg is not None
        assert isinstance(msg.content, MediaContent)
        assert msg.content.url == "data:image/jpeg;base64,"


# =============================================================================
# Test WhatsAppPersonalSourceProvider initialization
# =============================================================================


class TestSourceInit:
    def test_default_initialization(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        assert source.name == "neonize:whatsapp-session.db"
        assert source.status == SourceStatus.STOPPED
        assert source._db == "whatsapp-session.db"
        assert source._channel_id == "whatsapp-personal"
        assert source._device_name == "RoomKit"

    def test_custom_config(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider(
            db="custom.db",
            channel_id="my-wa",
            device_name="MyBot",
        )

        assert source.name == "neonize:custom.db"
        assert source._channel_id == "my-wa"
        assert source._device_name == "MyBot"

    def test_custom_parser(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        async def my_parser(client: object, event: object) -> InboundMessage | None:
            return InboundMessage(
                channel_id="custom",
                sender_id="custom",
                content=TextContent(body="custom"),
            )

        source = WhatsAppPersonalSourceProvider(parser=my_parser)

        assert source._parser is my_parser

    def test_custom_on_event(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        async def handler(event_type: str, data: dict) -> None:
            pass

        source = WhatsAppPersonalSourceProvider(on_event=handler)

        assert source._on_event is handler


# =============================================================================
# Test event callbacks
# =============================================================================


class TestEventCallbacks:
    async def test_fire_event_calls_on_event(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        received: list[tuple[str, dict]] = []

        async def handler(event_type: str, data: dict) -> None:
            received.append((event_type, data))

        source = WhatsAppPersonalSourceProvider(on_event=handler)
        await source._fire_event("qr", {"codes": ["code1"]})

        assert len(received) == 1
        assert received[0] == ("qr", {"codes": ["code1"]})

    async def test_fire_event_handles_sync_callback(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        received: list[tuple[str, dict]] = []

        def handler(event_type: str, data: dict) -> None:
            received.append((event_type, data))

        source = WhatsAppPersonalSourceProvider(on_event=handler)
        await source._fire_event("connected", {})

        assert len(received) == 1

    async def test_fire_event_noop_when_no_handler(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()
        # Should not raise
        await source._fire_event("qr", {"codes": []})

    async def test_fire_event_multiple_types(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        received: list[str] = []

        async def handler(event_type: str, data: dict) -> None:
            received.append(event_type)

        source = WhatsAppPersonalSourceProvider(on_event=handler)
        await source._fire_event("authenticated", {"jid": "1234@s.whatsapp.net"})
        await source._fire_event("connected", {})
        await source._fire_event("receipt", {"type": "read"})

        assert received == ["authenticated", "connected", "receipt"]


# =============================================================================
# Test receive loop / message emission
# =============================================================================


class TestReceiveLoop:
    async def test_start_raises_import_error_when_neonize_missing(self) -> None:
        import roomkit.sources.neonize as nz_module
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        original = nz_module.HAS_NEONIZE
        nz_module.HAS_NEONIZE = False

        try:

            async def emit(msg: InboundMessage) -> InboundResult:
                return InboundResult()

            with pytest.raises(ImportError, match="neonize is required"):
                await source.start(emit)
        finally:
            nz_module.HAS_NEONIZE = original

    async def test_messages_emitted_via_emit(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()
        received: list[InboundMessage] = []

        async def emit(msg: InboundMessage) -> InboundResult:
            received.append(msg)
            return InboundResult()

        import roomkit.sources.neonize as nz_module

        original_has = nz_module.HAS_NEONIZE
        original_client_cls = nz_module.NewAClient
        nz_module.HAS_NEONIZE = True

        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = MagicMock()

        # Capture event handlers registered via @client.event(EventClass)
        handlers: dict[object, object] = {}

        def mock_event(event_cls: object) -> object:
            def decorator(fn: object) -> object:
                handlers[event_cls] = fn
                return fn

            return decorator

        mock_client.event = mock_event
        nz_module.NewAClient = MagicMock(return_value=mock_client)  # type: ignore[assignment]

        # Create fake event classes that the local import will resolve to
        class FakeQREv:
            pass

        class FakePairStatusEv:
            pass

        class FakeConnectedEv:
            pass

        class FakeLoggedOutEv:
            pass

        class FakeDisconnectedEv:
            pass

        class FakeMessageEv:
            pass

        class FakeReceiptEv:
            pass

        class FakeChatPresenceEv:
            pass

        import sys

        fake_module = MagicMock()
        fake_module.QREv = FakeQREv
        fake_module.PairStatusEv = FakePairStatusEv
        fake_module.ConnectedEv = FakeConnectedEv
        fake_module.LoggedOutEv = FakeLoggedOutEv
        fake_module.DisconnectedEv = FakeDisconnectedEv
        fake_module.MessageEv = FakeMessageEv
        fake_module.ReceiptEv = FakeReceiptEv
        fake_module.ChatPresenceEv = FakeChatPresenceEv

        sys.modules["neonize"] = MagicMock()
        sys.modules["neonize.aioze"] = MagicMock()
        sys.modules["neonize.aioze.client"] = MagicMock(NewAClient=nz_module.NewAClient)
        sys.modules["neonize.aioze.events"] = fake_module

        try:
            task = asyncio.create_task(source.start(emit))
            await asyncio.sleep(0.05)

            # Find the MessageEv handler
            msg_handler = handlers.get(FakeMessageEv)
            if msg_handler is not None:
                msg_event = _make_message_event(conversation="Test message")
                await msg_handler(mock_client, msg_event)  # type: ignore[misc]

            await asyncio.sleep(0.05)
            await source.stop()

            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            assert len(received) == 1
            assert received[0].content.body == "Test message"  # type: ignore[union-attr]
        finally:
            nz_module.HAS_NEONIZE = original_has
            nz_module.NewAClient = original_client_cls  # type: ignore[assignment]
            for key in list(sys.modules):
                if key.startswith("neonize"):
                    del sys.modules[key]

    async def test_stop_signal_breaks_loop(self) -> None:
        import sys

        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        import roomkit.sources.neonize as nz_module

        original_has = nz_module.HAS_NEONIZE
        original_client_cls = nz_module.NewAClient
        nz_module.HAS_NEONIZE = True

        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.disconnect = MagicMock()
        mock_client.event = lambda cls: lambda fn: fn
        nz_module.NewAClient = MagicMock(return_value=mock_client)  # type: ignore[assignment]

        # Mock neonize events module
        fake_events = MagicMock()
        sys.modules["neonize"] = MagicMock()
        sys.modules["neonize.aioze"] = MagicMock()
        sys.modules["neonize.aioze.client"] = MagicMock(NewAClient=nz_module.NewAClient)
        sys.modules["neonize.aioze.events"] = fake_events

        try:

            async def emit(msg: InboundMessage) -> InboundResult:
                return InboundResult()

            task = asyncio.create_task(source.start(emit))
            await asyncio.sleep(0.05)

            assert source.status in (SourceStatus.CONNECTING, SourceStatus.CONNECTED)

            await source.stop()
            await asyncio.sleep(1.5)

            assert task.done() or source._should_stop()
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            nz_module.HAS_NEONIZE = original_has
            nz_module.NewAClient = original_client_cls  # type: ignore[assignment]
            for key in list(sys.modules):
                if key.startswith("neonize"):
                    del sys.modules[key]

    async def test_message_counting(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()
        source._set_status(SourceStatus.CONNECTED)

        assert source._messages_received == 0

        source._record_message()
        source._record_message()
        source._record_message()

        assert source._messages_received == 3

        health = await source.healthcheck()
        assert health.messages_received == 3
        assert health.last_message_at is not None


# =============================================================================
# Test error handling
# =============================================================================


class TestErrorHandling:
    async def test_connection_error_sets_error_status(self) -> None:
        import sys

        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        import roomkit.sources.neonize as nz_module

        original_has = nz_module.HAS_NEONIZE
        original_client_cls = nz_module.NewAClient
        nz_module.HAS_NEONIZE = True

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(side_effect=ConnectionError("Connection refused"))
        mock_client.disconnect = MagicMock()
        mock_client.event = lambda cls: lambda fn: fn
        nz_module.NewAClient = MagicMock(return_value=mock_client)  # type: ignore[assignment]

        # Mock the neonize events module so the local import succeeds
        fake_events = MagicMock()
        sys.modules["neonize"] = MagicMock()
        sys.modules["neonize.aioze"] = MagicMock()
        sys.modules["neonize.aioze.client"] = MagicMock(NewAClient=nz_module.NewAClient)
        sys.modules["neonize.aioze.events"] = fake_events

        try:

            async def emit(msg: InboundMessage) -> InboundResult:
                return InboundResult()

            with pytest.raises(ConnectionError):
                await source.start(emit)

            assert source.status == SourceStatus.ERROR
        finally:
            nz_module.HAS_NEONIZE = original_has
            nz_module.NewAClient = original_client_cls  # type: ignore[assignment]
            for key in list(sys.modules):
                if key.startswith("neonize"):
                    del sys.modules[key]

    async def test_status_transitions(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        assert source.status == SourceStatus.STOPPED

        source._set_status(SourceStatus.CONNECTING)
        assert source.status == SourceStatus.CONNECTING

        source._set_status(SourceStatus.CONNECTED)
        assert source.status == SourceStatus.CONNECTED
        assert source._connected_at is not None

        source._set_status(SourceStatus.RECONNECTING)
        assert source.status == SourceStatus.RECONNECTING

        source._set_status(SourceStatus.ERROR, "test error")
        assert source.status == SourceStatus.ERROR
        assert source._error == "test error"


# =============================================================================
# Test health
# =============================================================================


class TestHealth:
    async def test_initial_health(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        health = await source.healthcheck()
        assert health.status == SourceStatus.STOPPED
        assert health.messages_received == 0
        assert health.connected_at is None
        assert health.last_message_at is None
        assert health.error is None

    async def test_connected_health(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        source._set_status(SourceStatus.CONNECTED)
        source._record_message()

        health = await source.healthcheck()
        assert health.status == SourceStatus.CONNECTED
        assert health.messages_received == 1
        assert health.connected_at is not None
        assert health.last_message_at is not None


# =============================================================================
# Test send convenience method
# =============================================================================


class TestSendMethod:
    async def test_send_raises_when_not_connected(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        with pytest.raises(RuntimeError, match="not connected"):
            await source.send("1234567890@s.whatsapp.net", "Hello")

    async def test_send_delegates_to_client(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()
        mock_client = AsyncMock()
        mock_client.send_message = AsyncMock()
        source._client = mock_client
        source._set_status(SourceStatus.CONNECTED)

        await source.send("1234567890@s.whatsapp.net", "Hello")

        mock_client.send_message.assert_called_once_with(
            "1234567890@s.whatsapp.net",
            "Hello",
        )


# =============================================================================
# Test client property
# =============================================================================


class TestClientProperty:
    def test_client_initially_none(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()

        assert source.client is None

    def test_client_set_after_mock_init(self) -> None:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        source = WhatsAppPersonalSourceProvider()
        mock_client = MagicMock()
        source._client = mock_client

        assert source.client is mock_client
