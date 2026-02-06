"""Tests for WhatsApp Personal provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from roomkit.models.enums import ChannelType
from roomkit.models.event import (
    AudioContent,
    EventSource,
    LocationContent,
    MediaContent,
    RoomEvent,
    SystemContent,
    TextContent,
    VideoContent,
)
from roomkit.providers.whatsapp.personal import WhatsAppPersonalProvider, _build_jid
from roomkit.sources.base import SourceStatus

# =============================================================================
# Helpers
# =============================================================================


def _make_source(*, connected: bool = True) -> MagicMock:
    """Build a mock WhatsAppPersonalSourceProvider."""
    source = MagicMock()
    source.status = SourceStatus.CONNECTED if connected else SourceStatus.STOPPED
    client = AsyncMock()
    client.send_message = AsyncMock()
    client.send_image = AsyncMock()
    client.send_document = AsyncMock()
    client.send_audio = AsyncMock()
    client.send_video = AsyncMock()
    client.send_location = AsyncMock()
    source.client = client
    return source


def _make_event(content: object) -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(
            channel_id="wa-personal",
            channel_type=ChannelType.WHATSAPP_PERSONAL,
        ),
        content=content,  # type: ignore[arg-type]
    )


# =============================================================================
# Test JID building
# =============================================================================


class TestBuildJid:
    def test_plain_phone(self) -> None:
        assert _build_jid("1234567890") == "1234567890@s.whatsapp.net"

    def test_phone_with_plus(self) -> None:
        assert _build_jid("+1234567890") == "1234567890@s.whatsapp.net"

    def test_already_jid(self) -> None:
        assert _build_jid("1234567890@s.whatsapp.net") == "1234567890@s.whatsapp.net"


# =============================================================================
# Test provider
# =============================================================================


class TestWhatsAppPersonalProvider:
    def test_constructor_stores_source(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        assert provider._source is source

    def test_name_property(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        assert provider.name == "whatsapp-personal"

    async def test_send_text_content(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(TextContent(body="Hello"))

        result = await provider.send(event, "+1234567890")

        assert result.success is True
        assert result.provider_message_id is not None
        source.client.send_message.assert_called_once_with(
            "1234567890@s.whatsapp.net",
            "Hello",
        )

    async def test_send_media_content_image(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(
            MediaContent(
                url="https://example.com/photo.jpg",
                mime_type="image/jpeg",
                caption="A photo",
            )
        )

        result = await provider.send(event, "1234567890")

        assert result.success is True
        source.client.send_image.assert_called_once_with(
            "1234567890@s.whatsapp.net",
            "https://example.com/photo.jpg",
            caption="A photo",
        )

    async def test_send_media_content_document(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(
            MediaContent(
                url="https://example.com/report.pdf",
                mime_type="application/pdf",
                filename="report.pdf",
            )
        )

        result = await provider.send(event, "1234567890")

        assert result.success is True
        source.client.send_document.assert_called_once_with(
            "1234567890@s.whatsapp.net",
            "https://example.com/report.pdf",
            filename="report.pdf",
        )

    async def test_send_audio_content(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(
            AudioContent(url="https://example.com/voice.ogg", mime_type="audio/ogg")
        )

        result = await provider.send(event, "1234567890")

        assert result.success is True
        source.client.send_audio.assert_called_once_with(
            "1234567890@s.whatsapp.net",
            "https://example.com/voice.ogg",
            ptt=True,
        )

    async def test_send_video_content(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(VideoContent(url="https://example.com/video.mp4"))

        result = await provider.send(event, "1234567890")

        assert result.success is True
        source.client.send_video.assert_called_once_with(
            "1234567890@s.whatsapp.net",
            "https://example.com/video.mp4",
        )

    async def test_send_location_content(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(LocationContent(latitude=45.5, longitude=-73.5, label="Montreal"))

        result = await provider.send(event, "1234567890")

        assert result.success is True
        source.client.send_location.assert_called_once_with(
            "1234567890@s.whatsapp.net",
            45.5,
            -73.5,
            name="Montreal",
        )

    async def test_send_when_disconnected(self) -> None:
        source = _make_source(connected=False)
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(TextContent(body="Hello"))

        result = await provider.send(event, "1234567890")

        assert result.success is False
        assert "not connected" in (result.error or "")

    async def test_send_unsupported_content(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(SystemContent(body="system message"))

        result = await provider.send(event, "1234567890")

        assert result.success is False
        assert "Unsupported" in (result.error or "")

    async def test_send_handles_exception(self) -> None:
        source = _make_source()
        source.client.send_message = AsyncMock(side_effect=Exception("network error"))
        provider = WhatsAppPersonalProvider(source)
        event = _make_event(TextContent(body="Hello"))

        result = await provider.send(event, "1234567890")

        assert result.success is False
        assert "network error" in (result.error or "")

    async def test_close_is_noop(self) -> None:
        source = _make_source()
        provider = WhatsAppPersonalProvider(source)
        # Should not raise
        await provider.close()
