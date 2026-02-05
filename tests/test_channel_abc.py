"""Tests for abstract base classes and DefaultContentTranscoder."""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.core.router import ContentTranscoder
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.identity.base import IdentityResolver
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import (
    LocationContent,
    MediaContent,
    RichContent,
    TextContent,
)
from roomkit.providers.ai.base import AIProvider
from roomkit.providers.email.base import EmailProvider
from roomkit.providers.sms.base import SMSProvider
from roomkit.providers.whatsapp.base import WhatsAppProvider
from roomkit.store.base import ConversationStore


class TestABCsNotInstantiable:
    def test_channel_abc(self) -> None:
        with pytest.raises(TypeError):
            Channel("ch1")  # type: ignore[abstract]

    def test_store_abc(self) -> None:
        with pytest.raises(TypeError):
            ConversationStore()  # type: ignore[abstract]

    def test_identity_resolver_abc(self) -> None:
        with pytest.raises(TypeError):
            IdentityResolver()  # type: ignore[abstract]

    def test_content_transcoder_abc(self) -> None:
        with pytest.raises(TypeError):
            ContentTranscoder()  # type: ignore[abstract]

    def test_sms_provider_abc(self) -> None:
        with pytest.raises(TypeError):
            SMSProvider()  # type: ignore[abstract]

    def test_email_provider_abc(self) -> None:
        with pytest.raises(TypeError):
            EmailProvider()  # type: ignore[abstract]

    def test_ai_provider_abc(self) -> None:
        with pytest.raises(TypeError):
            AIProvider()  # type: ignore[abstract]

    def test_whatsapp_provider_abc(self) -> None:
        with pytest.raises(TypeError):
            WhatsAppProvider()  # type: ignore[abstract]


def _make_binding(
    channel_id: str = "ch1",
    media_types: list[ChannelMediaType] | None = None,
) -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id="r1",
        channel_type=ChannelType.SMS,
        capabilities=ChannelCapabilities(media_types=media_types or [ChannelMediaType.TEXT]),
    )


class TestDefaultContentTranscoder:
    async def test_text_passthrough(self) -> None:
        t = DefaultContentTranscoder()
        src = _make_binding("src")
        tgt = _make_binding("tgt")
        content = TextContent(body="hello")
        result = await t.transcode(content, src, tgt)
        assert isinstance(result, TextContent)
        assert result.body == "hello"

    async def test_rich_to_text_fallback(self) -> None:
        t = DefaultContentTranscoder()
        src = _make_binding("src", [ChannelMediaType.RICH])
        tgt = _make_binding("tgt", [ChannelMediaType.TEXT])
        content = RichContent(body="**bold**", plain_text="bold")
        result = await t.transcode(content, src, tgt)
        assert isinstance(result, TextContent)
        assert result.body == "bold"

    async def test_rich_passthrough_when_supported(self) -> None:
        t = DefaultContentTranscoder()
        src = _make_binding("src", [ChannelMediaType.RICH])
        tgt = _make_binding("tgt", [ChannelMediaType.RICH])
        content = RichContent(body="**bold**")
        result = await t.transcode(content, src, tgt)
        assert isinstance(result, RichContent)

    async def test_media_to_text_fallback(self) -> None:
        t = DefaultContentTranscoder()
        src = _make_binding("src", [ChannelMediaType.MEDIA])
        tgt = _make_binding("tgt", [ChannelMediaType.TEXT])
        content = MediaContent(
            url="https://example.com/photo.jpg",
            mime_type="image/jpeg",
            caption="My photo",
        )
        result = await t.transcode(content, src, tgt)
        assert isinstance(result, TextContent)
        assert "My photo" in result.body

    async def test_location_to_text_fallback(self) -> None:
        t = DefaultContentTranscoder()
        src = _make_binding("src", [ChannelMediaType.LOCATION])
        tgt = _make_binding("tgt", [ChannelMediaType.TEXT])
        content = LocationContent(latitude=45.5, longitude=-73.6, label="Montreal")
        result = await t.transcode(content, src, tgt)
        assert isinstance(result, TextContent)
        assert "Montreal" in result.body
        assert "45.5" in result.body


class TestChannelInfoProperty:
    def test_sms_channel_info(self) -> None:
        from roomkit.channels import SMSChannel

        ch = SMSChannel("sms1", from_number="+15551234567")
        assert ch.info == {"from_": "+15551234567"}

    def test_email_channel_info(self) -> None:
        from roomkit.channels import EmailChannel

        ch = EmailChannel("email1", from_address="bot@example.com")
        assert ch.info == {"from_": "bot@example.com"}

    def test_websocket_channel_info(self) -> None:
        from roomkit.channels.websocket import WebSocketChannel

        ch = WebSocketChannel("ws1")
        assert ch.info == {"connection_count": 0}

    def test_ai_channel_info(self) -> None:
        from roomkit.channels.ai import AIChannel
        from roomkit.providers.ai.mock import MockAIProvider

        provider = MockAIProvider(responses=["hi"])
        ch = AIChannel("ai1", provider=provider)
        assert ch.info["provider"] == "MockAIProvider"

    def test_whatsapp_channel_info(self) -> None:
        from roomkit.channels import WhatsAppChannel

        ch = WhatsAppChannel("wa1")
        assert isinstance(ch.info, dict)


class TestChannelCapabilities:
    def test_sms_capabilities(self) -> None:
        from roomkit.channels import SMSChannel

        ch = SMSChannel("sms1")
        caps = ch.capabilities()
        assert ChannelMediaType.TEXT in caps.media_types
        assert caps.max_length == 1600

    def test_email_capabilities(self) -> None:
        from roomkit.channels import EmailChannel

        ch = EmailChannel("email1")
        caps = ch.capabilities()
        assert caps.supports_rich_text is True
        assert caps.supports_media is True

    def test_websocket_capabilities(self) -> None:
        from roomkit.channels.websocket import WebSocketChannel

        ch = WebSocketChannel("ws1")
        caps = ch.capabilities()
        assert caps.supports_rich_text is True
        assert caps.supports_buttons is True
        assert caps.supports_cards is True
        assert caps.supports_quick_replies is True

    def test_ai_capabilities(self) -> None:
        from roomkit.channels.ai import AIChannel
        from roomkit.providers.ai.mock import MockAIProvider

        provider = MockAIProvider(responses=["hi"])
        ch = AIChannel("ai1", provider=provider)
        caps = ch.capabilities()
        assert caps.supports_rich_text is True

    def test_whatsapp_capabilities(self) -> None:
        from roomkit.channels import WhatsAppChannel

        ch = WhatsAppChannel("wa1")
        caps = ch.capabilities()
        assert caps.supports_rich_text is True
        assert caps.supports_buttons is True
        assert caps.max_buttons == 3
        assert caps.supports_quick_replies is True
        assert caps.supports_media is True
