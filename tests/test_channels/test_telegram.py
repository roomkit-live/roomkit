"""Tests for Telegram channel."""

from __future__ import annotations

from roomkit.channels import TelegramChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.providers.telegram.mock import MockTelegramProvider
from tests.conftest import make_event


class TestTelegramChannel:
    async def test_handle_inbound(self) -> None:
        ch = TelegramChannel("tg1")
        ctx = RoomContext(room=Room(id="r1"))
        msg = InboundMessage(
            channel_id="tg1",
            sender_id="12345",
            content=TextContent(body="hello"),
        )
        event = await ch.handle_inbound(msg, ctx)
        assert event.source.channel_type == ChannelType.TELEGRAM

    async def test_deliver_with_provider(self) -> None:
        provider = MockTelegramProvider()
        ch = TelegramChannel("tg1", provider=provider)
        binding = ChannelBinding(
            channel_id="tg1",
            room_id="r1",
            channel_type=ChannelType.TELEGRAM,
            metadata={"telegram_chat_id": "12345"},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(channel_id="tg1", channel_type=ChannelType.TELEGRAM)
        await ch.deliver(event, binding, ctx)
        assert len(provider.sent) == 1

    async def test_capabilities(self) -> None:
        ch = TelegramChannel("tg1")
        caps = ch.capabilities()
        assert ChannelMediaType.LOCATION in caps.media_types
        assert caps.supports_edit is True
        assert caps.supports_delete is True
        assert caps.supports_reactions is True
        assert caps.supports_media is True
        assert caps.max_length == 4096
