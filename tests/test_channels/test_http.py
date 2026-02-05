"""Tests for HTTP webhook channel."""

from __future__ import annotations

from roomkit.channels import HTTPChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import TextContent
from roomkit.models.room import Room
from roomkit.providers.http.mock import MockHTTPProvider
from tests.conftest import make_event


class TestHTTPChannel:
    async def test_handle_inbound(self) -> None:
        ch = HTTPChannel("http1")
        ctx = RoomContext(room=Room(id="r1"))
        msg = InboundMessage(
            channel_id="http1",
            sender_id="user1",
            content=TextContent(body="hello"),
        )
        event = await ch.handle_inbound(msg, ctx)
        assert event.room_id == "r1"
        assert event.source.channel_type == ChannelType.WEBHOOK

    async def test_deliver_with_provider(self) -> None:
        provider = MockHTTPProvider()
        ch = HTTPChannel("http1", provider=provider)
        binding = ChannelBinding(
            channel_id="http1",
            room_id="r1",
            channel_type=ChannelType.WEBHOOK,
            metadata={"recipient_id": "user-456"},
        )
        ctx = RoomContext(room=Room(id="r1"))
        event = make_event(channel_id="http1", channel_type=ChannelType.WEBHOOK)
        await ch.deliver(event, binding, ctx)
        assert len(provider.sent) == 1
        assert provider.sent[0]["to"] == "user-456"

    async def test_capabilities(self) -> None:
        ch = HTTPChannel("http1")
        caps = ch.capabilities()
        assert ChannelMediaType.TEXT in caps.media_types
        assert ChannelMediaType.RICH in caps.media_types
