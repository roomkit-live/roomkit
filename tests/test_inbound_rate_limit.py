"""Tests for framework-level inbound rate limiting."""

from __future__ import annotations

from roomkit import InboundMessage, RateLimit, RoomKit, TextContent
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent


class SimpleChannel(Channel):
    """Minimal channel for testing."""

    channel_type = ChannelType.SMS

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
            ),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def test_messages_within_limit_pass() -> None:
    """Messages within the rate limit should be processed normally."""
    kit = RoomKit(inbound_rate_limit=RateLimit(max_per_second=10.0))
    ch = SimpleChannel("ch1")
    kit.register_channel(ch)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "ch1")

    # Send 5 messages — all should pass with a 10/sec limit
    for i in range(5):
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ch1",
                sender_id="user",
                content=TextContent(body=f"msg-{i}"),
            )
        )
        assert not result.blocked, f"Message {i} should not be blocked"

    await kit.close()


async def test_excess_messages_blocked_with_rate_limited_reason() -> None:
    """Messages exceeding the rate limit should be blocked with reason='rate_limited'."""
    # Allow only 1 message per second (burst=1)
    kit = RoomKit(inbound_rate_limit=RateLimit(max_per_second=1.0))
    ch = SimpleChannel("ch1")
    kit.register_channel(ch)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "ch1")

    # First message passes
    result1 = await kit.process_inbound(
        InboundMessage(
            channel_id="ch1",
            sender_id="user",
            content=TextContent(body="first"),
        )
    )
    assert not result1.blocked

    # Second message should be rate limited
    result2 = await kit.process_inbound(
        InboundMessage(
            channel_id="ch1",
            sender_id="user",
            content=TextContent(body="second"),
        )
    )
    assert result2.blocked
    assert result2.reason == "rate_limited"

    await kit.close()


async def test_per_channel_isolation() -> None:
    """Rate limiting should be per-channel, not global."""
    kit = RoomKit(inbound_rate_limit=RateLimit(max_per_second=1.0))
    ch1 = SimpleChannel("ch1")
    ch2 = SimpleChannel("ch2")
    kit.register_channel(ch1)
    kit.register_channel(ch2)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "ch1")
    await kit.attach_channel(room.id, "ch2")

    # First message on ch1 passes
    r1 = await kit.process_inbound(
        InboundMessage(channel_id="ch1", sender_id="u", content=TextContent(body="a"))
    )
    assert not r1.blocked

    # Second message on ch1 blocked
    r2 = await kit.process_inbound(
        InboundMessage(channel_id="ch1", sender_id="u", content=TextContent(body="b"))
    )
    assert r2.blocked

    # First message on ch2 passes (separate bucket)
    r3 = await kit.process_inbound(
        InboundMessage(channel_id="ch2", sender_id="u", content=TextContent(body="c"))
    )
    assert not r3.blocked

    await kit.close()


async def test_no_rate_limit_means_no_blocking() -> None:
    """When no inbound_rate_limit is set, nothing is blocked."""
    kit = RoomKit()  # No rate limit
    ch = SimpleChannel("ch1")
    kit.register_channel(ch)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "ch1")

    for i in range(20):
        result = await kit.process_inbound(
            InboundMessage(
                channel_id="ch1",
                sender_id="user",
                content=TextContent(body=f"msg-{i}"),
            )
        )
        assert not result.blocked

    await kit.close()
