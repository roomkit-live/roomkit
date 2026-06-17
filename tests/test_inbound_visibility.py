"""InboundMessage.visibility — deliver to transports without waking the agent.

A proactive notification posted into a room (e.g. a Telegram HITL prompt) must
reach the external transport so the human sees it, but must NOT trigger the
room's intelligence channel into replying. ``visibility="transport"`` on the
inbound expresses exactly that.
"""

from __future__ import annotations

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider


async def _kit() -> tuple[RoomKit, MockAIProvider]:
    kit = RoomKit()
    provider = MockAIProvider(responses=["Hello!"])
    kit.register_channel(WebSocketChannel("ws-user"))
    kit.register_channel(AIChannel("ai-bot", provider=provider))
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "ws-user")
    await kit.attach_channel("room-1", "ai-bot", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


class TestInboundVisibility:
    async def test_default_visibility_triggers_the_agent(self) -> None:
        kit, provider = await _kit()
        await kit.process_inbound(
            InboundMessage(channel_id="ws-user", sender_id="u1", content=TextContent(body="Hi")),
            room_id="room-1",
        )
        assert len(provider.calls) == 1  # the agent generated a reply

    async def test_transport_visibility_does_not_trigger_the_agent(self) -> None:
        kit, provider = await _kit()
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws-user",
                sender_id="u1",
                content=TextContent(body="Notification"),
                visibility="transport",
            ),
            room_id="room-1",
        )
        assert len(provider.calls) == 0  # intelligence channel was skipped
        # The message is still stored (the transport/human sees it).
        events = await kit.store.list_events("room-1")
        assert any(
            isinstance(e.content, TextContent) and e.content.body == "Notification" for e in events
        )
