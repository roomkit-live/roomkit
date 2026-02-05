"""Integration: Read-only observer producing side effects."""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
)
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.task import Observation


class ObserverChannel(Channel):
    """Read-only observer that produces side effect observations."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.observed: list[str] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.observed.append("observed")
        obs = Observation(
            id=f"obs_{event.id}",
            room_id=event.room_id,
            channel_id=self.channel_id,
            content="sentiment positive",
            category="sentiment",
        )
        return ChannelOutput(observations=[obs])

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class TestObserver:
    async def test_observer_receives_events(self) -> None:
        """Read-only observer receives all events."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        observer = ObserverChannel("observer1")

        kit.register_channel(ws)
        kit.register_channel(observer)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel(
            "r1",
            "observer1",
            category=ChannelCategory.INTELLIGENCE,
            access=Access.READ_ONLY,
        )

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="Hello"),
        )
        await kit.process_inbound(msg)
        assert len(observer.observed) == 1

    async def test_observer_cannot_originate(self) -> None:
        """Observer with READ_ONLY access cannot originate events."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        observer = ObserverChannel("observer1")
        ws_received: list[RoomEvent] = []

        async def ws_send(conn_id: str, event: RoomEvent) -> None:
            ws_received.append(event)

        ws.register_connection("conn1", ws_send)

        kit.register_channel(ws)
        kit.register_channel(observer)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel(
            "r1",
            "observer1",
            category=ChannelCategory.INTELLIGENCE,
            access=Access.READ_ONLY,
        )

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="test"),
        )
        await kit.process_inbound(msg)
        assert len(observer.observed) == 1

    async def test_observer_side_effects_stored(self) -> None:
        """Observer side effects (observations) are collected during broadcast."""
        kit = RoomKit()
        ws = WebSocketChannel("ws1")
        observer = ObserverChannel("observer1")

        kit.register_channel(ws)
        kit.register_channel(observer)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")
        await kit.attach_channel(
            "r1",
            "observer1",
            category=ChannelCategory.INTELLIGENCE,
        )

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="analyze this"),
        )
        await kit.process_inbound(msg)
        assert len(observer.observed) == 1
