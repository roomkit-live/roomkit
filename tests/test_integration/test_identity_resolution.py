"""Integration: Identity resolution scenarios."""

from __future__ import annotations

from roomkit.channels.websocket import WebSocketChannel
from roomkit.core.framework import RoomKit
from roomkit.identity.mock import MockIdentityResolver
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.models.identity import Identity


class TestIdentityResolution:
    async def test_known_identity_resolved(self) -> None:
        """Known sender gets identity resolved on event."""
        identity = Identity(id="alice", display_name="Alice", phone="+15551234567")
        resolver = MockIdentityResolver(mapping={"sender1": identity})
        kit = RoomKit(identity_resolver=resolver)

        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="sender1",
            content=TextContent(body="Hello"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert result.event is not None
        assert result.event.source.participant_id == "alice"

    async def test_unknown_identity_pending(self) -> None:
        """Unknown sender leaves participant_id as None."""
        resolver = MockIdentityResolver()
        kit = RoomKit(identity_resolver=resolver)

        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="unknown_user",
            content=TextContent(body="Who am I?"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked
        assert result.event is not None
        # participant_id retains the channel-assigned value for unknown identity
        assert result.event.source.participant_id == "unknown_user"

    async def test_no_resolver_configured(self) -> None:
        """Without a resolver, identity resolution is skipped."""
        kit = RoomKit()  # No resolver
        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        msg = InboundMessage(
            channel_id="ws1",
            sender_id="user1",
            content=TextContent(body="No resolver"),
        )
        result = await kit.process_inbound(msg)
        assert not result.blocked

    async def test_multiple_identities(self) -> None:
        """Different senders get different identities."""
        resolver = MockIdentityResolver(
            mapping={
                "alice": Identity(id="alice-id", display_name="Alice"),
                "bob": Identity(id="bob-id", display_name="Bob"),
            }
        )
        kit = RoomKit(identity_resolver=resolver)

        ws = WebSocketChannel("ws1")
        kit.register_channel(ws)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws1")

        for sender_id in ["alice", "bob"]:
            msg = InboundMessage(
                channel_id="ws1",
                sender_id=sender_id,
                content=TextContent(body=f"Hello from {sender_id}"),
            )
            result = await kit.process_inbound(msg)
            assert result.event is not None
            assert result.event.source.participant_id == f"{sender_id}-id"
