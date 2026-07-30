"""A binding is never widened implicitly (RFC §7.5-6, §7.5-7).

Two ways the framework can hand out more access than an integrator granted:
by deriving a binding from another and letting the gaps fall back to defaults,
and by re-attaching a channel that was deliberately detached.
"""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import Access, ChannelType, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent


class _Transport(Channel):
    channel_type = ChannelType.WEBSOCKET

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class TestSharedChannelInheritsPermissions:
    """Sharing a channel into a delegated room must not promote it."""

    async def _kit(self) -> RoomKit:
        kit = RoomKit()
        kit.register_channel(_Transport("observer"))
        await kit.create_room(room_id="parent")
        return kit

    async def test_read_only_stays_read_only_in_the_child_room(self) -> None:
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", access=Access.READ_ONLY)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.access == Access.READ_ONLY

    async def test_restricted_visibility_is_carried_over(self) -> None:
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", visibility=Visibility.NONE)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.visibility == Visibility.NONE

    async def test_a_muted_channel_stays_muted(self) -> None:
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", muted=True)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.muted is True

    async def test_a_full_participant_is_still_shared_as_one(self) -> None:
        """Narrowing is not the goal — faithfulness is."""
        kit = await self._kit()
        await kit.attach_channel("parent", "observer", access=Access.READ_WRITE)

        handle = await kit.delegate(
            "parent", agent_id="observer", task="t", share_channels=["observer"]
        )

        child = await kit._store.get_binding(handle.child_room_id, "observer")
        assert child is not None
        assert child.access == Access.READ_WRITE
        assert child.muted is False


class TestDetachIsNotUndoneByAutoAttach:
    """Detaching is how access is revoked; a message must not restore it."""

    async def _kit(self) -> RoomKit:
        kit = RoomKit()
        kit.register_channel(_Transport("ws"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "ws")
        return kit

    async def test_a_detached_channel_is_not_re_attached_by_a_message(self) -> None:
        kit = await self._kit()
        await kit.detach_channel("r1", "ws")

        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="mallory", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert await kit._store.get_binding("r1", "ws") is None

    async def test_an_explicit_attach_re_grants_access(self) -> None:
        """Revocation is not permanent — it is the integrator's to undo."""
        kit = await self._kit()
        await kit.detach_channel("r1", "ws")
        await kit.attach_channel("r1", "ws", access=Access.READ_ONLY)

        binding = await kit._store.get_binding("r1", "ws")
        assert binding is not None
        assert binding.access == Access.READ_ONLY

        # And the auto-attach path leaves that binding alone afterwards.
        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )
        binding = await kit._store.get_binding("r1", "ws")
        assert binding is not None
        assert binding.access == Access.READ_ONLY

    async def test_a_never_attached_channel_is_still_auto_attached(self) -> None:
        """The convenience this guard narrows must keep working."""
        kit = RoomKit()
        kit.register_channel(_Transport("ws"))
        await kit.create_room(room_id="r1")

        await kit.process_inbound(
            InboundMessage(channel_id="ws", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert await kit._store.get_binding("r1", "ws") is not None
