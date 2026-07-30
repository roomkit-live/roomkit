"""Authorization on inbound EDIT/DELETE (RFC §10.3).

The framework, not the payload, decides who may rewrite a room's history. A
channel that lets a remote party choose ``edit_source`` or ``delete_type``
must not thereby let them choose their own authority.
"""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, DeleteType, EventType, ParticipantRole
from roomkit.models.event import (
    DeleteContent,
    EditContent,
    EventSource,
    RoomEvent,
    TextContent,
)


class _Transport(Channel):
    """Transport that forwards ``event_type``/``content`` verbatim.

    This is the realistic shape of a WebSocket or transport channel: the
    remote party controls the payload, so it also controls ``edit_source``
    and ``delete_type``.
    """

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


class _SystemTransport(_Transport):
    channel_type = ChannelType.SYSTEM


async def _setup() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(_Transport("src"))
    kit.register_channel(_SystemTransport("sys"))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    await kit.attach_channel("r1", "sys")
    return kit


async def _post(kit: RoomKit, body: str, sender_id: str) -> RoomEvent:
    result = await kit.process_inbound(
        InboundMessage(channel_id="src", sender_id=sender_id, content=TextContent(body=body))
    )
    assert result.event is not None
    return result.event


async def _edit(
    kit: RoomKit,
    target: RoomEvent,
    *,
    sender_id: str,
    edit_source: str | None,
    channel_id: str = "src",
):
    return await kit.process_inbound(
        InboundMessage(
            channel_id=channel_id,
            sender_id=sender_id,
            event_type=EventType.EDIT,
            content=EditContent(
                target_event_id=target.id,
                new_content=TextContent(body="rewritten"),
                edit_source=edit_source,
            ),
        )
    )


async def _delete(
    kit: RoomKit,
    target: RoomEvent,
    *,
    sender_id: str,
    delete_type: DeleteType,
    channel_id: str = "src",
):
    return await kit.process_inbound(
        InboundMessage(
            channel_id=channel_id,
            sender_id=sender_id,
            event_type=EventType.DELETE,
            content=DeleteContent(target_event_id=target.id, delete_type=delete_type),
        )
    )


class TestEditAuthorization:
    async def test_author_may_edit_own_message(self) -> None:
        """The ordinary path must keep working."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(kit, original, sender_id="alice", edit_source="sender")

        assert result.blocked is False
        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content.body == "rewritten"

    async def test_non_author_cannot_edit(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(kit, original, sender_id="mallory", edit_source="sender")

        assert result.blocked is True
        assert result.reason == "not_original_author"

    async def test_unknown_edit_source_does_not_skip_the_author_check(self) -> None:
        """An ``edit_source`` outside the RFC vocabulary is not a privilege."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        for forged in ("admin", "system_", "Sender", "alice", ""):
            result = await _edit(kit, original, sender_id="mallory", edit_source=forged)

            assert result.blocked is True, f"edit_source={forged!r} bypassed authorization"
            assert result.reason == "not_original_author"

        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content.body == "v1"

    async def test_participant_id_as_edit_source_still_requires_authorship(self) -> None:
        """The pattern the examples used to show must not grant authority."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        # Alice editing her own message with her id as edit_source: allowed.
        mine = await _edit(kit, original, sender_id="alice", edit_source="alice")
        assert mine.blocked is False
        # Mallory doing the same on Alice's message: refused.
        theirs = await _edit(kit, original, sender_id="mallory", edit_source="mallory")
        assert theirs.blocked is True

    async def test_system_edit_source_requires_a_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(kit, original, sender_id="mallory", edit_source="system")

        assert result.blocked is True
        assert result.reason == "not_authorized"

    async def test_system_edit_source_allowed_from_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(
            kit, original, sender_id="moderator", edit_source="system", channel_id="sys"
        )

        assert result.blocked is False


class TestDeleteAuthorization:
    async def test_author_may_delete_own_message(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)

        assert result.blocked is False

    async def test_non_author_cannot_delete_as_sender(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="mallory", delete_type=DeleteType.SENDER)

        assert result.blocked is True
        assert result.reason == "not_original_author"

    async def test_admin_delete_requires_verified_authority(self) -> None:
        """Claiming ADMIN is not the same as holding it."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="mallory", delete_type=DeleteType.ADMIN)

        assert result.blocked is True
        assert result.reason == "not_authorized"
        stored = await kit._store.get_event(original.id)
        assert stored is not None
        assert stored.metadata.get("deleted") is not True

    async def test_admin_delete_allowed_for_room_owner(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        await kit.add_member("r1", "src", "owner-1", role=ParticipantRole.OWNER)

        result = await _delete(kit, original, sender_id="owner-1", delete_type=DeleteType.ADMIN)

        assert result.blocked is False

    async def test_system_delete_requires_a_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="mallory", delete_type=DeleteType.SYSTEM)

        assert result.blocked is True
        assert result.reason == "not_authorized"

    async def test_system_delete_allowed_from_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(
            kit, original, sender_id="janitor", delete_type=DeleteType.SYSTEM, channel_id="sys"
        )

        assert result.blocked is False
