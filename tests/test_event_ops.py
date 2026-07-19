"""Tests for EventOpsMixin — direct event mutation APIs and the RFC §10.3
mutation triggers (ON_EVENT_UPDATED / ON_EVENT_DELETED)."""

from __future__ import annotations

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, EventType, HookExecution, HookTrigger
from roomkit.models.event import (
    DeleteContent,
    EditContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.hook import HookResult
from tests.conftest import make_event


class RecordingTransport(Channel):
    """Transport that honors ``message.event_type`` and records deliveries."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []

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
        self.delivered.append(event)
        return ChannelOutput.empty()


async def _setup() -> tuple[RoomKit, RecordingTransport]:
    kit = RoomKit()
    src = RecordingTransport("src")
    kit.register_channel(src)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    return kit, src


def _track(kit: RoomKit, trigger: HookTrigger) -> list[RoomEvent]:
    calls: list[RoomEvent] = []

    @kit.hook(trigger, execution=HookExecution.ASYNC)
    async def observe(event: RoomEvent, ctx: RoomContext) -> HookResult:
        calls.append(event)
        return HookResult.allow()

    return calls


async def _send(kit: RoomKit, body: str, sender_id: str = "u1") -> RoomEvent:
    result = await kit.process_inbound(
        InboundMessage(channel_id="src", sender_id=sender_id, content=TextContent(body=body))
    )
    assert result.event is not None
    return result.event


class TestUpdateEventApi:
    async def test_updates_content_and_fires_hook(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "v1")
        updated_calls = _track(kit, HookTrigger.ON_EVENT_UPDATED)

        updated = await kit.update_event("r1", original.id, content=TextContent(body="v2"))

        assert updated is not None
        assert isinstance(updated.content, TextContent)
        assert updated.content.body == "v2"
        stored = await kit.store.get_event(original.id)
        assert stored is not None
        assert stored.content.body == "v2"
        assert len(updated_calls) == 1
        assert updated_calls[0].id == original.id
        assert updated_calls[0].content.body == "v2"

    async def test_replaces_source_and_metadata(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "task result")

        new_source = original.source.model_copy(
            update={"channel_type": ChannelType.SYSTEM, "participant_id": "task_result"}
        )
        updated = await kit.update_event(
            "r1", original.id, source=new_source, metadata={"message_type": "task_result"}
        )

        assert updated is not None
        assert updated.source.channel_type is ChannelType.SYSTEM
        assert updated.metadata == {"message_type": "task_result"}

    async def test_no_fields_returns_target_without_hook(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "v1")
        updated_calls = _track(kit, HookTrigger.ON_EVENT_UPDATED)

        updated = await kit.update_event("r1", original.id)

        assert updated is not None
        assert updated.content.body == "v1"
        assert updated_calls == []

    async def test_unknown_or_foreign_event_returns_none(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "v1")
        updated_calls = _track(kit, HookTrigger.ON_EVENT_UPDATED)

        assert await kit.update_event("r1", "missing", content=TextContent(body="x")) is None
        assert await kit.update_event("r2", original.id, content=TextContent(body="x")) is None
        assert updated_calls == []


class TestDeleteEventApi:
    async def test_deletes_root_and_replies_and_fires_hook(self) -> None:
        kit, _src = await _setup()
        root = await _send(kit, "root")
        reply = make_event(room_id="r1", body="reply", parent_event_id=root.id)
        await kit.store.add_event(reply)
        deleted_calls = _track(kit, HookTrigger.ON_EVENT_DELETED)

        deleted = await kit.delete_event("r1", root.id)

        assert deleted == [root.id, reply.id]
        assert await kit.store.get_event(root.id) is None
        assert await kit.store.get_event(reply.id) is None
        assert len(deleted_calls) == 1
        # The hook receives the pre-delete snapshot of the root event.
        assert deleted_calls[0].id == root.id
        assert deleted_calls[0].content.body == "root"

    async def test_no_cascade_keeps_replies(self) -> None:
        kit, _src = await _setup()
        root = await _send(kit, "root")
        reply = make_event(room_id="r1", body="reply", parent_event_id=root.id)
        await kit.store.add_event(reply)

        deleted = await kit.delete_event("r1", root.id, cascade_replies=False)

        assert deleted == [root.id]
        assert await kit.store.get_event(reply.id) is not None

    async def test_unknown_or_foreign_event_returns_empty(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "keep me")
        deleted_calls = _track(kit, HookTrigger.ON_EVENT_DELETED)

        assert await kit.delete_event("r1", "missing") == []
        assert await kit.delete_event("r2", original.id) == []
        assert await kit.store.get_event(original.id) is not None
        assert deleted_calls == []


class TestInboundMutationTriggers:
    """RFC §10.3 — the inbound EDIT/DELETE path fires the same triggers."""

    async def test_inbound_edit_fires_on_event_updated(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "v1")
        updated_calls = _track(kit, HookTrigger.ON_EVENT_UPDATED)

        await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="u1",
                event_type=EventType.EDIT,
                content=EditContent(
                    target_event_id=original.id,
                    new_content=TextContent(body="v2"),
                    edit_source="u1",
                ),
            )
        )

        assert len(updated_calls) == 1
        assert updated_calls[0].id == original.id
        assert updated_calls[0].content.body == "v2"
        assert updated_calls[0].metadata.get("edited") is True

    async def test_inbound_delete_fires_on_event_deleted(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "bye")
        deleted_calls = _track(kit, HookTrigger.ON_EVENT_DELETED)

        await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="u1",
                event_type=EventType.DELETE,
                content=DeleteContent(target_event_id=original.id),
            )
        )

        assert len(deleted_calls) == 1
        assert deleted_calls[0].id == original.id
        # Inbound DELETE is soft (RFC §10.3): the target is flagged, not removed.
        assert deleted_calls[0].metadata.get("deleted") is True
        assert await kit.store.get_event(original.id) is not None

    async def test_blocked_edit_fires_no_mutation_trigger(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "v1")
        updated_calls = _track(kit, HookTrigger.ON_EVENT_UPDATED)

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def block_edits(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.type is EventType.EDIT:
                return HookResult.block(reason="moderated")
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="u1",
                event_type=EventType.EDIT,
                content=EditContent(
                    target_event_id=original.id,
                    new_content=TextContent(body="v2"),
                    edit_source="u1",
                ),
            )
        )

        assert updated_calls == []
