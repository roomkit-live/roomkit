"""Tests for EventOpsMixin — direct event mutation APIs and the RFC §10.3
mutation triggers (ON_EVENT_UPDATED / ON_EVENT_DELETED)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

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
from roomkit.models.room import Room, RoomTimers
from roomkit.store.memory import InMemoryStore
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
        await kit.hook_engine.drain_observers()

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

    @pytest.mark.parametrize("fields", [(), ("content",), ("source", "metadata")])
    async def test_an_update_that_changes_nothing_writes_and_fires_nothing(
        self, fields: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RFC §10.3: no fields, or the values the event holds, return the
        event as stored, with no write and no ON_EVENT_UPDATED."""
        kit, _src = await _setup()
        original = await _send(kit, "v1")
        stored = await kit.store.get_event(original.id)
        assert stored is not None
        updated_calls = _track(kit, HookTrigger.ON_EVENT_UPDATED)
        writes: list[RoomEvent] = []

        async def update_event(event: RoomEvent) -> RoomEvent:
            writes.append(event)
            return event

        monkeypatch.setattr(kit.store, "update_event", update_event)

        same = {name: getattr(stored, name) for name in fields}
        updated = await kit.update_event("r1", original.id, **same)

        assert updated == stored
        assert writes == []
        assert updated_calls == []
        await kit.close()

    async def test_unknown_or_foreign_event_returns_none(self) -> None:
        kit, _src = await _setup()
        original = await _send(kit, "v1")
        updated_calls = _track(kit, HookTrigger.ON_EVENT_UPDATED)

        assert await kit.update_event("r1", "missing", content=TextContent(body="x")) is None
        assert await kit.update_event("r2", original.id, content=TextContent(body="x")) is None
        assert updated_calls == []


class TestAnotherOrganizationsRoom:
    @pytest.mark.parametrize("operation", ["update", "delete"])
    async def test_it_acts_as_on_a_missing_room(self, operation: str) -> None:
        """RFC §17.2, §10.3: scoped to an organization, an update or a delete
        in another organization's room writes nothing, its due timers
        included, and fires no hook; the room's own organization acts."""
        kit = RoomKit()
        kit.register_channel(RecordingTransport("src"))
        await kit.create_room(room_id="r1", organization_id="acme")
        await kit.attach_channel("r1", "src")
        original = await _send(kit, "v1")
        room = await kit.store.get_room("r1")
        assert room is not None
        # Its pause is due: a read that applied it would write the room.
        idle = datetime.now(UTC) - timedelta(hours=1)
        timers = RoomTimers(inactive_after_seconds=60, last_activity_at=idle)
        await kit.store.update_room(room.model_copy(update={"timers": timers}))
        before = (await kit.store.get_room("r1"), await kit.store.get_event(original.id))
        fired = _track(kit, HookTrigger.ON_EVENT_UPDATED) + _track(
            kit, HookTrigger.ON_EVENT_DELETED
        )
        acts = {
            "update": lambda org: kit.update_event(
                "r1", original.id, content=TextContent(body="v2"), organization_id=org
            ),
            "delete": lambda org: kit.delete_event("r1", original.id, organization_id=org),
        }

        refused = await acts[operation]("globex")

        assert not refused
        after = (await kit.store.get_room("r1"), await kit.store.get_event(original.id))
        assert (after, fired) == (before, [])
        assert await acts[operation]("acme")
        await kit.close()


class TestDeleteEventApi:
    async def test_deletes_root_replies_and_changes_and_fires_per_event(self) -> None:
        """RFC §10.3: a hard delete takes the thread and the edits of every
        event it removes, and ON_EVENT_DELETED fires once per removed event,
        with the event as it was stored."""
        kit, _src = await _setup()
        root = await _send(kit, "root")
        reply = make_event(room_id="r1", body="reply", parent_event_id=root.id, index=1)
        await kit.store.add_event(reply)
        edit = await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="u1",
                event_type=EventType.EDIT,
                content=EditContent(target_event_id=root.id, new_content=TextContent(body="v2")),
            )
        )
        assert edit.event is not None
        as_stored = [await kit.store.get_event(e) for e in (root.id, reply.id, edit.event.id)]
        deleted_calls = _track(kit, HookTrigger.ON_EVENT_DELETED)

        deleted = await kit.delete_event("r1", root.id)

        assert deleted == [root.id, reply.id, edit.event.id]
        await kit.hook_engine.drain_observers()
        assert [await kit.store.get_event(e) for e in deleted] == [None] * 3
        assert deleted_calls == as_stored
        assert deleted_calls[0].content == TextContent(body="v2")

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
                    edit_source="sender",
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
                    edit_source="sender",
                ),
            )
        )

        assert updated_calls == []


class _ReadsFailAfterAChange(InMemoryStore):
    """A store that cannot read a room once it stored a change, as one that
    lost its connection right after the write would."""

    def __init__(self) -> None:
        super().__init__()
        self.reads_fail = False

    async def update_event(self, event: RoomEvent) -> RoomEvent:
        stored = await super().update_event(event)
        self.reads_fail = True
        return stored

    async def delete_event(self, room_id: str, event_id: str, **kw: Any) -> list[RoomEvent]:
        removed = await super().delete_event(room_id, event_id, **kw)
        self.reads_fail = True
        return removed

    async def get_room(self, room_id: str) -> Room | None:
        if self.reads_fail:
            raise ConnectionError("the store went away")
        return await super().get_room(room_id)


class TestAChangeWhoseRoomCannotBeReadAfter:
    @pytest.mark.parametrize("operation", ["update", "delete"])
    async def test_stands_and_fires_its_trigger(
        self, operation: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The room cannot be read once the change's lock is released: the
        call returns as the change stands, the failure is logged, and the
        trigger fires with the room as the change left it (RFC §10.3,
        §13.6)."""
        store = _ReadsFailAfterAChange()
        kit = RoomKit(store=store)
        kit.register_channel(RecordingTransport("src"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "src")
        original = await _send(kit, "v1")
        fired: list[tuple[str, list[str]]] = []

        @kit.hook(HookTrigger.ON_EVENT_UPDATED, execution=HookExecution.ASYNC, name="updated")
        @kit.hook(HookTrigger.ON_EVENT_DELETED, execution=HookExecution.ASYNC, name="deleted")
        async def observe(event: RoomEvent, ctx: RoomContext) -> None:
            fired.append((ctx.room.id, [_text(e) for e in ctx.recent_events]))

        with caplog.at_level(logging.WARNING):
            if operation == "update":
                result: Any = await kit.update_event(
                    "r1", original.id, content=TextContent(body="v2")
                )
                assert result is not None and _text(result) == "v2"
            else:
                assert await kit.delete_event("r1", original.id) == [original.id]

        await kit.hook_engine.drain_observers()
        texts = ["v2"] if operation == "update" else []
        assert [(room, [t for t in seen if t in ("v1", "v2")]) for room, seen in fired] == [
            ("r1", texts)
        ]
        assert any("could not be read" in r.getMessage() for r in caplog.records)
        store.reads_fail = False
        await kit.close()


def _text(event: RoomEvent) -> str:
    return getattr(event.content, "body", "")


class _CountsContextReads(InMemoryStore):
    """A store that counts the room contexts read from it."""

    def __init__(self) -> None:
        super().__init__()
        self.context_reads = 0

    async def load_room_context(self, room_id: str, **kw: Any) -> Any:
        self.context_reads += 1
        return await super().load_room_context(room_id, **kw)


class TestADirectChangesContext:
    @pytest.mark.parametrize("operation", ["update", "delete"])
    @pytest.mark.parametrize("hooked", [False, True], ids=["no-hook", "a-hook"])
    async def test_is_read_only_for_a_hook_of_its_trigger(
        self, operation: str, hooked: bool
    ) -> None:
        """A direct update or delete reads the room's context for its
        trigger's hooks only, under the lock and once it is released (RFC
        §10.3); with no hook of that trigger it reads none."""
        store = _CountsContextReads()
        kit = RoomKit(store=store)
        kit.register_channel(RecordingTransport("src"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "src")
        original = await _send(kit, "v1")
        trigger = (
            HookTrigger.ON_EVENT_UPDATED if operation == "update" else HookTrigger.ON_EVENT_DELETED
        )
        fired = _track(kit, trigger) if hooked else []
        store.context_reads = 0

        if operation == "update":
            await kit.update_event("r1", original.id, content=TextContent(body="v2"))
        else:
            await kit.delete_event("r1", original.id)
        await kit.hook_engine.drain_observers()

        assert (store.context_reads, len(fired)) == ((2, 1) if hooked else (0, 0))
        await kit.close()
