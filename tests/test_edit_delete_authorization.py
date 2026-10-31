"""Authorization on inbound EDIT/DELETE (RFC §10.3).

The framework, not the payload, decides who may rewrite a room's history. A
channel that lets a remote party choose ``edit_source`` or ``delete_type``
must not thereby let them choose their own authority.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from roomkit.channels.ai import AIChannel
from roomkit.channels.base import Channel
from roomkit.core.exceptions import ChangeRefusedError
from roomkit.core.framework import RoomKit
from roomkit.core.visibility import visible_events
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    DeleteType,
    EventDirection,
    EventStatus,
    EventType,
    HookExecution,
    HookTrigger,
    ParticipantRole,
    ParticipantStatus,
)
from roomkit.models.event import (
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
    EventSource,
    RoomEvent,
    TextContent,
)
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.store_filter import PersistencePolicy
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.store.memory import InMemoryStore


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
                # A provider's echo of the channel's own output travels outbound.
                direction=message.metadata.get("direction", EventDirection.INBOUND),
                participant_id=message.sender_id or None,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _Parsing(_Transport):
    """A transport that records the history it parses each message with."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.parsed_with: list[RoomContext] = []

    @property
    def recent_events_window(self) -> int:
        return 10

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        self.parsed_with.append(context)
        return await super().handle_inbound(message, context)


class _Editable(_Transport):
    """A transport that applies edits and deletes natively, and records what
    it is handed."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []
        self.contexts: list[RoomContext] = []

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(supports_edit=True, supports_delete=True)

    @property
    def recent_events_window(self) -> int:
        return 10

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        self.contexts.append(context)
        return ChannelOutput.empty()


async def _setup() -> RoomKit:
    kit = RoomKit()
    kit.register_channel(_Transport("src"))
    await kit.create_room(room_id="r1")
    # Author and non-author write on one channel: a group binding (RFC §10.4).
    await kit.attach_channel("r1", "src", group=True)
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
    direction: EventDirection = EventDirection.INBOUND,
):
    return await kit.process_inbound(
        InboundMessage(
            channel_id=channel_id,
            sender_id=sender_id,
            event_type=EventType.DELETE,
            content=DeleteContent(target_event_id=target.id, delete_type=delete_type),
            metadata={"direction": direction},
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

    @pytest.mark.parametrize("named", ["admin", "system_", "Sender", "alice", ""])
    def test_an_edit_naming_another_source_is_refused_as_it_is_built(self, named: str) -> None:
        """``edit_source`` is ``sender``, ``system`` or none (RFC §10.3), as
        roomkit-rs's ``EditSource``: any other value is refused before it
        can claim anything (RMK-1021 item 2). It was read as the sender's."""
        with pytest.raises(ValidationError, match="edit_source"):
            EditContent(target_event_id="e1", new_content=TextContent(body="x"), edit_source=named)

    async def test_system_edit_source_requires_a_system_channel(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _edit(kit, original, sender_id="mallory", edit_source="system")

        assert result.blocked is True
        assert result.reason == "not_authorized"


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


def _hidden_changes(target_id: str) -> dict[str, tuple[EventType, EventContent]]:
    """Events holding a change they do not make (RFC §10.3 rule 1), by name."""
    edit = EditContent(target_event_id=target_id, new_content=TextContent(body="rewritten"))
    delete = DeleteContent(target_event_id=target_id)
    return {
        "a message carrying an edit": (EventType.MESSAGE, edit),
        "a message carrying a delete": (EventType.MESSAGE, delete),
        "an edit carrying text": (EventType.EDIT, TextContent(body="rewritten")),
        "a delete carrying an edit": (EventType.DELETE, edit),
        "an edit carrying a delete": (EventType.EDIT, delete),
        "a composite part": (
            EventType.MESSAGE,
            CompositeContent(parts=[TextContent(body="hi"), edit]),
        ),
        "an edit's new content": (EventType.EDIT, edit.model_copy(update={"new_content": delete})),
        "a composite in an edit's new content": (
            EventType.EDIT,
            edit.model_copy(update={"new_content": CompositeContent(parts=[delete])}),
        ),
    }


class TestWhatAChangeIs:
    """RFC §10.3 rule 1: an event is an edit when its type is EDIT and its
    content an EditContent, a delete when DELETE and DeleteContent; any other
    event holding a change, at any depth, is refused before a channel can
    apply it natively."""

    @pytest.mark.parametrize("case", list(_hidden_changes("")))
    async def test_a_change_held_where_it_is_not_made_is_refused(self, case: str) -> None:
        kit = await _setup()
        peer = _Editable("peer")
        kit.register_channel(peer)
        await kit.attach_channel("r1", "peer")
        original = await _post(kit, "v1", sender_id="alice")
        event_type, content = _hidden_changes(original.id)[case]

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="src", sender_id="alice", event_type=event_type, content=content
            )
        )

        assert (result.blocked, result.reason) == (True, "target_event_not_found")
        assert result.event is None
        assert [e.id for e in peer.delivered] == [original.id]
        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content == TextContent(body="v1")
        assert not stored.metadata
        await kit.close()


class TestWhatAChangeMayTarget:
    """RFC §10.3 rule 2: the target, read under the room lock, is an event of
    the room that is not BLOCKED, not deleted and not itself a change."""

    async def test_a_refused_message_is_no_target(self) -> None:
        kit = await _setup()

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def no_spam(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if isinstance(event.content, TextContent) and event.content.body == "spam":
                return HookResult.block(reason="spam")
            return HookResult.allow()

        refused = (
            await kit.process_inbound(
                InboundMessage(
                    channel_id="src", sender_id="alice", content=TextContent(body="spam")
                )
            )
        ).event
        assert refused is not None and refused.status == EventStatus.BLOCKED

        edit = await _edit(kit, refused, sender_id="alice", edit_source="sender")
        delete = await _delete(kit, refused, sender_id="alice", delete_type=DeleteType.SENDER)

        assert (edit.reason, delete.reason) == ("target_event_not_found",) * 2
        stored = await kit._store.get_event(refused.id)
        assert stored is not None and not stored.metadata
        await kit.close()

    @pytest.mark.parametrize("change", ["edit", "delete"])
    async def test_a_deleted_message_is_no_target(self, change: str) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        first = await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)
        assert first.blocked is False

        if change == "edit":
            result = await _edit(kit, original, sender_id="alice", edit_source="sender")
        else:
            result = await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)

        assert (result.blocked, result.reason) == (True, "target_event_not_found")
        assert result.event is None
        await kit.close()

    @pytest.mark.parametrize("change", ["edit", "delete"])
    async def test_a_change_is_no_target(self, change: str) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        first = await _edit(kit, original, sender_id="alice", edit_source="sender")
        assert first.event is not None

        if change == "edit":
            result = await _edit(kit, first.event, sender_id="alice", edit_source="sender")
        else:
            result = await _delete(
                kit, first.event, sender_id="alice", delete_type=DeleteType.SENDER
            )

        assert (result.blocked, result.reason) == (True, "target_event_not_found")
        await kit.close()

    async def test_an_edited_message_may_be_edited_again(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        await _edit(kit, original, sender_id="alice", edit_source="sender")

        again = await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="alice",
                event_type=EventType.EDIT,
                content=EditContent(
                    target_event_id=original.id, new_content=TextContent(body="third")
                ),
            )
        )

        assert again.blocked is False
        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content == TextContent(body="third")
        await kit.close()

    async def test_a_retried_delete_is_answered_as_the_first(self) -> None:
        """A client that retries a delete gives it a key (RFC §13.4)."""
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        retry = InboundMessage(
            channel_id="src",
            sender_id="alice",
            event_type=EventType.DELETE,
            content=DeleteContent(target_event_id=original.id),
            idempotency_key="delete-1",
        )

        first = await kit.process_inbound(retry)
        second = await kit.process_inbound(retry)

        assert first.event is not None and first.blocked is False
        assert second.duplicate is True and second.event is not None
        assert second.event.id == first.event.id
        await kit.close()


class TestWhoMayChange:
    """RFC §10.3 rules 3 to 6: who authored an event, and who holds the
    authority a change claims."""

    @pytest.mark.parametrize(
        ("role", "status", "allowed"),
        [
            (ParticipantRole.OWNER, ParticipantStatus.ACTIVE, True),
            (ParticipantRole.AGENT, ParticipantStatus.ACTIVE, True),
            (ParticipantRole.MEMBER, ParticipantStatus.ACTIVE, False),
            (ParticipantRole.OWNER, ParticipantStatus.LEFT, False),
            (ParticipantRole.OWNER, ParticipantStatus.BANNED, False),
            (ParticipantRole.AGENT, ParticipantStatus.BANNED, False),
        ],
    )
    async def test_an_admin_delete_is_an_active_owner_s_or_agent_s(
        self, role: ParticipantRole, status: ParticipantStatus, allowed: bool
    ) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")
        moderator = await kit.add_member("r1", "src", "mod-1", role=role)
        await kit._store.update_participant(moderator.model_copy(update={"status": status}))

        result = await _delete(kit, original, sender_id="mod-1", delete_type=DeleteType.ADMIN)

        assert (result.blocked, result.reason) == (
            (False, None) if allowed else (True, "not_authorized")
        )
        await kit.close()

    async def test_an_admin_delete_from_no_one_needs_an_identity(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(kit, original, sender_id="", delete_type=DeleteType.ADMIN)

        assert (result.blocked, result.reason) == (True, "identity_required_for_edit")
        await kit.close()

    @pytest.mark.parametrize(
        ("delete_type", "sender_id", "reason"),
        [(DeleteType.ADMIN, "owner-1", "not_authorized"), (DeleteType.SENDER, "alice", None)],
        ids=["an owner's authority", "a participant's authorship"],
    )
    async def test_a_channel_s_own_output_speaks_as_no_participant(
        self, delete_type: DeleteType, sender_id: str, reason: str | None
    ) -> None:
        """An event that travelled outbound is its channel's, whatever
        participant it names (RFC §10.3 rule 3): it neither holds that
        participant's authority nor is the author of their messages."""
        kit = await _setup()
        await kit.add_member("r1", "src", "owner-1", role=ParticipantRole.OWNER)
        original = await _post(kit, "v1", sender_id="alice")

        result = await _delete(
            kit,
            original,
            sender_id=sender_id,
            delete_type=delete_type,
            direction=EventDirection.OUTBOUND,
        )

        assert (result.blocked, result.reason) == (True, reason or "not_original_author")
        stored = await kit._store.get_event(original.id)
        assert stored is not None and not stored.metadata
        await kit.close()

    async def test_a_system_change_needs_no_author(self) -> None:
        """A SYSTEM change claims the system's authority (rule 6), a hook's,
        since no channel is of the type SYSTEM: neither it nor its target
        needs an author, as a sender's change does (rule 4)."""
        kit = await _setup()
        notice = await kit.send_event("r1", "src", TextContent(body="notice"))
        assert notice.source.participant_id is None

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def janitor(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if not _said(event, "go"):
                return HookResult.allow()
            source = EventSource(channel_id="janitor", channel_type=ChannelType.SYSTEM)
            delete = DeleteContent(target_event_id=notice.id, delete_type=DeleteType.SYSTEM)
            return HookResult.allow([InjectedEvent(event=_change("r1", source, delete))])

        await _post(kit, "go", sender_id="alice")

        stored = await kit._store.get_event(notice.id)
        assert stored is not None and stored.metadata.get("deleted") is True
        await kit.close()


type _Answer = Callable[[RoomEvent, RoomContext], list[RoomEvent]]


class _Agent(Channel):
    """An intelligence channel answering each other channel's event with
    what *answer* returns."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str, answer: _Answer) -> None:
        super().__init__(channel_id)
        self._answer = answer

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        responses = self._answer(event, context)
        return ChannelOutput(responded=bool(responses), response_events=responses)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


def _said(event: RoomEvent, body: str) -> bool:
    return isinstance(event.content, TextContent) and event.content.body == body


def _change(room_id: str, source: EventSource, content: EditContent | DeleteContent) -> RoomEvent:
    kind = EventType.EDIT if isinstance(content, EditContent) else EventType.DELETE
    return RoomEvent(room_id=room_id, type=kind, source=source, content=content)


def _tracked(kit: RoomKit, trigger: HookTrigger) -> list[RoomEvent]:
    seen: list[RoomEvent] = []

    @kit.hook(trigger, execution=HookExecution.ASYNC)
    async def track(event: RoomEvent, ctx: RoomContext) -> None:
        seen.append(event)

    return seen


async def _with_peer(kit: RoomKit) -> _Editable:
    peer = _Editable("peer")
    kit.register_channel(peer)
    await kit.attach_channel("r1", "peer")
    return peer


def _changes_handed(peer: _Editable) -> list[RoomEvent]:
    return [e for e in peer.delivered if e.type in (EventType.EDIT, EventType.DELETE)]


class TestEveryPassValidates:
    """RFC §10.3: step 8a runs on every pass that commits an event, a
    response's and an injected event's included, and a refused response or
    injected change is logged and not stored."""

    async def test_an_agent_edits_its_own_response(self) -> None:
        kit = await _setup()

        def answer(event: RoomEvent, ctx: RoomContext) -> list[RoomEvent]:
            source = EventSource(channel_id="agent", channel_type=ChannelType.AI)
            if _said(event, "hi"):
                return [RoomEvent(room_id="r1", source=source, content=TextContent(body="draft"))]
            if _said(event, "fix it"):
                draft = next(e for e in ctx.recent_events if e.source.channel_id == "agent")
                edit = EditContent(target_event_id=draft.id, new_content=TextContent(body="final"))
                return [_change("r1", source, edit)]
            return []

        kit.register_channel(_Agent("agent", answer))
        await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
        updated = _tracked(kit, HookTrigger.ON_EVENT_UPDATED)

        await _post(kit, "hi", sender_id="alice")
        await _post(kit, "fix it", sender_id="alice")

        drafts = [e for e in await kit._store.list_events("r1") if e.source.channel_id == "agent"]
        draft = next(e for e in drafts if e.type == EventType.MESSAGE)
        assert (draft.content, draft.metadata.get("edited")) == (TextContent(body="final"), True)
        assert [e.id for e in updated] == [draft.id]
        await kit.close()

    async def test_a_response_naming_a_participant_does_not_edit_their_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        kit = await _setup()
        peer = await _with_peer(kit)
        original = await _post(kit, "v1", sender_id="alice")

        def answer(event: RoomEvent, ctx: RoomContext) -> list[RoomEvent]:
            if not _said(event, "go"):
                return []
            source = EventSource(
                channel_id="agent", channel_type=ChannelType.AI, participant_id="alice"
            )
            edit = EditContent(target_event_id=original.id, new_content=TextContent(body="x"))
            return [_change("r1", source, edit)]

        kit.register_channel(_Agent("agent", answer))
        await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)

        with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
            await _post(kit, "go", sender_id="alice")

        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content == TextContent(body="v1")
        timeline = await kit._store.list_events("r1")
        assert not [e for e in timeline if e.type == EventType.EDIT]
        assert _changes_handed(peer) == []
        assert "not_original_author" in caplog.text
        await kit.close()

    @pytest.mark.parametrize(
        "claimed",
        [
            DeleteContent(target_event_id="", delete_type=DeleteType.SENDER),
            DeleteContent(target_event_id="", delete_type=DeleteType.ADMIN),
            EditContent(target_event_id="", new_content=TextContent(body="x")),
            EditContent(
                target_event_id="", new_content=TextContent(body="x"), edit_source="sender"
            ),
        ],
        ids=["a sender's delete", "an admin delete", "a sourceless edit", "a sender's edit"],
    )
    async def test_a_hook_never_changes_in_a_participant_s_name(
        self, claimed: EditContent | DeleteContent, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An injected change is the hook's, whatever source it names (RFC
        §9.4, §10.3 rule 6): one claiming another authority than the
        system's is refused, even under the author's own source."""
        kit = await _setup()
        peer = await _with_peer(kit)
        await kit.add_member("r1", "src", "owner-1", role=ParticipantRole.OWNER)
        original = await _post(kit, "v1", sender_id="alice")
        change = claimed.model_copy(update={"target_event_id": original.id})
        source = original.source

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def rewriter(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if not _said(event, "go"):
                return HookResult.allow()
            injected = _change("r1", source, change)
            return HookResult.allow([InjectedEvent(event=injected, target_channel_ids=["peer"])])

        with caplog.at_level(logging.WARNING, logger="roomkit.framework"):
            await _post(kit, "go", sender_id="alice")

        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content == TextContent(body="v1")
        assert not stored.metadata
        timeline = await kit._store.list_events("r1")
        assert not [e for e in timeline if e.type in (EventType.EDIT, EventType.DELETE)]
        assert _changes_handed(peer) == []
        assert "not_authorized" in caplog.text
        await kit.close()

    async def test_a_hook_s_system_delete_marks_its_target(self) -> None:
        kit = await _setup()
        peer = await _with_peer(kit)
        original = await _post(kit, "v1", sender_id="alice")
        deleted = _tracked(kit, HookTrigger.ON_EVENT_DELETED)

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def janitor(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if not _said(event, "go"):
                return HookResult.allow()
            source = EventSource(channel_id="janitor", channel_type=ChannelType.SYSTEM)
            delete = DeleteContent(target_event_id=original.id, delete_type=DeleteType.SYSTEM)
            injected = _change("r1", source, delete)
            return HookResult.allow([InjectedEvent(event=injected, target_channel_ids=["peer"])])

        await _post(kit, "go", sender_id="alice")

        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.metadata.get("deleted") is True
        assert [e.id for e in deleted] == [original.id]
        assert [e.content for e in _changes_handed(peer)] == [
            DeleteContent(target_event_id=original.id, delete_type=DeleteType.SYSTEM)
        ]
        await kit.close()

    async def test_an_agent_edits_its_own_direct_injection(self) -> None:
        """A direct injection's source travels as its channel's category says
        (RFC §10.5): OUTBOUND for an intelligence channel, whose event is its
        own words going out, so it edits its message as its channel, with no
        participant named; a transport's stays INBOUND."""
        kit = await _setup()
        kit.register_channel(_Agent("agent", lambda event, ctx: []))
        await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)

        said = await kit.send_event("r1", "agent", TextContent(body="draft"))
        heard = await kit.send_event("r1", "src", TextContent(body="hi"), participant_id="alice")
        await kit.send_event(
            "r1",
            "agent",
            EditContent(target_event_id=said.id, new_content=TextContent(body="final")),
            event_type=EventType.EDIT,
        )

        assert (said.source.direction, heard.source.direction) == (
            EventDirection.OUTBOUND,
            EventDirection.INBOUND,
        )
        stored = await kit._store.get_event(said.id)
        assert stored is not None and stored.content == TextContent(body="final")
        await kit.close()

    async def test_a_refused_direct_injection_raises_with_its_reason(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        with pytest.raises(ChangeRefusedError) as refused:
            await kit.send_event(
                "r1",
                "src",
                DeleteContent(target_event_id=original.id, delete_type=DeleteType.SENDER),
                event_type=EventType.DELETE,
                participant_id="mallory",
            )

        assert refused.value.reason == "not_original_author"
        assert not [e for e in await kit._store.list_events("r1") if e.type == EventType.DELETE]
        await kit.close()


async def _advisor_room() -> tuple[RoomKit, _Editable, _Editable, RoomEvent]:
    """A room where alice wrote a message only the advisor's channel saw."""
    kit = await _setup()
    advisor, customer = _Editable("advisor"), _Editable("customer")
    for peer in (advisor, customer):
        kit.register_channel(peer)
        await kit.attach_channel("r1", peer.channel_id)
    await _post(kit, "hello", sender_id="alice")
    aside = await kit.send_event(
        "r1",
        "src",
        TextContent(body="for the advisor"),
        participant_id="alice",
        visibility="advisor",
    )
    return kit, advisor, customer, aside


class TestWhomAChangeReaches:
    """RFC §10.3 state updates 4 and 5: an accepted change reaches no one
    who could not see its target, and is delivered with its target as its
    commit left it."""

    async def test_a_correction_of_an_aside_never_reaches_the_customer(self) -> None:
        kit, advisor, customer, aside = await _advisor_room()

        result = await _edit(kit, aside, sender_id="alice", edit_source="sender")

        assert result.event is not None and result.event.visibility == "advisor"
        assert [e.id for e in _changes_handed(advisor)] == [result.event.id]
        assert _changes_handed(customer) == []
        await kit.close()

    async def test_an_injected_change_reaches_its_targets_that_saw_the_target(self) -> None:
        kit, advisor, customer, aside = await _advisor_room()

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def janitor(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if not _said(event, "clean up"):
                return HookResult.allow()
            source = EventSource(channel_id="janitor", channel_type=ChannelType.SYSTEM)
            delete = DeleteContent(target_event_id=aside.id, delete_type=DeleteType.SYSTEM)
            injected = InjectedEvent(
                event=_change("r1", source, delete), target_channel_ids=["advisor", "customer"]
            )
            return HookResult.allow([injected])

        await _post(kit, "clean up", sender_id="alice")

        [handed] = _changes_handed(advisor)
        assert handed.visibility == "advisor"
        assert _changes_handed(customer) == []
        await kit.close()

    async def test_a_change_is_delivered_with_its_target_as_committed(self) -> None:
        kit = await _setup()
        peer = await _with_peer(kit)
        original = await _post(kit, "v1", sender_id="alice")

        await _edit(kit, original, sender_id="alice", edit_source="sender")

        [edit] = _changes_handed(peer)
        context = peer.contexts[peer.delivered.index(edit)]
        [target] = [e for e in context.recent_events if e.id == original.id]
        assert (target.content, target.metadata) == (
            TextContent(body="rewritten"),
            {"edited": True},
        )
        await kit.close()


def _change_of(kind: str, target: RoomEvent, authority: str) -> EditContent | DeleteContent:
    """An edit or a delete of *target* made with *authority*, the sender's or
    the system's."""
    if kind == "edit":
        return EditContent(
            target_event_id=target.id, new_content=TextContent(body="final"), edit_source=authority
        )
    return DeleteContent(target_event_id=target.id, delete_type=DeleteType(authority))


async def _change_from(kit: RoomKit, origin: str, kind: str) -> None:
    """Change a message of the room by *origin*: its author's inbound
    message, an agent's response changing its own, or a hook's injected
    system change."""
    if origin == "response":

        def answer(event: RoomEvent, ctx: RoomContext) -> list[RoomEvent]:
            source = EventSource(channel_id="agent", channel_type=ChannelType.AI)
            if _said(event, "hi"):
                return [RoomEvent(room_id="r1", source=source, content=TextContent(body="draft"))]
            if _said(event, "go"):
                draft = next(e for e in ctx.recent_events if e.source.channel_id == "agent")
                return [_change("r1", source, _change_of(kind, draft, "sender"))]
            return []

        kit.register_channel(_Agent("agent", answer))
        await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
        await _post(kit, "hi", sender_id="alice")
        await _post(kit, "go", sender_id="alice")
        return
    original = await _post(kit, "v1", sender_id="alice")
    if origin == "inbound":
        change = _change_of(kind, original, "sender")
        await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="alice",
                event_type=EventType.EDIT if kind == "edit" else EventType.DELETE,
                content=change,
            )
        )
        return

    @kit.hook(HookTrigger.BEFORE_BROADCAST)
    async def janitor(event: RoomEvent, ctx: RoomContext) -> HookResult:
        if not _said(event, "go"):
            return HookResult.allow()
        source = EventSource(channel_id="janitor", channel_type=ChannelType.SYSTEM)
        injected = _change("r1", source, _change_of(kind, original, "system"))
        return HookResult.allow([InjectedEvent(event=injected)])

    await _post(kit, "go", sender_id="alice")


def _redacted(event: RoomEvent) -> HookResult:
    """A modify of a change that says another thing: an edit's new content
    redacted, a delete's reason; any other event allowed."""
    content = event.content
    if isinstance(content, EditContent):
        redacted = content.model_copy(update={"new_content": TextContent(body="REDACTED")})
    elif isinstance(content, DeleteContent):
        redacted = content.model_copy(update={"reason": "REDACTED"})
    else:
        return HookResult.allow()
    return HookResult.modify(event.model_copy(update={"content": redacted}))


class TestTheMutationTriggers:
    """RFC §10.3 Mutation triggers: ON_EVENT_UPDATED and ON_EVENT_DELETED
    read the room as the change's commit left it, whatever its origin: the
    target as rewritten, and the change itself as the newest event, at its
    committed index and as its hooks left it."""

    @pytest.mark.parametrize("nested", [False, True], ids=["alone", "after-a-nested-commit"])
    @pytest.mark.parametrize("kind", ["edit", "delete"])
    @pytest.mark.parametrize("origin", ["inbound", "response", "injected"])
    async def test_a_trigger_reads_the_room_as_the_change_left_it(
        self, origin: str, kind: str, nested: bool
    ) -> None:
        """*nested*: the change's own hook commits an event ahead of it
        (``send_event()``), which the trigger's history holds."""
        kit = await _setup()
        trigger = HookTrigger.ON_EVENT_UPDATED if kind == "edit" else HookTrigger.ON_EVENT_DELETED
        contexts: list[RoomContext] = []

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="redact")
        async def redact(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if nested and event.type in (EventType.EDIT, EventType.DELETE):
                await kit.send_event("r1", "src", TextContent(body="nested"))
            return _redacted(event)

        @kit.hook(trigger, execution=HookExecution.ASYNC)
        async def track(event: RoomEvent, ctx: RoomContext) -> None:
            contexts.append(ctx)

        await _change_from(kit, origin, kind)

        [change] = [
            e
            for e in await kit._store.list_events("r1")
            if e.type in (EventType.EDIT, EventType.DELETE)
        ]
        assert isinstance(change.content, EditContent | DeleteContent)
        target = await kit._store.get_event(change.content.target_event_id)
        assert "REDACTED" in change.content.model_dump_json()
        [context] = contexts
        assert context.recent_events[-1] == change
        assert [e for e in context.recent_events if e.id == change.content.target_event_id] == [
            target
        ]
        # The whole history and the counters as the change's commit left
        # them: the room's conversation, then the change.
        conversation = await kit._store.get_conversation("r1", limit=50)
        assert [_said(e, "nested") for e in conversation].count(True) == int(nested)
        assert context.recent_events == [*conversation, change]
        room = await kit._store.get_room("r1")
        assert room is not None
        assert (context.room.event_count, context.room.latest_index) == (
            room.event_count,
            room.latest_index,
        )
        await kit.close()


class TestTheTargetReadAgain:
    """RFC §10.1 step 9a: once a hook ran under the lock, an edit or a
    delete has its target read and validated again."""

    async def test_a_delete_committed_during_an_edit_s_hooks_refuses_the_edit(self) -> None:
        kit = await _setup()
        original = await _post(kit, "v1", sender_id="alice")

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def moderator(event: RoomEvent, ctx: RoomContext) -> HookResult:
            if event.type == EventType.EDIT:
                # Committed ahead of the edit, within its locked section: its
                # author deletes it.
                await kit.send_event(
                    "r1",
                    "src",
                    DeleteContent(target_event_id=original.id),
                    event_type=EventType.DELETE,
                    participant_id="alice",
                )
            return HookResult.allow()

        result = await _edit(kit, original, sender_id="alice", edit_source="sender")

        assert (result.blocked, result.reason) == (True, "target_event_not_found")
        stored = await kit._store.get_event(original.id)
        assert stored is not None
        assert (stored.content, stored.metadata) == (TextContent(body="v1"), {"deleted": True})
        await kit.close()


class TestWhatAChannelIsHandedOfADeletedEvent:
    """RFC §10.3: a soft-deleted event's content is withheld from every
    history a channel is handed, the one it parses a message with and the
    one a broadcast delivers with; the stored record keeps it."""

    async def test_a_delete_is_delivered_with_its_target_withheld(self) -> None:
        kit = await _setup()
        peer = await _with_peer(kit)
        original = await _post(kit, "v1", sender_id="alice")

        await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)

        [delete] = _changes_handed(peer)
        context = peer.contexts[peer.delivered.index(delete)]
        [target] = [e for e in context.recent_events if e.id == original.id]
        assert (target.content, target.metadata) == (TextContent(body=""), {"deleted": True})
        stored = await kit._store.get_event(original.id)
        assert stored is not None and stored.content == TextContent(body="v1")
        await kit.close()

    async def test_a_message_is_parsed_with_the_deleted_event_withheld(self) -> None:
        kit = RoomKit()
        parser = _Parsing("src")
        kit.register_channel(parser)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "src", group=True)
        original = await _post(kit, "v1", sender_id="alice")
        await _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)

        await _post(kit, "next", sender_id="alice")

        [held] = [e for e in parser.parsed_with[-1].recent_events if e.id == original.id]
        assert held.content == TextContent(body="")
        await kit.close()


class _WholeConversation(InMemoryStore):
    """A store whose conversation read serves every received row, changes
    included, as a custom store may."""

    async def get_conversation(
        self, room_id: str, *, limit: int = 50, after_index: int | None = None
    ) -> list[RoomEvent]:
        return await self.list_events(room_id, limit=limit, newest_first=True)


async def _agent_room(store: InMemoryStore | None = None) -> tuple[RoomKit, MockAIProvider]:
    kit = RoomKit(store=store) if store is not None else RoomKit()
    provider = MockAIProvider(responses=["noted"] * 10)
    kit.register_channel(_Transport("src"))
    kit.register_channel(AIChannel("ai", provider=provider))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src")
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, provider


def _last_prompt(provider: MockAIProvider) -> str:
    return " ".join(str(m.content) for m in provider.calls[-1].messages)


class TestNoReplayOfAWithdrawnMessage:
    """RFC §10.3: a regeneration re-runs the agents on a message still said,
    never on a deleted one nor on an edit or a delete row, which only step
    8a applies."""

    async def test_a_regeneration_skips_a_deleted_message(self) -> None:
        kit, provider = await _agent_room()
        first = await _post(kit, "first", sender_id="alice")
        secret = await _post(kit, "the code is 4417", sender_id="alice")
        await _delete(kit, secret, sender_id="alice", delete_type=DeleteType.SENDER)

        target = await kit.regenerate_target("r1")
        await kit.regenerate_response("r1")

        assert target is not None and target.id == first.id
        assert "4417" not in _last_prompt(provider)
        await kit.close()

    async def test_a_regeneration_skips_an_edit_row(self) -> None:
        kit, _provider = await _agent_room(_WholeConversation())
        original = await _post(kit, "v1", sender_id="alice")
        await _edit(kit, original, sender_id="alice", edit_source="sender")

        target = await kit.regenerate_target("r1")

        assert target is not None and target.id == original.id
        await kit.close()

    async def test_no_agent_is_re_run_on_a_change_row(self) -> None:
        """The door a held message's release takes (RFC §11.7.6)."""
        kit, provider = await _agent_room()
        original = await _post(kit, "v1", sender_id="alice")
        edit = await _edit(kit, original, sender_id="alice", edit_source="sender")
        assert edit.event is not None
        calls = len(provider.calls)

        rerun = await kit._rerun_agents_on("r1", edit.event.id)  # noqa: SLF001

        assert rerun is None and len(provider.calls) == calls
        await kit.close()


class _Flaky(_Transport):
    """A transport whose delivery of the message saying *failing* fails once
    *release* is set."""

    def __init__(self, channel_id: str, failing: str) -> None:
        super().__init__(channel_id)
        self.failing = failing
        self.started, self.release = asyncio.Event(), asyncio.Event()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if not _said(event, self.failing):
            return ChannelOutput.empty()
        self.started.set()
        await self.release.wait()
        raise RuntimeError("provider down")


async def _changed(kit: RoomKit, event_id: str, key: str) -> None:
    async with asyncio.timeout(2):
        while not (await kit._store.get_event(event_id)).metadata.get(key):  # type: ignore[union-attr]
            await asyncio.sleep(0.005)


class TestADeliveryRecordUndoesNoChange:
    """RFC §10.3, §14.3: the record of a failed delivery, written once the
    delivery set is done, writes the delivery results alone, never the
    event as it stood at its commit over a change committed since."""

    @pytest.mark.parametrize("change", ["edit", "delete"])
    async def test_a_change_committed_during_a_failing_delivery_stands(self, change: str) -> None:
        kit = await _setup()
        flaky = _Flaky("flaky", failing="v1")
        kit.register_channel(flaky)
        await kit.attach_channel("r1", "flaky")
        posted = asyncio.create_task(_post(kit, "v1", sender_id="alice"))
        await asyncio.wait_for(flaky.started.wait(), 2)
        [original] = [e for e in await kit._store.list_events("r1") if _said(e, "v1")]

        if change == "edit":
            changing = _edit(kit, original, sender_id="alice", edit_source="sender")
        else:
            changing = _delete(kit, original, sender_id="alice", delete_type=DeleteType.SENDER)
        changed = asyncio.create_task(changing)
        await _changed(kit, original.id, "edited" if change == "edit" else "deleted")
        flaky.release.set()
        await asyncio.wait_for(asyncio.gather(posted, changed), 2)

        stored = await kit._store.get_event(original.id)
        assert stored is not None
        if change == "edit":
            assert (stored.content, stored.metadata) == (
                TextContent(body="rewritten"),
                {"edited": True},
            )
        else:
            assert (stored.content, stored.metadata) == (TextContent(body="v1"), {"deleted": True})
        assert stored.delivery_results["flaky"]["status"] == "failed"
        await kit.close()


class _Reader(_Agent):
    """An agent that answers nothing and keeps the history it is handed,
    over a window of three events."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id, lambda event, ctx: [])
        self.histories: list[list[RoomEvent]] = []

    @property
    def recent_events_window(self) -> int:
        return 3

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.histories.append(visible_events(context, self.channel_id))
        return ChannelOutput.empty()


class TestAnEditOutsideItsTargetsWindow:
    """RFC §10.3: an edit of a deleted message holds a version of its text,
    withheld whether or not the history still holds the message."""

    @pytest.mark.parametrize(
        ("deleted", "stored"),
        [(True, True), (True, False), (False, True)],
        ids=["deleted", "deleted, the delete unstored", "not deleted"],
    )
    async def test_an_edit_is_withheld_with_the_message_it_corrected(
        self, deleted: bool, stored: bool
    ) -> None:
        """A delete the persistence policy keeps out of the timeline still
        marks its target and the target's edits."""
        policy = None if stored else PersistencePolicy(exclude_types={EventType.DELETE})
        kit = RoomKit(store=_WholeConversation(), persistence_policy=policy)
        kit.register_channel(_Transport("src"))
        reader = _Reader("reader")
        kit.register_channel(reader)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "src")
        await kit.attach_channel("r1", "reader", category=ChannelCategory.INTELLIGENCE)
        said = await _post(kit, "my pin is 1234", sender_id="alice")
        await kit.process_inbound(
            InboundMessage(
                channel_id="src",
                sender_id="alice",
                event_type=EventType.EDIT,
                content=EditContent(
                    target_event_id=said.id, new_content=TextContent(body="my pin is 5678")
                ),
            )
        )
        if deleted:
            await _delete(kit, said, sender_id="alice", delete_type=DeleteType.SENDER)
        if not deleted or not stored:
            # The row a stored delete takes, so the message leaves the window.
            await _post(kit, "noted", sender_id="alice")
        for filler in ("one", "two"):
            await _post(kit, filler, sender_id="alice")

        history = reader.histories[-1]
        assert said.id not in [e.id for e in history]
        [edit] = [e for e in history if e.type == EventType.EDIT]
        shown = str(edit.content)
        assert ("5678" in shown) is not deleted
        assert "1234" not in " ".join(str(e.content) for e in history)
        await kit.close()
