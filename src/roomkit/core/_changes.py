"""Edits and deletes (RFC §10.3): what a change is, what it may target, who
may make it, what its commit writes on its target, and the trigger it fires
once the room lock is released."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from roomkit.models.enums import (
    ChannelType,
    DeleteType,
    EditSource,
    EventDirection,
    EventStatus,
    EventType,
    HookTrigger,
    ParticipantRole,
    ParticipantStatus,
)
from roomkit.models.event import (
    CompositeContent,
    DeleteContent,
    EditContent,
    EventContent,
    RoomEvent,
    TextContent,
)

if TYPE_CHECKING:
    from roomkit.core._authors import People
    from roomkit.models.context import RoomContext

type Change = EditContent | DeleteContent

TARGET_NOT_FOUND = "target_event_not_found"
"""Step 8a's refusal of a change that names no target it may change."""
IDENTITY_REQUIRED = "identity_required_for_edit"
"""Step 8a's refusal of a change that needs an author it lacks."""
NOT_AUTHORIZED = "not_authorized"
"""Step 8a's refusal of a change claiming an authority its sender lacks."""
NOT_ORIGINAL_AUTHOR = "not_original_author"
"""Step 8a's refusal of a sender's change of another's event."""

CHANGE_REFUSALS = frozenset(
    {TARGET_NOT_FOUND, IDENTITY_REQUIRED, NOT_AUTHORIZED, NOT_ORIGINAL_AUTHOR}
)
"""Every reason step 8a refuses a change with (RFC §10.1)."""

_CHANGE_TYPES = frozenset({EventType.EDIT, EventType.DELETE})
_ADMINISTRATORS = frozenset({ParticipantRole.OWNER, ParticipantRole.AGENT})
_GONE = frozenset({ParticipantStatus.LEFT, ParticipantStatus.BANNED})
# RFC §10.3 fixes ``edit_source`` at "sender" | "system" | null, which
# ``EditSource`` holds: any other value is refused as the edit is built.
_SYSTEM_SOURCE = EditSource.SYSTEM


@dataclass(frozen=True, slots=True)
class ChangeCheck:
    """Step 8a's answer on an event (RFC §10.1): the *refusal* of a change,
    or the *target* of one it accepts; neither for an event that is no
    change."""

    target: RoomEvent | None = None
    refusal: str | None = None


NO_CHANGE = ChangeCheck()
"""Step 8a's answer on an event that is no change."""


def holds_change(content: EventContent) -> bool:
    """Whether *content* is or holds a change at any depth: an edit, a
    delete, or a composite with one among its parts."""
    if isinstance(content, EditContent | DeleteContent):
        return True
    if isinstance(content, CompositeContent):
        return any(holds_change(part) for part in content.parts)
    return False


def concerns_change(event: RoomEvent) -> bool:
    """Whether step 8a judges *event* (RFC §10.1): an EDIT or a DELETE, or
    an event holding a change anywhere."""
    return event.type in _CHANGE_TYPES or holds_change(event.content)


def change_of(event: RoomEvent) -> Change | None:
    """The change *event* makes (RFC §10.3 rule 1): its content, when its
    type is EDIT and its content an edit, or DELETE and a delete, with no
    change held in it besides; ``None`` otherwise. A type and a content that
    disagree, or a change held anywhere else (an edit's new content, a
    composite's part), name no target that can be checked."""
    content = event.content
    if event.type == EventType.EDIT and isinstance(content, EditContent):
        return None if holds_change(content.new_content) else content
    if event.type == EventType.DELETE and isinstance(content, DeleteContent):
        return content
    return None


def is_deleted(event: RoomEvent) -> bool:
    """Whether an accepted DELETE marked *event* deleted (``metadata.deleted``)."""
    return event.metadata.get("deleted") is True


def withdrawn(event: RoomEvent) -> bool:
    """Whether *event* is no message a turn may answer or replay (RFC
    §10.3): deleted by an accepted DELETE, or itself an edit or a delete,
    which only step 8a may apply. Its text is not handed to an agent as the
    event it answers."""
    return is_deleted(event) or concerns_change(event)


def target_refusal(target: RoomEvent | None, room_id: str) -> str | None:
    """Why *target* is no target a change of *room_id* may change (RFC §10.3
    rule 2), ``None`` when it is one: it is an event of the room, not
    BLOCKED, not deleted and not itself a change. An edited event may be
    edited again; a delete of a deleted one is refused, a client retrying a
    delete giving it an idempotency key (§13.4)."""
    if (
        target is None
        or target.room_id != room_id
        or target.status == EventStatus.BLOCKED
        or is_deleted(target)
        or concerns_change(target)
    ):
        return TARGET_NOT_FOUND
    return None


def author(event: RoomEvent, people: People) -> tuple[str, ...] | None:
    """Who authored *event* (RFC §10.3 rule 3): for an event that travelled
    outbound, its channel, whatever participant it names, since a channel
    never speaks as a participant; otherwise the participant its sender id
    names on its channel, else that sender id on its channel; else nobody.
    The participant's status does not matter: one who left, or was banned,
    still authored their messages."""
    source = event.source
    if source.direction == EventDirection.OUTBOUND:
        return ("channel", source.channel_id)
    person = people.of(event)
    if person is not None:
        return ("participant", person.id)
    if source.participant_id:
        return ("sender", source.channel_id, source.participant_id)
    return None


def claims_system(change: Change) -> bool:
    """Whether *change* claims the system's authority (RFC §10.3 rule 6): a
    SYSTEM delete, or an edit whose ``edit_source`` is ``system``."""
    if isinstance(change, DeleteContent):
        return change.delete_type == DeleteType.SYSTEM
    return change.edit_source == _SYSTEM_SOURCE


def authority_refusal(
    change: Change, event: RoomEvent, target: RoomEvent, people: People, *, by_hook: bool
) -> str | None:
    """Why *event*'s sender may not make *change* of *target* (RFC §10.3
    rules 4 to 6), ``None`` when they may. A change a hook injected
    (*by_hook*) is the hook's, and a SYSTEM change only. A SYSTEM change
    comes from a SYSTEM channel; an ADMIN delete from an active OWNER or
    AGENT; any other is the sender's, made of their own event."""
    if by_hook or claims_system(change):
        system = by_hook or event.source.channel_type == ChannelType.SYSTEM
        return None if system and claims_system(change) else NOT_AUTHORIZED
    if isinstance(change, DeleteContent) and change.delete_type == DeleteType.ADMIN:
        return _admin_refusal(event, people)
    mine, theirs = author(event, people), author(target, people)
    if mine is None or theirs is None:
        return IDENTITY_REQUIRED
    return None if mine == theirs else NOT_ORIGINAL_AUTHOR


def _admin_refusal(event: RoomEvent, people: People) -> str | None:
    """Why *event*'s sender holds no administrative authority (RFC §10.3
    rule 5), ``None`` when they do: an OWNER or an AGENT, neither LEFT nor
    BANNED, as the participant they are on their channel. A channel's own
    output holds no participant's authority, an id that names an owner on
    another channel none, and a sender who names no one is not identified.
    Moderation that outranks the roster belongs on the host's own calls
    (``update_event`` / ``delete_event``), whose authorization the host owns."""
    if event.source.direction == EventDirection.OUTBOUND:
        return NOT_AUTHORIZED
    if not event.source.participant_id:
        return IDENTITY_REQUIRED
    record = people.named(event)
    if record is None or record.role not in _ADMINISTRATORS or record.status in _GONE:
        return NOT_AUTHORIZED
    return None


def applied(change: Change, target: RoomEvent) -> tuple[RoomEvent, HookTrigger]:
    """*target* as the commit of *change* writes it, its other metadata kept
    (RFC §10.3 state updates 1 and 2), and the mutation trigger fired with
    it: an edit writes its new content and ``metadata.edited``, a delete
    ``metadata.deleted``."""
    if isinstance(change, EditContent):
        edited = target.model_copy(
            update={
                "content": change.new_content,
                "metadata": {**target.metadata, "edited": True},
            }
        )
        return edited, HookTrigger.ON_EVENT_UPDATED
    deleted = target.model_copy(update={"metadata": {**target.metadata, "deleted": True}})
    return deleted, HookTrigger.ON_EVENT_DELETED


def with_target(context: RoomContext, target: RoomEvent | None) -> RoomContext:
    """*context* holding *target* as its change's commit left it, in place of
    the copy its history read before (RFC §10.3 state update 5): a change is
    delivered with the target it changed. A target outside the history is
    not added."""
    if target is None or not any(e.id == target.id for e in context.recent_events):
        return context
    events = [target if e.id == target.id else e for e in context.recent_events]
    return context.model_copy(update={"recent_events": events})


def withheld(events: list[RoomEvent]) -> list[RoomEvent]:
    """*events* as a history a channel is handed holds them (RFC §7.5 rule
    8, §10.3): a soft-deleted event, and an edit of one, which holds a
    version of its text, stay with their content withheld.

    Deleted is what the event's mark says, ``metadata.deleted``, which only
    an accepted delete's commit writes, on its target and on each edit of it
    (or the host, which owns the record): an edit is withheld whether or not
    the history still holds the message it corrected. An edit of a message
    the history marks deleted is withheld too, for an edit stored before
    edits were marked. A DELETE row proves nothing by itself: one stored
    without step 8a (a host's ``commit_event``, a child room's trace) was
    never checked. The event keeps its id, index, type, source with the
    provider's message id, thread, metadata and ``channel_data``; its
    stored record keeps its content, and so do the histories hooks read.
    """
    deleted = {event.id for event in events if is_deleted(event)}
    if not deleted:
        return events
    return [
        _without_content(event)
        if event.id in deleted
        or (isinstance(event.content, EditContent) and event.content.target_event_id in deleted)
        else event
        for event in events
    ]


def for_channels(context: RoomContext) -> RoomContext:
    """*context* as a channel is handed it (RFC §10.3): its history
    :func:`withheld`. The same object when nothing in it was deleted; the
    context hooks receive stays whole."""
    events = withheld(context.recent_events)
    if events is context.recent_events:
        return context
    return context.model_copy(update={"recent_events": events})


def _without_content(event: RoomEvent) -> RoomEvent:
    """*event* with what was said withheld: an empty text in its place, or
    in an edit's new content, and none of the provider's original payload,
    which holds the text. An empty text rather than a placeholder, whose
    words a model would read as the sender's; the mark says what
    happened."""
    empty = TextContent(body="")
    content = event.content
    kept = (
        content.model_copy(update={"new_content": empty})
        if isinstance(content, EditContent)
        else empty
    )
    source = event.source.model_copy(update={"raw_payload": {}})
    return event.model_copy(update={"content": kept, "source": source})
