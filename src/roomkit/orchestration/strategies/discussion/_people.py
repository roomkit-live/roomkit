"""Who is who in a discussion: people, their labels, what an agent may read (RFC §19.7.5).

A person is told apart by the label the transcript gives them (§6.4): their
name kept to a name's characters, with the rank a look-alike of an earlier
source carries (``Alice (2)``), or the channel of a sender with no name
(``@sms1``), a form no name takes.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from roomkit._text import person_name
from roomkit.channels._speaker import channel_label, label_name, turn_labels
from roomkit.core.visibility import effective_visibility, visibility_allows
from roomkit.models.context import RoomContext
from roomkit.models.delivery import SYSTEM_SENDER_ID
from roomkit.models.enums import Access, ChannelCategory, ParticipantRole, ParticipantStatus
from roomkit.models.event import RoomEvent
from roomkit.models.participant import Participant

from ._names import NAME_LIMIT, name_key, same_name

_NOT_PEOPLE = frozenset({ParticipantRole.AGENT, ParticipantRole.BOT})
_READS = frozenset({Access.READ_WRITE, Access.READ_ONLY})

PeopleIndex = dict[str, list[str]]
"""Each name people are addressed by, ignoring case, and the labels it names."""


def sees(agent: str, event: RoomEvent, context: RoomContext) -> bool:
    """Whether *agent* may read *event*: a name is no way around visibility.
    The scope is the event's own, else its source binding's (§7.5)."""
    binding = context.get_binding(agent)
    if binding is None or binding.access not in _READS:
        return False
    scope = effective_visibility(event, context.get_binding(event.source.channel_id))
    return visibility_allows(scope, binding)


def is_person(event: RoomEvent, context: RoomContext) -> bool:
    """A transport's event from a participant that is neither an agent nor a
    bot, or with no participant record behind it (rule 2); never the
    framework's own sender of a delivery (§22)."""
    if event.source.participant_id == SYSTEM_SENDER_ID:
        return False
    binding = context.get_binding(event.source.channel_id)
    if binding is None or binding.category != ChannelCategory.TRANSPORT:
        return False
    participant = _participant(event, context)
    return participant is None or participant.role not in _NOT_PEOPLE


def records_people(context: RoomContext) -> bool:
    """Whether the room records a person among its participants."""
    return any(_active_person(p) for p in context.participants)


def person_label(event: RoomEvent, context: RoomContext) -> str:
    """The person a message is from, as the transcript labels them."""
    return turn_labels([event], context).get(event.id) or channel_label(event.source.channel_id)


def people_index(
    context: RoomContext, agents: Collection[str], listed: Sequence[str] | None
) -> PeopleIndex:
    """Each name agents address people by, ignoring case, and the people it
    names, as the transcript labels them (§6.4): the room's active people and
    the speakers of its recent messages; with *listed* (``people``), only
    those. A name is kept to a name's characters (``@AliceMartin``), never an
    agent's channel id nor longer than :data:`NAME_LIMIT`; two labels behind
    one name (``Alice Martin``, ``AliceMartin``) make it ambiguous, and a
    look-alike's ranked label (``Alice (2)``) is addressed by no name."""
    labels = [_label_of(p) for p in context.participants if _active_person(p)]
    spoken = turn_labels([e for e in context.recent_events if is_person(e, context)], context)
    labels += [label for label in spoken.values() if label and label == label_name(label)]
    allowed = None if listed is None else {name_key(p).casefold() for p in listed}
    taken = {a.casefold() for a in agents}
    index: PeopleIndex = {}
    for label in labels:
        name = name_key(label.removeprefix("@"))
        key = name.casefold()
        if not name or len(name) > NAME_LIMIT or key in taken:
            continue
        if allowed is not None and key not in allowed:
            continue
        named = index.setdefault(key, [])
        if not any(same_name(known, label) for known in named):
            named.append(label)
    return index


def names_of(index: PeopleIndex) -> list[str]:
    """The names an index addresses people by, each once."""
    return [name_key(labels[0].removeprefix("@")) for labels in index.values()]


def _participant(event: RoomEvent, context: RoomContext) -> Participant | None:
    pid = event.source.participant_id
    return next((p for p in context.participants if p.id == pid), None) if pid else None


def _active_person(participant: Participant) -> bool:
    return participant.status == ParticipantStatus.ACTIVE and participant.role not in _NOT_PEOPLE


def _label_of(participant: Participant) -> str:
    """The label the transcript gives a participant's turns: their name, else
    their channel's (``@sms1``)."""
    return person_name(participant.display_name) or channel_label(participant.channel_id)
