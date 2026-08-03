"""Who is speaking, as a transcript should name them.

Shared by the surfaces that write a room's conversation out for someone else
to read — the console transcript, and the room context an ACP session is
handed. Both answer the same question, and a room where the console says
"Marie · sms" while an agent is told "sms" names one person two ways.
"""

from __future__ import annotations

from collections.abc import Callable

from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent


def speaker_label(
    event: RoomEvent,
    context: RoomContext,
    agent_label: Callable[[str], str] | None = None,
) -> str:
    """Name the author of *event* the way a reader should see it.

    A person gets their own name and the channel they speak through —
    ``"Marie · sms"`` — because in a room holding several humans, the channel
    id names none of them: two colleagues texting in would otherwise share one
    handle. Anything without a participant (an agent, a system event) keeps the
    channel-derived label.

    ``agent_label`` renames those channel-derived labels for presentation; it
    defaults to the channel id itself, which is what a room addressed by
    ``@channel-id`` should show.

    ``source.participant_id`` holds a ``Participant.id`` when the channel names
    its own sender, and an ``Identity.id`` when the identity pipeline resolved
    one (RFC §11) — two namespaces in one field, so both are tried.
    """
    label = agent_label if agent_label is not None else _channel_id_label
    participant_id = event.source.participant_id
    if not participant_id:
        return label(event.source.channel_id)

    person = next(
        (p for p in context.participants if p.id == participant_id),
        None,
    ) or next(
        (p for p in context.participants if p.identity_id == participant_id),
        None,
    )
    if person is None:
        return label(event.source.channel_id)
    return f"{person.display_name or person.id} · {event.source.channel_id}"


def _channel_id_label(channel_id: str) -> str:
    """The channel id, unchanged — the default way to name a non-person."""
    return channel_id
