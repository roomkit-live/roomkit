"""The channels a participant has been reached through (RFC §5.5).

A participant is one record per (room, id): the same person reached by SMS and
then by email is one participant, not two, which is what a cross-channel
identity is for. So a caller naming a channel the record was not created on gets
that record back — legitimate, and it must not fail. What it must not be is
silent, because the record handed over still names another channel in
``channel_id``, and a caller that goes on to keep a lifecycle or a status on it
is driving another channel's record without having been told. That is how
leaving a conference came to erase a team-channel membership.

Two separate things, kept separate: the list every writer of a record owes it,
and the warning only a caller who *named* a channel is owed. A backend
reporting an arrival on its own channel writes the first and not the second —
it named nothing, and a warning on every arrival would repeat what the caller
who did name one was already told.
"""

from __future__ import annotations

import logging

from roomkit.models.participant import Participant

logger = logging.getLogger("roomkit.framework")


def channels_reached(existing: Participant, channel_id: str) -> list[str] | None:
    """The ``connected_via`` to store now that *channel_id* has reached *existing*.

    The primary channel comes first and the rest keep their order of first
    sight. Returns ``None`` when the record already lists them all — there is
    nothing to write, and a record whose channels have not changed should not
    be rewritten to say so.
    """
    channels = list(dict.fromkeys([existing.channel_id, *existing.connected_via, channel_id]))
    return channels if channels != existing.connected_via else None


def warn_cross_channel(existing: Participant, channel_id: str, *, rehomed: bool) -> None:
    """Say that *channel_id* has been handed a record homed on another channel.

    A no-op when the caller named the record's own channel: there is nothing to
    warn about, and this runs wherever a participant is looked up.

    *rehomed* is for a deliberate join, the one caller that may move the primary
    channel to the one being joined through — the record still exists once, and
    the channel it replaces stays on the list, but a caller reading
    ``channel_id`` afterwards sees only the new one.
    """
    if channel_id == existing.channel_id:
        return
    if rehomed:
        logger.warning(
            "Participant %s of room %s is joining through channel %r; the primary "
            "channel of their record was %r and becomes %r — one record still, and "
            "%r stays in connected_via (RFC 5.5).",
            existing.id,
            existing.room_id,
            channel_id,
            existing.channel_id,
            channel_id,
            existing.channel_id,
            extra={"room_id": existing.room_id},
        )
        return
    logger.warning(
        "Participant %s of room %s is recorded on channel %r; %r asked for them "
        "and gets that record as it stands, primary channel included (RFC 5.5). "
        "Reaching one participant on several channels is the point — but a "
        "lifecycle or a status kept on this record from %r moves it for %r too.",
        existing.id,
        existing.room_id,
        existing.channel_id,
        channel_id,
        channel_id,
        existing.channel_id,
        extra={"room_id": existing.room_id},
    )
