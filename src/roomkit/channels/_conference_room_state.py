"""What a conference channel knows about one room.

A channel serves many rooms at once and holds a dozen things about each: which
generation of the attachment is current, the bot in the conference and the ones
still leaving it, what is subscribed, what work is in flight. They are facets of
one thing — an attachment to a room — and the guarantees are written across
them: a lane may only resume if the generation *and* the bot session are still
the ones it started on; a credential is only handed out if the room is still
attached at the same generation it was minted for.

Held together here so that reading them together is the only way to read them.

Lifetime is deliberate: a record is created for any room the channel has ever
served and never removed. The join lock and the generation counter must outlive
the attachment — a re-attach that minted a second lock would let two joins
publish the AI on two tracks, and a generation reset to zero would make stale
background work look current. The fields with shorter lives are emptied by the
detach that ends them. Nothing reads a record's *existence*: an unattached room
is ``attached=False``, not a missing key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.models.enums import Access

if TYPE_CHECKING:
    from roomkit.conference.models import (
        BotSession,
        ConferenceAccess,
        ConferenceGrants,
        ConferenceTrack,
    )
    from roomkit.models.channel import ChannelBinding


@dataclass
class LeavingSession:
    """A bot session on its way out of a conference, and why it is not out yet.

    The two travel together because the second is only readable through the
    first: a session ``leave()`` refused to take out is still in the meeting,
    and the room reporting it present is the whole of what an integrator has to
    go on (RFC 17.7).
    """

    bot: BotSession
    error: str | None = None
    """What ``leave()`` raised, when it did. ``None`` while it is still ahead."""
    task: asyncio.Task[bool] | None = None
    """The one ``leave()`` in flight for this session, when one is.

    Departures are exact-once (RFC 12.10.4): a path that wants this session
    out while the task runs joins it instead of asking the backend a second
    time — two concurrent ``leave()`` calls for one session ask the SFU to
    remove a participant twice, and whichever answer arrives second is about
    a session that no longer exists. A *finished* task is not a lock: a
    departure that failed leaves ``error`` behind, and the close's retry is a
    new departure.
    """
    owed_an_end: bool = False
    """Whether a detach still has a ``conference_ended`` to announce for it.

    A detach owes one to the session it took out of the room, and holds it back
    for as long as the bot is still in the conference — the event says a
    departure happened. A session abandoned mid-join is not owed one: nothing
    announced its start either.
    """


class ConferenceRoomState:
    """One room's share of a conference channel."""

    def __init__(self) -> None:
        # Per room rather than per channel: held across the backend's network
        # join, and one conference taking its time must not decide when every
        # other room may connect or disconnect.
        self.lock = asyncio.Lock()
        # The conference room's own existence: `ensure_room` and `close_room`,
        # and nothing else. A teardown deferred past a re-attach reaches its
        # `close_room` with the room already recreated under the same name, and
        # a generation read outside this would be read before the create and
        # acted on after it — which destroys the conference that replaced the
        # one being torn down.
        #
        # Separate from `lock`, which a join holds across the backend's connect.
        # A teardown queued behind a network join would be waiting on a call it
        # has no budget for, and the whole point of the teardown is that it is
        # bounded.
        self.sfu_room = asyncio.Lock()
        self.generation = 0
        self.attached = False
        self.binding: ChannelBinding | None = None
        self.bot: BotSession | None = None
        # What the SFU applied to `bot` when it joined — the grants the session
        # actually holds, kept so a hot-plug can tell whether the session must
        # be re-permissioned or replaced (RFC 12.10.4). Written at every join;
        # meaningful only while `bot` is.
        self.bot_grants: ConferenceGrants | None = None
        # Set for the whole of a join. The bot exists in the conference before
        # it exists here, and that gap is the only reason the configured
        # identity has to stand in for the session's.
        self.joining = False
        # A squatter on the bot's identity publishes track after track; the
        # collision is worth saying once, not once per event.
        self.collision_reported = False
        # Sessions out of `bot` that have not left the conference yet, by
        # session id. A room can be leaving more than once: a teardown held open
        # is still running when a re-attach brings a second bot in and a second
        # detach sends it out behind the first.
        #
        # An entry stays until the session has *actually* left, which is not the
        # same as until the teardown has run: a `leave()` the backend refused
        # leaves the bot sitting in the conference, and the entry carries the
        # reason from then on.
        self.leaving: dict[str, LeavingSession] = {}
        self.pending_teardown: asyncio.Task[None] | None = None
        self.tasks: set[asyncio.Task[None]] = set()
        self.mints: set[asyncio.Task[ConferenceAccess]] = set()
        # Bumped when a track is unpublished, which is what lets a subscription
        # still in flight discover that its track is gone. The room generation
        # cannot answer that: the room is current, the track stopped existing.
        self.track_epochs: dict[str, int] = {}
        # What the bot is subscribed to. The lanes cannot stand in for it: a
        # channel that records without transcribing has no lanes, and the
        # subscriptions would be left behind by every path reading the lanes to
        # find them.
        self.subscribed: dict[str, ConferenceTrack] = {}

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def bump(self) -> int:
        """Close the current generation of the attachment and return the new one."""
        self.generation += 1
        return self.generation

    def is_current(self, generation: int, bot: BotSession) -> bool:
        """Whether the room is unchanged since a background step started.

        Cancellation is not a guarantee: a backend may well shield its network
        call, so a task can resume after the detach that cancelled it. Every
        await in background work is followed by this check.
        """
        return self.generation == generation and self.bot is bot

    def may_collect(self) -> bool:
        """Whether the channel is currently allowed to take audio from the room.

        A room the channel is not attached to fails closed. The binding alone
        cannot decide it: detaching drops the binding, and an absent binding
        means "attached, binding not yet seen" — which is open. So a lane
        suspended in speech recognition would resume mid-teardown, pass this
        gate on a room the channel had left, and deliver a transcription into
        the end announcement.
        """
        if not self.attached:
            return False
        if self.binding is None:
            return True
        return not self.binding.muted and self.binding.access in (
            Access.READ_WRITE,
            Access.WRITE_ONLY,
        )

    # -------------------------------------------------------------------------
    # Tracks
    # -------------------------------------------------------------------------

    def track_token(self, track_id: str) -> int:
        """Current generation of a track's publication."""
        return self.track_epochs.get(track_id, 0)

    def bump_track(self, track_id: str) -> None:
        """Record that a track has been unpublished."""
        self.track_epochs[track_id] = self.track_epochs.get(track_id, 0) + 1

    def is_subscribed(self, track_id: str) -> bool:
        """Whether the bot is consuming a track."""
        return track_id in self.subscribed

    def subscribe(self, track: ConferenceTrack) -> None:
        """Start routing a track's frames."""
        self.subscribed[track.id] = track

    def forget_subscription(self, track_id: str) -> bool:
        """Stop routing one track's frames. Returns whether it was being routed."""
        return self.subscribed.pop(track_id, None) is not None

    def forget_subscriptions(self) -> list[str]:
        """Stop routing every track, and say which those were."""
        track_ids = list(self.subscribed)
        self.subscribed.clear()
        return track_ids

    # -------------------------------------------------------------------------
    # Work in flight
    # -------------------------------------------------------------------------

    def spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run background work for the room, tracked so a detach can cancel it."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(log_task_exception)

    def cancel_tasks(self) -> None:
        """Cancel the room's background work."""
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()

    # -------------------------------------------------------------------------
    # Leaving
    # -------------------------------------------------------------------------

    def start_leaving(self, bot: BotSession) -> None:
        """Record that a detach is taking one session out, and it is not out yet."""
        self.leaving[bot.id] = LeavingSession(bot=bot, owed_an_end=True)

    def start_closing(self, bot: BotSession) -> None:
        """Move an active bot onto the close's durable departure ledger.

        Closing inventories every room before its first external await. That
        way a budget expiring in one room cannot make a later room's bot
        disappear when ``bot`` is cleared. Unlike a detach, a channel close
        never announced that it owed this session a ``conference_ended``.
        """
        self.leaving.setdefault(bot.id, LeavingSession(bot=bot))

    def record_leave_failure(self, bot: BotSession, reason: str) -> None:
        """Record that ``leave()`` did not take a session out of the conference.

        Onto ``leaving`` as well as into the reason, because the callers arrive
        from opposite directions: a detach has already put the session there,
        while a close and an abandoned join reach this holding a session the
        room has no other record of — and one the channel cannot remove must be
        reported present whichever way it got here.
        """
        entry = self.leaving.get(bot.id)
        if entry is None:
            entry = self.leaving[bot.id] = LeavingSession(bot=bot)
        entry.error = reason

    def forget_leaving(self, bot: BotSession | None) -> None:
        """Record that one session has finished leaving the room."""
        if bot is not None:
            self.leaving.pop(bot.id, None)

    def stuck_sessions(self) -> list[LeavingSession]:
        """Sessions still in the conference because ``leave()`` refused to work.

        Only those: a session whose teardown has not reached its ``leave()`` yet
        is on its way out under its own power, and a second attempt behind it
        would be two ``leave()`` calls for one session.
        """
        return [entry for entry in self.leaving.values() if entry.error is not None]

    def closing_sessions(self) -> list[LeavingSession]:
        """Sessions the channel close owns or must retry.

        An entry with a departure still running belongs to whichever path
        started it — retrying would be a second ``leave()`` for one session,
        and announcing its end is that path's own obligation. An entry with
        ``owed_an_end`` and no error belongs to a detach whose teardown is
        still running, for the same reason. Entries created by
        :meth:`start_closing` owe no announcement; failed detach sessions
        carry an error and are retried at the channel's last opportunity.
        """
        return [
            entry
            for entry in self.leaving.values()
            if (entry.task is None or entry.task.done())
            and (not entry.owed_an_end or entry.error is not None)
        ]

    def leave_failures(self) -> dict[str, str]:
        """Why each session that could not be removed is still in the conference."""
        return {
            session_id: entry.error
            for session_id, entry in sorted(self.leaving.items())
            if entry.error is not None
        }
