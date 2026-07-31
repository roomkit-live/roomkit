"""What a conference channel subscribes to, and when it stops.

Subscription is consumer-driven: a track reaches the process only when
something configured on the channel consumes it, which is what keeps a
ten-person meeting affordable — a camera nobody analyses is bandwidth and CPU
spent on frames nobody reads. The bot excludes itself, because some backends
report it back through the very callbacks it registered.

The rest is what a subscription has to survive. A room can be detached and a
track unpublished while a subscription is in flight, and the two fail
differently, so both are read again on the far side of the call rather than
assumed unchanged across it.

Split from ConferenceChannel for room, not for isolation: everything here reads
the channel it is mixed into, and the host contract says how much of it.

See RFC section 12.10.4.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels._conference_operations import ConferenceResource
from roomkit.conference.models import ConferenceTrack, TrackKind
from roomkit.models.enums import HookTrigger

if TYPE_CHECKING:
    from roomkit.channels._conference_activity import RoomActivity
    from roomkit.conference.base import ConferenceBackend
    from roomkit.conference.models import BotSession, ConferenceRecordingConfig
    from roomkit.voice.stt.base import STTProvider

logger = logging.getLogger("roomkit.channels.conference")


class ConferenceSubscriptionMixin:
    """Which tracks a conference channel consumes.

    Host contract — what ConferenceChannel provides:
        channel_id, _backend, _activity: the channel and what it talks to.
        _stt, _recording: what decides whether a kind of track is consumed.
        _room / _attached_room: the per-room record (ConferenceRoomState).
        _ensure_bot, _is_own_bot, _fire: reached on the way to a subscription.
        _open_lane, _stop_consuming: what a subscription starts and ends.
    """

    channel_id: str
    _backend: ConferenceBackend
    _operations: Any
    _activity: RoomActivity
    _stt: STTProvider | None
    _recording: ConferenceRecordingConfig | None

    # Provided by ConferenceChannel and ConferenceLanesMixin — see above
    _room: Any
    _attached_room: Any
    _ensure_bot: Any
    _is_own_bot: Any
    _fire: Any
    _open_lane: Any
    _stop_consuming: Any

    def _consumes(self, kind: TrackKind) -> bool:
        """Whether anything configured on this channel consumes a track kind.

        Video stays out of the process unless something looks at it: subscribing
        to every camera in a ten-person meeting costs bandwidth and CPU for
        frames nobody reads.
        """
        if kind is TrackKind.AUDIO:
            return self._stt is not None or self._recording is not None
        return False

    async def _on_track_published(self, room_id: str, track: ConferenceTrack) -> None:
        room = self._attached_room(room_id)
        if room is None:
            return
        if self._is_own_bot(room_id, track.participant_id):
            return
        # Read before the first await, not before the subscription. Everything
        # below — the hooks, bringing the bot in — is somewhere an unpublish can
        # land, and a generation read afterwards would take that unpublish for
        # the starting state and subscribe to a track nobody is publishing.
        track_token = room.track_token(track.id)
        async with self._activity.track(room_id):
            if not room.attached:
                return
            await self._fire(
                room_id,
                HookTrigger.ON_CONFERENCE_TRACK_PUBLISHED,
                "conference_track_published",
                f"Track {track.id} published",
                # The kind is what an interface acts on — a VIDEO publication
                # is "the camera came on" — and it matches the mute pair's
                # payload, so both read the same way.
                {
                    "track_id": track.id,
                    "participant_id": track.participant_id,
                    "kind": track.kind.value,
                },
            )
            if track.kind is TrackKind.SCREEN_SHARE:
                await self._fire(
                    room_id,
                    HookTrigger.ON_SCREEN_SHARE_STARTED,
                    "screen_share_started",
                    f"Screen share started by {track.participant_id}",
                    {"track_id": track.id, "participant_id": track.participant_id},
                )
        if not self._consumes(track.kind) or not room.may_collect():
            return
        generation = room.generation
        bot = await self._ensure_bot(room_id)
        await self._subscribe_track(room_id, track, bot, generation, track_token)

    async def _on_track_unpublished(self, room_id: str, track: ConferenceTrack) -> None:
        room = self._attached_room(room_id)
        if room is None:
            return
        if self._is_own_bot(room_id, track.participant_id):
            return
        # Recorded before anything is awaited, so a subscription still in
        # flight for this track sees the unpublish when it re-reads the token.
        # Closing the lane only reaches a track that already has one.
        room.bump_track(track.id)
        async with self._activity.track(room_id):
            if not room.attached:
                return
            if room.forget_subscription(track.id):
                await self._release_track(room.bot, track.id)
            await self._fire(
                room_id,
                HookTrigger.ON_CONFERENCE_TRACK_UNPUBLISHED,
                "conference_track_unpublished",
                f"Track {track.id} unpublished",
                {
                    "track_id": track.id,
                    "participant_id": track.participant_id,
                    "kind": track.kind.value,
                },
            )
            if track.kind is TrackKind.SCREEN_SHARE:
                await self._fire(
                    room_id,
                    HookTrigger.ON_SCREEN_SHARE_STOPPED,
                    "screen_share_stopped",
                    f"Screen share stopped by {track.participant_id}",
                    {"track_id": track.id, "participant_id": track.participant_id},
                )

    async def _on_track_muted(self, room_id: str, track: ConferenceTrack) -> None:
        """Relay a publisher muting their track — "camera off" included.

        Presence, not media: most clients express a camera toggle as a muted
        VIDEO track rather than an unpublish, so this pair and the track's
        kind are what a management interface reads its microphone and camera
        indicators from (RFC 12.10.4). Not gated by the binding's collection
        state, like the other SFU signals; the bot's own tracks are excluded
        exactly as they are from every other track event.
        """
        await self._relay_mute(
            room_id, track, HookTrigger.ON_CONFERENCE_TRACK_MUTED, "conference_track_muted"
        )

    async def _on_track_unmuted(self, room_id: str, track: ConferenceTrack) -> None:
        await self._relay_mute(
            room_id, track, HookTrigger.ON_CONFERENCE_TRACK_UNMUTED, "conference_track_unmuted"
        )

    async def _relay_mute(
        self, room_id: str, track: ConferenceTrack, trigger: HookTrigger, code: str
    ) -> None:
        room = self._attached_room(room_id)
        if room is None:
            return
        if self._is_own_bot(room_id, track.participant_id):
            return
        async with self._activity.track(room_id):
            if not room.attached:
                return
            await self._fire(
                room_id,
                trigger,
                code,
                f"Track {track.id} of {track.participant_id} is "
                f"{'muted' if track.muted else 'unmuted'}",
                {
                    "track_id": track.id,
                    "participant_id": track.participant_id,
                    "kind": track.kind.value,
                },
            )

    async def _release_track(self, bot: BotSession | None, track_id: str) -> None:
        """Stop a track arriving, and close what it was feeding.

        The order is the useful one — the frames stop before the lane that
        consumed them goes — but only the second half is this channel's to
        guarantee. A backend that refuses the unsubscribe has left the channel
        receiving frames it now drops, which is waste; skipping the teardown
        behind it would leave the lane, its pipeline stage state and the track's
        recording alive with nothing able to reach them again, since the room
        has already forgotten the subscription they were found through.

        RFC 12.10.4 step 4 asks for all three of unsubscribe, teardown and
        hooks, and the one that failed is the one that gets logged.
        """
        await self._unsubscribe_quietly(bot, track_id)
        await self._stop_consuming(track_id)

    async def _unsubscribe_quietly(self, bot: BotSession | None, track_id: str) -> None:
        """Ask the backend to stop delivering a track, and carry on if it will not."""
        if bot is None:
            return
        try:
            with self._operations.use(
                ConferenceResource.BACKEND, what=f"unsubscribing track {track_id}"
            ):
                await self._backend.unsubscribe_track(bot, track_id)
        except Exception:
            logger.exception(
                "Conference channel %r could not unsubscribe from track %s. Its lane and "
                "recording are being closed regardless; frames the backend keeps sending "
                "are dropped on arrival",
                self.channel_id,
                track_id,
            )

    async def _on_active_speaker_changed(self, room_id: str, participant_id: str) -> None:
        room = self._attached_room(room_id)
        if room is None:
            return
        async with self._activity.track(room_id):
            if not room.attached:
                return
            await self._fire(
                room_id,
                HookTrigger.ON_ACTIVE_SPEAKER_CHANGED,
                "conference_active_speaker",
                f"Active speaker is {participant_id}",
                {"participant_id": participant_id},
            )

    async def _on_connection_quality(
        self, room_id: str, participant_id: str, quality: str
    ) -> None:
        """Relay the SFU's view of a participant's connection, per participant.

        Not collection — no media is read to relay it — so it is not gated by
        the binding's collection state, exactly like the active-speaker signal
        above (RFC 12.10.4). A quality bar in a management interface is the
        consumer.
        """
        room = self._attached_room(room_id)
        if room is None:
            return
        async with self._activity.track(room_id):
            if not room.attached:
                return
            await self._fire(
                room_id,
                HookTrigger.ON_CONNECTION_QUALITY_CHANGED,
                "conference_connection_quality",
                f"Connection quality of {participant_id} is {quality}",
                {"participant_id": participant_id, "quality": quality},
            )

    async def _apply_collection_state(self, room_id: str) -> None:
        """Bring subscriptions in line with what the binding now allows."""
        room = self._room(room_id)
        bot = room.bot
        if bot is None:
            return
        if not room.may_collect():
            # Forget, then unsubscribe, then tear down. Forgetting the
            # subscription is what stops frames being routed and unsubscribing
            # is what stops them arriving; neither should queue behind a task
            # cancellation, which is only cleanup.
            track_ids = room.forget_subscriptions()
            for track_id in track_ids:
                await self._unsubscribe_quietly(bot, track_id)
            for track_id in track_ids:
                await self._stop_consuming(track_id)
            return
        generation = room.generation
        with self._operations.use(
            ConferenceResource.BACKEND, what=f"listing participants of room {room_id}"
        ):
            participants = await self._backend.list_participants(room_id)
        if not room.is_current(generation, bot):
            return
        for participant in participants:
            for track in participant.tracks:
                if self._is_own_bot(room_id, track.participant_id):
                    continue
                if not self._consumes(track.kind) or room.is_subscribed(track.id):
                    continue
                if not await self._subscribe_track(room_id, track, bot, generation):
                    return

    async def _subscribe_track(
        self,
        room_id: str,
        track: ConferenceTrack,
        bot: BotSession,
        generation: int,
        track_token: int | None = None,
    ) -> bool:
        """Subscribe to a track and start consuming it, or undo the subscription.

        Cancellation is not a guarantee — an SDK may shield its network call —
        so the room *and the track* are read again once the subscription lands
        rather than assumed unchanged across it. Both can move underneath it,
        and they fail differently: the room can be detached, and the track can
        be unpublished while the room carries on. The unpublish callback can
        only close what the channel has registered, so a track that goes away
        during the subscription would otherwise leave behind a subscription and
        a lane for something nobody is publishing.

        A lane opened on a room the channel has left costs more than a stray
        subscription: it owns a task nobody will cancel and holds pipeline
        stage state, some of it native, that nobody will release.

        Returns whether the room is still the one the caller started on, so a
        caller working through a list stops rather than subscribing the rest
        into a room that has moved on. A track that vanished is not that: the
        room is fine, so the caller carries on with the others.

        ``track_token`` is the track's generation as of the caller's *first*
        await, which is not always this method's. A caller that ran hooks and
        brought the bot in before getting here has already given the unpublish
        several places to land, and reading the generation now would adopt the
        unpublish as the baseline instead of noticing it. A caller with no
        await ahead of this one may leave it out.
        """
        room = self._room(room_id)
        if track_token is None:
            track_token = room.track_token(track.id)
        with self._operations.use(
            ConferenceResource.BACKEND, what=f"subscribing track {track.id}"
        ):
            await self._backend.subscribe_track(bot, track.id)
            room_current = room.is_current(generation, bot) and room.may_collect()
            if room_current and room.track_token(track.id) == track_token:
                room.subscribe(track)
                self._open_lane(room_id, track)
                return True
            await self._backend.unsubscribe_track(bot, track.id)
            return room_current
