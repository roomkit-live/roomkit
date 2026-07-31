"""Hot-plugging intelligence into a running conference channel.

The configuration first need is read from — stt, tts, recording — is not fixed
at construction (RFC 12.10.4): each can be plugged into and unplugged from a
channel that is serving live conferences. A plug is a first need in its own
right — the occupancy probe is re-run, an occupied conference is joined at
once, and the tracks already published are subscribed — and unplugging the
last need takes the bot out: a session kept past the last consumer and the
last voice is the silent observer RFC section 17.7 refuses.

The grants follow. A derived grant set is computed from the configuration in
force at the join, so a plug that widens what a live session must do brings
its grants in line: through the backend's ``update_bot_grants`` where the
BOT_GRANT_UPDATE capability is declared — the session and the event bridge
survive — and by re-joining otherwise. An explicit ``bot_grants`` is never
rewritten; the caller who set it took coverage on themselves.

Changes are serialised on one lock: everything a change triggers — the grants
derived, the subscriptions re-evaluated, the join or leave decided — is read
from the configuration as a whole, and two changes interleaving would act on
a configuration neither of them describes.

Split from ConferenceChannel for room, not for isolation: everything here reads
the channel it is mixed into, and the host contract says how much of it.

See RFC sections 12.10.4 and 17.7.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Any

# Read through the module rather than bound into this one, so a deployment's
# or a test's override of the drain budget applies here too.
from roomkit.channels import _conference_activity
from roomkit.channels._conference_operations import ConferenceResource
from roomkit.conference.models import ConferenceCapability, ConferenceGrants
from roomkit.core.exceptions import RoomNotAttachedError
from roomkit.core.task_utils import log_task_exception
from roomkit.voice.pipeline.engine import AudioPipeline

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.channels._conference_recording import ConferenceRecording
    from roomkit.channels._conference_voice import ConferenceVoice
    from roomkit.conference.base import ConferenceBackend
    from roomkit.conference.models import ConferenceRecordingConfig
    from roomkit.recorder.base import MediaRecorder
    from roomkit.voice.pipeline.config import AudioPipelineConfig
    from roomkit.voice.stt.base import STTProvider
    from roomkit.voice.tts.base import TTSProvider

logger = logging.getLogger("roomkit.channels.conference")


class ConferencePlugMixin:
    """The public plug/unplug surface and what each change sets in motion.

    Host contract — what ConferenceChannel provides:
        channel_id, _backend, _voice, _operations: the channel and what it
            talks to.
        _stt, _pipeline, _pipeline_config, _recording, _recorder: the slots a
            plug fills and an unplug empties.
        _explicit_bot_grants, _bot_grants, _transport_only: the derivations a
            change moves (the last two are properties over the configuration).
        _e2ee, _close_providers, _max_queued_frames, _plug_lock,
            _deferred_closes: the constructor decisions a plug honours.
        _rooms / _room: the per-room records (ConferenceRoomState).
        _consumes, _resolve_pipeline, _resolve_recorder, _record_bot_audio:
            the constructor's own validation and wiring, reused so a plug can
            refuse exactly what construction refuses.
        _ensure_bot, _ensure_bot_for_resume, _apply_collection_state,
            _open_lane, _unsubscribe_quietly, _stop_consuming,
            _close_lane_instance, _leave_and_record, _announce_end: the join,
            subscription and departure machinery every change steers.
    """

    channel_id: str
    _backend: ConferenceBackend
    _voice: ConferenceVoice
    _stt: STTProvider | None
    _pipeline: AudioPipeline | None
    _pipeline_config: AudioPipelineConfig | None
    _recording: ConferenceRecordingConfig | None
    _recorder: ConferenceRecording | None
    _explicit_bot_grants: ConferenceGrants | None
    _recording_events: Any
    _e2ee: bool
    _close_providers: bool
    _max_queued_frames: int
    _plug_lock: asyncio.Lock
    _deferred_closes: set[asyncio.Task[None]]
    _operations: Any

    # Provided by ConferenceChannel and its other mixins — see above
    _rooms: Any
    _room: Any
    _lanes: Any
    _bot_grants: Any
    _transport_only: Any
    _consumes: Any
    _resolve_pipeline: Any
    _resolve_recorder: Any
    _record_bot_audio: Any
    _ensure_bot: Any
    _ensure_bot_for_resume: Any
    _apply_collection_state: Any
    _open_lane: Any
    _unsubscribe_quietly: Any
    _stop_consuming: Any
    _close_lane_instance: Any
    _leave_and_record: Any
    _announce_end: Any

    # -------------------------------------------------------------------------
    # Plugging — a need arrives
    # -------------------------------------------------------------------------

    async def plug_stt(
        self, stt: STTProvider, *, pipeline: AudioPipelineConfig | None = None
    ) -> None:
        """Plug speech recognition into the running channel.

        A first need in its own right: every attached room whose conference
        already holds participants is joined before this returns, and the
        tracks already published are subscribed — the meeting is transcribed
        from the plug forward, not from the next publication (RFC 12.10.4).
        An empty conference stays unjoined, exactly as at construction.

        Refuses what the constructor refuses, before touching any state: an
        E2EE conference cannot be transcribed, a pipeline without a VAD cannot
        segment, and a slot already holding a recognizer is not silently
        replaced — a swap is a teardown and a rebuild whatever single verb
        offers it, so the observation gap belongs in the open: unplug, then
        plug.
        """
        async with self._plug_lock:
            if self._stt is not None:
                raise ValueError(
                    "A speech recognizer is already plugged into this channel; call "
                    "unplug_stt() first. A swap is a teardown and a rebuild of the "
                    "lanes whatever verb offers it, so the two halves stay separate "
                    "operations (RFC 12.10.4)."
                )
            if self._e2ee:
                raise ValueError(
                    "STT cannot be plugged into an end-to-end encrypted conference: the "
                    "bot receives ciphertext it has no key for, so the lanes would "
                    "transcribe noise. The constructor's refusal holds identically at "
                    "the plug (RFC 12.10.4)."
                )
            config = self._resolve_pipeline(pipeline, stt)
            self._stt = stt
            self._pipeline_config = config
            self._pipeline = AudioPipeline(config) if config is not None else None
            await self._wake_rooms()

    async def plug_tts(self, tts: TTSProvider) -> None:
        """Plug a voice into the running channel.

        The bot can speak from the moment this returns. In a room where it is
        already sitting, its session's grants are widened first — in place
        when the backend can (BOT_GRANT_UPDATE), by re-joining when it cannot
        (RFC 12.10.4) — because a voice on a session without ``publish_audio``
        is a bot the SFU will silence.
        """
        async with self._plug_lock:
            if self._voice.tts is not None:
                raise ValueError(
                    "A synthesizer is already plugged into this channel; call "
                    "unplug_tts() first. A swap is two operations on purpose "
                    "(RFC 12.10.4)."
                )
            self._voice.set_tts(tts)
            await self._wake_rooms()

    async def plug_recording(
        self, recording: ConferenceRecordingConfig, *, recorder: MediaRecorder
    ) -> None:
        """Plug recording into the running channel.

        Subject to the constructor's own refusals — E2EE, a mode with no
        egress surface — via the same resolution path, so a configuration
        refused at construction is refused identically here. Consent gating
        is unchanged: each track's recording opens on its first frame, after
        ON_RECORDING_STARTED has been heard (RFC 17.6).
        """
        async with self._plug_lock:
            if self._recording is not None:
                raise ValueError(
                    "A recording is already plugged into this channel; call "
                    "unplug_recording() first. A swap is two operations on purpose "
                    "(RFC 12.10.4)."
                )
            resolved = self._resolve_recorder(
                recording, recorder, e2ee=self._e2ee, max_queued_frames=self._max_queued_frames
            )
            self._recording = recording
            self._recorder = resolved
            self._voice.set_on_published(self._record_bot_audio)
            await self._wake_rooms()

    # -------------------------------------------------------------------------
    # Unplugging — a need leaves
    # -------------------------------------------------------------------------

    async def unplug_stt(self) -> None:
        """Unplug speech recognition. Idempotent: an empty slot is left as asked.

        The lanes close — all of them, they exist only for recognition — and
        the tracks nothing else consumes are unsubscribed. When recognition
        was the last need, the bot leaves every conference it was in, with
        ``conference_ended`` announced: the channel is pure transport again
        (RFC 12.10.4). The recognizer and the pipeline are closed when the
        channel owns its providers (``close_providers``), once nothing the
        channel admitted still runs inside them.
        """
        async with self._plug_lock:
            if self._stt is None:
                return
            stt = self._stt
            pipeline = self._pipeline
            self._stt = None
            self._pipeline = None
            self._pipeline_config = None
            lanes = list(self._lanes.values())
            self._lanes.clear()
            results = await asyncio.gather(
                *(self._close_lane_instance(lane) for lane in lanes), return_exceptions=True
            )
            for lane, result in zip(lanes, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "Conference channel %r could not close the lane of track %s while "
                        "unplugging its recognizer: %s",
                        self.channel_id,
                        lane.track_id,
                        result,
                    )
            await self._settle_rooms()
            if self._close_providers:
                await self._close_retired(
                    resources=(ConferenceResource.PIPELINE, ConferenceResource.STT),
                    closer=partial(self._close_stt_pair, stt, pipeline),
                    what="the unplugged recognizer and its pipeline",
                )

    async def unplug_tts(self) -> None:
        """Unplug the voice. Idempotent: an empty slot is left as asked.

        An utterance in flight is ended the way a barge-in ends one —
        ``stop_playback`` and a terminal chunk — because the conference is
        live and the bot may be staying in it: the turn genuinely ended (RFC
        12.10.4). When the voice was the last need, the bot then leaves. The
        synthesizer is closed when the channel owns its providers.
        """
        async with self._plug_lock:
            tts = self._voice.set_tts(None)
            if tts is None:
                return
            await self._voice.interrupt_all()
            await self._settle_rooms()
            if self._close_providers:
                await self._close_retired(
                    resources=(ConferenceResource.TTS,),
                    closer=tts.close,
                    what="the unplugged synthesizer",
                )

    async def unplug_recording(self) -> None:
        """Unplug recording. Idempotent: an empty slot is left as asked.

        Frames stop reaching the recorder the moment the slot empties; the
        recordings already open are then finalized and announced, exactly as
        a detach finalizes them (RFC 12.10.8). The recorder itself is closed
        when the channel owns its providers, and left to the caller otherwise
        — the recordings this channel started are finished either way.
        """
        async with self._plug_lock:
            if self._recorder is None:
                return
            recorder = self._recorder
            self._recorder = None
            self._recording = None
            self._voice.set_on_published(None)
            await self._settle_rooms()
            await self._finalize_unplugged_recorder(recorder)

    # -------------------------------------------------------------------------
    # What a change sets in motion, room by room
    # -------------------------------------------------------------------------

    async def _wake_rooms(self) -> None:
        """Bring every attached room in line with a configuration that grew.

        Awaited rather than spawned: the plug's promise is that its effects
        are in force on return. A join or probe that fails does not fail the
        plug — the configuration stands and the lazy join remains, the same
        discipline as every other first-need trigger (RFC 12.10.4) — so
        nothing here raises past its room.
        """
        for room_id in list(self._rooms):
            room = self._rooms.get(room_id)
            if room is None or not room.attached:
                continue
            try:
                if room.bot is None:
                    # The plug is a first need: the occupancy probe of the
                    # attach is re-run, with all of its own re-reads and its
                    # quiet failure modes.
                    await self._ensure_bot_for_resume(room_id, room.generation, trigger="plug")
                else:
                    await self._align_grants(room_id)
                if room.bot is not None:
                    await self._apply_collection_state(room_id)
                    # The re-evaluation subscribes what nothing consumed before
                    # and opens lanes on the way — but a track subscribed *by
                    # the recording* before this plug is skipped there, and it
                    # is exactly the track a plugged recognizer must reach:
                    # transcribed from the plug forward, whoever subscribed
                    # first. _open_lane refuses duplicates and no-ops without
                    # a pipeline, so this is the missing half and nothing more.
                    for track in list(room.subscribed.values()):
                        self._open_lane(room_id, track)
            except Exception:
                logger.exception(
                    "Conference channel %r could not bring room %s in line with a newly "
                    "plugged need; the lazy join remains and the next trigger retries",
                    self.channel_id,
                    room_id,
                )

    async def _settle_rooms(self) -> None:
        """Bring every attached room in line with a configuration that shrank."""
        for room_id in list(self._rooms):
            room = self._rooms.get(room_id)
            if room is None or not room.attached:
                continue
            try:
                if self._transport_only:
                    await self._retire_bot(room_id)
                else:
                    await self._drop_unconsumed(room_id)
                    await self._align_grants(room_id)
            except Exception:
                logger.exception(
                    "Conference channel %r could not bring room %s in line after an unplug",
                    self.channel_id,
                    room_id,
                )

    async def _drop_unconsumed(self, room_id: str) -> None:
        """Unsubscribe the tracks nothing plugged in consumes any more.

        The unplug half of the runtime re-evaluation clause (RFC 12.10.4):
        forgetting first is what stops frames being routed, unsubscribing is
        what stops them arriving, and the teardown closes what they fed —
        the same order ``_apply_collection_state`` uses when collection ends.
        """
        room = self._room(room_id)
        bot = room.bot
        dropped = [
            track_id
            for track_id, track in list(room.subscribed.items())
            if not self._consumes(track.kind)
        ]
        for track_id in dropped:
            room.forget_subscription(track_id)
        for track_id in dropped:
            await self._unsubscribe_quietly(bot, track_id)
        for track_id in dropped:
            await self._stop_consuming(track_id)

    async def _retire_bot(self, room_id: str) -> None:
        """Take a room's bot out because nothing needs it any more.

        The unplug-of-the-last-need exit (RFC 12.10.4): a session kept past
        the last consumer and the last voice is the silent observer section
        17.7 refuses, so it leaves, with the ``conference_ended`` a departure
        owes. The generation bump is what makes a join still in flight abandon
        the session it opened — the next generation is pure transport, and
        nothing may hand it a bot.
        """
        room = self._room(room_id)
        if room.bot is None and not room.joining:
            return
        room.bump()
        bot = room.bot
        room.bot = None
        if bot is None:
            # A join in flight owns its own compensation: it re-reads the
            # generation after connecting, sees this bump, and leaves the
            # session it opened.
            return
        self._voice.forget_room(room_id)
        track_ids = room.forget_subscriptions()
        for track_id in track_ids:
            await self._unsubscribe_quietly(bot, track_id)
        for track_id in track_ids:
            await self._stop_consuming(track_id)
        room.start_leaving(bot)
        if await self._leave_and_record(room_id, bot):
            await self._announce_end(room_id, bot)

    # -------------------------------------------------------------------------
    # Grants — the session must cover the configured needs
    # -------------------------------------------------------------------------

    async def _align_grants(self, room_id: str) -> None:
        """Bring a live session's grants in line with the configuration.

        In place when the backend can (BOT_GRANT_UPDATE) — the session, its
        subscriptions and the event bridge survive — and by re-joining when
        it cannot and the change widens what the session must do. A narrowing
        the backend cannot apply is left standing: an unused privilege
        against a cut in the event bridge is the trade RFC 12.10.4 settles
        for continuity. An explicit ``bot_grants`` is never rewritten.
        """
        room = self._room(room_id)
        bot = room.bot
        if bot is None or self._explicit_bot_grants is not None:
            return
        wanted = self._bot_grants
        held = room.bot_grants
        if held == wanted:
            return
        generation = room.generation
        if ConferenceCapability.BOT_GRANT_UPDATE in self._backend.capabilities:
            try:
                with self._operations.use(
                    ConferenceResource.BACKEND,
                    what=f"updating the bot's grants in room {room_id}",
                ):
                    await self._backend.update_bot_grants(bot, wanted)
            except Exception:
                logger.exception(
                    "Conference channel %r could not update its bot's grants in room %s",
                    self.channel_id,
                    room_id,
                )
                if self._widens(held, wanted):
                    await self._rejoin_for_grants(room_id)
                return
            if room.is_current(generation, bot):
                room.bot_grants = wanted
            return
        if self._widens(held, wanted):
            await self._rejoin_for_grants(room_id)
        else:
            logger.debug(
                "Conference channel %r leaves its bot's wider grants standing in room %s: "
                "backend %r cannot update a connected session's grants, and a re-join "
                "would cut the event bridge to remove a privilege nobody uses",
                self.channel_id,
                room_id,
                self._backend.name,
            )

    @staticmethod
    def _widens(held: ConferenceGrants | None, wanted: ConferenceGrants) -> bool:
        """Whether ``wanted`` lets the bot do something ``held`` does not.

        Only a widening forces a re-join on a backend that cannot update in
        place: without the new permission the plugged need cannot function at
        all, where a leftover permission merely goes unused (RFC 12.10.4).
        """
        if held is None:
            return True
        return (
            (wanted.publish_audio and not held.publish_audio)
            or (wanted.subscribe and not held.subscribe)
            or (wanted.publish_video and not held.publish_video)
            or (wanted.publish_screen_share and not held.publish_screen_share)
        )

    async def _rejoin_for_grants(self, room_id: str) -> None:
        """Replace a session whose grants the SFU will not change.

        A leave and a join, each announced as the session event it is (RFC
        12.10.4): the old session's end is real — its lanes close, its
        recordings finalize — and the new session joins with grants derived
        from the configuration in force. The join's failure is the lazy
        join's ordinary quiet failure; the caller re-subscribes afterwards.
        """
        await self._retire_bot(room_id)
        try:
            await self._ensure_bot(room_id)
        except RoomNotAttachedError:
            return
        except Exception:
            logger.exception(
                "Conference channel %r took its bot out of room %s to change its grants "
                "but could not bring the replacement in. The conference runs without the "
                "framework's own media session until a later join succeeds",
                self.channel_id,
                room_id,
            )

    # -------------------------------------------------------------------------
    # Closing what an unplug retired
    # -------------------------------------------------------------------------

    async def _close_retired(
        self,
        *,
        resources: tuple[ConferenceResource, ...],
        closer: Callable[[], Awaitable[None]],
        what: str,
    ) -> None:
        """Close a retired provider once nothing the channel admitted uses it.

        The unplug's small sibling of the shutdown's ``close_resource``. The
        common case is idle at once — the lanes or playbacks were closed just
        above — and the provider closes here. An operation that survived its
        own grace keeps its lease, and the provider follows it in the
        background, off the unplug's clock (RFC 12.10.4): waiting longer here
        would hold the caller on a call that already ignored a cancellation.
        """
        if await self._operations.wait_idle(
            *resources, timeout=_conference_activity.DRAIN_TIMEOUT_S
        ):
            await self._close_quietly(closer, what)
            return
        holders = sorted(
            {holder for resource in resources for holder in self._operations.holders(resource)}
        )
        logger.error(
            "Conference channel %r is retaining %s past its unplug: still in use by %s. "
            "It closes in the background once those operations truly end",
            self.channel_id,
            what,
            "; ".join(holders) or "an operation that outlived its budget",
        )
        task = asyncio.create_task(
            self._close_once_idle(resources, closer, what),
            name=f"roomkit-conference-retired-{self.channel_id}",
        )
        self._deferred_closes.add(task)
        task.add_done_callback(self._deferred_closes.discard)
        task.add_done_callback(log_task_exception)

    async def _close_once_idle(
        self,
        resources: tuple[ConferenceResource, ...],
        closer: Callable[[], Awaitable[None]],
        what: str,
    ) -> None:
        await self._operations.when_idle(*resources)
        await self._close_quietly(closer, what)

    async def _close_quietly(self, closer: Callable[[], Awaitable[None]], what: str) -> None:
        try:
            await closer()
        except Exception:
            logger.exception("Conference channel %r could not close %s", self.channel_id, what)

    async def _close_stt_pair(self, stt: STTProvider, pipeline: AudioPipeline | None) -> None:
        """Close an unplugged recognizer and the pipeline built for it.

        The pipeline's close is synchronous and aggregates its providers'
        failures, exactly as at channel close; a partial failure is logged
        and the recognizer is still closed — each independently of the other.
        """
        if pipeline is not None:
            try:
                await asyncio.to_thread(pipeline.close)
            except BaseExceptionGroup as failures:
                logger.error(
                    "Conference channel %r could not close %d provider(s) of the unplugged "
                    "pipeline; the others were closed",
                    self.channel_id,
                    len(failures.exceptions),
                    exc_info=failures,
                )
        await stt.close()

    async def _finalize_unplugged_recorder(self, recorder: ConferenceRecording) -> None:
        """Finish what an unplugged recording subsystem still holds, and say so.

        The recordings this channel started are its own to finish whoever owns
        the recorder, so they are finalized here and their ends announced —
        the same obligations the channel's close honours, reported to the log
        rather than to a shutdown that is not running.
        """
        report = await recorder.close(close_recorder=self._close_providers)
        for detail in report.unfinished:
            logger.error(
                "Conference channel %r could not finalize a recording while unplugging: %s",
                self.channel_id,
                detail,
            )
        if report.recorder_retained:
            logger.error(
                "Conference channel %r is retaining the unplugged recorder: call(s) the "
                "framework gave up on are still running inside it; it closes once they end",
                self.channel_id,
            )
        if report.recorder_close_error is not None:
            logger.error(
                "Conference channel %r could not close the unplugged recorder: %s",
                self.channel_id,
                report.recorder_close_error,
            )
        try:
            async with asyncio.timeout(_conference_activity.DRAIN_TIMEOUT_S):
                await self._recording_events.stopped_all(report.finished)
        except TimeoutError:
            logger.error(
                "Conference channel %r unplugged its recording without announcing %d "
                "finished recording(s): the announcements did not return within %.1fs. "
                "The recordings themselves are finalized",
                self.channel_id,
                len(report.finished),
                _conference_activity.DRAIN_TIMEOUT_S,
            )
