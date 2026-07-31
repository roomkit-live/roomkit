"""What a conference channel does with the audio it subscribed to.

One processing lane per subscribed audio track, so the recogniser runs per
utterance and every transcription is attributed to the participant that
published the track rather than to whoever the meeting's loudest voice was. The
recorder writes each track to a file of its own, the bot's own speech included —
it publishes rather than receives, so nothing else in the conference carries it.

A frame does as little as possible on arrival: the backend awaits each
subscriber in turn, so transcribing inline would stall every other
participant's frames behind one provider's latency.

Split from ConferenceChannel for room, not for isolation: everything here reads
the channel it is mixed into, and the host contract says how much of it.

See RFC section 12.10.4.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels._conference_lane import ConferenceLane, ConferenceTranscription
from roomkit.channels._conference_operations import ConferenceResource
from roomkit.channels._conference_recording import TrackFormat
from roomkit.conference.models import ConferenceTrack, TrackKind
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import HookTrigger
from roomkit.models.event import TextContent
from roomkit.voice.audio_frame import AudioFrame

if TYPE_CHECKING:
    from roomkit.channels._conference_activity import RoomActivity
    from roomkit.channels._conference_realtime import ConferenceRealtime
    from roomkit.channels._conference_recording import ConferenceRecording, TrackRecording
    from roomkit.channels._conference_recording_events import ConferenceRecordingEvents
    from roomkit.channels._conference_voice import ConferenceVoice
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import AudioChunk
    from roomkit.voice.pipeline.engine import AudioPipeline
    from roomkit.voice.stt.base import STTProvider

logger = logging.getLogger("roomkit.channels.conference")


class ConferenceLanesMixin:
    """Lanes, recording and transcription for a conference channel.

    Host contract — what ConferenceChannel provides:
        channel_id, _framework, _activity, _voice: the channel and what it
            talks to.
        _stt, _pipeline, _recorder, _max_queued_frames, _bot_identity: what a
            subscribed track is fed through.
        _recording_events: announces what the recordings did, since the
            recorder itself is synchronous and cannot.
        _lanes: open lanes by track id, channel-wide because a frame arrives
            with a track and not a room.
        _room / _attached_room: the per-room record (ConferenceRoomState).
        _fire: how the speech edges are announced on their lifecycle hooks.
    """

    channel_id: str
    _framework: RoomKit | None
    _activity: RoomActivity
    _voice: ConferenceVoice
    _realtime: ConferenceRealtime
    _stt: STTProvider | None
    _pipeline: AudioPipeline | None
    _recorder: ConferenceRecording | None
    _recording_events: ConferenceRecordingEvents
    _max_queued_frames: int
    _bot_identity: str
    _lanes: dict[str, ConferenceLane]
    _operations: Any

    # Provided by ConferenceChannel — see the host contract above
    _room: Any
    _attached_room: Any
    _fire: Any

    def _lane_ids(self, room_id: str) -> list[str]:
        """Tracks with a lane in a room, as a snapshot safe to close over."""
        return [track_id for track_id, lane in self._lanes.items() if lane.room_id == room_id]

    def _open_lane(self, room_id: str, track: ConferenceTrack) -> None:
        """Start a processing lane for a subscribed track.

        The lane holds a lease on the shared pipeline — and the recognizer,
        when there is one — from before it starts until nothing of its own can
        still be inside either. The lease, not any list the channel keeps, is
        what stops a close from freeing those providers under a lane whose
        provider call ignored its cancellation.
        """
        if self._pipeline is None or track.id in self._lanes:
            return
        resources = [ConferenceResource.PIPELINE]
        if self._stt is not None:
            resources.append(ConferenceResource.STT)
        lane = ConferenceLane(
            track_id=track.id,
            room_id=room_id,
            participant_id=track.participant_id,
            pipeline=self._pipeline,
            on_speech=self._voice.consider_interruption,
            on_utterance=self._on_lane_utterance,
            on_speech_start=self._on_lane_speech_start,
            on_speech_end=self._on_lane_speech_end,
            # Wired unconditionally: the tap is a no-op while no realtime
            # provider is configured, and a provider plugged mid-meeting must
            # hear the lanes that were already open (RFC 12.10.12).
            on_frame=self._realtime.mixer.feed,
            max_queued_frames=self._max_queued_frames,
            lease=self._operations.acquire(*resources, what=f"lane for track {track.id}"),
        )
        self._lanes[track.id] = lane
        lane.start()

    async def _close_lane(self, track_id: str) -> bool:
        """Stop a lane and release the stage state its track held.

        Returns whether there was one to close.
        """
        lane = self._lanes.pop(track_id, None)
        if lane is None:
            return False
        await self._close_lane_instance(lane)
        return True

    async def _close_lane_instance(self, lane: ConferenceLane) -> bool:
        """Close one lane, and say whether its task outlived the grace.

        An abandoned lane needs no tracking here: it holds its lease on the
        shared pipeline and recognizer until its task truly ends, and the
        close retains those providers for exactly as long as any lease on
        them remains.
        """
        return await lane.aclose()

    async def _stop_consuming(self, track_id: str) -> None:
        """Close whatever a track was feeding: its lane, its recording, or both.

        A track that carried no frame closes nothing and announces nothing:
        the recording opens on the first one, so a participant who stayed
        silent leaves no file to report.
        """
        lane = self._lanes.get(track_id)
        if lane is not None:
            self._realtime.mixer.drop_track(lane.room_id, track_id)
        await self._close_lane(track_id)
        if self._recorder is None:
            return
        finished = await self._recorder.close_track(track_id)
        if finished is not None:
            await self._recording_events.stopped(finished)

    async def _on_track_audio(self, track: ConferenceTrack, frame: AudioFrame) -> None:
        """Hand a track's audio to its lane and to its recording.

        Deliberately does no work of its own: the backend awaits each
        subscriber in turn, so transcribing — or opening, writing or closing a
        file — here would stall the arrival of every other participant's frames
        behind one provider's latency. Both collaborators take the frame and
        return; both do the slow part on their own schedule, and the recording
        does all of its own on another thread besides (RFC sections 12.10.4 and
        12.10.8).

        The lane is fed first, and that order is the point rather than a
        preference: recording and transcription are independent, and a
        conference that could not be recorded went on being unrecorded *and*
        untranscribed for as long as the file write sat in front of the lane.
        ``feed`` does not raise either — the guarantee belongs to the recorder,
        which knows how it fails — so the two are independent both ways round.

        Gated on the subscription rather than on the lane, because a channel
        that records without transcribing has no lane to find and would drop
        every frame. The subscription is also what makes a stray frame — a
        backend delivering a track nobody asked for — nothing this channel
        records.
        """
        room = self._attached_room(track.room_id)
        if room is None or not room.is_subscribed(track.id):
            return
        if not room.may_collect():
            return
        lane = self._lanes.get(track.id)
        if lane is not None:
            lane.submit(frame)
        if self._recorder is not None:
            self._recorder.feed(track, frame.data, TrackFormat.of_frame(frame))

    async def _announce_recording_started(self, recording: TrackRecording) -> None:
        """Say a track's recording has opened, from where the opening happened.

        Not from the frame callback, which is what handed the frame over: the
        announcement runs integrator code, and the callback is the one place in
        this channel that must not. It is the recording's own writer that calls
        this, once the recorder has actually accepted the recording — which is
        also the only moment at which there is an id to name.

        Registered as room activity, like every other announcement this channel
        makes. A handler that detaches the channel is ordinary code — a
        disclosure policy refusing to be recorded is the realistic one — and
        without this the detach finds no enclosing work, tears down inline, and
        closes from inside the announcement the very recording it is announcing.
        """
        async with self._activity.track(recording.room_id):
            await self._recording_events.started(recording)

    async def _record_bot_audio(self, room_id: str, chunk: AudioChunk) -> None:
        """Record what the bot published, on a track of its own.

        The bot publishes rather than receives, so its audio never comes back
        through ``on_track_audio`` and nothing else in the conference would
        carry it. It is recorded as a track like any other, attributed to the
        bot's identity and unmixed: the only track that resembles an outbound
        direction is still a participant's, and what the AI said is part of
        what was said.

        The track is built per chunk rather than kept: a re-attach brings a new
        bot session and therefore a different track, and a cached one would go
        on recording the new conference into the old session's file.
        """
        room = self._attached_room(room_id)
        if self._recorder is None or room is None or not room.may_collect():
            return
        bot = room.bot
        if bot is None:
            return
        self._recorder.feed(
            ConferenceTrack(
                id=f"bot:{bot.id}",
                room_id=room_id,
                participant_id=self._bot_identity,
                kind=TrackKind.AUDIO,
            ),
            chunk.data,
            TrackFormat.of_chunk(chunk),
        )

    async def _on_lane_speech_start(self, lane: ConferenceLane) -> None:
        """Announce that a lane's track went from silence to speech.

        The real-time half of "who is speaking right now" (RFC 12.10.4): the
        SFU's dominant-speaker signal cannot say that nobody is, and the
        transcription arrives only after the recognizer's round trip. Named
        per participant and track, because a management interface lights an
        indicator on a person, not on a room.
        """
        await self._announce_speech_edge(
            lane, HookTrigger.ON_SPEECH_START, "speech_start", "started speaking"
        )

    async def _on_lane_speech_end(self, lane: ConferenceLane) -> None:
        """Announce that a lane's utterance closed — before it is transcribed.

        "They stopped speaking" is true the moment the VAD closes the
        utterance; recognition is a round trip that has not happened yet.
        """
        await self._announce_speech_edge(
            lane, HookTrigger.ON_SPEECH_END, "speech_end", "stopped speaking"
        )

    async def _announce_speech_edge(
        self, lane: ConferenceLane, trigger: HookTrigger, code: str, what: str
    ) -> None:
        """Fire one speech-boundary hook, without costing the lane its frame.

        Runs on the lane's own task, upstream of the utterance hand-off: an
        error escaping here would be caught by the lane's per-frame guard and
        cost the utterance behind it, so nothing is allowed to escape. The
        announcement registers as room activity like every other, so a detach
        drains it rather than contradicting it.
        """
        room = self._attached_room(lane.room_id)
        if room is None:
            return
        try:
            async with self._activity.track(lane.room_id):
                if not room.attached:
                    return
                await self._fire(
                    lane.room_id,
                    trigger,
                    code,
                    f"Participant {lane.participant_id} {what}",
                    {"participant_id": lane.participant_id, "track_id": lane.track_id},
                )
        except Exception:
            logger.exception(
                "Conference channel %r could not announce a speech boundary of track %s",
                self.channel_id,
                lane.track_id,
            )

    async def _on_lane_utterance(
        self, lane: ConferenceLane, audio: bytes, sample_rate: int
    ) -> None:
        """Transcribe one utterance and route it, attributed to its speaker.

        One event per utterance rather than one per frame: the VAD decided
        where the utterance ended, and this is the audio it accumulated.
        """
        if self._stt is None or self._framework is None:
            return
        room = self._attached_room(lane.room_id)
        if room is None or not room.may_collect():
            return
        # A per-call lease besides the lane's own: a lane opened before
        # recognition was plugged holds no STT lease of its own, and this is
        # what keeps the recognizer from being closed under its call.
        with self._operations.use(
            ConferenceResource.STT, what=f"transcribing track {lane.track_id}"
        ):
            result = await self._stt.transcribe(AudioFrame(data=audio, sample_rate=sample_rate))
        if not room.may_collect() or lane.track_id not in self._lanes:
            return
        text = (result.text or "").strip()
        if not text:
            return
        # ON_TRANSCRIPTION is integrator code, and it runs on the lane's own
        # task. A handler that detaches the channel is ordinary — it is how a
        # keyword ends a meeting — and without this the teardown would run
        # inside that handler, cancel this very lane, and take the chain it is
        # standing on down with it. Registered as room activity, the detach
        # recognises itself as nested and defers, exactly as it does for an
        # announcement.
        async with self._activity.track(lane.room_id):
            if not room.may_collect():
                return
            text = await self._run_transcription_hook(lane, text)
        if not text:
            return
        # Read once more, and outside the block: delivery takes the room lock,
        # which a detach holds while it runs its own hooks, so it cannot be part
        # of what a teardown drains without the two waiting on each other. Which
        # leaves the room free to have gone during the hook — and a transcript
        # delivered then would arrive after the end announcement.
        if not room.may_collect():
            return
        await self._framework.process_inbound(
            InboundMessage(
                channel_id=self.channel_id,
                sender_id=lane.participant_id,
                content=TextContent(body=text),
                metadata={"conference_track_id": lane.track_id, "source": "conference"},
            ),
            room_id=lane.room_id,
        )

    async def _run_transcription_hook(self, lane: ConferenceLane, text: str) -> str:
        """Let hooks inspect, rewrite or block a lane's transcription.

        Synchronous, because the point is to decide before the text reaches the
        room: a redaction hook that ran afterwards would be too late.

        A hook may block the text or return a rewritten ConferenceTranscription.
        One that raises blocks too: ON_TRANSCRIPTION fails closed in the hook
        engine, because carrying on with the original would publish exactly what
        the hook existed to suppress.
        """
        if self._framework is None:
            return text
        context = await self._framework._build_context(lane.room_id)
        payload = ConferenceTranscription(
            track_id=lane.track_id,
            participant_id=lane.participant_id,
            room_id=lane.room_id,
            text=text,
        )
        result = await self._framework.hook_engine.run_sync_hooks(
            lane.room_id,
            HookTrigger.ON_TRANSCRIPTION,
            payload,
            context,
            skip_event_filter=True,
        )
        if not result.allowed:
            return ""
        if isinstance(result.event, ConferenceTranscription):
            return result.event.text
        return text
