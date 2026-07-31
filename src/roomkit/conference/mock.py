"""Mock conference backend for testing.

Scripts the events a real SFU would produce, so the channel can be exercised
without an SFU, credentials or a network. Two of its behaviours matter more
than the rest, because they are what make the framework's own rules testable:
it refuses to deliver frames for a track nobody subscribed to, and it can echo
the bot back through its own callbacks.

It is also deliberately strict about what it accepts: a mock more permissive
than the SFU it stands in for lets a channel pass its tests and fail against
the real thing.

And it can be made to misbehave. A backend that always succeeds, instantly, in
one audio format, proves only the happy path — so failures, latency, per-track
formats and utterance boundaries are all configurable here rather than in a
second mock that would drift from this one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from roomkit.conference._mock_faults import ErrorSpec, MockFaults
from roomkit.conference._mock_media import (
    MockDelivery,
    MockTrackFormat,
    MockUtterance,
    pcm_frame,
)
from roomkit.conference.base import ConferenceBackend
from roomkit.conference.models import (
    BotSession,
    ConferenceAccess,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceParticipant,
    ConferenceTrack,
    TrackKind,
)
from roomkit.core.exceptions import ConferenceCapabilityError
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

INJECTABLE_METHODS = frozenset(
    {
        "ensure_room",
        "close_room",
        "mint_access",
        "list_participants",
        "remove_participant",
        "mute_track",
        "unmute_track",
        "join_as_bot",
        "leave",
        "subscribe_track",
        "unsubscribe_track",
        "publish_audio",
        "stop_playback",
        "publish_video",
        "close",
    }
)
"""Backend calls that can be made to fail or to take time."""

INJECTABLE_EMISSIONS = frozenset(
    {
        "participant_joined",
        "participant_left",
        "track_published",
        "track_unpublished",
        "track_muted",
        "track_unmuted",
        "track_audio",
        "track_video",
        "active_speaker_changed",
        "connection_quality",
        "bot_session_ended",
    }
)
"""Callback fanouts that can be made to take time. They cannot be made to fail:
a backend logs what its subscribers raise and carries on."""


@dataclass
class MockConferenceCall:
    """Record of a call made to MockConferenceBackend."""

    method: str
    args: dict[str, Any] = field(default_factory=dict)


class MockConferenceBackend(ConferenceBackend):
    """Conference backend that scripts SFU events for tests.

    Example::

        backend = MockConferenceBackend()
        bot = await backend.join_as_bot("room-1", "roomkit", ConferenceGrants())

        track = await backend.simulate_track_published("room-1", "p-alice")
        await backend.subscribe_track(bot, track.id)
        await backend.simulate_audio(track, AudioFrame(data=b"..."))

    Made to misbehave::

        backend.fail("join_as_bot", TimeoutError)      # the SFU refuses the bot
        backend.delay("track_audio", 0.05)             # slow delivery
        dial_in = MockTrackFormat(sample_rate=8_000, sample_width=1)
        track = await backend.simulate_track_published(
            "room-1", "p-bob", audio_format=dial_in
        )
        await backend.simulate_audio(track, backend.frame_for(track))
    """

    def __init__(self, *, capabilities: ConferenceCapability = ConferenceCapability.NONE) -> None:
        super().__init__()
        self._capabilities = capabilities
        self.calls: list[MockConferenceCall] = []
        self.rooms: dict[str, dict[str, Any]] = {}
        self.participants: dict[str, dict[str, ConferenceParticipant]] = {}
        self.tracks: dict[str, ConferenceTrack] = {}
        self.track_formats: dict[str, MockTrackFormat] = {}
        self.subscriptions: set[str] = set()
        self.published_audio: list[AudioChunk] = []
        self.utterances: list[MockUtterance] = []
        self._open_utterances: dict[str, MockUtterance] = {}
        self.published_video: list[VideoFrame] = []
        self.playback_stops: list[str] = []
        self.bots: list[BotSession] = []
        self.dropped_frames: list[str] = []
        self.deliveries: list[MockDelivery] = []
        self.faults = MockFaults(methods=INJECTABLE_METHODS, emissions=INJECTABLE_EMISSIONS)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> ConferenceCapability:
        return self._capabilities

    # -------------------------------------------------------------------------
    # Injection — what makes the unhappy paths reachable
    # -------------------------------------------------------------------------

    def fail(
        self,
        method: str,
        error: ErrorSpec | None = None,
        *,
        times: int | None = None,
    ) -> None:
        """Make a backend call raise. See :meth:`MockFaults.fail`."""
        self.faults.fail(method, error, times=times)

    def delay(self, operation: str, seconds: float) -> None:
        """Make a backend call or a callback fanout take time.

        See :meth:`MockFaults.delay`.
        """
        self.faults.delay(operation, seconds)

    def _record(self, method: str, **args: Any) -> None:
        self.calls.append(MockConferenceCall(method=method, args=args))

    async def _enter(self, method: str, **args: Any) -> None:
        """Record the call, then apply whatever was injected on it.

        Recorded before it can fail, because the request did go out: a test
        asserting the channel tried to join needs the trace of the attempt.
        """
        self._record(method, **args)
        await self.faults.apply(method)

    async def _emit(self, event: str, callbacks: Sequence[Callable[..., Any]], *args: Any) -> None:
        await self.faults.apply(event)
        await super()._emit(event, callbacks, *args)

    def _require(self, capability: ConferenceCapability, operation: str) -> None:
        if capability not in self._capabilities:
            raise ConferenceCapabilityError(
                f"{operation} requires {capability.name}, which backend "
                f"{self.name!r} does not declare"
            )

    # -------------------------------------------------------------------------
    # Control plane
    # -------------------------------------------------------------------------

    async def ensure_room(
        self,
        room_id: str,
        metadata: dict[str, Any] | None = None,
        e2ee: bool = False,
    ) -> None:
        await self._enter("ensure_room", room_id=room_id, metadata=metadata, e2ee=e2ee)
        if e2ee:
            self._require(ConferenceCapability.E2EE, "End-to-end encryption")
        self.rooms.setdefault(room_id, {"metadata": metadata or {}, "e2ee": e2ee})
        self.participants.setdefault(room_id, {})

    async def close_room(self, room_id: str) -> None:
        await self._enter("close_room", room_id=room_id)
        self.rooms.pop(room_id, None)
        self.participants.pop(room_id, None)

    async def mint_access(
        self,
        room_id: str,
        participant_id: str,
        grants: ConferenceGrants,
        *,
        display_name: str | None = None,
    ) -> ConferenceAccess:
        await self._enter(
            "mint_access",
            room_id=room_id,
            participant_id=participant_id,
            grants=grants,
            display_name=display_name,
        )
        return ConferenceAccess(
            url="wss://mock.conference.invalid",
            token=f"mock-token-{participant_id}",
            provider_data={"room_id": room_id},
        )

    async def list_participants(self, room_id: str) -> list[ConferenceParticipant]:
        await self._enter("list_participants", room_id=room_id)
        return list(self.participants.get(room_id, {}).values())

    async def remove_participant(self, room_id: str, participant_id: str) -> None:
        await self._enter("remove_participant", room_id=room_id, participant_id=participant_id)
        self.participants.get(room_id, {}).pop(participant_id, None)

    async def mute_track(self, room_id: str, track_id: str) -> None:
        await self._enter("mute_track", room_id=room_id, track_id=track_id)
        if (track := self.tracks.get(track_id)) is not None:
            track.muted = True
            # A server-side mute is a mute transition like any other: the SFU
            # observes its own moderation and reports it (RFC 12.10.3).
            await self._emit_track_muted(room_id, track)

    async def unmute_track(self, room_id: str, track_id: str) -> None:
        await self._enter("unmute_track", room_id=room_id, track_id=track_id)
        self._require(ConferenceCapability.REMOTE_UNMUTE, "Remote unmute")
        if (track := self.tracks.get(track_id)) is not None:
            track.muted = False
            await self._emit_track_unmuted(room_id, track)

    # -------------------------------------------------------------------------
    # Bot participant
    # -------------------------------------------------------------------------

    async def join_as_bot(
        self,
        room_id: str,
        identity: str,
        grants: ConferenceGrants,
    ) -> BotSession:
        await self._enter("join_as_bot", room_id=room_id, identity=identity, grants=grants)
        bot = BotSession(id=f"bot-{uuid4().hex[:8]}", room_id=room_id, identity=identity)
        self.bots.append(bot)
        return bot

    async def leave(self, bot: BotSession) -> None:
        await self._enter("leave", bot=bot.id)
        if bot in self.bots:
            self.bots.remove(bot)
        # The record itself stays, incomplete, because that is the assertion a
        # test makes about a bot cut off mid-utterance. What goes is the claim
        # that the track is still open.
        self._open_utterances.pop(bot.id, None)

    async def subscribe_track(self, bot: BotSession, track_id: str) -> None:
        await self._enter("subscribe_track", bot=bot.id, track_id=track_id)
        self.subscriptions.add(track_id)

    async def unsubscribe_track(self, bot: BotSession, track_id: str) -> None:
        await self._enter("unsubscribe_track", bot=bot.id, track_id=track_id)
        self.subscriptions.discard(track_id)

    async def publish_audio(self, bot: BotSession, chunk: AudioChunk) -> None:
        await self._enter("publish_audio", bot=bot.id, is_final=chunk.is_final)
        if not chunk.format.startswith("pcm"):
            raise ValueError(
                f"publish_audio expects decoded PCM, got format {chunk.format!r}. "
                "Encoding belongs to the backend: a caller choosing the wire "
                "format defeats the abstraction boundary."
            )
        self.published_audio.append(chunk)
        self._append_to_utterance(bot, chunk)

    async def stop_playback(self, bot: BotSession) -> None:
        """Record the barge-in gesture. There is nothing here to discard.

        The mock publishes synchronously — nothing queues, so a real flush
        would have no observable effect, and inventing one would have the mock
        outdo every real SFU. What a test asserts is that the gesture reached
        the backend at all, and for which bot. The open utterance is
        deliberately left open: a stop is not a boundary, and the closing
        chunk that follows is still owed (RFC section 12.10.3).
        """
        await self._enter("stop_playback", bot=bot.id)
        self.playback_stops.append(bot.id)

    async def publish_video(self, bot: BotSession, frame: VideoFrame) -> None:
        await self._enter("publish_video", bot=bot.id)
        self._require(ConferenceCapability.VIDEO_PUBLISH, "Bot video publishing")
        if frame.is_encoded:
            raise ValueError(
                f"publish_video expects a raw frame, got codec {frame.codec!r}. "
                "The backend owns encoding, symmetrically with the decoding it "
                "owns inbound."
            )
        self.published_video.append(frame)

    async def close(self) -> None:
        await self._enter("close")

    def utterances_for(self, bot: BotSession) -> list[MockUtterance]:
        """What was published on one bot's track, in order.

        A channel serving several rooms holds a bot session per room, and
        ``utterances`` interleaves them the way the calls arrived. Asking per
        bot is asking about one track.
        """
        return [utterance for utterance in self.utterances if utterance.bot_id == bot.id]

    def _append_to_utterance(self, bot: BotSession, chunk: AudioChunk) -> None:
        """File a published chunk under the utterance it belongs to.

        Kept per bot, because a bot is a track: a new utterance opens once that
        bot's last one was closed by ``is_final``, and two published
        concurrently on one bot share a record — which is how a test sees that
        two answers ran together. Two bots publishing at the same time are two
        rooms, and get a record each.
        """
        utterance = self._open_utterances.get(bot.id)
        if utterance is None:
            utterance = MockUtterance(bot_id=bot.id)
            self.utterances.append(utterance)
            self._open_utterances[bot.id] = utterance
        utterance.chunks.append(chunk)
        utterance.complete = chunk.is_final
        if chunk.is_final:
            del self._open_utterances[bot.id]

    # -------------------------------------------------------------------------
    # Simulation — drive the callbacks as a real SFU would
    # -------------------------------------------------------------------------

    async def simulate_participant_joined(
        self,
        room_id: str,
        participant_id: str,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        client_metadata: dict[str, Any] | None = None,
        asserts_provenance: bool = True,
    ) -> ConferenceParticipant:
        """Announce a participant joining the media session.

        ``metadata`` is what the SFU itself asserts — a dial-in's caller number
        as the trunk reported it, which is what identity resolution consumes.
        ``client_metadata`` is what the participant's own client supplied at
        join: surfaced like any other attribute, never vouched for, and so
        never an address (RFC §12.10.2). A key given in both is the SFU's.

        ``asserts_provenance=False`` is the third kind of backend — one that
        cannot tell the two apart and says so. Everything it surfaces becomes
        unvouched, whichever argument it arrived in.
        """
        asserted = dict(metadata or {})
        surfaced = {**(client_metadata or {}), **asserted}
        participant = ConferenceParticipant(
            participant_id=participant_id,
            display_name=display_name,
            metadata=surfaced,
            asserted_metadata=asserted if asserts_provenance else None,
        )
        self.participants.setdefault(room_id, {})[participant_id] = participant
        await self._emit_participant_joined(room_id, participant)
        return participant

    async def simulate_participant_left(
        self, room_id: str, participant_id: str
    ) -> ConferenceParticipant | None:
        participant = self.participants.get(room_id, {}).pop(participant_id, None)
        if participant is None:
            return None
        await self._emit_participant_left(room_id, participant)
        return participant

    async def simulate_bot_disconnected(
        self, bot: BotSession, reason: str = "connection lost"
    ) -> None:
        """End the bot's session the way an SFU does: without a ``leave()``.

        The session is forgotten first — a dropped connection is not a
        participant, and a later ``leave()`` for it finds nothing to do — and
        then reported, which is the order the contract promises (RFC 12.10.3).
        """
        if bot in self.bots:
            self.bots.remove(bot)
        self._open_utterances.pop(bot.id, None)
        await self._emit_bot_session_ended(bot, reason)

    async def simulate_track_published(
        self,
        room_id: str,
        participant_id: str,
        kind: TrackKind = TrackKind.AUDIO,
        *,
        track_id: str | None = None,
        audio_format: MockTrackFormat | None = None,
    ) -> ConferenceTrack:
        """Publish a track, optionally in a format of its own.

        ``audio_format`` is what this publisher negotiated with the SFU.
        Participants negotiate separately and nothing obliges them to agree, so
        a conference of three can carry three formats — and a track that
        declares one only accepts frames in it.
        """
        track = ConferenceTrack(
            id=track_id or f"tr-{uuid4().hex[:8]}",
            room_id=room_id,
            participant_id=participant_id,
            kind=kind,
        )
        self.tracks[track.id] = track
        if audio_format is not None:
            self.track_formats[track.id] = audio_format
        if (participant := self.participants.get(room_id, {}).get(participant_id)) is not None:
            participant.tracks.append(track)
        await self._emit_track_published(room_id, track)
        return track

    async def simulate_track_unpublished(self, track_id: str) -> ConferenceTrack | None:
        track = self.tracks.pop(track_id, None)
        if track is None:
            return None
        self.subscriptions.discard(track_id)
        self.track_formats.pop(track_id, None)
        await self._emit_track_unpublished(track.room_id, track)
        return track

    def frame_for(
        self, track: ConferenceTrack, *, ms: int = 20, amplitude: float = 0.25
    ) -> AudioFrame:
        """Build a frame in the format ``track`` was published in.

        ``amplitude`` is a fraction of full scale: the default is loud enough
        for an energy VAD to call speech, and ``0.0`` gives the silence that
        ends an utterance.
        """
        return pcm_frame(
            self.track_formats.get(track.id, MockTrackFormat()), ms=ms, amplitude=amplitude
        )

    async def simulate_audio(self, track: ConferenceTrack, frame: AudioFrame) -> bool:
        """Deliver an audio frame, if the bot subscribed to the track.

        Returns whether it was delivered. An unsubscribed track produces no
        frame at all — a real SFU forwards nothing to a subscriber that did not
        ask, and a mock that delivered anyway would make selective subscription
        untestable.
        """
        if (audio_format := self.track_formats.get(track.id)) is not None and not (
            audio_format.matches(frame)
        ):
            raise ValueError(
                f"track {track.id} was published as {audio_format.describe()}, and a "
                f"frame of {frame.sample_rate} Hz, {frame.channels} ch, "
                f"{frame.sample_width * 8}-bit cannot arrive on it. An SFU forwards "
                "what the publisher sent. Use frame_for(track) to build one."
            )
        if track.id not in self.subscriptions:
            self.dropped_frames.append(track.id)
            return False
        await self._timed_delivery(track, self._emit_track_audio(track, frame))
        return True

    async def simulate_video(self, track: ConferenceTrack, frame: VideoFrame) -> bool:
        """Deliver a video frame, if the bot subscribed to the track."""
        if track.id not in self.subscriptions:
            self.dropped_frames.append(track.id)
            return False
        await self._timed_delivery(track, self._emit_track_video(track, frame))
        return True

    async def _timed_delivery(self, track: ConferenceTrack, emission: Awaitable[None]) -> None:
        """Deliver a frame and record how long it took.

        RFC section 12.10.4 makes lane isolation checkable from outside — "by
        delaying recognition on one track and measuring frame delivery on
        another" — and this is the measurement. Subscribers are awaited in
        sequence, so a lane doing its work inline shows up here as delivery
        time on everyone else's frames.
        """
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        try:
            await emission
        finally:
            self.deliveries.append(
                MockDelivery(
                    track_id=track.id,
                    kind=track.kind,
                    started_at=started_at,
                    elapsed=loop.time() - started_at,
                )
            )

    async def simulate_track_muted(self, track_id: str) -> None:
        """The publisher mutes their own track — a camera toggled off included."""
        track = self.tracks[track_id]
        track.muted = True
        await self._emit_track_muted(track.room_id, track)

    async def simulate_track_unmuted(self, track_id: str) -> None:
        track = self.tracks[track_id]
        track.muted = False
        await self._emit_track_unmuted(track.room_id, track)

    async def simulate_active_speaker(self, room_id: str, participant_id: str) -> None:
        await self._emit_active_speaker_changed(room_id, participant_id)

    async def simulate_connection_quality(
        self, room_id: str, participant_id: str, quality: str
    ) -> None:
        await self._emit_connection_quality(room_id, participant_id, quality)

    async def simulate_bot_echo(self, bot: BotSession) -> ConferenceTrack:
        """Report the bot back through its own callbacks, as some SFUs do.

        Announces the bot as a participant and publishes a track in its name.
        Without self-exclusion the framework would then create a participant
        record for its own bot and transcribe the AI's own speech, so this is
        the scripted event sequence that proves the rule holds.
        """
        await self.simulate_participant_joined(bot.room_id, bot.identity)
        return await self.simulate_track_published(bot.room_id, bot.identity)
