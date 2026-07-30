"""ConferenceBackend abstract base class.

A ConferenceBackend adapts an external SFU to RoomKit. It owns the media plane
and everything that goes with it — SDP negotiation, ICE, codec selection,
simulcast layers, bitrate management — and exposes none of it: the framework
sees decoded frames, opaque credentials, and participant lifecycle events.

See RFC section 12.10.3. The section is STABLE — validated on paper against
the published server APIs of several SFUs, then revised against the first
conforming backend (LiveKit) — and follows normal stability rules.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from roomkit.conference.models import (
    BotSession,
    ConferenceAccess,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceParticipant,
    ConferenceTrack,
)
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk

logger = logging.getLogger("roomkit.conference.backend")

# Participant lifecycle: (room_id, participant).
ParticipantCallback = Callable[[str, ConferenceParticipant], Any]

# Track lifecycle: (room_id, track).
TrackCallback = Callable[[str, ConferenceTrack], Any]

# Decoded audio from a subscribed track: (track, frame).
# The track carries both the publishing participant and the room, which is what
# makes the frame attributable and routable.
TrackAudioCallback = Callable[[ConferenceTrack, AudioFrame], Any]

# Video from a subscribed track: (track, frame).
TrackVideoCallback = Callable[[ConferenceTrack, VideoFrame], Any]

# Dominant speaker change: (room_id, participant_id).
ActiveSpeakerCallback = Callable[[str, str], Any]

# Per-participant quality report: (room_id, participant_id, quality).
# Quality is a backend-supplied label such as "excellent", "good", "poor" or
# "lost"; it is not normalised because SFUs do not agree on the scale.
ConnectionQualityCallback = Callable[[str, str, str], Any]

# The bot's own session ended without a leave(): (session, reason).
# The SFU dropped the connection, evicted the bot, or deleted the room under
# it. A backend that observes such an end reports it here, forgets the
# session, and refuses further media calls for it (RFC 12.10.3); the channel
# treats the report as the session's end in fact.
BotSessionEndedCallback = Callable[[BotSession, str], Any]


class ConferenceBackend(ABC):
    """Abstract base class for SFU conference backends.

    The backend does three things: it administers conference rooms, it mints
    the credentials human clients use to join the SFU directly, and it gives
    the framework one bot connection through which to subscribe to tracks and
    publish the AI's voice.

    What it deliberately does not do is carry human-to-human media. Clients
    connect to the SFU themselves; the framework never proxies their signalling
    or their packets.

    Example::

        backend = MyConferenceBackend()
        backend.on_track_audio(handle_audio)

        await backend.ensure_room("room-1")
        access = await backend.mint_access("room-1", "p-alice", ConferenceGrants())
        bot = await backend.join_as_bot("room-1", "roomkit", ConferenceGrants.observer())
    """

    def __init__(self) -> None:
        self._participant_joined: list[ParticipantCallback] = []
        self._participant_left: list[ParticipantCallback] = []
        self._track_published: list[TrackCallback] = []
        self._track_unpublished: list[TrackCallback] = []
        self._track_audio: list[TrackAudioCallback] = []
        self._track_video: list[TrackVideoCallback] = []
        self._active_speaker_changed: list[ActiveSpeakerCallback] = []
        self._connection_quality: list[ConnectionQualityCallback] = []
        self._bot_session_ended: list[BotSessionEndedCallback] = []

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, used in logs and configuration errors."""

    @property
    @abstractmethod
    def capabilities(self) -> ConferenceCapability:
        """What this backend supports.

        The framework refuses configurations the backend cannot honour rather
        than discovering it at runtime — egress recording without
        ``EGRESS_RECORDING``, remote unmute without ``REMOTE_UNMUTE``, bot
        video without ``VIDEO_PUBLISH``.
        """

    # -------------------------------------------------------------------------
    # Control plane
    # -------------------------------------------------------------------------

    @abstractmethod
    async def ensure_room(
        self,
        room_id: str,
        metadata: dict[str, Any] | None = None,
        e2ee: bool = False,
    ) -> None:
        """Create the conference room if it does not exist.

        Idempotent: called whenever a channel attaches, including for a room
        that is already conferring.

        Args:
            room_id: Room this conference belongs to, one-to-one.
            metadata: Provider-specific room configuration.
            e2ee: Request end-to-end encryption. Must raise a configuration
                error when the backend lacks ``ConferenceCapability.E2EE``.
                With E2EE active the bot receives ciphertext, so transcription,
                vision and framework recording are unavailable unless the bot
                is admitted as a key holder.
        """

    @abstractmethod
    async def close_room(self, room_id: str) -> None:
        """Tear down the conference room and disconnect its participants."""

    @abstractmethod
    async def mint_access(
        self,
        room_id: str,
        participant_id: str,
        grants: ConferenceGrants,
    ) -> ConferenceAccess:
        """Mint credentials for a participant to join the SFU directly.

        The framework passes its own ``Participant.id``. A backend whose SFU
        cannot carry a caller-supplied identity must keep the mapping itself
        and translate at this boundary, because every attribution guarantee
        downstream depends on ``participant_id`` meaning the same thing on both
        sides.

        The returned credential is opaque to the framework: the integrator
        hands it to its client application, and the provider's SDK consumes it.
        """

    @abstractmethod
    async def list_participants(self, room_id: str) -> list[ConferenceParticipant]:
        """Return the participants currently connected to the conference."""

    @abstractmethod
    async def remove_participant(self, room_id: str, participant_id: str) -> None:
        """Disconnect a participant from the conference."""

    @abstractmethod
    async def mute_track(self, room_id: str, track_id: str) -> None:
        """Mute a published track as a moderator.

        Always available, unlike unmuting.
        """

    @abstractmethod
    async def unmute_track(self, room_id: str, track_id: str) -> None:
        """Unmute a published track as a moderator.

        Requires ``ConferenceCapability.REMOTE_UNMUTE``, and must raise a
        configuration error without it rather than failing silently or
        appearing to succeed. Unmuting someone else's microphone is a privacy
        decision rather than a technical one, and SFUs commonly refuse it
        unless explicitly enabled server-side.
        """

    # -------------------------------------------------------------------------
    # Bot participant — the framework's only crossing of the media boundary
    # -------------------------------------------------------------------------

    @abstractmethod
    async def join_as_bot(
        self,
        room_id: str,
        identity: str,
        grants: ConferenceGrants,
    ) -> BotSession:
        """Connect the framework to the conference as a participant.

        Grants are applied to the bot session rather than assumed: a speaking
        bot needs ``publish_audio``, while an observer is subscribe-only and
        hidden.

        The backend must not auto-subscribe the bot to anything. The
        framework's subscription set is authoritative, and it is expressed
        through ``subscribe_track()`` alone.
        """

    @abstractmethod
    async def leave(self, bot: BotSession) -> None:
        """Disconnect the bot session, leaving the conference running."""

    @abstractmethod
    async def subscribe_track(self, bot: BotSession, track_id: str) -> None:
        """Start delivering a track's frames to the bot.

        The only way frames begin arriving. A backend must not deliver
        ``on_track_audio`` or ``on_track_video`` for a track that was never
        passed here — selective subscription is what keeps unconsumed video out
        of the framework process entirely.
        """

    @abstractmethod
    async def unsubscribe_track(self, bot: BotSession, track_id: str) -> None:
        """Stop delivering a track's frames to the bot."""

    @abstractmethod
    async def publish_audio(self, bot: BotSession, chunk: AudioChunk) -> None:
        """Publish decoded PCM audio on the bot's track.

        AudioChunk is the outbound stream type: it names its own encoding in
        ``format`` and marks the end of an utterance with ``is_final``.
        Implementations must reject a chunk that is not PCM rather than
        forwarding it, because encoding belongs to the backend — a caller
        choosing the wire format would defeat this interface.

        One bot track, heard by every participant. Targeted per-participant
        audio is not supported: the AI is synthesized once and published once.

        What that single track guarantees a backend, in return: an utterance
        arrives contiguously — the framework never interleaves two — and ends on
        a chunk whose ``is_final`` is set, so ``is_final`` is a boundary an
        implementation can rely on rather than a hint. An utterance a
        participant cut short ends the same way, and that closing chunk may
        carry no audio at all (``data=b""``): there is nothing left to play,
        only an end to declare.

        With one exception, which is the end of the session rather than the end
        of an utterance. An utterance the channel abandons because it has left
        the conference is *not* closed: the session the terminal chunk would
        name is on its way out, so publishing into it would race the
        :meth:`leave` behind it. Nothing in band can carry that boundary — a
        process that crashed or a connection that dropped announces nothing
        either — so an implementation must take :meth:`leave`, and a session
        disconnecting by any other means, as ending whatever utterance was in
        flight on it (RFC section 12.10.4).
        """

    @abstractmethod
    async def publish_video(self, bot: BotSession, frame: VideoFrame) -> None:
        """Publish a raw video frame on the bot's track.

        Requires ``ConferenceCapability.VIDEO_PUBLISH``.

        The frame must be raw (``frame.is_raw``). The backend owns encoding,
        symmetrically with the decoding it owns inbound: requiring an encoded
        frame would mean the framework choosing a codec the SFU accepts, which
        is exactly what this interface exists to avoid.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release backend resources. Idempotent."""

    # -------------------------------------------------------------------------
    # Callback registration
    # -------------------------------------------------------------------------

    def on_participant_joined(self, callback: ParticipantCallback) -> None:
        """Register a callback for participants joining the media session."""
        self._participant_joined.append(callback)

    def on_participant_left(self, callback: ParticipantCallback) -> None:
        """Register a callback for participants leaving the media session."""
        self._participant_left.append(callback)

    def on_track_published(self, callback: TrackCallback) -> None:
        """Register a callback for tracks being published."""
        self._track_published.append(callback)

    def on_track_unpublished(self, callback: TrackCallback) -> None:
        """Register a callback for tracks being unpublished."""
        self._track_unpublished.append(callback)

    def on_track_audio(self, callback: TrackAudioCallback) -> None:
        """Register a callback for decoded audio from subscribed tracks."""
        self._track_audio.append(callback)

    def on_track_video(self, callback: TrackVideoCallback) -> None:
        """Register a callback for video from subscribed tracks."""
        self._track_video.append(callback)

    def on_active_speaker_changed(self, callback: ActiveSpeakerCallback) -> None:
        """Register a callback for dominant-speaker changes."""
        self._active_speaker_changed.append(callback)

    def on_connection_quality(self, callback: ConnectionQualityCallback) -> None:
        """Register a callback for per-participant quality reports."""
        self._connection_quality.append(callback)

    def on_bot_session_ended(self, callback: BotSessionEndedCallback) -> None:
        """Register a callback for the bot's session ending without a leave().

        Reported when the backend observes the end — a dropped connection, an
        eviction, the room deleted underneath the bot. A backend that cannot
        observe the loss reports nothing, and knowingly inherits the failure
        mode: a dropped bot its channel goes on reporting present.
        """
        self._bot_session_ended.append(callback)

    # -------------------------------------------------------------------------
    # Emission — for backend implementations
    # -------------------------------------------------------------------------

    async def _emit(self, event: str, callbacks: Sequence[Callable[..., Any]], *args: Any) -> None:
        """Invoke every registered callback, awaiting the coroutines.

        Observers are decoupled from each other and from the backend: one
        failing callback is logged and the rest still run, because a subscriber
        raising must not tear down the media session that fed it. This is
        best-effort fanout, so it is the subscriber's job to handle its own
        failures — a lane that swallows its errors here goes quiet silently.
        """
        for callback in callbacks:
            try:
                result = callback(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Conference callback failed on %s: %r", event, callback)

    async def _emit_participant_joined(
        self, room_id: str, participant: ConferenceParticipant
    ) -> None:
        await self._emit("participant_joined", self._participant_joined, room_id, participant)

    async def _emit_participant_left(
        self, room_id: str, participant: ConferenceParticipant
    ) -> None:
        await self._emit("participant_left", self._participant_left, room_id, participant)

    async def _emit_track_published(self, room_id: str, track: ConferenceTrack) -> None:
        await self._emit("track_published", self._track_published, room_id, track)

    async def _emit_track_unpublished(self, room_id: str, track: ConferenceTrack) -> None:
        await self._emit("track_unpublished", self._track_unpublished, room_id, track)

    async def _emit_track_audio(self, track: ConferenceTrack, frame: AudioFrame) -> None:
        await self._emit("track_audio", self._track_audio, track, frame)

    async def _emit_track_video(self, track: ConferenceTrack, frame: VideoFrame) -> None:
        await self._emit("track_video", self._track_video, track, frame)

    async def _emit_active_speaker_changed(self, room_id: str, participant_id: str) -> None:
        await self._emit(
            "active_speaker_changed", self._active_speaker_changed, room_id, participant_id
        )

    async def _emit_connection_quality(
        self, room_id: str, participant_id: str, quality: str
    ) -> None:
        await self._emit(
            "connection_quality", self._connection_quality, room_id, participant_id, quality
        )

    async def _emit_bot_session_ended(self, bot: BotSession, reason: str) -> None:
        await self._emit("bot_session_ended", self._bot_session_ended, bot, reason)
