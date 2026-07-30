"""LiveKit conference backend — the first ConferenceBackend on a real SFU.

LiveKit is used as an SFU and nothing else: the media plane. The server API
administers rooms, mints credentials and moderates; the realtime SDK gives the
framework one bot participant to subscribe tracks on and publish the AI's voice
through. That is the whole surface.

What is deliberately absent is ``livekit-agents``. RoomKit already owns VAD,
speech-to-text, synthesis, turn detection and interruption, and a transport that
segmented speech would break the separation RFC section 12.10.1 principle 4
draws — the same rule that makes VoiceBackend a pure transport. It would also
defeat the point of having a real backend at all: this exists so RoomKit's own
lanes meet audio RoomKit did not manufacture, and a library that normalised and
segmented upstream would hide exactly the bugs worth finding.

Requires the ``livekit`` optional dependency::

    pip install roomkit[livekit]

Usage::

    from roomkit.channels import ConferenceChannel
    from roomkit.conference.livekit import LiveKitConfig, LiveKitConferenceBackend

    backend = LiveKitConferenceBackend(
        LiveKitConfig(url="wss://my-project.livekit.cloud")  # key/secret from env
    )
    conference = ConferenceChannel("conf", backend=backend, stt=stt, tts=tts)
    kit.register_channel(conference)

Self-hosting, for a local run. Two things about this invocation are not
optional, and both cost an afternoon to discover: ``--dev`` binds to loopback
*inside* the container, where a published port cannot reach it, and LiveKit
advertises the ICE candidate a client is expected to dial — which has to be an
address on the host side of the NAT, not the container's::

    cat > livekit.yaml <<'YAML'
    port: 7880
    rtc:
      tcp_port: 7881
      udp_port: 7882          # one port instead of the default 50000-60000 range,
      node_ip: 127.0.0.1      # which macOS publishes very slowly
      use_external_ip: false
    YAML

    docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \\
        -e LIVEKIT_CONFIG="$(cat livekit.yaml)" \\
        livekit/livekit-server --dev --bind 0.0.0.0

``--dev`` then prints the placeholder credentials it generated, and
``LiveKitConfig(url="ws://127.0.0.1:7880", api_key="devkey",
api_secret="secret")`` connects to it. ``tests/conference/test_livekit_live.py``
is the suite that runs against it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from roomkit.conference._livekit_mapping import (
    capabilities_for,
    participant_record,
    rtc_participant_kind_name,
    rtc_track_kind_name,
    rtc_track_source_name,
    track_record,
    video_grant_kwargs,
)
from roomkit.conference._livekit_session import ConferenceEmissions, LiveKitBotSession
from roomkit.conference.base import ConferenceBackend
from roomkit.conference.models import (
    BotSession,
    ConferenceAccess,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceParticipant,
    ConferenceTrack,
)
from roomkit.core.exceptions import ConferenceCapabilityError
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import AudioChunk

logger = logging.getLogger("roomkit.conference.livekit")

INSTALL_HINT = "Install it with: pip install roomkit[livekit]"


def _import_livekit() -> tuple[Any, Any]:
    """Import the LiveKit SDKs, raising a clear error if either is missing.

    Deferred so that this module — and therefore the backend's whole contract —
    stays importable without the extra. That is what lets the translation and
    conformance tests run wherever RoomKit is installed.
    """
    try:
        from livekit import api, rtc
    except ImportError as exc:
        raise ImportError(
            f"livekit and livekit-api are required for LiveKitConferenceBackend. {INSTALL_HINT}"
        ) from exc
    return api, rtc


@dataclass
class LiveKitConfig:
    """Connection and behaviour settings for :class:`LiveKitConferenceBackend`.

    ``url``, ``api_key`` and ``api_secret`` fall back to LiveKit's own
    environment variables — ``LIVEKIT_URL``, ``LIVEKIT_API_KEY``,
    ``LIVEKIT_API_SECRET`` — so a deployment that already sets them needs no
    RoomKit-specific configuration. The URL is the signalling one (``ws://`` or
    ``wss://``); it is what clients are handed, and the server API derives its
    own endpoint from it.
    """

    url: str | None = None
    """Signalling endpoint. Defaults to ``LIVEKIT_URL``."""

    api_key: str | None = field(default=None, repr=False)
    """API key. Defaults to ``LIVEKIT_API_KEY``."""

    api_secret: str | None = field(default=None, repr=False)
    """API secret. Defaults to ``LIVEKIT_API_SECRET``. Kept out of ``repr()``."""

    access_ttl: timedelta = timedelta(minutes=15)
    """How long a minted credential stays valid.

    Short by default: the credential leaves the process, and RFC section 12.10.2
    recommends short-lived ones. Long enough that a client which fetches its
    token then asks the user for microphone permission still gets in.
    """

    audio_sample_rate: int = 48_000
    """Rate to ask LiveKit's decoder for on subscribed audio tracks.

    48 kHz is Opus's own rate and LiveKit's default, so it is the closest thing
    to "what the publisher sent". Normalising it to what a recognizer wants is
    the lane's job (RFC section 12.10.4), not this backend's — every frame
    declares the format it arrives in.
    """

    audio_channels: int = 1
    """Channel count to ask for on subscribed audio tracks.

    Mono because that is what the pipeline works in and what LiveKit defaults
    to. Set it to 2 to have the SFU hand over a stereo publisher's audio
    unmixed, and the lane's downmix do the work instead.
    """

    publish_queue_ms: int = 300
    """How much of the AI's voice LiveKit may buffer ahead of playout.

    Two things trade off here. The buffer is what keeps the bot's speech
    gap-free when synthesis stutters — and it is also the audio that keeps
    playing after a participant interrupts, because the interface has no way to
    say "stop talking" (see :meth:`LiveKitBotSession.publish`). 300 ms keeps an
    interruption within the range a person reads as responsive.
    """

    remote_unmute: bool = False
    """Whether the server allows unmuting someone else's track.

    Off by default because LiveKit's own default is off: unmuting a remote
    microphone needs ``room.enable_remote_unmute`` in the server configuration.
    Setting this without that is how a moderation UI comes to offer a button the
    server refuses, so the capability is declared only when an integrator says
    the server was configured for it.
    """

    sip_gateway: bool = False
    """Whether PSTN participants can dial into this deployment's conferences.

    Off by default for the same reason: LiveKit's SIP service needs a trunk and
    a dispatch rule before a phone can reach a room. When it is on, dial-ins
    arrive as ordinary participants whose ``sip.`` attributes this backend
    asserts, and identity resolution can reach them.
    """

    room_metadata_key: str = "roomkit"
    """Key under which ``ensure_room`` metadata is stored in LiveKit's room.

    LiveKit's room metadata is one opaque string. Nesting under a key rather
    than writing the mapping at the top level leaves room for whatever else a
    deployment keeps there.
    """


class LiveKitConferenceBackend(ConferenceBackend):
    """ConferenceBackend backed by a LiveKit SFU.

    Example::

        backend = LiveKitConferenceBackend(LiveKitConfig(url="ws://127.0.0.1:7880"))
        await backend.ensure_room("room-1")
        access = await backend.mint_access("room-1", "p-alice", ConferenceGrants())
        bot = await backend.join_as_bot("room-1", "roomkit", ConferenceGrants.for_bot())
    """

    def __init__(self, config: LiveKitConfig | None = None) -> None:
        super().__init__()
        self._config = config or LiveKitConfig()
        self._api_module, self._rtc = _import_livekit()
        url = self._config.url or os.getenv("LIVEKIT_URL")
        api_key = self._config.api_key or os.getenv("LIVEKIT_API_KEY")
        api_secret = self._config.api_secret or os.getenv("LIVEKIT_API_SECRET")
        if not url:
            raise ValueError(
                "LiveKitConferenceBackend needs a signalling URL: pass "
                "LiveKitConfig(url=...) or set LIVEKIT_URL"
            )
        if not api_key or not api_secret:
            raise ValueError(
                "LiveKitConferenceBackend needs an API key and secret: pass them on "
                "LiveKitConfig or set LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
            )
        self._url: str = url
        self._api_key: str = api_key
        self._api_secret: str = api_secret
        if self._config.audio_channels not in (1, 2):
            raise ValueError(f"audio_channels must be 1 or 2, got {self._config.audio_channels}")
        if self._config.publish_queue_ms <= 0:
            raise ValueError(
                f"publish_queue_ms must be positive, got {self._config.publish_queue_ms}: "
                "the bot's voice needs somewhere to be buffered before playout"
            )
        self._sessions: dict[str, LiveKitBotSession] = {}
        self._api: Any | None = None
        self._api_lock = asyncio.Lock()
        self._closed = False

    @property
    def name(self) -> str:
        return "livekit"

    @property
    def capabilities(self) -> ConferenceCapability:
        """What this backend has wired, which is not everything LiveKit sells.

        Declared: separate screen-share tracks, dominant-speaker events and
        per-participant quality reports — each one an event this backend
        translates and forwards. Plus, when the deployment says so,
        ``REMOTE_UNMUTE`` and ``SIP_GATEWAY``, because both depend on server
        configuration this backend cannot see.

        Absent, and each for its own reason. ``EGRESS_RECORDING``: LiveKit can
        do it, nothing here asks it to, and framework-side recording already
        works through the lanes. ``VIDEO_PUBLISH``: the bot has nothing to show
        until an avatar gives it something, so the source is not built and the
        capability is not claimed. ``E2EE``: admitting the bot to a conference's
        key exchange is a contract ConferenceBackend does not have, so declaring
        it would promise a bot that can hear an encrypted room when it cannot.
        """
        return capabilities_for(
            remote_unmute=self._config.remote_unmute,
            sip_gateway=self._config.sip_gateway,
        )

    def _require(self, capability: ConferenceCapability, operation: str) -> None:
        if capability not in self.capabilities:
            raise ConferenceCapabilityError(
                f"{operation} requires {capability.name}, which backend "
                f"{self.name!r} does not declare"
            )

    async def _client(self) -> Any:
        """The server API client, built on first use.

        Not in ``__init__``: it opens an ``aiohttp`` session, which wants a
        running loop, and a backend is constructed while the application is
        being wired up rather than inside one.
        """
        async with self._api_lock:
            if self._api is None:
                self._api = self._api_module.LiveKitAPI(self._url, self._api_key, self._api_secret)
            return self._api

    # -------------------------------------------------------------------------
    # Control plane
    # -------------------------------------------------------------------------

    async def ensure_room(
        self,
        room_id: str,
        metadata: dict[str, Any] | None = None,
        e2ee: bool = False,
    ) -> None:
        """Create the LiveKit room if it is not there.

        ``create_room`` is idempotent on the name, which is what the interface
        asks for: a channel attaching to a room that is already conferring
        re-issues this call and nothing changes.
        """
        if e2ee:
            self._require(ConferenceCapability.E2EE, "End-to-end encryption")
        client = await self._client()
        request = self._api_module.CreateRoomRequest(name=room_id)
        if metadata:
            request.metadata = json.dumps({self._config.room_metadata_key: metadata})
        await client.room.create_room(request)

    async def close_room(self, room_id: str) -> None:
        client = await self._client()
        await client.room.delete_room(self._api_module.DeleteRoomRequest(room=room_id))

    async def mint_access(
        self,
        room_id: str,
        participant_id: str,
        grants: ConferenceGrants,
    ) -> ConferenceAccess:
        """Mint a join credential for a participant the framework named.

        ``participant_id`` becomes LiveKit's participant identity verbatim, so
        the value comes back on every participant and track LiveKit reports.
        That is rule 2 of RFC section 12.10.2 satisfied by the SFU itself rather
        than by a mapping table this backend would have to keep.
        """
        return self._access(room_id, participant_id, grants, publish_data=True)

    def _access(
        self,
        room_id: str,
        identity: str,
        grants: ConferenceGrants,
        *,
        publish_data: bool,
    ) -> ConferenceAccess:
        video_grants = self._api_module.VideoGrants(
            **video_grant_kwargs(room_id, grants, publish_data=publish_data)
        )
        token = (
            self._api_module.AccessToken(self._api_key, self._api_secret)
            .with_identity(identity)
            .with_grants(video_grants)
            .with_ttl(self._config.access_ttl)
        )
        return ConferenceAccess(
            url=self._url,
            token=token.to_jwt(),
            expires_at=datetime.now(UTC) + self._config.access_ttl,
            provider_data={"room": room_id, "identity": identity},
        )

    async def list_participants(self, room_id: str) -> list[ConferenceParticipant]:
        """Who is connected, as the server sees it.

        Read through the server API rather than off the bot's own room, so the
        answer is available before a bot has joined and does not depend on one
        being connected. The names LiveKit's two protocols use differ, so they
        are brought to the realtime dialect here — provenance is decided on
        those names, and a control-plane spelling would leave a dial-in
        unresolvable.
        """
        client = await self._client()
        response = await client.room.list_participants(
            self._api_module.ListParticipantsRequest(room=room_id)
        )
        return [self._participant(room_id, info) for info in response.participants]

    def _participant(self, room_id: str, info: Any) -> ConferenceParticipant:
        kind_name = rtc_participant_kind_name(
            self._api_module.ParticipantInfo.Kind.Name(info.kind)
        )
        participant = participant_record(
            identity=info.identity,
            sid=info.sid,
            kind_name=kind_name,
            name=info.name,
            metadata=info.metadata,
            attributes=dict(info.attributes),
            connected_at=_joined_at(info),
        )
        participant.tracks = [
            track
            for track in (self._track(room_id, info.identity, t) for t in info.tracks)
            if track is not None
        ]
        return participant

    def _track(self, room_id: str, identity: str, info: Any) -> ConferenceTrack | None:
        try:
            return track_record(
                sid=info.sid,
                room_id=room_id,
                participant_id=identity,
                kind_name=rtc_track_kind_name(self._api_module.TrackType.Name(info.type)),
                source_name=rtc_track_source_name(self._api_module.TrackSource.Name(info.source)),
                muted=info.muted,
                name=info.name,
                mime_type=info.mime_type,
            )
        except ValueError:
            logger.debug(
                "Skipping LiveKit track %s in room %s: RoomKit has no kind for it",
                info.sid,
                room_id,
                exc_info=True,
            )
            return None

    async def remove_participant(self, room_id: str, participant_id: str) -> None:
        client = await self._client()
        await client.room.remove_participant(
            self._api_module.RoomParticipantIdentity(room=room_id, identity=participant_id)
        )

    async def mute_track(self, room_id: str, track_id: str) -> None:
        await self._set_muted(room_id, track_id, muted=True)

    async def unmute_track(self, room_id: str, track_id: str) -> None:
        self._require(ConferenceCapability.REMOTE_UNMUTE, "Remote unmute")
        await self._set_muted(room_id, track_id, muted=False)

    async def _set_muted(self, room_id: str, track_id: str, *, muted: bool) -> None:
        identity = await self._publisher_identity(room_id, track_id)
        client = await self._client()
        await client.room.mute_published_track(
            self._api_module.MuteRoomTrackRequest(
                room=room_id, identity=identity, track_sid=track_id, muted=muted
            )
        )

    async def _publisher_identity(self, room_id: str, track_id: str) -> str:
        """Find who publishes a track, because LiveKit's mute API needs to know.

        The interface passes a track alone, and LiveKit is keyed on the
        participant as well. The bot session watched the track be published, so
        it is asked first; failing that — a track published before the bot
        arrived, or moderation on a room with no bot in it at all, which is a
        control-plane call and does not require one — the server is.
        """
        for session in self._sessions.values():
            if session.room_id != room_id:
                continue
            if (identity := session.publisher_identity(track_id)) is not None:
                return identity
        for participant in await self.list_participants(room_id):
            for track in participant.tracks:
                if track.id == track_id:
                    return participant.participant_id
        raise ValueError(
            f"no participant in room {room_id!r} publishes track {track_id!r}, so there is "
            "nobody to moderate"
        )

    # -------------------------------------------------------------------------
    # Bot participant
    # -------------------------------------------------------------------------

    async def join_as_bot(
        self,
        room_id: str,
        identity: str,
        grants: ConferenceGrants,
    ) -> BotSession:
        """Connect the framework to the conference as a participant.

        The bot's token carries the grants it was given and nothing more, and it
        joins with auto-subscription off: the framework's subscription set is
        the authoritative one, and a bot the SDK subscribed on its own behalf
        would deliver media nobody asked for.
        """
        if self._closed:
            raise RuntimeError("LiveKitConferenceBackend is closed")
        session = BotSession(id=f"lk-{uuid4().hex[:8]}", room_id=room_id, identity=identity)
        bot = LiveKitBotSession(
            rtc=self._rtc,
            session=session,
            config=self._config,
            emissions=self._emissions(),
        )
        access = self._access(room_id, identity, grants, publish_data=False)
        try:
            await bot.connect(access.url, access.token)
        except BaseException:
            # A join that failed still built a room object and started the task
            # that drains its events, and nothing else holds either: the session
            # is never registered, so the channel gets no handle to close. Torn
            # down here or not at all.
            #
            # On its own task, shielded, because a cancellation is one of the
            # ways to arrive here — the channel abandons a join whose room was
            # detached underneath it — and cleanup that awaited normally would
            # be cancelled at its first await, which is where the disconnect is.
            # Suppressed so that what the caller is told is what went wrong with
            # the join, not what went wrong tidying up after it.
            cleanup = asyncio.create_task(bot.leave())
            with contextlib.suppress(BaseException):
                await asyncio.shield(cleanup)
            raise
        # Registered once connected, because the session takes its identifier
        # from the participant LiveKit created for it.
        self._sessions[session.id] = bot
        return session

    def _emissions(self) -> ConferenceEmissions:
        return ConferenceEmissions(
            participant_joined=self._emit_participant_joined,
            participant_left=self._emit_participant_left,
            track_published=self._emit_track_published,
            track_unpublished=self._emit_track_unpublished,
            track_audio=self._emit_track_audio,
            track_video=self._emit_track_video,
            active_speaker_changed=self._emit_active_speaker_changed,
            connection_quality=self._emit_connection_quality,
            bot_session_ended=self._bot_session_gone,
        )

    async def _bot_session_gone(self, bot: BotSession, reason: str) -> None:
        """Forget a session the SFU ended, then report it (RFC 12.10.3).

        Forgotten first: the connection is gone, so there is nothing a later
        ``leave()`` could do for it, and a registry still carrying it would
        refuse the re-join the report exists to make possible.
        """
        self._sessions.pop(bot.id, None)
        await self._emit_bot_session_ended(bot, reason)

    def _session(self, bot: BotSession) -> LiveKitBotSession:
        session = self._sessions.get(bot.id)
        if session is None:
            raise ValueError(f"bot session {bot.id!r} is not connected to backend {self.name!r}")
        return session

    async def leave(self, bot: BotSession) -> None:
        """Take the bot out, and forget the session only once it *is* out.

        Popping first was how a failed disconnect became invisible: the
        registry had already forgotten the session, so a retry found nothing
        to leave and the channel's books called the bot gone while it may
        still have been in the meeting. The session stays registered until
        the disconnect returns, and the failure propagates for the channel's
        leaving ledger to record (RFC 12.10.4).
        """
        session = self._sessions.get(bot.id)
        if session is None:
            return
        await session.leave()
        self._sessions.pop(bot.id, None)

    async def subscribe_track(self, bot: BotSession, track_id: str) -> None:
        await self._session(bot).subscribe(track_id)

    async def unsubscribe_track(self, bot: BotSession, track_id: str) -> None:
        await self._session(bot).unsubscribe(track_id)

    async def publish_audio(self, bot: BotSession, chunk: AudioChunk) -> None:
        await self._session(bot).publish(chunk)

    async def publish_video(self, bot: BotSession, frame: VideoFrame) -> None:
        """Refused: this backend does not publish the bot's video.

        LiveKit can carry it. Nothing here builds the source that would, because
        the bot has nothing to show until an avatar gives it something — so the
        capability is not declared and this is what not declaring it means
        (RFC section 12.10.3).
        """
        self._require(ConferenceCapability.VIDEO_PUBLISH, "Bot video publishing")

    async def close(self) -> None:
        """Release the sessions and the API client. Idempotent.

        A session whose disconnect fails stays registered and is *raised*,
        together, once every session has been attempted and the client is
        released — a close that only logged them reported bots possibly still
        in their meetings as a clean shutdown, which is the one answer the
        channel's books must never get (RFC 12.10.4). The channel records the
        failure against its own close and keeps naming the sessions.
        """
        if self._closed:
            return
        self._closed = True
        failures: list[Exception] = []
        for bot_id, session in list(self._sessions.items()):
            try:
                await session.leave()
            except Exception as exc:
                failures.append(exc)
                logger.exception("Leaving conference room %s during close failed", session.room_id)
            else:
                self._sessions.pop(bot_id, None)
        api, self._api = self._api, None
        if api is not None:
            await api.aclose()
        if failures:
            raise ExceptionGroup(
                f"closing the LiveKit backend could not take {len(failures)} bot session(s) "
                "out of their conference room(s)",
                failures,
            )


def _joined_at(info: Any) -> datetime | None:
    """When a participant joined, from whichever field LiveKit filled.

    ``joined_at_ms`` is the precise one and the newer one; ``joined_at`` carries
    whole seconds. Both are zero on a participant that has not finished joining,
    and that is not a moment in 1970 — it is no answer, so it is reported as
    none rather than as the epoch.
    """
    if getattr(info, "joined_at_ms", 0):
        return datetime.fromtimestamp(info.joined_at_ms / 1000, tz=UTC)
    if getattr(info, "joined_at", 0):
        return datetime.fromtimestamp(info.joined_at, tz=UTC)
    return None
