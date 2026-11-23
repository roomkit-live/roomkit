"""LiveKit backend behaviour that needs the SDK but no server.

Control-plane calls run against a recording stand-in for ``LiveKitAPI``, and the
participants it returns are real protobuf messages — the same types the server
sends — so the translation is checked against the wire shape rather than a
hand-made object that agrees with it. Minting needs nothing stood in for at all:
a token is signed locally, so the claims can be decoded and read.

What genuinely needs a server — joining, subscribing, frames, teardown — is in
``test_livekit_live.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from roomkit.channels import _conference_activity
from roomkit.conference import _livekit_departure as departure_module
from roomkit.conference._livekit_bridge import MAX_QUEUED_EVENTS, EventBridge
from roomkit.conference._livekit_departure import SessionDeparture
from roomkit.conference._livekit_media import TrackPumps
from roomkit.conference._livekit_room_view import RoomView
from roomkit.conference._livekit_session import ConferenceEmissions, LiveKitBotSession
from roomkit.conference._livekit_voice import BotVoiceTrack, VoicePublicationError
from roomkit.conference.livekit import LiveKitConferenceBackend, LiveKitConfig
from roomkit.conference.models import BotSession, ConferenceGrants, ConferenceTrack, TrackKind
from roomkit.core.exceptions import ConferenceCapabilityError
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.base import AudioChunk

api = pytest.importorskip("livekit.api")
jwt = pytest.importorskip("jwt")

API_KEY = "devkey"
API_SECRET = "secret-that-is-long-enough-for-hs256"


@dataclass
class _FakeRoomService:
    """Records what the backend asked the server to do."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    participants: list[Any] = field(default_factory=list)
    remove_error: Exception | None = None

    async def create_room(self, request: Any) -> Any:
        self.calls.append(("create_room", request))
        return api.Room(name=request.name)

    async def delete_room(self, request: Any) -> Any:
        self.calls.append(("delete_room", request))
        return api.DeleteRoomResponse()

    async def list_participants(self, request: Any) -> Any:
        self.calls.append(("list_participants", request))
        return api.ListParticipantsResponse(participants=self.participants)

    async def remove_participant(self, request: Any) -> Any:
        self.calls.append(("remove_participant", request))
        if self.remove_error is not None:
            raise self.remove_error
        return api.RemoveParticipantResponse()

    async def mute_published_track(self, request: Any) -> Any:
        self.calls.append(("mute_published_track", request))
        return api.MuteRoomTrackResponse()

    async def update_participant(self, request: Any) -> Any:
        self.calls.append(("update_participant", request))
        return api.ParticipantInfo(identity=request.identity)


@dataclass
class _FakeAPI:
    room: _FakeRoomService = field(default_factory=_FakeRoomService)
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


def _backend(**config: Any) -> LiveKitConferenceBackend:
    settings = {"url": "ws://127.0.0.1:7880", "api_key": API_KEY, "api_secret": API_SECRET}
    settings.update(config)
    return LiveKitConferenceBackend(LiveKitConfig(**settings))


def _served(backend: LiveKitConferenceBackend) -> _FakeRoomService:
    """Give a backend a stand-in server and hand back its record of calls."""
    fake = _FakeAPI()
    backend._api = fake
    return fake.room


def _claims(token: str) -> dict[str, Any]:
    return jwt.decode(token, API_SECRET, algorithms=["HS256"])


async def _ignore(*args: Any) -> None:
    return None


def _emissions(**named: Callable[..., Awaitable[None]]) -> ConferenceEmissions:
    """A session's emissions, every one ignored unless named."""
    fields = {name: _ignore for name in ConferenceEmissions.__dataclass_fields__}
    return ConferenceEmissions(**{**fields, **named})


def _session_for(
    rtc: Any,
    *,
    session: BotSession | None = None,
    ended: Callable[..., Awaitable[None]] = _ignore,
    evict: Callable[[], Awaitable[None]] = _ignore,
    **emitted: Callable[..., Awaitable[None]],
) -> LiveKitBotSession:
    """A LiveKitBotSession on a stand-in rtc, every emission ignored unless named."""
    return LiveKitBotSession(
        rtc=rtc,
        session=session or BotSession(id="lk-1", room_id="room-1", identity="roomkit"),
        config=SimpleNamespace(publish_queue_ms=200),
        emissions=_emissions(bot_session_ended=ended, **emitted),
        evict=evict,
    )


class TestConfiguration:
    def test_a_missing_url_is_refused_at_construction(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("LIVEKIT_URL", raising=False)

        with pytest.raises(ValueError, match="signalling URL"):
            LiveKitConferenceBackend(LiveKitConfig(api_key=API_KEY, api_secret=API_SECRET))

    def test_missing_credentials_are_refused_at_construction(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

        with pytest.raises(ValueError, match="API key and secret"):
            LiveKitConferenceBackend(LiveKitConfig(url="ws://127.0.0.1:7880"))

    def test_the_environment_supplies_what_the_config_leaves_out(self, monkeypatch: Any) -> None:
        """LiveKit's own variables, so a deployment that already sets them needs
        no RoomKit-specific configuration.
        """
        monkeypatch.setenv("LIVEKIT_URL", "ws://from-env:7880")
        monkeypatch.setenv("LIVEKIT_API_KEY", API_KEY)
        monkeypatch.setenv("LIVEKIT_API_SECRET", API_SECRET)

        backend = LiveKitConferenceBackend()

        assert backend._url == "ws://from-env:7880"

    def test_an_unusable_channel_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="audio_channels"):
            _backend(audio_channels=3)

    def test_a_source_with_no_queue_is_refused(self) -> None:
        with pytest.raises(ValueError, match="publish_queue_ms"):
            _backend(publish_queue_ms=0)

    def test_the_secret_stays_out_of_the_config_repr(self) -> None:
        """A config that surfaces in a log or a traceback must not carry it."""
        assert API_SECRET not in repr(LiveKitConfig(api_secret=API_SECRET))

    def test_the_backend_names_itself(self) -> None:
        assert _backend().name == "livekit"


class TestMintedAccess:
    def test_the_framework_identity_is_the_livekit_identity(self) -> None:
        """Rule 2 of RFC section 12.10.2: the value goes in, and LiveKit echoes
        it back on every participant and track.
        """
        access = _backend()._access("room-1", "p-alice", ConferenceGrants(), publish_data=True)

        assert _claims(access.token)["sub"] == "p-alice"

    def test_the_grants_reach_the_token(self) -> None:
        access = _backend()._access(
            "room-1",
            "p-alice",
            ConferenceGrants(publish_video=False, publish_screen_share=False, moderate=True),
            publish_data=True,
        )

        video = _claims(access.token)["video"]
        assert video["room"] == "room-1"
        assert video["roomJoin"] is True
        assert video["roomAdmin"] is True
        assert video["canPublishSources"] == ["microphone"]

    def test_the_sound_of_a_screen_share_reaches_the_token_when_granted(self) -> None:
        access = _backend()._access(
            "room-1",
            "p-alice",
            ConferenceGrants(publish_video=False, publish_screen_share_audio=True),
            publish_data=True,
        )

        video = _claims(access.token)["video"]
        assert video["canPublishSources"] == ["microphone", "screen_share", "screen_share_audio"]

    async def test_a_default_mint_does_not_carry_the_sound_of_a_screen_share(self) -> None:
        access = await _backend().mint_access("room-1", "p-alice", ConferenceGrants())

        video = _claims(access.token)["video"]
        assert video["canPublishSources"] == ["microphone", "camera", "screen_share"]

    def test_a_listening_bot_asks_the_server_for_no_publish_right(self) -> None:
        access = _backend()._access(
            "room-1", "roomkit", ConferenceGrants.observer(), publish_data=False
        )

        video = _claims(access.token)["video"]
        assert video["canPublish"] is False
        assert video["hidden"] is True
        assert "canPublishSources" not in video

    async def test_the_credential_expires_and_says_when(self) -> None:
        backend = _backend(access_ttl=timedelta(minutes=5))
        before = datetime.now(UTC)

        access = await backend.mint_access("room-1", "p-alice", ConferenceGrants())

        assert access.expires_at is not None
        assert timedelta(minutes=4) < access.expires_at - before < timedelta(minutes=6)
        assert _claims(access.token)["exp"] > before.timestamp()

    def test_the_client_is_pointed_at_the_signalling_url(self) -> None:
        access = _backend()._access("room-1", "p-alice", ConferenceGrants(), publish_data=True)

        assert access.url == "ws://127.0.0.1:7880"
        assert access.provider_data == {"room": "room-1", "identity": "p-alice"}

    async def test_the_attributes_the_caller_asked_for_ride_the_token(self) -> None:
        """RFC §12.10.3: the identity is not the only field that travels.

        LiveKit reports the claim back in the attribute map ``list_participants``
        reads, which is what makes the value reachable by the SFU's own clients.
        """
        access = await _backend().mint_access(
            "room-1", "p-alice", ConferenceGrants(), attributes={"app.user": "user-42"}
        )

        assert _claims(access.token)["attributes"] == {"app.user": "user-42"}

    @pytest.mark.parametrize("attributes", [None, {}], ids=["none", "empty-map"])
    async def test_a_mint_that_asks_for_no_attribute_carries_no_claim(
        self, attributes: dict[str, str] | None
    ) -> None:
        access = await _backend().mint_access(
            "room-1", "p-alice", ConferenceGrants(), attributes=attributes
        )

        assert "attributes" not in _claims(access.token)

    async def test_a_name_and_attributes_travel_together(self) -> None:
        access = await _backend().mint_access(
            "room-1",
            "p-alice",
            ConferenceGrants(),
            display_name="Alice",
            attributes={"app.user": "user-42"},
        )

        claims = _claims(access.token)
        assert claims["name"] == "Alice"
        assert claims["attributes"] == {"app.user": "user-42"}

    def test_the_bots_own_token_carries_none(self) -> None:
        """The bot is the framework's own session, and nothing asks it to
        present attributes to the room.
        """
        access = _backend()._access(
            "room-1", "roomkit", ConferenceGrants.observer(), publish_data=False
        )

        assert "attributes" not in _claims(access.token)


class TestControlPlane:
    async def test_a_room_is_created_under_the_roomkit_room_id(self) -> None:
        backend = _backend()
        room = _served(backend)

        await backend.ensure_room("room-1")

        assert room.calls[0][0] == "create_room"
        assert room.calls[0][1].name == "room-1"

    async def test_metadata_is_nested_rather_than_written_flat(self) -> None:
        """LiveKit's room metadata is one opaque string, and a deployment may
        keep its own things in it.
        """
        backend = _backend()
        room = _served(backend)

        await backend.ensure_room("room-1", {"tenant": "acme"})

        assert room.calls[0][1].metadata == '{"roomkit": {"tenant": "acme"}}'

    async def test_a_second_attach_re_issues_the_same_idempotent_call(self) -> None:
        backend = _backend()
        room = _served(backend)

        await backend.ensure_room("room-1")
        await backend.ensure_room("room-1")

        assert [call[0] for call in room.calls] == ["create_room", "create_room"]

    async def test_end_to_end_encryption_is_refused_rather_than_ignored(self) -> None:
        """Admitting the bot to a key exchange is a contract ConferenceBackend
        does not have, so the capability is not declared — and this is what not
        declaring it has to mean.
        """
        backend = _backend()
        _served(backend)

        with pytest.raises(ConferenceCapabilityError, match="E2EE"):
            await backend.ensure_room("room-1", e2ee=True)

    async def test_the_refusal_happens_before_the_room_is_created(self) -> None:
        backend = _backend()
        room = _served(backend)

        with pytest.raises(ConferenceCapabilityError):
            await backend.ensure_room("room-1", e2ee=True)

        assert room.calls == []

    async def test_closing_a_room_deletes_it(self) -> None:
        backend = _backend()
        room = _served(backend)

        await backend.close_room("room-1")

        assert room.calls == [("delete_room", api.DeleteRoomRequest(room="room-1"))]

    async def test_removing_a_participant_names_it_by_identity(self) -> None:
        backend = _backend()
        room = _served(backend)

        await backend.remove_participant("room-1", "p-alice")

        assert room.calls[0][1].identity == "p-alice"


class TestListedParticipants:
    def _info(self, **overrides: Any) -> Any:
        fields: dict[str, Any] = {
            "identity": "p-alice",
            "sid": "PA_1",
            "name": "Alice",
            "kind": api.ParticipantInfo.Kind.STANDARD,
            "joined_at_ms": 1_785_000_000_000,
        }
        fields.update(overrides)
        return api.ParticipantInfo(**fields)

    async def test_a_participant_is_translated_from_the_wire_shape(self) -> None:
        backend = _backend()
        _served(backend).participants = [self._info()]

        participants = await backend.list_participants("room-1")

        assert [p.participant_id for p in participants] == ["p-alice"]
        assert participants[0].metadata["livekit.sid"] == "PA_1"

    async def test_the_reported_join_time_is_used_and_is_aware(self) -> None:
        backend = _backend()
        _served(backend).participants = [self._info()]

        participant = (await backend.list_participants("room-1"))[0]

        assert participant.connected_at == datetime.fromtimestamp(1_785_000_000, tz=UTC)
        assert participant.connected_at.tzinfo is not None

    async def test_a_participant_still_joining_reports_no_join_time(self) -> None:
        """Zero is not a moment in 1970 — it is no answer, and the field's own
        default is a better one than the epoch.
        """
        backend = _backend()
        _served(backend).participants = [self._info(joined_at_ms=0, joined_at=0)]

        participant = (await backend.list_participants("room-1"))[0]

        assert participant.connected_at.year > 2000

    async def test_whole_second_join_times_are_still_read(self) -> None:
        backend = _backend()
        _served(backend).participants = [self._info(joined_at_ms=0, joined_at=1_785_000_000)]

        participant = (await backend.list_participants("room-1"))[0]

        assert participant.connected_at == datetime.fromtimestamp(1_785_000_000, tz=UTC)

    async def test_a_dial_in_arrives_with_its_caller_number_asserted(self) -> None:
        """The control-plane kind is spelled differently from the realtime one,
        and provenance is decided on the realtime spelling — so a dial-in listed
        through the API must be as resolvable as one that arrived as an event.
        """
        backend = _backend()
        _served(backend).participants = [
            self._info(
                kind=api.ParticipantInfo.Kind.SIP,
                attributes={"sip.phoneNumber": "+15145550123"},
            )
        ]

        participant = (await backend.list_participants("room-1"))[0]

        assert (participant.asserted_metadata or {})["sip.phoneNumber"] == "+15145550123"

    def test_the_backend_declares_the_key_of_a_caller_s_number(self) -> None:
        """LiveKit's key is the backend's to provide (RFC §12.10.2): a channel
        on it reads ``sip.phoneNumber`` before its generic keys, and never the
        number the caller dialled."""
        assert _backend().address_keys == ("sip.phoneNumber",)

    async def test_a_participants_tracks_come_with_it(self) -> None:
        backend = _backend()
        _served(backend).participants = [
            self._info(
                tracks=[
                    api.TrackInfo(
                        sid="TR_1",
                        type=api.TrackType.AUDIO,
                        source=api.TrackSource.MICROPHONE,
                        name="mic",
                        muted=True,
                    ),
                    api.TrackInfo(
                        sid="TR_2",
                        type=api.TrackType.VIDEO,
                        source=api.TrackSource.SCREEN_SHARE,
                    ),
                ]
            )
        ]

        tracks = (await backend.list_participants("room-1"))[0].tracks

        assert [t.id for t in tracks] == ["TR_1", "TR_2"]
        assert tracks[0].kind is TrackKind.AUDIO
        assert tracks[0].muted is True
        assert tracks[1].kind is TrackKind.SCREEN_SHARE

    async def test_a_track_kind_roomkit_has_no_lane_for_is_left_out(self) -> None:
        backend = _backend()
        _served(backend).participants = [
            self._info(
                tracks=[
                    api.TrackInfo(
                        sid="TR_3", type=api.TrackType.DATA, source=api.TrackSource.UNKNOWN
                    )
                ]
            )
        ]

        assert (await backend.list_participants("room-1"))[0].tracks == []


class TestModeration:
    def _room_with_track(self, backend: LiveKitConferenceBackend) -> _FakeRoomService:
        room = _served(backend)
        room.participants = [
            api.ParticipantInfo(
                identity="p-alice",
                sid="PA_1",
                tracks=[
                    api.TrackInfo(
                        sid="TR_1",
                        type=api.TrackType.AUDIO,
                        source=api.TrackSource.MICROPHONE,
                    )
                ],
            )
        ]
        return room

    async def test_muting_finds_the_publisher_the_interface_did_not_name(self) -> None:
        """LiveKit's mute API is keyed on the participant as well as the track,
        and the interface passes only the track — so the server is asked who
        publishes it. No capability is declared: muting needs none.
        """
        backend = _backend()
        room = self._room_with_track(backend)

        await backend.mute_track("room-1", "TR_1")

        request = next(r for name, r in room.calls if name == "mute_published_track")
        assert request.identity == "p-alice"
        assert request.track_sid == "TR_1"
        assert request.muted is True

    async def test_unmuting_without_the_capability_is_refused(self) -> None:
        """SFUs commonly refuse remote unmute unless enabled server-side, so a
        backend that appeared to succeed would be worse than one that says no.
        """
        backend = _backend(remote_unmute=False)
        room = self._room_with_track(backend)

        with pytest.raises(ConferenceCapabilityError, match="REMOTE_UNMUTE"):
            await backend.unmute_track("room-1", "TR_1")

        assert not [name for name, _ in room.calls if name == "mute_published_track"]

    async def test_unmuting_works_when_the_server_was_configured_for_it(self) -> None:
        backend = _backend(remote_unmute=True)
        room = self._room_with_track(backend)

        await backend.unmute_track("room-1", "TR_1")

        request = next(r for name, r in room.calls if name == "mute_published_track")
        assert request.muted is False

    async def test_moderating_a_track_nobody_publishes_says_so(self) -> None:
        backend = _backend()
        _served(backend)

        with pytest.raises(ValueError, match="nobody to moderate"):
            await backend.mute_track("room-1", "TR_missing")


class TestBotSurface:
    async def test_publishing_video_is_refused_as_a_missing_capability(self) -> None:
        """LiveKit can carry it; nothing here builds the source that would.
        Refused as an undeclared capability, which is the interface's own way of
        saying so — not as an unimplemented method.
        """
        backend = _backend()
        frame = VideoFrame(data=b"\x00" * 6, codec="raw_rgb24", width=2, height=1)

        with pytest.raises(ConferenceCapabilityError, match="VIDEO_PUBLISH"):
            await backend.publish_video(BotSession(id="lk-1", room_id="r", identity="b"), frame)

    @pytest.mark.parametrize("operation", ["subscribe", "unsubscribe", "publish"])
    async def test_a_session_this_backend_did_not_open_is_refused(self, operation: str) -> None:
        backend = _backend()
        stranger = BotSession(id="lk-stranger", room_id="room-1", identity="roomkit")

        with pytest.raises(ValueError, match="not connected"):
            if operation == "subscribe":
                await backend.subscribe_track(stranger, "TR_1")
            elif operation == "unsubscribe":
                await backend.unsubscribe_track(stranger, "TR_1")
            else:
                await backend.publish_audio(stranger, AudioChunk(data=b"\x00\x00"))

    async def test_leaving_a_session_that_is_not_there_is_quiet(self) -> None:
        """Teardown runs on paths where the join failed, and a leave that raised
        would turn a failed join into a failed detach.
        """
        await _backend().leave(BotSession(id="lk-gone", room_id="room-1", identity="roomkit"))

    async def test_stopping_playback_for_a_session_that_is_not_there_is_quiet(self) -> None:
        """Deliberately unlike the refused trio above: a barge-in can race the
        SFU dropping the bot, and the silence a stop asks for is already true
        of a session that is gone (RFC §12.10.3).
        """
        await _backend().stop_playback(
            BotSession(id="lk-gone", room_id="room-1", identity="roomkit")
        )

    async def test_stopping_playback_reaches_the_session_it_names(self) -> None:
        backend = _backend()

        class _Session:
            def __init__(self) -> None:
                self.stops = 0

            def stop_playback(self) -> None:
                self.stops += 1

        session = _Session()
        backend._sessions["lk-1"] = session  # type: ignore[assignment]

        await backend.stop_playback(BotSession(id="lk-1", room_id="room-1", identity="roomkit"))

        assert session.stops == 1

    async def test_a_grant_update_sends_the_sources_as_server_enum_values(self) -> None:
        """The update's carrier: TrackSource values on ``UpdateParticipant``,
        converted from the same list the token carries (RFC 12.10.3).
        """
        backend = _backend()
        room = _served(backend)
        bot = BotSession(id="lk-1", room_id="room-1", identity="roomkit")
        backend._sessions[bot.id] = object()  # type: ignore[assignment]

        await backend.update_bot_grants(
            bot,
            ConferenceGrants(
                publish_audio=True,
                publish_video=False,
                publish_screen_share=False,
                publish_screen_share_audio=True,
            ),
        )

        [(name, request)] = room.calls
        assert name == "update_participant"
        assert request.identity == "roomkit"
        assert request.permission.can_publish is True
        assert list(request.permission.can_publish_sources) == [
            api.TrackSource.MICROPHONE,
            api.TrackSource.SCREEN_SHARE_AUDIO,
        ]

    async def test_joining_a_closed_backend_is_refused(self) -> None:
        backend = _backend()
        _served(backend)
        await backend.close()

        with pytest.raises(RuntimeError, match="closed"):
            await backend.join_as_bot("room-1", "roomkit", ConferenceGrants.for_bot())


class TestClose:
    async def test_closing_releases_the_server_client(self) -> None:
        backend = _backend()
        fake = _FakeAPI()
        backend._api = fake

        await backend.close()

        assert fake.closed is True

    async def test_closing_twice_is_idempotent(self) -> None:
        backend = _backend()
        backend._api = _FakeAPI()

        await backend.close()
        await backend.close()

    async def test_closing_a_backend_that_never_talked_to_a_server_is_quiet(self) -> None:
        await _backend().close()


class _RefusingSession:
    """A stand-in bot session whose disconnect the SFU refuses on demand."""

    def __init__(self) -> None:
        self.attempts = 0
        self.refuse = True
        self.room_id = "room-1"

    async def leave(self) -> None:
        self.attempts += 1
        if self.refuse:
            raise RuntimeError("the SFU refused the disconnect")


class TestLeaveFailureIsNotSwallowed:
    """RFC 12.10.4: failing to remove a session is failing to close. The
    channel's entire departure bookkeeping — the leaving ledger, info's
    bot_present, the close's final raise — is built on leave() telling the
    truth, so a backend that swallows a failed disconnect defeats all of it.
    """

    async def test_a_failed_disconnect_propagates_and_keeps_the_session(self) -> None:
        backend = _backend()
        bot = BotSession(id="lk-1", room_id="room-1", identity="roomkit")
        session = _RefusingSession()
        backend._sessions[bot.id] = session  # type: ignore[assignment]

        with pytest.raises(RuntimeError):
            await backend.leave(bot)

        # The registry still carries the session, so the close's retry has
        # something to leave — popping first was how the bot got stranded.
        assert bot.id in backend._sessions
        session.refuse = False
        await backend.leave(bot)
        assert bot.id not in backend._sessions
        assert session.attempts == 2

    async def test_close_raises_for_sessions_it_could_not_take_out(self) -> None:
        backend = _backend()
        backend._api = _FakeAPI()
        bot = BotSession(id="lk-1", room_id="room-1", identity="roomkit")
        session = _RefusingSession()
        backend._sessions[bot.id] = session  # type: ignore[assignment]

        with pytest.raises(ExceptionGroup) as failure:
            await backend.close()

        assert "1 bot session(s)" in str(failure.value)
        assert bot.id in backend._sessions, "close forgot a bot it could not remove"

    async def test_the_session_itself_retries_the_disconnect(self) -> None:
        class _FakeRoom:
            def __init__(self) -> None:
                self.disconnects = 0
                self.refuse = True
                self.local_participant = None

            def on(self, *args: Any, **kwargs: Any) -> None:
                return None

            async def disconnect(self) -> None:
                self.disconnects += 1
                if self.refuse:
                    raise RuntimeError("signalling connection already gone")

        room = _FakeRoom()
        rtc = SimpleNamespace(Room=lambda: room)

        session = _session_for(rtc)

        with pytest.raises(RuntimeError):
            await session.leave()
        # A failed leave is not terminal: the retry reattempts the disconnect.
        room.refuse = False
        await session.leave()
        assert room.disconnects == 2
        # A *successful* one is: later calls are no-ops.
        await session.leave()
        assert room.disconnects == 2


def _departure_for(
    room: Any,
    *,
    session: BotSession | None = None,
    ended: Callable[..., Awaitable[None]] = _ignore,
    evict: Callable[[], Awaitable[None]] = _ignore,
) -> SessionDeparture:
    """A SessionDeparture on a stand-in room, with real voice, pumps and bridge."""
    bot = session or BotSession(id="lk-1", room_id="room-1", identity="roomkit")
    return SessionDeparture(
        room=room,
        session=bot,
        voice=BotVoiceTrack(
            rtc=SimpleNamespace(),
            room=room,
            identity=bot.identity,
            room_id=bot.room_id,
            queue_ms=200,
        ),
        pumps=TrackPumps(
            rtc=SimpleNamespace(),
            room_id=bot.room_id,
            audio_sink=_ignore,
            video_sink=_ignore,
            config=SimpleNamespace(),
        ),
        bridge=EventBridge(bot.room_id),
        report_end=ended,
        evict=evict,
    )


def _bridge_session(**emitted: Callable[..., Awaitable[None]]) -> Any:
    """A LiveKitBotSession with a fake rtc, for exercising the event bridge."""

    class _FakeRoom:
        local_participant = None

        def on(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def disconnect(self) -> None:
            return None

    return _session_for(SimpleNamespace(Room=_FakeRoom), **emitted)


class _ClearableSource:
    """A stand-in ``rtc.AudioSource`` that records what happens to its queue."""

    def __init__(self, sample_rate: int, num_channels: int, queue_size_ms: int = 0) -> None:
        self.captured: list[Any] = []
        self.cleared = 0
        self.closed = False

    async def capture_frame(self, frame: Any) -> None:
        self.captured.append(frame)

    def clear_queue(self) -> None:
        self.cleared += 1

    async def aclose(self) -> None:
        self.closed = True


def _voice_track(
    publish_error: Exception | None = None,
) -> tuple[Any, list[_ClearableSource]]:
    """A BotVoiceTrack on a fake rtc, and every source it creates.

    ``publish_error`` is what the SDK raises when asked to publish the track;
    the fake participant keeps the sids it was asked to unpublish.
    """
    sources: list[_ClearableSource] = []

    def _source(sample_rate: int, num_channels: int, queue_size_ms: int) -> _ClearableSource:
        source = _ClearableSource(sample_rate, num_channels, queue_size_ms)
        sources.append(source)
        return source

    class _LocalParticipant:
        def __init__(self) -> None:
            self.unpublished: list[Any] = []

        async def publish_track(self, track: Any, options: Any) -> None:
            if publish_error is not None:
                raise publish_error

        async def unpublish_track(self, sid: Any) -> None:
            self.unpublished.append(sid)

    rtc = SimpleNamespace(
        AudioSource=_source,
        LocalAudioTrack=SimpleNamespace(
            create_audio_track=lambda name, source: SimpleNamespace(sid="TR_BOT")
        ),
        TrackPublishOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        TrackSource=SimpleNamespace(SOURCE_MICROPHONE=2),
        AudioFrame=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    track = BotVoiceTrack(
        rtc=rtc,
        room=SimpleNamespace(local_participant=_LocalParticipant()),
        identity="roomkit",
        room_id="room-1",
        queue_ms=200,
    )
    return track, sources


class TestStopPlaybackDiscardsTheQueue:
    async def test_the_stop_empties_the_sources_queue(self) -> None:
        """``clear_queue`` is the whole point of the gesture: the ``queue_ms``
        of audio the framework ran ahead of playout is what a participant who
        cut the bot off would otherwise sit through.
        """
        track, sources = _voice_track()
        await track.publish(AudioChunk(data=b"\x00\x00" * 160, sample_rate=48_000))

        track.discard_queued()

        assert [source.cleared for source in sources] == [1]

    async def test_a_stop_before_any_audio_touches_nothing(self) -> None:
        """No source yet means nothing was ever published: nothing to discard,
        and no source sprung into being by the discard itself.
        """
        track, sources = _voice_track()

        track.discard_queued()

        assert sources == []

    async def test_a_stop_is_not_a_boundary(self) -> None:
        """The utterance stays open across the stop: the closing chunk is
        still owed after it (RFC §12.10.3), and it is what closes the track's
        books, not the flush.
        """
        track, _ = _voice_track()
        await track.publish(AudioChunk(data=b"\x00\x00", sample_rate=48_000))

        track.discard_queued()

        assert track.abandon_utterance() is True, "the stop closed the utterance"

    async def test_a_left_session_stops_nothing_and_raises_nothing(self) -> None:
        """Unlike ``publish``, which raises for a departed session: the
        silence a stop asks for is already true of one.
        """
        session = _bridge_session()
        await session.leave()

        session.stop_playback()


class TestTheEventBridgeIsBounded:
    """The consumer awaits the framework's fanout — identity, hooks — so an
    authorised participant generating control events faster than that returns
    must cost bounded memory, never unbounded growth.
    """

    async def test_state_events_coalesce_to_one_entry_per_key(self) -> None:
        session = _bridge_session()
        participant = SimpleNamespace(identity="p-flappy")

        for _ in range(1000):
            session._view.on_connection_quality_changed(
                participant, SimpleNamespace(name="QUALITY_EXCELLENT")
            )
            session._view.on_connection_quality_changed(
                participant, SimpleNamespace(name="QUALITY_POOR")
            )

        assert session._bridge.queued == 1

    async def test_a_coalesced_state_delivers_its_latest_value_once(self) -> None:
        seen: list[tuple[Any, ...]] = []

        async def _record(*args: Any) -> None:
            seen.append(args)

        session = _bridge_session(connection_quality=_record)
        participant = SimpleNamespace(identity="p-1")
        session._view.on_connection_quality_changed(
            participant, SimpleNamespace(name="QUALITY_POOR")
        )
        session._view.on_connection_quality_changed(
            participant, SimpleNamespace(name="QUALITY_EXCELLENT")
        )

        session._bridge.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5.0
            while not seen:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            await session._bridge.stop()

        assert seen == [("room-1", "p-1", "excellent")]

    async def test_a_lifecycle_overflow_ends_the_session_rather_than_losing_facts(
        self,
    ) -> None:
        """An arrival or a track dropped silently is a roster that lies and a
        track never transcribed. Past the bound the session ends through the
        bot_session_ended contract — the channel re-joins, and the fresh
        catch-up announces the current truth instead of a holed history.
        """

        reported: list[str] = []

        async def _ended(bot: Any, reason: str) -> None:
            reported.append(reason)

        session = _bridge_session(ended=_ended)

        for _ in range(MAX_QUEUED_EVENTS + 50):
            session._put(_ignore, "room-1")

        assert session._bridge.queued <= MAX_QUEUED_EVENTS
        assert not session._departure.admitting, "the overflow did not end the session"
        assert session._departure._ender is not None
        await asyncio.wait_for(session._departure._ender, timeout=5.0)
        assert len(reported) == 1
        assert "overflow" in reported[0]


class TestASfuSideDisconnectIsReported:
    async def test_the_session_reports_its_own_loss_and_ends(self) -> None:
        class _FakeRoom:
            local_participant = None

            def on(self, *args: Any, **kwargs: Any) -> None:
                return None

            async def disconnect(self) -> None:
                raise AssertionError("a dropped session has nothing to disconnect")

        reported: list[tuple[Any, str]] = []

        async def _ended(bot: Any, reason: str) -> None:
            reported.append((bot, reason))

        bot = BotSession(id="lk-1", room_id="room-1", identity="roomkit")
        session = _session_for(SimpleNamespace(Room=_FakeRoom), session=bot, ended=_ended)

        session._departure.dropped("SIGNAL_CLOSE")
        assert session._departure._ender is not None
        await asyncio.wait_for(session._departure._ender, timeout=5.0)

        assert reported == [(bot, "SIGNAL_CLOSE")]
        # The end is terminal: a later leave() has nothing to do, and the
        # disconnect callback firing again reports nothing twice.
        await session.leave()
        session._departure.dropped("SIGNAL_CLOSE")
        assert len(reported) == 1

    async def test_the_backend_forgets_the_session_it_was_told_about(self) -> None:
        backend = _backend()
        bot = BotSession(id="lk-1", room_id="room-1", identity="roomkit")
        backend._sessions[bot.id] = object()  # type: ignore[assignment]

        await backend._bot_session_gone(bot, "connection lost")

        assert bot.id not in backend._sessions


class TestAnOverflowEndIsConfirmedBeforeReported:
    """Ending a session empties the registry and seats a replacement. For an
    overflow the old connection is still live, so the report MUST wait for
    the disconnect — reported early, the old bot sits in the meeting beside
    the replacement the supervisor brings in.
    """

    def _session_with_room(self) -> tuple[Any, Any, list[str]]:
        class _FakeRoom:
            local_participant = None

            def __init__(self) -> None:
                self.refuse = True
                self.disconnects = 0

            def on(self, *args: Any, **kwargs: Any) -> None:
                return None

            async def disconnect(self) -> None:
                self.disconnects += 1
                if self.refuse:
                    raise RuntimeError("signalling wedged")

        room = _FakeRoom()
        reported: list[str] = []

        async def _ended(bot: Any, reason: str) -> None:
            reported.append(reason)

        session = _session_for(SimpleNamespace(Room=lambda: room), ended=_ended)
        return session, room, reported

    async def test_a_refused_disconnect_keeps_the_session_unreported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(departure_module, "OVERFLOW_DISCONNECT_DELAYS_S", (0.0,))
        session, room, reported = self._session_with_room()

        for _ in range(MAX_QUEUED_EVENTS + 1):
            session._put(_ignore, "room-1")
        assert session._departure._ender is not None
        await asyncio.wait_for(session._departure._ender, timeout=5.0)

        # Every attempt failed: the end is NOT reported, the session is kept,
        # and nothing will seat a replacement beside a live connection.
        assert room.disconnects == 2
        assert reported == []
        assert session._departure._disconnected is False

        # leave() retries the disconnect — failure was not terminal — and a
        # requested leave reports nothing: its caller owns the books.
        room.refuse = False
        await session.leave()
        assert session._departure._disconnected is True
        assert reported == []

    async def test_a_confirmed_disconnect_reports_with_the_loss_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(departure_module, "OVERFLOW_DISCONNECT_DELAYS_S", (0.0,))
        session, room, reported = self._session_with_room()
        room.refuse = False

        for _ in range(MAX_QUEUED_EVENTS + 1):
            session._put(_ignore, "room-1")
        assert session._departure._ender is not None
        await asyncio.wait_for(session._departure._ender, timeout=5.0)

        assert len(reported) == 1
        assert "overflow" in reported[0]
        assert "discarded undelivered" in reported[0]
        assert room.disconnects == 1


class TestTheDisconnectIsSingleFlight:
    async def test_a_leave_during_the_unhealthy_disconnect_shares_it_and_silences_the_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One call on the wire, and the requested leave owns the books: the
        unhealthy end that loses the race reports nothing.
        """

        monkeypatch.setattr(departure_module, "OVERFLOW_DISCONNECT_DELAYS_S", (0.0,))
        gate = asyncio.Event()

        class _FakeRoom:
            local_participant = None

            def __init__(self) -> None:
                self.disconnects = 0

            def on(self, *args: Any, **kwargs: Any) -> None:
                return None

            async def disconnect(self) -> None:
                self.disconnects += 1
                await gate.wait()

        room = _FakeRoom()
        reported: list[str] = []

        async def _ended(bot: Any, reason: str) -> None:
            reported.append(reason)

        session = _session_for(SimpleNamespace(Room=lambda: room), ended=_ended)

        # Overflow: the unhealthy end starts its disconnect and suspends in it.
        for _ in range(MAX_QUEUED_EVENTS + 1):
            session._put(_ignore, "room-1")
        assert session._departure._ender is not None
        deadline = asyncio.get_running_loop().time() + 5.0
        while room.disconnects == 0:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0)

        # A requested leave arrives while the call is on the wire. It joins
        # the same call rather than issuing a second one.
        leaving = asyncio.create_task(session.leave())
        for _ in range(20):
            await asyncio.sleep(0)
        assert room.disconnects == 1, "two concurrent disconnects reached the SDK"

        gate.set()
        await asyncio.wait_for(leaving, timeout=5.0)
        await asyncio.wait_for(session._departure._ender, timeout=5.0)

        assert room.disconnects == 1
        assert session._departure._disconnected is True
        assert reported == [], "the unhealthy end reported over a requested leave"


class _WedgedRoom:
    """A room whose disconnect waits on a listener nothing will release.

    What livekit-rtc does after a failed publish or unpublish: ``disconnect()``
    awaits the room's listener task, which waits forever on an event nobody
    acknowledges.
    """

    local_participant = None

    def __init__(self) -> None:
        self.disconnects = 0
        self._task: asyncio.Task[None] | None = None

    def on(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def disconnect(self) -> None:
        self.disconnects += 1
        if self._task is None:
            self._task = asyncio.create_task(asyncio.Event().wait())
        await self._task


class _Evictions:
    """A stand-in for the backend's eviction: counts calls, fails the first ``failures``."""

    def __init__(self, failures: int = 0) -> None:
        self.calls = 0
        self._failures = failures

    async def __call__(self) -> None:
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError("server unreachable")


class TestTheDepartureIsBounded:
    """A departure the SDK cannot finish is settled by the server (RMK-350)."""

    @pytest.fixture(autouse=True)
    def _short_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(departure_module, "DISCONNECT_TIMEOUT_S", 0.05)

    async def test_a_disconnect_that_never_returns_is_settled_by_the_server(self) -> None:
        room = _WedgedRoom()
        evictions = _Evictions()
        departure = _departure_for(room, evict=evictions)

        await asyncio.wait_for(departure.leave(), timeout=2.0)

        assert evictions.calls == 1
        assert departure._disconnected is True
        await asyncio.sleep(0)
        assert room._task is not None and room._task.cancelled(), "the stuck listener stayed"
        assert room.disconnects == 1

    async def test_a_disconnect_that_returns_asks_nothing_of_the_server(self) -> None:
        evictions = _Evictions()
        departure = _departure_for(_FakeRoom(), evict=evictions)

        await asyncio.wait_for(departure.leave(), timeout=2.0)

        assert evictions.calls == 0
        assert departure._disconnected is True

    async def test_a_failed_eviction_is_a_failed_departure_and_the_next_one_retries(
        self,
    ) -> None:
        """Not terminal, like a refused disconnect: the session stays on the books
        and a later leave() asks the server again, over the same SDK call.
        """
        room = _WedgedRoom()
        evictions = _Evictions(failures=1)
        departure = _departure_for(room, evict=evictions)

        with pytest.raises(RuntimeError, match="server unreachable"):
            await asyncio.wait_for(departure.leave(), timeout=2.0)
        assert departure._disconnected is False

        await asyncio.wait_for(departure.leave(), timeout=2.0)

        assert evictions.calls == 2
        assert departure._disconnected is True
        assert room.disconnects == 1, "a second disconnect reached the SDK beside the first"

    async def test_a_caller_cancelled_during_the_eviction_finds_the_departure_done(
        self,
    ) -> None:
        """The departure outlives the caller a budget cancelled: the retry gets
        its answer instead of a second disconnect into an SDK that already let
        the room go, and the bridge is still taken down.
        """
        room = _WedgedRoom()
        evicted = asyncio.Event()

        async def _slow_eviction() -> None:
            await asyncio.sleep(0.2)
            evicted.set()

        departure = _departure_for(room, evict=_slow_eviction)
        departure._bridge.start()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(departure.leave(), timeout=0.1)
        await asyncio.wait_for(evicted.wait(), timeout=2.0)

        await asyncio.wait_for(departure.leave(), timeout=2.0)

        assert room.disconnects == 1
        assert departure._disconnected is True
        assert departure._bridge._consumer is None

    async def test_an_eviction_that_outlasts_its_bound_is_a_failed_departure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(departure_module, "EVICTION_TIMEOUT_S", 0.05)
        room = _WedgedRoom()
        attempts = 0

        async def _eviction() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.Event().wait()

        departure = _departure_for(room, evict=_eviction)

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(departure.leave(), timeout=2.0)
        assert departure._disconnected is False

        await asyncio.wait_for(departure.leave(), timeout=2.0)

        assert attempts == 2
        assert departure._disconnected is True

    async def test_an_unhealthy_end_on_a_stuck_sdk_is_settled_and_reported(self) -> None:
        """The overflow and refused-voice ends disconnect through the same
        bounded departure, so a stuck SDK delays their report by the bound
        rather than forever.
        """
        room = _WedgedRoom()
        evictions = _Evictions()
        reported: list[str] = []

        async def _ended(bot: Any, reason: str) -> None:
            reported.append(reason)

        departure = _departure_for(room, ended=_ended, evict=evictions)

        departure.end_unhealthy("voice publication failed: refused")
        assert departure._ender is not None
        await asyncio.wait_for(departure._ender, timeout=2.0)

        assert reported == ["voice publication failed: refused"]
        assert evictions.calls == 1

    async def test_closing_the_backend_settles_a_stuck_session(self) -> None:
        backend = _backend()
        service = _served(backend)
        room = _WedgedRoom()
        bot = BotSession(id="lk-1", room_id="room-1", identity="roomkit")
        session = _session_for(
            SimpleNamespace(Room=lambda: room),
            session=bot,
            evict=partial(backend._evict, "room-1", "roomkit"),
        )
        backend._sessions[bot.id] = session

        await asyncio.wait_for(backend.close(), timeout=2.0)

        assert [name for name, _ in service.calls] == ["remove_participant"]
        assert bot.id not in backend._sessions
        assert backend._api is None

    async def test_an_sfu_side_end_is_reported_though_the_disconnect_never_returns(
        self,
    ) -> None:
        """The SFU already dropped the bot, so nothing is evicted; what must not
        happen is the end going unreported behind a disconnect that never returns.
        """
        room = _WedgedRoom()
        evictions = _Evictions()
        reported: list[str] = []

        async def _ended(bot: Any, reason: str) -> None:
            reported.append(reason)

        departure = _departure_for(room, ended=_ended, evict=evictions)

        departure.dropped("SIGNAL_CLOSE")
        assert departure._ender is not None
        await asyncio.wait_for(departure._ender, timeout=2.0)

        assert reported == ["SIGNAL_CLOSE"]
        assert evictions.calls == 0
        assert room._task is not None and room._task.cancelled()


def test_the_departure_bounds_fit_inside_the_channels_detach_budget() -> None:
    """The backend settles a stuck departure before the channel gives up on it."""
    settled_within = departure_module.DISCONNECT_TIMEOUT_S + departure_module.EVICTION_TIMEOUT_S

    assert settled_within < _conference_activity.DRAIN_TIMEOUT_S


class _FakeRoom:
    """A room whose disconnect returns at once."""

    local_participant = None

    def on(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def disconnect(self) -> None:
        return None


class TestTheVoiceTrackAtTheEnd:
    async def test_closing_releases_the_source_without_unpublishing(self) -> None:
        """The disconnect takes the track down; an unpublish is one more SDK call
        that can fail and leave the room's listener stuck.
        """
        track, sources = _voice_track()
        await track.publish(AudioChunk(data=b"\x00\x00" * 160, sample_rate=48_000))

        await track.close()

        assert track._room.local_participant.unpublished == []
        assert [source.closed for source in sources] == [True]

    async def test_a_refused_publication_is_named_and_releases_its_source(self) -> None:
        track, sources = _voice_track(publish_error=RuntimeError("insufficient permissions"))

        with pytest.raises(VoicePublicationError, match="insufficient permissions"):
            await track.publish(AudioChunk(data=b"\x00\x00" * 160, sample_rate=48_000))

        assert [source.closed for source in sources] == [True]

    @pytest.mark.parametrize("fmt", ["opus", "pcm_f32le"])
    async def test_a_chunk_that_is_not_16_bit_pcm_is_refused_before_a_source(
        self, fmt: str
    ) -> None:
        """rtc.AudioFrame is 16-bit signed PCM: an encoded chunk, or another width,
        would publish noise (RFC section 12.10.3).
        """
        track, sources = _voice_track()

        with pytest.raises(ValueError, match=f"'{fmt}'"):
            await track.publish(AudioChunk(data=b"\x00\x00" * 160, format=fmt))

        assert sources == []


class TestARefusedVoiceEndsTheSession:
    async def test_the_session_ends_as_unhealthy_and_says_why(self) -> None:
        """The SDK stops delivering the room's events after a refused
        publication, so the session ends and the channel re-joins (RFC 12.10.3).
        """
        reported: list[str] = []

        async def _ended(bot: Any, reason: str) -> None:
            reported.append(reason)

        session = _session_for(SimpleNamespace(Room=_FakeRoom), ended=_ended)
        voice, _ = _voice_track(publish_error=RuntimeError("insufficient permissions"))
        session._voice = voice

        with pytest.raises(VoicePublicationError):
            await session.publish(AudioChunk(data=b"\x00\x00" * 160, sample_rate=48_000))

        assert session._departure._ender is not None
        await asyncio.wait_for(session._departure._ender, timeout=2.0)
        assert len(reported) == 1
        assert reported[0].startswith("voice publication failed")
        assert "insufficient permissions" in reported[0]
        with pytest.raises(RuntimeError, match="has left"):
            await session.publish(AudioChunk(data=b"\x00\x00" * 160, sample_rate=48_000))


class TestTheBackendEvictsThroughTheServer:
    async def test_a_participant_already_gone_counts_as_removed(self) -> None:
        backend = _backend()
        room = _served(backend)
        room.remove_error = api.ServerError("not_found", "participant does not exist", status=404)

        await backend._evict("room-1", "roomkit")

        [(name, request)] = room.calls
        assert name == "remove_participant"
        assert request.identity == "roomkit"

    async def test_any_other_server_error_is_a_failed_eviction(self) -> None:
        backend = _backend()
        room = _served(backend)
        room.remove_error = api.ServerError("unavailable", "down", status=503)

        with pytest.raises(api.ServerError):
            await backend._evict("room-1", "roomkit")


class TestTheEventBridge:
    """The bridge on its own: ordered, bounded, and honest about being full."""

    async def test_events_are_emitted_in_the_order_they_were_queued(self) -> None:
        bridge = EventBridge("room-1")
        seen: list[int] = []

        async def _record(value: int) -> None:
            seen.append(value)

        for value in range(5):
            assert bridge.put(_record, value)
        bridge.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5.0
            while len(seen) < 5:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0)
        finally:
            await bridge.stop()

        assert seen == [0, 1, 2, 3, 4]

    def test_a_full_bridge_refuses_a_fact_rather_than_dropping_one(self) -> None:
        bridge = EventBridge("room-1")
        for _ in range(MAX_QUEUED_EVENTS):
            assert bridge.put(_ignore)

        assert bridge.put(_ignore) is False
        assert bridge.put_state("speaker", _ignore) is False
        assert bridge.queued == MAX_QUEUED_EVENTS

    async def test_a_refused_state_leaves_its_key_free_for_the_next_one(self) -> None:
        """A state the full bridge refused is not kept: the next value for the
        same key, once there is room, is queued and delivered.
        """
        bridge = EventBridge("room-1")
        seen: list[str] = []

        async def _record(value: str) -> None:
            seen.append(value)

        for _ in range(MAX_QUEUED_EVENTS):
            bridge.put(_ignore)
        assert bridge.put_state("speaker", _record, "p-1") is False

        bridge.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5.0
            while bridge.queued:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0)
            assert bridge.put_state("speaker", _record, "p-2") is True
            while not seen:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0)
        finally:
            await bridge.stop()

        assert seen == ["p-2"]

    async def test_stopping_counts_what_was_never_delivered(self) -> None:
        bridge = EventBridge("room-1")
        bridge.put(_ignore)
        bridge.put_state("speaker", _ignore, "p-1")
        bridge.put_state("speaker", _ignore, "p-2")

        assert await bridge.stop() == 2
        assert bridge.queued == 0


def _pumps() -> TrackPumps:
    return TrackPumps(
        rtc=SimpleNamespace(),
        room_id="room-1",
        audio_sink=_ignore,
        video_sink=_ignore,
        config=SimpleNamespace(),
    )


class TestTrackPumps:
    async def test_no_pump_starts_once_the_pumps_are_closed(self) -> None:
        """A track the SDK reports subscribed while the session is leaving
        must not start a pump nothing would ever stop.
        """
        pumps = _pumps()
        record = ConferenceTrack(
            id="TR_1", room_id="room-1", participant_id="p", kind=TrackKind.AUDIO
        )

        await pumps.close()
        pumps.start(record, SimpleNamespace())

        assert pumps._pumps == {}


class _Publication:
    """A stand-in LiveKit track publication that records subscription requests."""

    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.kind = "KIND_AUDIO"
        self.source = "SOURCE_MICROPHONE"
        self.muted = False
        self.name = ""
        self.mime_type = ""
        self.subscribed: list[bool] = []

    def set_subscribed(self, value: bool) -> None:
        self.subscribed.append(value)


class _NeverEndingStream:
    """An ``rtc.AudioStream`` that delivers nothing until it is cancelled."""

    def __aiter__(self) -> _NeverEndingStream:
        return self

    async def __anext__(self) -> Any:
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        return None


_VIEW_RTC = SimpleNamespace(
    ParticipantKind=SimpleNamespace(Name=lambda value: value),
    TrackKind=SimpleNamespace(Name=lambda value: value),
    TrackSource=SimpleNamespace(Name=lambda value: value),
    AudioStream=SimpleNamespace(from_track=lambda **kwargs: _NeverEndingStream()),
)


def _participant(identity: str, *publications: _Publication) -> Any:
    return SimpleNamespace(
        identity=identity,
        sid=f"PA_{identity}",
        kind="PARTICIPANT_KIND_STANDARD",
        name=identity,
        metadata="",
        attributes={},
        joined_at=None,
        track_publications={publication.sid: publication for publication in publications},
    )


def _view(queued: list[tuple[Any, tuple[Any, ...]]]) -> tuple[RoomView, TrackPumps]:
    """A RoomView whose queued emissions land in ``queued``, and its pumps."""
    pumps = TrackPumps(
        rtc=_VIEW_RTC,
        room_id="room-1",
        audio_sink=_ignore,
        video_sink=_ignore,
        config=SimpleNamespace(audio_sample_rate=48_000, audio_channels=1),
    )
    emissions = _emissions()

    def _enqueue(emit: Any, *args: Any) -> None:
        queued.append((emit, args))

    def _enqueue_state(key: Any, emit: Any, *args: Any) -> None:
        queued.append((emit, args))

    view = RoomView(
        rtc=_VIEW_RTC,
        room_id="room-1",
        emissions=emissions,
        pumps=pumps,
        enqueue=_enqueue,
        enqueue_state=_enqueue_state,
    )
    return view, pumps


class TestRoomView:
    """The room's books on fakes: what the live suite alone used to cover."""

    def test_an_arrival_the_catch_up_already_saw_is_announced_once(self) -> None:
        queued: list[tuple[Any, tuple[Any, ...]]] = []
        view, _ = _view(queued)
        publication = _Publication("TR_1")
        alice = _participant("p-alice", publication)

        view.catch_up(SimpleNamespace(remote_participants={"p-alice": alice}))
        view.on_participant_connected(alice)
        view.on_track_published(publication, alice)

        announced = [args[1] for _, args in queued]
        assert [type(item).__name__ for item in announced] == [
            "ConferenceParticipant",
            "ConferenceTrack",
        ]

    def test_a_subscription_nobody_asked_for_is_undone(self) -> None:
        view, pumps = _view([])
        publication = _Publication("TR_1")
        alice = _participant("p-alice", publication)
        view.on_track_published(publication, alice)

        view.on_track_subscribed(SimpleNamespace(), publication, alice)

        assert publication.subscribed == [False]
        assert pumps._pumps == {}

    async def test_a_departed_publisher_takes_its_tracks_off_the_books(self) -> None:
        """Nothing left to answer a moderation call about, and no pump left
        running for a track that is gone.
        """
        view, pumps = _view([])
        publication = _Publication("TR_1")
        alice = _participant("p-alice", publication)
        view.on_track_published(publication, alice)
        await view.subscribe("TR_1")
        view.on_track_subscribed(SimpleNamespace(), publication, alice)
        assert publication.subscribed == [True]
        assert view.publisher_identity("TR_1") == "p-alice"
        assert "TR_1" in pumps._pumps

        view.on_participant_disconnected(alice)

        assert view.publisher_identity("TR_1") is None
        assert pumps._pumps == {}
        await pumps.close()
