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

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from roomkit.conference.livekit import LiveKitConferenceBackend, LiveKitConfig
from roomkit.conference.models import BotSession, ConferenceGrants, TrackKind
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
        return api.RemoveParticipantResponse()

    async def mute_published_track(self, request: Any) -> Any:
        self.calls.append(("mute_published_track", request))
        return api.MuteRoomTrackResponse()


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

    def test_the_token_stays_out_of_the_access_repr(self) -> None:
        access = _backend()._access("room-1", "p-alice", ConferenceGrants(), publish_data=True)

        assert access.token not in repr(access)


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
        publishes it.
        """
        backend = _backend()
        room = self._room_with_track(backend)

        await backend.mute_track("room-1", "TR_1")

        request = next(r for name, r in room.calls if name == "mute_published_track")
        assert request.identity == "p-alice"
        assert request.track_sid == "TR_1"
        assert request.muted is True

    async def test_muting_is_available_without_any_capability(self) -> None:
        backend = _backend(remote_unmute=False)
        self._room_with_track(backend)

        await backend.mute_track("room-1", "TR_1")

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
