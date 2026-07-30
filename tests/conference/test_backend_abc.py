"""ConferenceBackend interface contract (RFC §12.10.3).

Covers what the abstract base class itself guarantees: that a backend cannot
half-implement the interface, and that the callback fanout the base provides
behaves. The behaviour of a real backend — selective subscription, frame
attribution, capability gating — is exercised through the mock backend suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import (
    BotSession,
    ConferenceAccess,
    ConferenceBackend,
    ConferenceCapability,
    ConferenceGrants,
    ConferenceParticipant,
    ConferenceTrack,
    TrackKind,
)
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import AudioChunk


class _Stub(ConferenceBackend):
    """Minimal conforming backend — enough to exercise the base class."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def capabilities(self) -> ConferenceCapability:
        return ConferenceCapability.NONE

    async def ensure_room(
        self, room_id: str, metadata: dict[str, Any] | None = None, e2ee: bool = False
    ) -> None:
        return None

    async def close_room(self, room_id: str) -> None:
        return None

    async def mint_access(
        self, room_id: str, participant_id: str, grants: ConferenceGrants
    ) -> ConferenceAccess:
        return ConferenceAccess(url="wss://stub", token="t")

    async def list_participants(self, room_id: str) -> list[ConferenceParticipant]:
        return []

    async def remove_participant(self, room_id: str, participant_id: str) -> None:
        return None

    async def mute_track(self, room_id: str, track_id: str) -> None:
        return None

    async def unmute_track(self, room_id: str, track_id: str) -> None:
        return None

    async def join_as_bot(
        self, room_id: str, identity: str, grants: ConferenceGrants
    ) -> BotSession:
        return BotSession(id="bs-1", room_id=room_id, identity=identity)

    async def leave(self, bot: BotSession) -> None:
        return None

    async def subscribe_track(self, bot: BotSession, track_id: str) -> None:
        return None

    async def unsubscribe_track(self, bot: BotSession, track_id: str) -> None:
        return None

    async def publish_audio(self, bot: BotSession, chunk: AudioChunk) -> None:
        return None

    async def publish_video(self, bot: BotSession, frame: VideoFrame) -> None:
        return None

    async def close(self) -> None:
        return None


def _track(track_id: str = "tr-1", kind: TrackKind = TrackKind.AUDIO) -> ConferenceTrack:
    return ConferenceTrack(id=track_id, room_id="room-1", participant_id="p-alice", kind=kind)


class TestAbstractSurface:
    def test_incomplete_backend_cannot_be_instantiated(self) -> None:
        """A backend that skips part of the interface must fail loudly at
        construction, not at the first call in production.
        """

        class Partial(ConferenceBackend):
            @property
            def name(self) -> str:
                return "partial"

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_complete_backend_instantiates(self) -> None:
        assert _Stub().name == "stub"

    def test_control_plane_and_bot_surface_are_abstract(self) -> None:
        """The whole interface is required — a backend cannot silently omit
        subscription or unmuting and still claim conformance.
        """
        required = ConferenceBackend.__abstractmethods__

        for member in (
            "name",
            "capabilities",
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
            "publish_video",
            "close",
        ):
            assert member in required

    def test_callback_registration_is_provided_by_the_base(self) -> None:
        """Registration is not abstract: backends emit, they do not each
        reimplement observer bookkeeping.
        """
        required = ConferenceBackend.__abstractmethods__

        for member in (
            "on_participant_joined",
            "on_track_published",
            "on_track_audio",
            "on_active_speaker_changed",
        ):
            assert member not in required


class TestCallbackFanout:
    async def test_sync_and_async_callbacks_both_run(self) -> None:
        backend = _Stub()
        seen: list[str] = []

        def sync_cb(track: ConferenceTrack, frame: AudioFrame) -> None:
            seen.append(f"sync:{track.id}")

        async def async_cb(track: ConferenceTrack, frame: AudioFrame) -> None:
            seen.append(f"async:{track.id}")

        backend.on_track_audio(sync_cb)
        backend.on_track_audio(async_cb)

        await backend._emit_track_audio(_track(), AudioFrame(data=b""))

        assert seen == ["sync:tr-1", "async:tr-1"]

    async def test_a_failing_subscriber_does_not_stop_the_others(self) -> None:
        """Best-effort fanout: a lane raising must not tear down the media
        session that feeds every other lane.
        """
        backend = _Stub()
        seen: list[str] = []

        def boom(track: ConferenceTrack, frame: AudioFrame) -> None:
            raise RuntimeError("lane exploded")

        def survivor(track: ConferenceTrack, frame: AudioFrame) -> None:
            seen.append("survivor")

        backend.on_track_audio(boom)
        backend.on_track_audio(survivor)

        await backend._emit_track_audio(_track(), AudioFrame(data=b""))

        assert seen == ["survivor"]

    async def test_failing_subscriber_is_logged_with_the_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _Stub()

        def boom(room_id: str, track: ConferenceTrack) -> None:
            raise RuntimeError("nope")

        backend.on_track_published(boom)

        with caplog.at_level("ERROR", logger="roomkit.conference.backend"):
            await backend._emit_track_published("room-1", _track())

        assert "track_published" in caplog.text

    async def test_participant_events_carry_room_and_participant(self) -> None:
        backend = _Stub()
        seen: list[tuple[str, str]] = []

        def cb(room_id: str, participant: ConferenceParticipant) -> None:
            seen.append((room_id, participant.participant_id))

        backend.on_participant_joined(cb)

        await backend._emit_participant_joined(
            "room-1", ConferenceParticipant(participant_id="p-alice")
        )

        assert seen == [("room-1", "p-alice")]

    async def test_audio_frames_are_attributable_to_participant_and_room(self) -> None:
        """The frame callback receives only a track, so the track is what makes
        the frame routable and attributable.
        """
        backend = _Stub()
        seen: list[tuple[str, str]] = []

        def cb(track: ConferenceTrack, frame: AudioFrame) -> None:
            seen.append((track.room_id, track.participant_id))

        backend.on_track_audio(cb)

        await backend._emit_track_audio(_track(), AudioFrame(data=b"\x00\x00"))

        assert seen == [("room-1", "p-alice")]

    async def test_unregistered_event_is_a_no_op(self) -> None:
        await _Stub()._emit_connection_quality("room-1", "p-alice", "poor")

    async def test_each_event_reaches_only_its_own_subscribers(self) -> None:
        backend = _Stub()
        audio: list[str] = []
        video: list[str] = []

        backend.on_track_audio(lambda track, frame: audio.append(track.id))
        backend.on_track_video(lambda track, frame: video.append(track.id))

        await backend._emit_track_audio(_track("tr-audio"), AudioFrame(data=b""))

        assert audio == ["tr-audio"]
        assert video == []
