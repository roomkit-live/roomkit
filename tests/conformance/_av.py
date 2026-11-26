"""An audio/video channel over a scripted combined transport, with a scripted
avatar (RFC §12.8.3, §12.8.8).

The transport is a voice transport whose calls may carry video: its ``video``
facet is roomkit's own CallVideoBackend, which the transport tells when a
call's video starts, goes live and ends. The channel is also a voice fixture,
so the voice steps (``voice.say``, ``voice.join``, ``voice.read``…) drive it.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit import AudioVideoChannel
from roomkit.video import (
    AudioVideoBackend,
    CallVideoBackend,
    MockAvatarProvider,
    VideoChunk,
    VideoFrame,
    VideoSession,
    VideoSessionState,
)
from roomkit.video.pipeline.encoder import VideoEncoderProvider
from roomkit.voice import VoiceSession
from tests.conformance._voice import VoiceFixture
from tests.conformance._voice_outbound import OutboundTransport
from tests.conformance._voice_synthesis import UnsupportedError, scripted_tts
from tests.conformance._voice_transport import PRIMARY

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness

AVATAR_FRAME = b"\x07\x07\x07"
"""The one frame the scripted avatar makes: ``BwcH`` in base64."""

_SETTLE_S = 2.0
"""How long ``av.read`` waits for the avatar's work: what it was fed, the
frames it made sent. Past it the read reports what reached them."""


class _Settling:
    """A record other threads add to, which a step awaits until it holds enough."""

    def __init__(self) -> None:
        self._changed = threading.Condition()

    def changed(self) -> None:
        with self._changed:
            self._changed.notify_all()

    async def until(self, ready: Callable[[], bool]) -> None:
        def wait() -> None:
            with self._changed:
                self._changed.wait_for(ready, timeout=_SETTLE_S)

        await asyncio.to_thread(wait)


class _CallVideo(CallVideoBackend):
    """The video of the scripted transport's calls: it records what it is sent."""

    def __init__(self, owner: AudioVideoBackend, settling: _Settling) -> None:
        super().__init__(owner)
        self.sent: list[tuple[str, bytes]] = []
        self._settling = settling

    def _record(self, session: VideoSession, data: bytes) -> None:
        self.sent.append((session.id, data))
        self._settling.changed()

    async def send_video(
        self, session: VideoSession, video: bytes | AsyncIterator[VideoChunk]
    ) -> None:
        if isinstance(video, bytes):
            self._record(session, video)
            return
        async for chunk in video:
            self._record(session, chunk.data)

    def send_video_sync(self, session: VideoSession, frame: VideoFrame) -> None:
        self._record(session, frame.data)


class _CombinedTransport(OutboundTransport, AudioVideoBackend):
    """A voice transport whose calls may carry video (§12.8.3). It reports a
    call's video live as it takes the call, or at ``av.video_live`` for one in
    ``video_later``, and the video ended before the call's client gone."""

    def __init__(self, spec: Mapping[str, Any], settling: _Settling) -> None:
        super().__init__(spec)
        self._video = _CallVideo(self, settling)
        self._with_video = set(spec.get("video", []))
        self._later = set(spec.get("video_later", []))

    @property
    def video(self) -> _CallVideo:
        return self._video

    async def connect(
        self,
        room_id: str,
        participant_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VoiceSession:
        call = await super().connect(room_id, participant_id, channel_id, metadata=metadata)
        if participant_id in self._with_video:
            self._video._add(
                VideoSession(
                    id=call.id,
                    room_id=room_id,
                    participant_id=participant_id,
                    channel_id=channel_id,
                    state=VideoSessionState.ACTIVE,
                )
            )
            if participant_id not in self._later:
                self._video._ready(call.id)
        return call

    async def simulate_client_disconnected(self, session: VoiceSession) -> None:
        self._video._ended(session.id)
        await super().simulate_client_disconnected(session)


class _Avatar(MockAvatarProvider):
    """A started avatar whose first ``feed_audio`` makes one 1×1 frame, every
    later one none, and which has no idle frame; it records what it is fed."""

    def __init__(self, settling: _Settling) -> None:
        super().__init__()
        self.fed = bytearray()
        self.made = 0
        self._settling = settling

    def feed_audio(self, pcm_data: bytes, sample_rate: int = 16000) -> list[VideoFrame]:
        self.fed.extend(pcm_data)
        frames = [] if self.made else [VideoFrame(data=AVATAR_FRAME, codec="raw_rgb24")]
        self.made += len(frames)
        self._settling.changed()
        return frames

    def get_idle_frame(self) -> VideoFrame | None:
        return None


class _Encoder(VideoEncoderProvider):
    """An encoder that encodes a frame as its own bytes."""

    @property
    def name(self) -> str:
        return "scripted"

    def encode(self, frame: VideoFrame) -> list[bytes]:
        return [frame.data]

    def flush(self) -> list[bytes]:
        return []


@dataclass
class AVFixture(VoiceFixture):
    transport: _CombinedTransport | None = None
    avatar: _Avatar | None = None
    settling: _Settling | None = None


def _fixture(harness: Harness, channel_id: str) -> AVFixture:
    fixture = harness.voice_fixtures[channel_id]
    assert isinstance(fixture, AVFixture), f"{channel_id} is no av.open channel"
    return fixture


async def _open(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    settling = _Settling()
    transport = _CombinedTransport(spec, settling)
    avatar = None
    if spec.get("avatar", False):
        avatar = _Avatar(settling)
        await avatar.start(b"", width=1, height=1)
    tts = None if spec.get("tts", {}) is None else scripted_tts(spec.get("tts", {}), harness.gates)
    channel = AudioVideoChannel(
        spec["channel_id"],
        backend=transport,
        tts=tts,
        avatar=avatar,
        avatar_encoder=_Encoder() if avatar is not None else None,
    )
    kit.register_channel(channel)
    await kit.attach_channel(spec["room_id"], channel.channel_id)
    fixture = AVFixture(
        channel,
        transport,  # ty: ignore[invalid-argument-type]
        tts,
        {},
        spec["room_id"],
        [],
        {},
        transport=transport,
        avatar=avatar,
        settling=settling,
    )
    fixture.transports[PRIMARY] = transport
    harness.voice_fixtures[channel.channel_id] = fixture
    for participant in spec["participants"]:
        session = await kit.join(spec["room_id"], channel.channel_id, participant_id=participant)
        assert isinstance(session, VoiceSession)
        fixture.sessions[participant] = session
        fixture.joined[participant] = [session]


def _video_of(fixture: AVFixture, participant: str) -> VideoSession:
    assert fixture.transport is not None
    video = fixture.transport.video.get_session(fixture.sessions[participant].id)
    assert video is not None, f"{participant}'s call carries no video"
    return video


async def _video_live(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = _fixture(harness, spec["channel_id"])
    assert fixture.transport is not None
    fixture.transport.video._ready(_video_of(fixture, spec["participant_id"]).id)


async def _video_connect(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = _fixture(harness, spec["channel_id"])
    assert fixture.transport is not None
    try:
        await fixture.transport.video.connect(
            fixture.room_id, spec["participant_id"], fixture.channel.channel_id
        )
    except NotImplementedError as error:
        raise UnsupportedError(str(error)) from error


async def _video_disconnect(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    fixture = _fixture(harness, spec["channel_id"])
    assert fixture.transport is not None
    await fixture.transport.video.disconnect(_video_of(fixture, spec["participant_id"]))


async def _read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Once the avatar was fed every byte the calls were sent, and the frames it
    made were sent (or ``_SETTLE_S`` passed): what it was fed, what was sent."""
    fixture = _fixture(harness, spec["channel_id"])
    transport, avatar, settling = fixture.transport, fixture.avatar, fixture.settling
    assert transport is not None and settling is not None
    if avatar is not None:
        spoken = sum(len(audio) for _, audio in transport.sent_audio)
        await settling.until(lambda: len(avatar.fed) >= spoken)
        await settling.until(lambda: len(transport.video.sent) >= avatar.made)
    owners = {session.id: name for name, session in fixture.sessions.items()}
    sent: dict[str, list[str]] = {name: [] for name in fixture.sessions}
    for session_id, data in transport.video.sent:
        name = owners.get(session_id, session_id)
        sent.setdefault(name, []).append(base64.b64encode(data).decode("ascii"))
    fed = bytes(avatar.fed) if avatar is not None else b""
    return {"avatar_audio": base64.b64encode(fed).decode("ascii"), "sent": sent}


AV_CALLS = {
    "av.open": _open,
    "av.video_live": _video_live,
    "av.video_connect": _video_connect,
    "av.video_disconnect": _video_disconnect,
    "av.read": _read,
}
AV_ARGUMENTS = {
    "av.open": {"room_id", "channel_id", "participants", "video", "video_later", "avatar", "tts"},
    "av.video_live": {"channel_id", "participant_id"},
    "av.video_connect": {"channel_id", "participant_id"},
    "av.video_disconnect": {"channel_id", "participant_id"},
    "av.read": {"channel_id"},
}
