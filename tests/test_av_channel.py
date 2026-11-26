"""Tests for AudioVideoChannel — combined audio+video channel."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import AudioVideoChannel, ChannelType, HookExecution, HookTrigger, RoomKit, _clock
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelMediaType
from roomkit.recorder.base import (
    ChannelRecordingConfig,
    MediaRecordingConfig,
    RoomRecorderBinding,
)
from roomkit.recorder.mock import MockMediaRecorder
from roomkit.video.avatar.mock import MockAvatarProvider
from roomkit.video.backends.av import AudioVideoBackend, CallVideoBackend
from roomkit.video.base import VideoSession, VideoSessionState
from roomkit.video.bridge import VideoBridgeConfig
from roomkit.video.pipeline.encoder.base import VideoEncoderProvider
from roomkit.video.video_frame import VideoFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.bridge import AudioBridgeConfig
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider
from tests.channel_tasks import settled


def _make_av_backend() -> MagicMock:
    """Build a mock that quacks like a combined A/V backend: a voice backend
    whose calls' video is its ``video`` facet."""
    from roomkit.video.backends.av import AudioVideoBackend

    backend = MagicMock()
    backend.__class__ = type("MockAVBackend", (AudioVideoBackend,), {})
    backend.name = "mock-av"
    backend.auto_connect = False
    backend.capabilities = MagicMock(return_value=0)
    backend.feeds_aec_reference = False
    backend.supports_playback_callback = False
    backend.close = AsyncMock()

    # Track registered callbacks
    backend._video_received_cb = None
    backend._video_taps = []

    def on_video_received(cb):
        backend._video_received_cb = cb

    def add_video_tap(cb):
        backend._video_taps.append(cb)

    backend.video.on_video_received = on_video_received
    backend.video.add_video_tap = add_video_tap

    # Default: no call, no video session
    backend.get_session.return_value = None
    backend.video.get_session.return_value = None
    return backend


def _make_video_session(session_id: str = "session-1") -> VideoSession:
    return VideoSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice",
        state=VideoSessionState.ACTIVE,
    )


def _make_voice_session(session_id: str = "session-1") -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice",
        state=VoiceSessionState.ACTIVE,
    )


def _make_video_frame(timestamp_ms: float | None = 0.0) -> VideoFrame:
    return VideoFrame(
        data=b"\x00" * 100,
        codec="h264",
        timestamp_ms=timestamp_ms,
        sequence=1,
        keyframe=True,
    )


class _SetClock:
    """A media clock that reads what the test sets (roomkit's ``_clock`` seam)."""

    now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        raise AssertionError("nothing here waits on the clock")

    async def wait_for(self, awaitable: Any, timeout: float | None) -> Any:
        raise AssertionError("nothing here waits on the clock")


class TestAudioVideoChannel:
    def test_register_av_channel(self) -> None:
        """AudioVideoChannel registers with correct channel_type."""
        kit = RoomKit()
        backend = _make_av_backend()
        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )
        kit.register_channel(av)

        assert av.channel_type == ChannelType.AUDIO_VIDEO

    def test_capabilities_audio_and_video(self) -> None:
        """capabilities() reports both audio and video support."""
        backend = _make_av_backend()
        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )
        caps = av.capabilities()
        # RFC §6.3: it takes text in and speaks it, as a voice channel does.
        assert set(caps.media_types) == {
            ChannelMediaType.AUDIO,
            ChannelMediaType.VIDEO,
            ChannelMediaType.TEXT,
        }
        assert caps.supports_audio is True
        assert caps.supports_video is True

    def test_video_frames_delivered_to_taps(self) -> None:
        """Video frames from the backend are delivered to registered taps."""
        backend = _make_av_backend()
        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )

        # Bind session so frames are routed
        session = _make_voice_session()
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="av",
            channel_type=ChannelType.AUDIO_VIDEO,
        )
        av.bind_session(session, "room-1", binding)

        # Add a tap
        received_frames: list[VideoFrame] = []
        av.add_video_media_tap(lambda sess, frame: received_frames.append(frame))

        # Simulate a video frame from the backend
        video_session = _make_video_session()
        frame = _make_video_frame()
        # Call the callback that was registered on the backend
        assert backend._video_received_cb is not None
        backend._video_received_cb(video_session, frame)

        assert len(received_frames) == 1
        assert received_frames[0] is frame

    @pytest.mark.parametrize("timed_by", ["frame", "clock"])
    def test_vision_analysis_throttled(self, timed_by: str) -> None:
        """Vision analysis is throttled by vision_interval_ms, on the frames'
        timestamps, or on the media clock for frames that carry none."""
        backend = _make_av_backend()
        mock_vision = MagicMock()
        mock_vision.name = "mock-vision"

        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            vision=mock_vision,
            vision_interval_ms=1000,
        )

        session = _make_voice_session()
        binding = ChannelBinding(
            room_id="room-1",
            channel_id="av",
            channel_type=ChannelType.AUDIO_VIDEO,
        )
        av.bind_session(session, "room-1", binding)

        video_session = _make_video_session()
        cb = backend._video_received_cb
        assert cb is not None

        clock = _SetClock()
        with _clock.installed(clock):
            for at_ms in (0, 500, 1100):
                # 0: analysed; 500: within the interval, not; 1100: analysed.
                clock.now = at_ms / 1000
                stamp = at_ms if timed_by == "frame" else None
                cb(video_session, _make_video_frame(timestamp_ms=stamp))

        # Check last_vision_ts was updated for frames 1 and 3
        assert av._last_vision_ts[video_session.id] == pytest.approx(1100)

    def test_info_includes_vision(self) -> None:
        """info property includes vision fields."""
        backend = _make_av_backend()
        mock_vision = MagicMock()
        mock_vision.name = "test-vision"

        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            vision=mock_vision,
        )

        info = av.info
        assert info["vision"] == "test-vision"
        assert "vision_interval_ms" in info

    async def test_close_cleans_video_state(self) -> None:
        """close() clears video-specific state."""
        backend = _make_av_backend()
        mock_vision = MagicMock()
        mock_vision.name = "mock-vision"
        mock_vision.close = AsyncMock()

        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            vision=mock_vision,
        )
        av.add_video_media_tap(lambda s, f: None)
        av._last_vision_results["s1"] = MagicMock()

        await av.close()

        assert len(av._video_media_taps) == 0
        assert len(av._last_vision_results) == 0
        mock_vision.close.assert_called_once()


class TestAVRecordingWiring:
    async def test_bind_wires_audio_and_video(self) -> None:
        """bind_voice_session on AudioVideoChannel wires both audio and video tracks."""
        kit = RoomKit()
        recorder = MockMediaRecorder()
        binding = RoomRecorderBinding(
            recorder=recorder,
            config=MediaRecordingConfig(),
        )

        backend = _make_av_backend()
        video_session = _make_video_session()
        backend.video.get_session.return_value = video_session

        av = AudioVideoChannel(
            "voice",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
            recording=ChannelRecordingConfig(audio=True, video=True),
        )
        kit.register_channel(av)

        await kit.create_room(room_id="room-1", recorders=[binding])
        await kit.attach_channel("room-1", "voice")

        session = _make_voice_session()
        await kit.bind_voice_session(session, "room-1", "voice")

        track_ids = [t.id for t in recorder.tracks]
        assert "audio:session-1" in track_ids
        assert "video:session-1" in track_ids

        # Verify video tap was added on the channel (not the backend)
        assert len(av._video_media_taps) > 0
        # Backend's add_video_tap should NOT be called
        assert len(backend._video_taps) == 0


class _CallVideo(CallVideoBackend):
    """The video of the calls of :class:`_CallsWithVideo`: it records each
    frame it is sent with the thread that sent it."""

    def __init__(self, owner: AudioVideoBackend) -> None:
        super().__init__(owner)
        self.sent: list[tuple[str, VideoFrame]] = []
        self.threads: list[int] = []
        self.first_sent = threading.Event()

    def _record(self, session_id: str, frame: VideoFrame) -> None:
        self.sent.append((session_id, frame))
        self.threads.append(threading.get_ident())
        self.first_sent.set()

    async def send_video(self, session: VideoSession, video: Any) -> None:
        raise AssertionError("the avatar's frames go out one at a time")

    def send_video_sync(self, session: VideoSession, frame: VideoFrame) -> None:
        self._record(session.id, frame)


class _AsyncOnlyCallVideo(_CallVideo):
    """A facet that keeps VideoBackend's own ``send_video_sync``, which hands
    the frame's bytes to ``send_video`` on the running loop."""

    async def send_video(self, session: VideoSession, video: Any) -> None:
        self._record(session.id, VideoFrame(data=video))

    send_video_sync = CallVideoBackend.send_video_sync


class _CallsWithVideo(MockVoiceBackend, AudioVideoBackend):
    """A combined backend written on the contract: each call it connects
    carries video, live at once, served by its ``video`` facet, but those of
    *without_video*."""

    def __init__(
        self, video: type[_CallVideo] = _CallVideo, without_video: frozenset[str] = frozenset()
    ) -> None:
        super().__init__()
        self._video = video(self)
        self._without_video = without_video

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
        if participant_id not in self._without_video:
            self._video._started(
                VideoSession(
                    id=call.id,
                    room_id=room_id,
                    participant_id=participant_id,
                    channel_id=channel_id,
                    state=VideoSessionState.ACTIVE,
                )
            )
        return call


class _NalEncoder(VideoEncoderProvider):
    """An encoder that encodes every frame as the same NAL units."""

    def __init__(self, nals: list[bytes]) -> None:
        self.nals = nals

    @property
    def name(self) -> str:
        return "nal"

    def encode(self, frame: VideoFrame) -> list[bytes]:
        return list(self.nals)

    def flush(self) -> list[bytes]:
        return []


class TestAvatarVideoGoesThroughTheCallsVideo:
    """RFC §12.8.8: the avatar's encoded frames are sent through the video
    backend, the call's ``video`` facet, whatever transport serves it."""

    @staticmethod
    async def _room(
        backend: _CallsWithVideo, nals: list[bytes]
    ) -> tuple[RoomKit, AudioVideoChannel]:
        avatar = MockAvatarProvider(idle_color=None)
        await avatar.start(b"", width=1, height=1)
        av = AudioVideoChannel(
            "av",
            backend=backend,
            tts=MockTTSProvider(),
            avatar=avatar,
            avatar_encoder=_NalEncoder(nals),
        )
        kit = RoomKit()
        kit.register_channel(av)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "av")
        return kit, av

    @pytest.mark.parametrize(
        ("nals", "sent", "keyframe"),
        [
            ([b"\x41\x9a"], b"\x41\x9a", False),
            (
                [b"\x67\x42", b"\x65\x88"],
                b"\x00\x00\x00\x01\x67\x42\x00\x00\x00\x01\x65\x88",
                True,
            ),
        ],
        ids=["one NAL unit, as it is", "several, in Annex B form"],
    )
    async def test_an_avatar_frame_is_sent_to_the_call_as_one_encoded_frame(
        self, nals: list[bytes], sent: bytes, keyframe: bool
    ) -> None:
        backend = _CallsWithVideo()
        kit, av = await self._room(backend, nals)
        session = await kit.join("r1", "av", participant_id="alice")

        await av.say(session, "Hello there")
        assert await asyncio.to_thread(backend.video.first_sent.wait, 2.0)
        await kit.close()

        session_id, frame = backend.video.sent[0]
        assert session_id == session.id
        assert (frame.data, frame.keyframe) == (sent, keyframe)

    @pytest.mark.parametrize(
        "facet", [_CallVideo, _AsyncOnlyCallVideo], ids=["its own sync send", "the default one"]
    )
    async def test_an_avatar_frame_is_sent_on_the_event_loop(
        self, facet: type[_CallVideo]
    ) -> None:
        backend = _CallsWithVideo(facet)
        kit, av = await self._room(backend, [b"\x41\x9a"])
        session = await kit.join("r1", "av", participant_id="alice")

        await av.say(session, "Hello there")
        assert await asyncio.to_thread(backend.video.first_sent.wait, 2.0)
        await kit.close()

        assert backend.video.sent[0][1].data == b"\x41\x9a"
        assert backend.video.threads[0] == threading.get_ident()

    async def test_each_call_s_avatar_frames_run_on_a_clock_of_its_own(self) -> None:
        backend = _CallsWithVideo()
        kit, av = await self._room(backend, [b"\x41\x9a"])
        alice = await kit.join("r1", "av", participant_id="alice")
        bob = await kit.join("r1", "av", participant_id="bob")

        await av.say(alice, "Hello there")
        assert await asyncio.to_thread(backend.video.first_sent.wait, 2.0)
        backend.video.first_sent.clear()
        await av.say(bob, "Hello there")
        assert await asyncio.to_thread(backend.video.first_sent.wait, 2.0)
        await kit.close()

        first = {sid: frame for sid, frame in reversed(backend.video.sent)}
        step = 1000.0 / MockAvatarProvider().fps
        assert first[alice.id].timestamp_ms == first[bob.id].timestamp_ms == step


class TestAVVideoBridgeCapacity:
    """RFC §12.8.9: a call whose video the full video bridge has no room for is
    refused before it is bound, as a voice join past the audio bridge's."""

    async def test_a_call_past_the_video_bridge_s_max_participants_is_not_bound(self) -> None:
        backend = _CallsWithVideo()
        av = AudioVideoChannel(
            "av",
            backend=backend,
            tts=MockTTSProvider(),
            video_bridge=VideoBridgeConfig(max_participants=1),
        )
        kit = RoomKit()
        kit.register_channel(av)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "av")
        started: list[str] = []

        @kit.hook(HookTrigger.ON_SESSION_STARTED, execution=HookExecution.ASYNC)
        async def on_started(event: object, ctx: object) -> None:
            started.append(event.participant_id)  # type: ignore[attr-defined]

        await kit.join("r1", "av", participant_id="alice")
        with pytest.raises(RuntimeError, match="max bridge participants"):
            await kit.join("r1", "av", participant_id="bob")
        bob = next(s for s in backend.list_sessions("r1") if s.participant_id == "bob")
        await backend.simulate_session_ready(bob)
        await settled(av)
        await kit.close()

        assert started == ["alice"]
        assert bob.state == VoiceSessionState.ENDED

    async def test_a_call_the_audio_bridge_refuses_leaves_its_video_bridge_seat_free(
        self,
    ) -> None:
        backend = _CallsWithVideo(without_video=frozenset({"bob"}))
        av = AudioVideoChannel(
            "av",
            backend=backend,
            tts=MockTTSProvider(),
            bridge=AudioBridgeConfig(max_participants=2),
            video_bridge=VideoBridgeConfig(max_participants=2),
        )
        kit = RoomKit()
        kit.register_channel(av)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "av")
        await kit.join("r1", "av", participant_id="alice")
        bob = await kit.join("r1", "av", participant_id="bob")
        with pytest.raises(RuntimeError, match="max bridge participants"):
            await kit.join("r1", "av", participant_id="carol")
        await kit.leave(bob)

        await kit.join("r1", "av", participant_id="dave")
        await kit.close()
