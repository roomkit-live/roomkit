"""Tests for video subsystem — VideoFrame, VideoBackend, VisionProvider."""

from __future__ import annotations

import pytest

from roomkit.video import (
    FaceDetection,
    MockVideoBackend,
    MockVisionProvider,
    VideoCapability,
    VideoChunk,
    VideoFrame,
    VideoSession,
    VideoSessionState,
)
from roomkit.video.backends.base import VideoBackend
from roomkit.video.vision.base import VisionProvider, VisionResult

# ---------------------------------------------------------------------------
# VideoFrame
# ---------------------------------------------------------------------------


class TestVideoFrame:
    def test_create_encoded_frame(self) -> None:
        frame = VideoFrame(data=b"\x00" * 100, codec="h264", width=1920, height=1080)
        assert frame.is_encoded
        assert not frame.is_raw
        assert frame.width == 1920
        assert frame.height == 1080
        assert frame.codec == "h264"
        assert frame.keyframe is False
        assert frame.sequence == 0

    def test_create_raw_frame(self) -> None:
        # RGB24: 3 bytes per pixel
        data = b"\x00" * (640 * 480 * 3)
        frame = VideoFrame(data=data, codec="raw_rgb24", width=640, height=480)
        assert frame.is_raw
        assert not frame.is_encoded

    def test_all_encoded_codecs(self) -> None:
        for codec in ("h264", "vp8", "vp9", "av1"):
            frame = VideoFrame(data=b"\x00", codec=codec)
            assert frame.is_encoded

    def test_all_raw_codecs(self) -> None:
        for codec in ("raw_rgb24", "raw_bgr24", "raw_yuv420p", "raw_nv12"):
            frame = VideoFrame(data=b"\x00", codec=codec)
            assert frame.is_raw

    def test_invalid_codec_raises(self) -> None:
        with pytest.raises(ValueError, match="codec must be one of"):
            VideoFrame(data=b"\x00", codec="invalid")

    def test_data_must_be_bytes(self) -> None:
        with pytest.raises(ValueError, match="must be bytes"):
            VideoFrame(data="not bytes")  # type: ignore[arg-type]

    def test_dimensions_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            VideoFrame(data=b"\x00", width=0, height=480)
        with pytest.raises(ValueError, match="must be positive"):
            VideoFrame(data=b"\x00", width=640, height=-1)

    def test_negative_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence must be >= 0"):
            VideoFrame(data=b"\x00", sequence=-1)

    def test_keyframe_flag(self) -> None:
        frame = VideoFrame(data=b"\x00", keyframe=True)
        assert frame.keyframe is True

    def test_metadata(self) -> None:
        frame = VideoFrame(data=b"\x00", metadata={"source": "camera"})
        assert frame.metadata["source"] == "camera"

    def test_timestamp(self) -> None:
        frame = VideoFrame(data=b"\x00", timestamp_ms=1500.0)
        assert frame.timestamp_ms == 1500.0

    def test_sequence(self) -> None:
        frame = VideoFrame(data=b"\x00", sequence=42)
        assert frame.sequence == 42


# ---------------------------------------------------------------------------
# VideoChunk
# ---------------------------------------------------------------------------


class TestVideoChunk:
    def test_create_chunk(self) -> None:
        chunk = VideoChunk(data=b"\x00" * 50, codec="vp8", width=320, height=240)
        assert chunk.codec == "vp8"
        assert chunk.width == 320
        assert chunk.is_final is False

    def test_final_chunk(self) -> None:
        chunk = VideoChunk(data=b"", is_final=True)
        assert chunk.is_final is True

    def test_invalid_codec_raises(self) -> None:
        with pytest.raises(ValueError, match="codec must be one of"):
            VideoChunk(data=b"\x00", codec="invalid")

    def test_all_valid_codecs(self) -> None:
        for codec in ("h264", "vp8", "vp9", "av1"):
            chunk = VideoChunk(data=b"\x00", codec=codec)
            assert chunk.codec == codec


# ---------------------------------------------------------------------------
# VideoSession
# ---------------------------------------------------------------------------


class TestVideoSession:
    def test_create_session(self) -> None:
        session = VideoSession(
            id="sess-1",
            room_id="room-1",
            participant_id="user-1",
            channel_id="video-1",
        )
        assert session.state == VideoSessionState.CONNECTING
        assert session.provider_session_id is None
        assert session.created_at is not None

    def test_session_states(self) -> None:
        for state in VideoSessionState:
            session = VideoSession(
                id="s", room_id="r", participant_id="p", channel_id="c", state=state
            )
            assert session.state == state


# ---------------------------------------------------------------------------
# VideoCapability
# ---------------------------------------------------------------------------


class TestVideoCapability:
    def test_none(self) -> None:
        assert VideoCapability.NONE.value == 0
        assert not VideoCapability.NONE

    def test_combine_flags(self) -> None:
        caps = VideoCapability.SIMULCAST | VideoCapability.SCREEN_SHARE
        assert VideoCapability.SIMULCAST in caps
        assert VideoCapability.SCREEN_SHARE in caps
        assert VideoCapability.SVC not in caps

    def test_all_flags(self) -> None:
        all_caps = (
            VideoCapability.SIMULCAST
            | VideoCapability.SVC
            | VideoCapability.SCREEN_SHARE
            | VideoCapability.RECORDING
            | VideoCapability.BANDWIDTH_ESTIMATION
        )
        assert VideoCapability.RECORDING in all_caps


# ---------------------------------------------------------------------------
# MockVideoBackend
# ---------------------------------------------------------------------------


class TestMockVideoBackend:
    async def test_connect_creates_session(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        assert session.state == VideoSessionState.ACTIVE
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert backend.calls[-1].method == "connect"

    async def test_disconnect(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.disconnect(session)
        stored = backend.get_session(session.id)
        assert stored is not None
        assert stored.state == VideoSessionState.ENDED
        assert backend.calls[-1].method == "disconnect"

    async def test_send_video_bytes(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")
        await backend.send_video(session, b"\x00" * 100)
        assert len(backend.sent_video) == 1
        assert backend.sent_video[0] == (session.id, b"\x00" * 100)

    async def test_send_video_stream(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1")

        async def chunks():
            yield VideoChunk(data=b"\x01" * 50)
            yield VideoChunk(data=b"\x02" * 50)

        await backend.send_video(session, chunks())
        assert len(backend.sent_video) == 1
        assert backend.sent_video[0][1] == b"\x01" * 50 + b"\x02" * 50

    async def test_list_sessions(self) -> None:
        backend = MockVideoBackend()
        await backend.connect("room-1", "user-1", "video-1")
        await backend.connect("room-1", "user-2", "video-1")
        await backend.connect("room-2", "user-3", "video-1")

        room1_sessions = backend.list_sessions("room-1")
        assert len(room1_sessions) == 2

        room2_sessions = backend.list_sessions("room-2")
        assert len(room2_sessions) == 1

    async def test_simulate_video_received(self) -> None:
        backend = MockVideoBackend()
        received_frames: list[tuple[VideoSession, VideoFrame]] = []

        def on_frame(session: VideoSession, frame: VideoFrame) -> None:
            received_frames.append((session, frame))

        backend.on_video_received(on_frame)
        session = await backend.connect("room-1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264", width=640, height=480)

        await backend.simulate_video_received(session, frame)
        assert len(received_frames) == 1
        assert received_frames[0][0].id == session.id
        assert received_frames[0][1] is frame

    async def test_simulate_session_ready(self) -> None:
        backend = MockVideoBackend()
        ready_sessions: list[VideoSession] = []

        def on_ready(session: VideoSession) -> None:
            ready_sessions.append(session)

        backend.on_session_ready(on_ready)
        session = await backend.connect("room-1", "user-1", "video-1")

        # connect fires session_ready automatically, but let's test explicit
        ready_sessions.clear()
        await backend.simulate_session_ready(session)
        assert len(ready_sessions) == 1

    async def test_simulate_client_disconnected(self) -> None:
        backend = MockVideoBackend()
        disconnected: list[VideoSession] = []

        def on_disconnect(session: VideoSession) -> None:
            disconnected.append(session)

        backend.on_client_disconnected(on_disconnect)
        session = await backend.connect("room-1", "user-1", "video-1")

        await backend.simulate_client_disconnected(session)
        assert len(disconnected) == 1

    async def test_close(self) -> None:
        backend = MockVideoBackend()
        await backend.connect("room-1", "user-1", "video-1")
        await backend.close()
        assert backend.calls[-1].method == "close"
        assert backend.list_sessions("room-1") == []

    async def test_capabilities(self) -> None:
        backend = MockVideoBackend(
            capabilities=VideoCapability.SIMULCAST | VideoCapability.RECORDING
        )
        assert VideoCapability.SIMULCAST in backend.capabilities
        assert VideoCapability.RECORDING in backend.capabilities

    async def test_get_session_not_found(self) -> None:
        backend = MockVideoBackend()
        assert backend.get_session("nonexistent") is None

    async def test_connect_with_metadata(self) -> None:
        backend = MockVideoBackend()
        session = await backend.connect("room-1", "user-1", "video-1", metadata={"codec": "vp9"})
        assert session.metadata == {"codec": "vp9"}
        assert backend.calls[-1].args["metadata"] == {"codec": "vp9"}


# ---------------------------------------------------------------------------
# VideoBackend ABC
# ---------------------------------------------------------------------------


class TestVideoBackendABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VideoBackend()  # type: ignore[abstract]

    def test_default_capabilities(self) -> None:
        class MinimalBackend(VideoBackend):
            @property
            def name(self) -> str:
                return "minimal"

            async def disconnect(self, session: VideoSession) -> None:
                pass

            async def send_video(self, session, video) -> None:
                pass

        backend = MinimalBackend()
        assert backend.capabilities == VideoCapability.NONE
        assert backend.get_session("x") is None
        assert backend.list_sessions("x") == []

    async def test_connect_not_implemented(self) -> None:
        class MinimalBackend(VideoBackend):
            @property
            def name(self) -> str:
                return "minimal"

            async def disconnect(self, session: VideoSession) -> None:
                pass

            async def send_video(self, session, video) -> None:
                pass

        backend = MinimalBackend()
        with pytest.raises(NotImplementedError, match="minimal does not implement connect"):
            await backend.connect("r", "p", "c")

    async def test_accept_not_implemented(self) -> None:
        class MinimalBackend(VideoBackend):
            @property
            def name(self) -> str:
                return "minimal"

            async def disconnect(self, session: VideoSession) -> None:
                pass

            async def send_video(self, session, video) -> None:
                pass

        backend = MinimalBackend()
        session = VideoSession(id="s", room_id="r", participant_id="p", channel_id="c")
        with pytest.raises(NotImplementedError, match="minimal does not implement accept"):
            await backend.accept(session, None)

    async def test_close_is_noop(self) -> None:
        class MinimalBackend(VideoBackend):
            @property
            def name(self) -> str:
                return "minimal"

            async def disconnect(self, session: VideoSession) -> None:
                pass

            async def send_video(self, session, video) -> None:
                pass

        backend = MinimalBackend()
        await backend.close()  # should not raise


# ---------------------------------------------------------------------------
# VisionProvider ABC
# ---------------------------------------------------------------------------


class TestVisionProviderABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VisionProvider()  # type: ignore[abstract]

    def test_default_name(self) -> None:
        class MyVision(VisionProvider):
            async def analyze_frame(self, frame):
                return VisionResult(description="test")

        provider = MyVision()
        assert provider.name == "MyVision"
        assert provider.supports_streaming is False

    async def test_warmup_and_close_are_noop(self) -> None:
        class MyVision(VisionProvider):
            async def analyze_frame(self, frame):
                return VisionResult(description="test")

        provider = MyVision()
        await provider.warmup()  # should not raise
        await provider.close()  # should not raise


# ---------------------------------------------------------------------------
# VisionResult
# ---------------------------------------------------------------------------


class TestVisionResult:
    def test_create_minimal(self) -> None:
        result = VisionResult(description="A person")
        assert result.description == "A person"
        assert result.labels == []
        assert result.confidence == 1.0
        assert result.faces == []
        assert result.text is None
        assert result.metadata == {}

    def test_create_full(self) -> None:
        face = FaceDetection(x=10, y=20, width=100, height=100, confidence=0.95)
        result = VisionResult(
            description="A person at a desk",
            labels=["person", "desk", "monitor"],
            confidence=0.88,
            faces=[face],
            text="Hello World",
            metadata={"model": "gpt-4o"},
        )
        assert len(result.faces) == 1
        assert result.faces[0].confidence == 0.95
        assert result.text == "Hello World"


# ---------------------------------------------------------------------------
# FaceDetection
# ---------------------------------------------------------------------------


class TestFaceDetection:
    def test_create(self) -> None:
        face = FaceDetection(x=0, y=0, width=50, height=50, confidence=0.99, label="Alice")
        assert face.label == "Alice"
        assert face.confidence == 0.99


# ---------------------------------------------------------------------------
# MockVisionProvider
# ---------------------------------------------------------------------------


class TestMockVisionProvider:
    async def test_default_descriptions(self) -> None:
        provider = MockVisionProvider()
        frame = VideoFrame(data=b"\x00", codec="h264")

        r1 = await provider.analyze_frame(frame)
        assert r1.description == "A video frame"
        assert r1.labels == ["person"]

        r2 = await provider.analyze_frame(frame)
        assert r2.description == "A person in a room"
        assert r2.labels == ["room"]

    async def test_custom_descriptions(self) -> None:
        provider = MockVisionProvider(descriptions=["Custom"])
        frame = VideoFrame(data=b"\x00", codec="vp8")

        result = await provider.analyze_frame(frame)
        assert result.description == "Custom"

    async def test_tracks_calls(self) -> None:
        provider = MockVisionProvider()
        frame1 = VideoFrame(data=b"\x01", codec="h264")
        frame2 = VideoFrame(data=b"\x02", codec="vp9")

        await provider.analyze_frame(frame1)
        await provider.analyze_frame(frame2)
        assert len(provider.calls) == 2
        assert provider.calls[0] is frame1
        assert provider.calls[1] is frame2

    async def test_round_robin(self) -> None:
        provider = MockVisionProvider(descriptions=["A", "B"])
        frame = VideoFrame(data=b"\x00", codec="h264")

        results = [await provider.analyze_frame(frame) for _ in range(4)]
        assert [r.description for r in results] == ["A", "B", "A", "B"]

    async def test_analyze_stream(self) -> None:
        provider = MockVisionProvider(descriptions=["Frame"])

        async def frames():
            for i in range(5):
                yield VideoFrame(data=b"\x00", codec="h264", timestamp_ms=float(i * 1000))

        results = []
        async for result in provider.analyze_stream(frames(), interval_ms=2000):
            results.append(result)

        # With interval_ms=2000, frames at 0, 2000, 4000 should be analyzed
        assert len(results) == 3

    async def test_analyze_stream_without_timestamps(self) -> None:
        """When frames lack timestamps, synthetic ~30fps timing is used."""
        provider = MockVisionProvider(descriptions=["Frame"])

        async def frames():
            # 100 frames without timestamps (~3.3s at synthetic 30fps)
            for _ in range(100):
                yield VideoFrame(data=b"\x00", codec="h264")

        results = []
        async for result in provider.analyze_stream(frames(), interval_ms=1000):
            results.append(result)

        # At ~33ms per frame synthetic, 1000ms interval means ~1 per 30 frames
        # 100 frames / 30 ≈ 3-4 analyses (first at 0, then ~30, ~60, ~90)
        assert 3 <= len(results) <= 5


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_imports_from_roomkit(self) -> None:
        """Verify video channel is accessible from the top-level package."""
        import roomkit

        assert hasattr(roomkit, "VideoChannel")

    def test_video_types_from_subpackages(self) -> None:
        """Verify video types are accessible from subpackages."""
        from roomkit.video.backends.mock import MockVideoBackend
        from roomkit.video.video_frame import VideoFrame
        from roomkit.video.vision.mock import MockVisionProvider

        assert VideoFrame is not None
        assert MockVideoBackend is not None
        assert MockVisionProvider is not None

    def test_hook_triggers_exist(self) -> None:
        """Verify video hook triggers are defined."""
        from roomkit.models.enums import HookTrigger

        assert HookTrigger.ON_VIDEO_SESSION_STARTED == "on_video_session_started"
        assert HookTrigger.ON_VIDEO_SESSION_ENDED == "on_video_session_ended"
        assert HookTrigger.ON_VIDEO_TRACK_ADDED == "on_video_track_added"
        assert HookTrigger.ON_VIDEO_TRACK_REMOVED == "on_video_track_removed"
        assert HookTrigger.ON_SCREEN_SHARE_STARTED == "on_screen_share_started"
        assert HookTrigger.ON_SCREEN_SHARE_STOPPED == "on_screen_share_stopped"

    def test_channel_type_video(self) -> None:
        """Verify ChannelType.VIDEO exists."""
        from roomkit.models.enums import ChannelType

        assert ChannelType.VIDEO == "video"
