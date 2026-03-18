"""Integration test: AudioVideoChannel + FastRTCVideoBackend."""

from __future__ import annotations

import numpy as np

from roomkit import AudioVideoChannel, RoomKit
from roomkit.video.backends.fastrtc import FastRTCVideoBackend
from roomkit.voice.pipeline import AudioPipelineConfig
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider


class TestAVChannelFastRTC:
    """AudioVideoChannel wired to FastRTCVideoBackend."""

    async def test_video_frames_reach_taps(self):
        backend = FastRTCVideoBackend()
        av = AudioVideoChannel(
            "av",
            stt=MockSTTProvider(),
            tts=MockTTSProvider(),
            backend=backend,
            pipeline=AudioPipelineConfig(),
        )

        kit = RoomKit()
        kit.register_channel(av)

        room = await kit.create_room(room_id="test-room")
        binding_result = await kit.attach_channel(room.id, "av")

        session = await backend.connect(room.id, "user-1", "av")
        session.metadata["websocket_id"] = "ws-test"
        av.bind_session(session, room.id, binding_result)

        # Register a video tap
        tapped = []
        av.add_video_media_tap(lambda s, f: tapped.append(f))

        # Simulate video frame from backend
        video_data = np.zeros((480, 640, 3), dtype=np.uint8)
        backend._handle_video_frame("ws-test", video_data, 640, 480)

        assert len(tapped) == 1
        assert tapped[0].codec == "raw_rgb24"
        assert tapped[0].width == 640
        assert tapped[0].height == 480

        await kit.close()

    async def test_backend_isinstance_video_backend(self):
        """AudioVideoChannel checks isinstance(backend, VideoBackend)."""
        from roomkit.video.backends.base import VideoBackend

        backend = FastRTCVideoBackend()
        assert isinstance(backend, VideoBackend)

    async def test_video_session_accessible(self):
        backend = FastRTCVideoBackend()
        session = await backend.connect("room-1", "user-1", "av")

        video_session = backend.get_video_session(session.id)
        assert video_session is not None
        assert video_session.id == session.id
