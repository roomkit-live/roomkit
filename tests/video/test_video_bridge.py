"""Tests for VideoBridge — session-to-session video forwarding."""

from __future__ import annotations

import asyncio

import pytest

from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.bridge import VideoBridge, VideoBridgeConfig
from roomkit.video.video_frame import VideoFrame


def _frame(data: bytes = b"\x00" * 100, *, keyframe: bool = True) -> VideoFrame:
    """Create a minimal test VideoFrame (keyframe by default for bridge tests)."""
    return VideoFrame(data=data, codec="h264", width=640, height=480, keyframe=keyframe)


class TestVideoBridge:
    """Core VideoBridge tests."""

    async def test_two_party_forward(self) -> None:
        """Video from session A is forwarded to session B."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        frame = _frame(b"\x01\x02" * 50)
        bridge.forward(s1, frame)

        # Wait for the async task created by send_video_sync
        await asyncio.sleep(0)

        # s2 should have received the video
        assert len(backend.sent_video) == 1
        assert backend.sent_video[0][0] == s2.id
        assert backend.sent_video[0][1] == frame.data

    async def test_bidirectional_forward(self) -> None:
        """Video flows both ways."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        frame_a = _frame(b"\x01" * 100)
        frame_b = _frame(b"\x02" * 100)

        bridge.forward(s1, frame_a)
        bridge.forward(s2, frame_b)
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 2
        # s1 sent frame_a → s2 received it
        assert backend.sent_video[0][0] == s2.id
        assert backend.sent_video[0][1] == frame_a.data
        # s2 sent frame_b → s1 received it
        assert backend.sent_video[1][0] == s1.id
        assert backend.sent_video[1][1] == frame_b.data

    async def test_no_self_forward(self) -> None:
        """A session should never receive its own frame."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        bridge.add_session(s1, "room-1", backend)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 0

    async def test_room_isolation(self) -> None:
        """Sessions in different rooms don't receive each other's frames."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-2", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-2", backend)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        # s2 is in a different room — should not receive
        assert len(backend.sent_video) == 0

    async def test_three_party_forward(self) -> None:
        """With 3 sessions, a frame is forwarded to the other 2."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")
        s3 = await backend.connect("room-1", "user-3", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.add_session(s3, "room-1", backend)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 2
        recipient_ids = {sid for sid, _ in backend.sent_video}
        assert recipient_ids == {s2.id, s3.id}

    async def test_remove_session(self) -> None:
        """Removed sessions don't receive frames."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.remove_session(s2.id)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 0

    async def test_max_participants(self) -> None:
        """Adding more than max_participants raises RuntimeError."""
        backend = MockVideoBackend()
        config = VideoBridgeConfig(max_participants=2)
        bridge = VideoBridge(config)

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")
        s3 = await backend.connect("room-1", "user-3", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        with pytest.raises(RuntimeError, match="max bridge participants"):
            bridge.add_session(s3, "room-1", backend)

    async def test_disabled_bridge(self) -> None:
        """Disabled bridge doesn't forward anything."""
        backend = MockVideoBackend()
        config = VideoBridgeConfig(enabled=False)
        bridge = VideoBridge(config)

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 0

    async def test_frame_filter_drops(self) -> None:
        """Frame filter returning None drops the frame."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # Filter drops all frames
        bridge.set_frame_filter(lambda session, frame: None)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 0

    async def test_frame_filter_passes(self) -> None:
        """Frame filter returning the frame lets it through."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # Filter passes all frames unchanged
        bridge.set_frame_filter(lambda session, frame: frame)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 1

    async def test_frame_processor(self) -> None:
        """Frame processor transforms frames for each target."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # Processor replaces data with marker
        marker = b"\xff" * 100
        bridge.set_frame_processor(
            lambda target, frame: VideoFrame(
                data=marker, codec=frame.codec, width=frame.width, height=frame.height
            )
        )

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        assert len(backend.sent_video) == 1
        assert backend.sent_video[0][1] == marker

    async def test_participant_count(self) -> None:
        """get_participant_count returns the correct count."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        assert bridge.get_participant_count("room-1") == 0

        s1 = await backend.connect("room-1", "user-1", "video")
        bridge.add_session(s1, "room-1", backend)
        assert bridge.get_participant_count("room-1") == 1

        s2 = await backend.connect("room-1", "user-2", "video")
        bridge.add_session(s2, "room-1", backend)
        assert bridge.get_participant_count("room-1") == 2

        bridge.remove_session(s1.id)
        assert bridge.get_participant_count("room-1") == 1

    async def test_get_bridged_sessions(self) -> None:
        """get_bridged_sessions returns correct session/backend pairs."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        sessions = bridge.get_bridged_sessions("room-1")
        assert len(sessions) == 2
        session_ids = {s.id for s, _ in sessions}
        assert session_ids == {s1.id, s2.id}

        # Non-existent room returns empty
        assert bridge.get_bridged_sessions("room-x") == []

    async def test_close_clears_all(self) -> None:
        """close() removes all sessions."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        bridge.add_session(s1, "room-1", backend)

        bridge.close()

        assert bridge.get_participant_count("room-1") == 0
        assert bridge.get_bridged_sessions("room-1") == []

    async def test_separate_backends(self) -> None:
        """Each session can use a different backend."""
        backend_a = MockVideoBackend()
        backend_b = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend_a.connect("room-1", "user-1", "video")
        s2 = await backend_b.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend_a)
        bridge.add_session(s2, "room-1", backend_b)

        bridge.forward(s1, _frame())
        await asyncio.sleep(0)

        # Frame should be sent via backend_b (s2's backend)
        assert len(backend_b.sent_video) == 1
        assert backend_b.sent_video[0][0] == s2.id
        # backend_a should NOT have received it back
        assert len(backend_a.sent_video) == 0

    async def test_delta_frames_forwarded_without_keyframe(self) -> None:
        """Delta frames are forwarded even without a preceding keyframe.

        In bridge topologies the remote decoder handles recovery;
        blocking delta frames would prevent video when the source
        endpoint does not respond to PLI (e.g. B2BUA that doesn't
        relay RTCP feedback).
        """
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")
        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)

        # Delta frame passes through immediately
        bridge.forward(s1, _frame(b"\x01" * 50, keyframe=False))
        await asyncio.sleep(0)
        assert len(backend.sent_video) == 1

    async def test_readd_session_after_remove(self) -> None:
        """A session can be removed and re-added."""
        backend = MockVideoBackend()
        bridge = VideoBridge()

        s1 = await backend.connect("room-1", "user-1", "video")
        s2 = await backend.connect("room-1", "user-2", "video")

        bridge.add_session(s1, "room-1", backend)
        bridge.add_session(s2, "room-1", backend)
        bridge.remove_session(s2.id)

        # No target
        bridge.forward(s1, _frame())
        await asyncio.sleep(0)
        assert len(backend.sent_video) == 0

        # Re-add
        bridge.add_session(s2, "room-1", backend)
        bridge.forward(s1, _frame())
        await asyncio.sleep(0)
        assert len(backend.sent_video) == 1
