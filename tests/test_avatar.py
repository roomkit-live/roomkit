"""Tests for AvatarProvider ABC and MockAvatarProvider."""

from __future__ import annotations

from roomkit.video.avatar import AvatarProvider, MockAvatarProvider


class TestMockAvatarProvider:
    async def test_name(self) -> None:
        avatar = MockAvatarProvider()
        assert avatar.name == "mock"

    async def test_fps(self) -> None:
        avatar = MockAvatarProvider(fps=25)
        assert avatar.fps == 25

    async def test_not_started_initially(self) -> None:
        avatar = MockAvatarProvider()
        assert avatar.is_started is False

    async def test_start(self) -> None:
        avatar = MockAvatarProvider()
        await avatar.start(b"fake-png-data", width=640, height=480)
        assert avatar.is_started is True

    async def test_stop(self) -> None:
        avatar = MockAvatarProvider()
        await avatar.start(b"fake-png-data")
        await avatar.stop()
        assert avatar.is_started is False

    async def test_close_alias(self) -> None:
        avatar = MockAvatarProvider()
        await avatar.start(b"fake-png-data")
        await avatar.close()
        assert avatar.is_started is False

    async def test_feed_audio_before_start_returns_empty(self) -> None:
        avatar = MockAvatarProvider()
        frames = avatar.feed_audio(b"\x00" * 640)
        assert frames == []

    async def test_feed_audio_produces_frame(self) -> None:
        avatar = MockAvatarProvider()
        await avatar.start(b"fake-png", width=64, height=48)

        frames = avatar.feed_audio(b"\x00" * 640, sample_rate=16000)
        assert len(frames) == 1
        assert frames[0].codec == "raw_rgb24"
        assert frames[0].width == 64
        assert frames[0].height == 48

    async def test_feed_audio_tracking(self) -> None:
        avatar = MockAvatarProvider()
        await avatar.start(b"fake-png")

        avatar.feed_audio(b"\x00" * 320)
        avatar.feed_audio(b"\x00" * 640)
        avatar.feed_audio(b"\x00" * 160)

        assert avatar.feed_count == 3
        assert avatar.total_audio_bytes == 320 + 640 + 160

    async def test_feed_audio_frame_color(self) -> None:
        avatar = MockAvatarProvider(color=(255, 0, 0))
        await avatar.start(b"fake-png", width=2, height=2)

        frames = avatar.feed_audio(b"\x00" * 640)
        data = frames[0].data
        # 2x2 pixels, each (255, 0, 0) = 12 bytes
        assert len(data) == 12
        assert data[:3] == b"\xff\x00\x00"

    async def test_idle_frame(self) -> None:
        avatar = MockAvatarProvider(idle_color=(100, 100, 100))
        await avatar.start(b"fake-png", width=4, height=4)

        frame = avatar.get_idle_frame()
        assert frame is not None
        assert frame.codec == "raw_rgb24"
        assert avatar.idle_count == 1

    async def test_idle_frame_none_when_not_started(self) -> None:
        avatar = MockAvatarProvider()
        assert avatar.get_idle_frame() is None

    async def test_idle_frame_none_when_disabled(self) -> None:
        avatar = MockAvatarProvider(idle_color=None)
        await avatar.start(b"fake-png")
        assert avatar.get_idle_frame() is None

    async def test_flush(self) -> None:
        avatar = MockAvatarProvider()
        await avatar.start(b"fake-png", width=8, height=8)

        frames = avatar.flush()
        assert len(frames) == 1
        assert avatar.flush_count == 1

    async def test_flush_before_start_returns_empty(self) -> None:
        avatar = MockAvatarProvider()
        assert avatar.flush() == []

    async def test_is_subclass(self) -> None:
        assert issubclass(MockAvatarProvider, AvatarProvider)

    async def test_default_dimensions(self) -> None:
        avatar = MockAvatarProvider()
        await avatar.start(b"fake-png")

        frames = avatar.feed_audio(b"\x00" * 640)
        assert frames[0].width == 512
        assert frames[0].height == 512
