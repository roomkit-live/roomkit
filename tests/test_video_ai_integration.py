"""Tests for video vision → AIChannel integration."""

from __future__ import annotations

import asyncio

import pytest

from roomkit import (
    MockAIProvider,
    MockVideoBackend,
    MockVisionProvider,
    RoomKit,
    VideoChannel,
    VideoFrame,
    setup_video_vision,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import ChannelCategory


@pytest.fixture
def kit() -> RoomKit:
    return RoomKit()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSetupVideoVision:
    async def test_vision_updates_ai_system_prompt(self, kit: RoomKit) -> None:
        """Vision results are injected into the AI's system prompt."""
        backend = MockVideoBackend()
        vision = MockVisionProvider(descriptions=["A cat sitting on a desk"])
        video = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=0)
        ai_provider = MockAIProvider(responses=["I see a cat!"])
        ai = AIChannel("ai-1", provider=ai_provider, system_prompt="You are helpful.")

        kit.register_channel(video)
        kit.register_channel(ai)

        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")
        await kit.attach_channel("r1", "ai-1", category=ChannelCategory.INTELLIGENCE)

        # Wire vision → AI (deprecated but still functional)
        setup_video_vision(kit, room_id="r1", ai_channel_id="ai-1")

        # Connect and send a frame
        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.2)

        # Verify the AI binding now has vision context in system_prompt
        binding = await kit._store.get_binding("r1", "ai-1")
        assert binding is not None
        prompt = binding.metadata.get("system_prompt", "")
        assert "A cat sitting on a desk" in prompt
        assert "You are helpful." in prompt

    async def test_vision_preserves_base_prompt(self, kit: RoomKit) -> None:
        """Multiple vision updates don't stack — base prompt is preserved."""
        backend = MockVideoBackend()
        vision = MockVisionProvider(descriptions=["Frame 1", "Frame 2"])
        video = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=0)
        ai = AIChannel("ai-1", provider=MockAIProvider(responses=["ok"]), system_prompt="Base.")

        kit.register_channel(video)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")
        await kit.attach_channel("r1", "ai-1", category=ChannelCategory.INTELLIGENCE)
        setup_video_vision(kit, room_id="r1", ai_channel_id="ai-1")

        session = await kit.connect_video("r1", "user-1", "video-1")

        # Send two frames with different timestamps to pass throttle
        for i in range(2):
            frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=float(i * 5000))
            await backend.simulate_video_received(session, frame)
            await asyncio.sleep(0.2)

        binding = await kit._store.get_binding("r1", "ai-1")
        prompt = binding.metadata.get("system_prompt", "")
        # Base prompt appears once, not stacked
        assert prompt.count("Base.") == 1
        # Latest vision is present
        assert "Frame 2" in prompt

    async def test_ignores_other_rooms(self, kit: RoomKit) -> None:
        """Vision events from other rooms are ignored."""
        backend = MockVideoBackend()
        vision = MockVisionProvider(descriptions=["Something"])
        video = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=0)
        ai = AIChannel("ai-1", provider=MockAIProvider(responses=["ok"]), system_prompt="Base.")

        kit.register_channel(video)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.create_room(room_id="r2")
        await kit.attach_channel("r1", "video-1")
        await kit.attach_channel("r2", "ai-1", category=ChannelCategory.INTELLIGENCE)

        # Wire vision for r2 only
        setup_video_vision(kit, room_id="r2", ai_channel_id="ai-1")

        # Send frame in r1
        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.2)

        # AI binding in r2 should be unchanged
        binding = await kit._store.get_binding("r2", "ai-1")
        prompt = binding.metadata.get("system_prompt", "")
        assert "Something" not in prompt

    async def test_custom_prefix(self, kit: RoomKit) -> None:
        """Custom context_prefix is used."""
        backend = MockVideoBackend()
        vision = MockVisionProvider(descriptions=["A person"])
        video = VideoChannel("video-1", backend=backend, vision=vision, vision_interval_ms=0)
        ai = AIChannel("ai-1", provider=MockAIProvider(responses=["ok"]))

        kit.register_channel(video)
        kit.register_channel(ai)
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "video-1")
        await kit.attach_channel("r1", "ai-1", category=ChannelCategory.INTELLIGENCE)

        setup_video_vision(
            kit,
            room_id="r1",
            ai_channel_id="ai-1",
            context_prefix="Camera shows:",
        )

        session = await kit.connect_video("r1", "user-1", "video-1")
        frame = VideoFrame(data=b"\x00" * 100, codec="h264", timestamp_ms=0.0)
        await backend.simulate_video_received(session, frame)
        await asyncio.sleep(0.2)

        binding = await kit._store.get_binding("r1", "ai-1")
        prompt = binding.metadata.get("system_prompt", "")
        assert "Camera shows: A person" in prompt


class TestExport:
    def test_importable_from_roomkit(self) -> None:
        import roomkit

        assert hasattr(roomkit, "setup_video_vision")
