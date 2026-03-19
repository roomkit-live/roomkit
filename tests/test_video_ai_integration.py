"""Tests for video vision → AIChannel integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import (
    RoomKit,
    VideoChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.video.ai_integration import setup_realtime_vision, setup_video_vision
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.mock import MockVisionProvider
from roomkit.voice.base import VoiceSession


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


class TestSetupRealtimeVision:
    async def test_injects_vision_via_inject_text(self, kit: RoomKit) -> None:
        """Vision results should be injected via inject_text(silent=True)."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        rtv.channel_id = "rtv-1"
        rtv.channel_type = ChannelType.REALTIME_VOICE
        rtv.close = AsyncMock()

        session = MagicMock(spec=VoiceSession)
        session.id = "sess-1"
        rtv.get_room_sessions = MagicMock(return_value=[session])
        rtv.inject_text = AsyncMock()

        kit._channels["rtv-1"] = rtv
        await kit.create_room(room_id="r1")

        setup_realtime_vision(kit, room_id="r1", voice_channel_id="rtv-1")

        # Fire a vision event
        await kit._emit_framework_event(
            "video_vision_result",
            room_id="r1",
            data={"description": "A cat on a desk"},
        )
        await asyncio.sleep(0.05)

        rtv.inject_text.assert_called_once()
        call_args = rtv.inject_text.call_args
        assert call_args[0][0] is session
        assert "A cat on a desk" in call_args[0][1]
        assert call_args[1]["silent"] is True

    async def test_dedup_skips_unchanged_description(self, kit: RoomKit) -> None:
        """Same description should not be re-injected."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        rtv.channel_id = "rtv-1"
        rtv.channel_type = ChannelType.REALTIME_VOICE
        rtv.close = AsyncMock()

        session = MagicMock(spec=VoiceSession)
        session.id = "sess-1"
        rtv.get_room_sessions = MagicMock(return_value=[session])
        rtv.inject_text = AsyncMock()

        kit._channels["rtv-1"] = rtv
        await kit.create_room(room_id="r1")

        setup_realtime_vision(kit, room_id="r1", voice_channel_id="rtv-1")

        # Fire same event twice
        for _ in range(2):
            await kit._emit_framework_event(
                "video_vision_result",
                room_id="r1",
                data={"description": "Same scene"},
            )
            await asyncio.sleep(0.05)

        # Only injected once (dedup)
        assert rtv.inject_text.call_count == 1

    async def test_ignores_other_rooms(self, kit: RoomKit) -> None:
        """Events from other rooms should not trigger injection."""
        rtv = MagicMock(spec=RealtimeVoiceChannel)
        rtv.channel_id = "rtv-1"
        rtv.channel_type = ChannelType.REALTIME_VOICE
        rtv.close = AsyncMock()
        rtv.inject_text = AsyncMock()

        kit._channels["rtv-1"] = rtv
        await kit.create_room(room_id="r1")
        await kit.create_room(room_id="r2")

        setup_realtime_vision(kit, room_id="r1", voice_channel_id="rtv-1")

        # Fire event in a different room
        await kit._emit_framework_event(
            "video_vision_result",
            room_id="r2",
            data={"description": "Something"},
        )
        await asyncio.sleep(0.05)

        rtv.inject_text.assert_not_called()


class TestExport:
    def test_importable_from_subpackage(self) -> None:
        from roomkit.video.ai_integration import setup_video_vision

        assert setup_video_vision is not None
