"""Video session isolation and shared channel resource ownership."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.av import AudioVideoChannel
from roomkit.channels.video import VideoChannel
from roomkit.models.channel import ChannelBinding
from roomkit.video.backends.mock import MockVideoBackend
from roomkit.video.base import VideoSession
from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.decoder.base import VideoDecoderProvider
from roomkit.video.pipeline.engine import VideoPipeline
from roomkit.video.pipeline.filter.censor import CensorVideoFilter
from roomkit.video.pipeline.filter.mock_face_touch import MockFaceTouchFilter
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.base import VisionResult
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.base import VoiceSession


def frame() -> VideoFrame:
    return VideoFrame(data=b"\xff" * 3, codec="raw_rgb24", width=1, height=1)


def test_censor_grace_and_reset_do_not_cross_sessions() -> None:
    pipeline = VideoPipeline(
        VideoPipelineConfig(filters=[CensorVideoFilter({"private"}, grace_frames=2)])
    )
    pipeline.update_filter_context("a", VisionResult(description="private", labels=["private"]))
    original = frame()
    assert pipeline.process_inbound("a", original).data == b"\0" * 3
    assert pipeline.process_inbound("b", original) is original
    pipeline.reset("b")
    pipeline.update_filter_context("a", VisionResult(description="clear", labels=[]))
    assert pipeline.process_inbound("a", original).data == b"\0" * 3
    assert pipeline.process_inbound("a", original).data == b"\0" * 3
    assert pipeline.process_inbound("a", original) is original
    pipeline.reset("a")
    assert not pipeline._filter_contexts


@pytest.mark.parametrize("kind", ["video", "av"])
@pytest.mark.parametrize("vision_mode", ["pipeline", "shared", "separate"])
async def test_channel_closes_pipeline_and_each_vision_once(kind: str, vision_mode: str) -> None:
    decoder = MagicMock(spec=VideoDecoderProvider)
    vision = AsyncMock()
    direct = (
        vision if vision_mode == "shared" else AsyncMock() if vision_mode == "separate" else None
    )
    config = VideoPipelineConfig(decoder=decoder, vision=vision)
    if kind == "video":
        channel = VideoChannel("v", backend=MockVideoBackend(), pipeline=config, vision=direct)
    else:
        channel = AudioVideoChannel(
            "av", backend=MockVoiceBackend(), video_pipeline=config, vision=direct
        )
    await channel.close()
    await channel.close()
    decoder.close.assert_called_once()
    vision.close.assert_awaited_once()
    if direct is not None:
        direct.close.assert_awaited_once()


@pytest.mark.parametrize("kind", ["video", "av"])
async def test_unbind_removes_only_its_video_context(kind: str) -> None:
    config = VideoPipelineConfig(filters=[CensorVideoFilter({"private"}, grace_frames=2)])
    if kind == "video":
        channel = VideoChannel("v", backend=MockVideoBackend(), pipeline=config)
        session = VideoSession(id="a", room_id="r", channel_id="v", participant_id="p")
    else:
        channel = AudioVideoChannel("v", backend=MockVoiceBackend(), video_pipeline=config)
        session = VoiceSession(id="a", room_id="r", channel_id="v", participant_id="p")
    try:
        channel.bind_session(
            session, "r", ChannelBinding(room_id="r", channel_id="v", channel_type="video")
        )
        pipeline = channel._video_pipeline
        pipeline.update_filter_context("a", VisionResult(description="a", labels=[]))
        pipeline.update_filter_context("b", VisionResult(description="b", labels=["private"]))
        channel.unbind_session(session)
        assert "a" not in pipeline._filter_contexts
        assert "b" in pipeline._filter_contexts
    finally:
        await channel.close()


async def test_pipeline_async_close_includes_vision_even_after_stage_failure() -> None:
    decoder = MagicMock(spec=VideoDecoderProvider)
    decoder.close.side_effect = RuntimeError("bad decoder close")
    vision = AsyncMock()
    pipeline = VideoPipeline(VideoPipelineConfig(decoder=decoder, vision=vision))
    with pytest.raises(ExceptionGroup):
        await pipeline.aclose()
    vision.close.assert_awaited_once()
    assert pipeline.process_inbound("a", frame()) is None


async def test_av_dispatches_and_drains_detection_events() -> None:
    from roomkit.video.events import VideoDetectionEvent

    detection = VideoDetectionEvent(
        session=None, kind="face_touch", labels=["cheek"], confidence=1.0
    )
    channel = AudioVideoChannel(
        "av",
        backend=MockVoiceBackend(),
        video_pipeline=VideoPipelineConfig(filters=[MockFaceTouchFilter({0: [detection]})]),
    )
    session = VideoSession(id="a", room_id="r", channel_id="av", participant_id="p")
    binding = ChannelBinding(room_id="r", channel_id="av", channel_type="audio_video")
    channel._session_bindings[session.id] = ("r", binding)
    hook = AsyncMock()
    channel._fire_video_detection_hook = hook
    try:
        channel._on_video_received(session, frame())
        import asyncio

        await asyncio.gather(*channel._scheduled_tasks)
        hook.assert_awaited_once_with(session, detection, "r")
        assert channel._video_pipeline.drain_events(session.id) == []
    finally:
        await channel.close()


async def test_vision_finishing_after_unbind_does_not_restore_session_state() -> None:
    import asyncio

    started, finish = asyncio.Event(), asyncio.Event()
    vision = AsyncMock()

    async def analyze(frame: VideoFrame) -> VisionResult:
        started.set()
        await finish.wait()
        return VisionResult(description="late", labels=["private"])

    vision.analyze_frame.side_effect = analyze
    channel = VideoChannel(
        "v",
        backend=MockVideoBackend(),
        pipeline=VideoPipelineConfig(vision=vision, filters=[CensorVideoFilter({"private"})]),
    )
    session = VideoSession(id="a", room_id="r", channel_id="v", participant_id="p")
    channel.bind_session(
        session, "r", ChannelBinding(room_id="r", channel_id="v", channel_type="video")
    )
    task = asyncio.create_task(channel._analyze_frame(session, frame(), "r"))
    await started.wait()
    channel.unbind_session(session)
    finish.set()
    try:
        await task
        assert "a" not in channel._last_vision_results
        assert "a" not in channel._last_vision_ts
        assert "a" not in channel._video_pipeline._filter_contexts
    finally:
        await channel.close()
