"""Real codec regressions: first keyframe, interleaved sources and session reset."""

from __future__ import annotations

from fractions import Fraction

import pytest

from roomkit.video.pipeline.config import VideoPipelineConfig
from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder
from roomkit.video.pipeline.engine import VideoPipeline
from roomkit.video.video_frame import VideoFrame


def encoded_sequence(value: int) -> list[VideoFrame]:
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    codec = av.CodecContext.create("libx264", "w")
    codec.width = codec.height = 32
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 25)
    codec.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "x264-params": "keyint=30:scenecut=0",
    }
    frames = []
    for index in range(4):
        raw = av.VideoFrame.from_ndarray(
            np.full((32, 32, 3), value, dtype=np.uint8), format="rgb24"
        )
        raw.pts = index
        for packet in codec.encode(raw):
            frames.append(
                VideoFrame(
                    data=bytes(packet),
                    keyframe=packet.is_keyframe,
                    sequence=index,
                    width=32,
                    height=32,
                )
            )
    assert len(frames) == 4
    assert [frame.keyframe for frame in frames] == [True, False, False, False]
    return frames


def test_first_keyframe_enables_following_p_frames() -> None:
    decoder = PyAVVideoDecoder()
    frames = encoded_sequence(90)
    try:
        for frame in frames:
            result = decoder.decode(frame)
            assert result is not None
            assert abs(sum(result.data) / len(result.data) - 90) < 5
        decoder.reset()
        assert decoder.decode(frames[1]) is None
        assert decoder.decode(frames[0]) is not None
        assert decoder.decode(frames[1]) is not None
    finally:
        decoder.close()


def test_interleaved_sources_and_reset_are_isolated() -> None:
    pipeline = VideoPipeline(VideoPipelineConfig(decoder=PyAVVideoDecoder()))
    first, second = encoded_sequence(40), encoded_sequence(200)
    try:
        assert pipeline.process_inbound("a", first[0]) is not None
        assert pipeline.process_inbound("b", second[1]) is None
        assert pipeline.process_inbound("b", second[0]) is not None
        for index in (1, 2):
            for sid, frames, expected in (("a", first, 40), ("b", second, 200)):
                result = pipeline.process_inbound(sid, frames[index])
                assert result is not None
                assert abs(sum(result.data) / len(result.data) - expected) < 5
        pipeline.reset("a")
        assert pipeline.process_inbound("a", first[3]) is None
        assert pipeline.process_inbound("b", second[3]) is not None
    finally:
        pipeline.close()
