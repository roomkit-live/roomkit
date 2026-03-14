"""Tests for the video pipeline engine, decoder, and resizer stages."""

from __future__ import annotations

import pytest

from roomkit.video.pipeline import (
    MockVideoDecoderProvider,
    MockVideoResizerProvider,
    VideoPipeline,
    VideoPipelineConfig,
)
from roomkit.video.video_frame import VideoFrame
from roomkit.video.vision.mock import MockVisionProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_frame(
    width: int = 640,
    height: int = 480,
    codec: str = "raw_rgb24",
    seq: int = 0,
) -> VideoFrame:
    """Create a raw pixel frame for testing."""
    # 3 bytes per pixel for RGB24
    data = b"\x00" * (width * height * 3)
    return VideoFrame(
        data=data,
        codec=codec,
        width=width,
        height=height,
        sequence=seq,
    )


def _encoded_frame(
    codec: str = "h264",
    keyframe: bool = False,
    seq: int = 0,
) -> VideoFrame:
    """Create an encoded frame stub for testing."""
    return VideoFrame(
        data=b"\x00\x00\x01" + b"\x65" * 100,
        codec=codec,
        width=640,
        height=480,
        keyframe=keyframe,
        sequence=seq,
    )


# ---------------------------------------------------------------------------
# MockVideoDecoderProvider tests
# ---------------------------------------------------------------------------


class TestMockVideoDecoder:
    def test_passthrough(self) -> None:
        decoder = MockVideoDecoderProvider()
        frame = _encoded_frame()
        result = decoder.decode(frame)
        assert result is frame
        assert decoder.call_count == 1

    def test_tracks_frames(self) -> None:
        decoder = MockVideoDecoderProvider()
        f1 = _encoded_frame(seq=0)
        f2 = _encoded_frame(seq=1)
        decoder.decode(f1)
        decoder.decode(f2)
        assert decoder.call_count == 2
        assert decoder.frames == [f1, f2]

    def test_reset_clears(self) -> None:
        decoder = MockVideoDecoderProvider()
        decoder.decode(_encoded_frame())
        decoder.reset()
        assert decoder.call_count == 0
        assert decoder.frames == []

    def test_name(self) -> None:
        assert MockVideoDecoderProvider().name == "MockVideoDecoder"


# ---------------------------------------------------------------------------
# MockVideoResizerProvider tests
# ---------------------------------------------------------------------------


class TestMockVideoResizer:
    def test_passthrough(self) -> None:
        resizer = MockVideoResizerProvider()
        frame = _raw_frame()
        result = resizer.resize(frame)
        assert result is frame
        assert resizer.call_count == 1

    def test_name(self) -> None:
        assert MockVideoResizerProvider().name == "MockVideoResizer"


# ---------------------------------------------------------------------------
# VideoPipeline — pass-through (no stages)
# ---------------------------------------------------------------------------


class TestVideoPipelinePassthrough:
    def test_no_stages_returns_frame(self) -> None:
        pipeline = VideoPipeline(VideoPipelineConfig())
        frame = _encoded_frame(keyframe=True)
        result = pipeline.process_inbound("s1", frame)
        assert result is frame

    def test_raw_frame_passthrough(self) -> None:
        pipeline = VideoPipeline(VideoPipelineConfig())
        frame = _raw_frame()
        result = pipeline.process_inbound("s1", frame)
        assert result is frame


# ---------------------------------------------------------------------------
# VideoPipeline — decoder only
# ---------------------------------------------------------------------------


class TestVideoPipelineDecoder:
    def test_decoder_called_for_encoded_frame(self) -> None:
        decoder = MockVideoDecoderProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(decoder=decoder))
        frame = _encoded_frame(keyframe=True)
        result = pipeline.process_inbound("s1", frame)
        assert result is frame
        assert decoder.call_count == 1

    def test_decoder_skipped_for_raw_frame(self) -> None:
        decoder = MockVideoDecoderProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(decoder=decoder))
        frame = _raw_frame()
        result = pipeline.process_inbound("s1", frame)
        assert result is frame
        # Decoder should NOT be called for raw frames.
        assert decoder.call_count == 0

    def test_decoder_returns_none_drops_frame(self) -> None:
        """When decoder returns None (e.g. waiting for keyframe), pipeline drops the frame."""

        class DroppingDecoder(MockVideoDecoderProvider):
            def decode(self, frame: VideoFrame) -> VideoFrame | None:
                super().decode(frame)
                return None

        decoder = DroppingDecoder()
        pipeline = VideoPipeline(VideoPipelineConfig(decoder=decoder))
        result = pipeline.process_inbound("s1", _encoded_frame())
        assert result is None


# ---------------------------------------------------------------------------
# VideoPipeline — resizer only
# ---------------------------------------------------------------------------


class TestVideoPipelineResizer:
    def test_resizer_called_for_raw_frame(self) -> None:
        resizer = MockVideoResizerProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(resizer=resizer))
        frame = _raw_frame()
        result = pipeline.process_inbound("s1", frame)
        assert result is frame
        assert resizer.call_count == 1

    def test_resizer_skipped_for_encoded_frame(self) -> None:
        """Resizer only runs on raw frames — encoded frames pass through."""
        resizer = MockVideoResizerProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(resizer=resizer))
        frame = _encoded_frame(keyframe=True)
        result = pipeline.process_inbound("s1", frame)
        assert result is frame
        assert resizer.call_count == 0


# ---------------------------------------------------------------------------
# VideoPipeline — decoder + resizer
# ---------------------------------------------------------------------------


class TestVideoPipelineDecoderResizer:
    def test_both_stages_run(self) -> None:
        decoder = MockVideoDecoderProvider()
        resizer = MockVideoResizerProvider()
        pipeline = VideoPipeline(
            VideoPipelineConfig(
                decoder=decoder,
                resizer=resizer,
            )
        )
        # Mock decoder passes through, so the "decoded" frame is still
        # marked as encoded (it's a mock). Force a raw frame to test
        # the full path: decoder receives encoded, resizer receives raw.
        raw = _raw_frame()
        result = pipeline.process_inbound("s1", raw)
        assert result is raw
        # Decoder skipped (raw frame), resizer ran.
        assert decoder.call_count == 0
        assert resizer.call_count == 1


# ---------------------------------------------------------------------------
# VideoPipeline — vision (async)
# ---------------------------------------------------------------------------


class TestVideoPipelineVision:
    async def test_vision_analysis(self) -> None:
        vision = MockVisionProvider(descriptions=["A test frame"])
        pipeline = VideoPipeline(VideoPipelineConfig(vision=vision))
        frame = _raw_frame()
        result = await pipeline.process_vision("s1", frame)
        assert result is not None
        assert result.description == "A test frame"
        assert len(vision.calls) == 1

    async def test_no_vision_returns_none(self) -> None:
        pipeline = VideoPipeline(VideoPipelineConfig())
        result = await pipeline.process_vision("s1", _raw_frame())
        assert result is None


# ---------------------------------------------------------------------------
# VideoPipeline — reset and close
# ---------------------------------------------------------------------------


class TestVideoPipelineLifecycle:
    def test_reset_resets_decoder(self) -> None:
        decoder = MockVideoDecoderProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(decoder=decoder))
        pipeline.process_inbound("s1", _encoded_frame(keyframe=True))
        assert decoder.call_count == 1
        pipeline.reset("s1")
        assert decoder.call_count == 0

    def test_close_closes_stages(self) -> None:
        decoder = MockVideoDecoderProvider()
        resizer = MockVideoResizerProvider()
        pipeline = VideoPipeline(
            VideoPipelineConfig(
                decoder=decoder,
                resizer=resizer,
            )
        )
        # close() should not raise.
        pipeline.close()


# ---------------------------------------------------------------------------
# PyAV decoder (requires av)
# ---------------------------------------------------------------------------


class TestPyAVVideoDecoder:
    @pytest.fixture(autouse=True)
    def _skip_without_av(self) -> None:
        pytest.importorskip("av", reason="PyAV not installed")

    def test_decode_vp8_keyframe(self) -> None:
        """Encode a VP8 frame with PyAV and decode it."""
        from fractions import Fraction

        import av
        import numpy as np

        from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder

        # Encode a synthetic frame.
        width, height = 64, 48
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[10:20, 10:30] = [255, 0, 0]  # red rectangle

        av_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        codec_ctx = av.codec.CodecContext.create("libvpx", "w")
        codec_ctx.width = width
        codec_ctx.height = height
        codec_ctx.pix_fmt = "yuv420p"
        codec_ctx.time_base = Fraction(1, 30)
        codec_ctx.open()

        packets = codec_ctx.encode(av_frame)
        packets += codec_ctx.encode(None)  # flush
        assert packets, "No packets produced"
        encoded_data = b"".join(bytes(p) for p in packets)

        frame = VideoFrame(
            data=encoded_data,
            codec="vp8",
            width=width,
            height=height,
            keyframe=True,
            sequence=0,
        )

        decoder = PyAVVideoDecoder(output_format="rgb24")
        result = decoder.decode(frame)

        assert result is not None
        assert result.codec == "raw_rgb24"
        assert result.width == width
        assert result.height == height
        assert result.metadata.get("decoder") == "PyAVVideoDecoder"
        assert len(result.data) == width * height * 3

        decoder.close()

    def test_keyframe_gating(self) -> None:
        """P-frames are dropped before the first keyframe arrives."""
        from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder

        decoder = PyAVVideoDecoder()
        # Non-keyframe should be dropped.
        p_frame = _encoded_frame(codec="h264", keyframe=False, seq=0)
        result = decoder.decode(p_frame)
        assert result is None

        decoder.close()

    def test_raw_frame_passthrough(self) -> None:
        """Raw frames pass through the decoder unchanged."""
        from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder

        decoder = PyAVVideoDecoder()
        frame = _raw_frame()
        result = decoder.decode(frame)
        assert result is frame

        decoder.close()

    def test_invalid_output_format(self) -> None:
        from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder

        with pytest.raises(ValueError, match="output_format must be one of"):
            PyAVVideoDecoder(output_format="jpeg")

    def test_reset_clears_codecs(self) -> None:
        from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder

        decoder = PyAVVideoDecoder()
        # Force codec context creation by decoding with bad data.
        # After reset, internal state should be clean.
        decoder._seen_keyframe["vp8"] = True
        decoder.reset()
        assert decoder._codecs == {}
        assert decoder._seen_keyframe == {}

    def test_name(self) -> None:
        from roomkit.video.pipeline.decoder.pyav import PyAVVideoDecoder

        assert PyAVVideoDecoder().name == "PyAVVideoDecoder"


# ---------------------------------------------------------------------------
# PyAV resizer (requires av + numpy)
# ---------------------------------------------------------------------------


class TestPyAVVideoResizer:
    @pytest.fixture(autouse=True)
    def _skip_without_av(self) -> None:
        pytest.importorskip("av", reason="PyAV not installed")
        pytest.importorskip("numpy", reason="numpy not installed")

    def test_scales_down(self) -> None:
        """Large frame is scaled to fit within target dimensions."""
        import numpy as np

        from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer

        resizer = PyAVVideoResizer(width=32, height=24)
        # 64x48 -> should scale down to 32x24 (exact 2x).
        width, height = 64, 48
        data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
        frame = VideoFrame(
            data=data,
            codec="raw_rgb24",
            width=width,
            height=height,
        )

        result = resizer.resize(frame)
        assert result.width == 32
        assert result.height == 24
        assert result.metadata.get("resizer") == "PyAVVideoResizer"

    def test_no_resize_if_within_target(self) -> None:
        """Frame already within target dimensions is returned unchanged."""
        from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer

        resizer = PyAVVideoResizer(width=640, height=480)
        frame = _raw_frame(width=320, height=240)
        result = resizer.resize(frame)
        assert result is frame

    def test_encoded_frame_passthrough(self) -> None:
        """Encoded frames cannot be resized — returned unchanged."""
        from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer

        resizer = PyAVVideoResizer(width=320, height=240)
        frame = _encoded_frame(keyframe=True)
        result = resizer.resize(frame)
        assert result is frame

    def test_aspect_ratio_preserved(self) -> None:
        """Aspect ratio is preserved when keep_aspect=True."""
        import numpy as np

        from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer

        resizer = PyAVVideoResizer(width=32, height=32)
        # 64x32 -> target 32x32 with aspect -> 32x16.
        width, height = 64, 32
        data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
        frame = VideoFrame(
            data=data,
            codec="raw_rgb24",
            width=width,
            height=height,
        )

        result = resizer.resize(frame)
        assert result.width == 32
        # Height should be scaled proportionally (16, rounded to even).
        assert result.height == 16

    def test_invalid_dimensions(self) -> None:
        from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer

        with pytest.raises(ValueError, match="positive"):
            PyAVVideoResizer(width=0, height=480)

    def test_name(self) -> None:
        from roomkit.video.pipeline.resizer.pyav import PyAVVideoResizer

        assert PyAVVideoResizer().name == "PyAVVideoResizer"


# ---------------------------------------------------------------------------
# VideoPipelineConfig — new fields
# ---------------------------------------------------------------------------


class TestVideoPipelineConfigNewFields:
    def test_default_none(self) -> None:
        config = VideoPipelineConfig()
        assert config.decoder is None
        assert config.resizer is None
        assert config.vision is None

    def test_with_all_stages(self) -> None:
        decoder = MockVideoDecoderProvider()
        resizer = MockVideoResizerProvider()
        vision = MockVisionProvider()
        config = VideoPipelineConfig(
            decoder=decoder,
            resizer=resizer,
            vision=vision,
        )
        assert config.decoder is decoder
        assert config.resizer is resizer
        assert config.vision is vision
