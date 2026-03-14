"""Tests for the video transform pipeline stage."""

from __future__ import annotations

import pytest

from roomkit.video.pipeline import (
    MockVideoTransformProvider,
    VideoPipeline,
    VideoPipelineConfig,
)
from roomkit.video.pipeline.filter.mock import MockVideoFilterProvider
from roomkit.video.video_frame import VideoFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_frame(
    width: int = 64,
    height: int = 48,
    codec: str = "raw_rgb24",
    seq: int = 0,
) -> VideoFrame:
    """Create a raw pixel frame for testing."""
    data = b"\x80" * (width * height * 3)
    return VideoFrame(
        data=data,
        codec=codec,
        width=width,
        height=height,
        sequence=seq,
        timestamp_ms=0.0,
    )


def _encoded_frame(seq: int = 0) -> VideoFrame:
    """Create an encoded frame stub for testing."""
    return VideoFrame(
        data=b"\x00\x00\x01" + b"\x65" * 100,
        codec="h264",
        width=640,
        height=480,
        keyframe=True,
        sequence=seq,
    )


# ---------------------------------------------------------------------------
# MockVideoTransformProvider
# ---------------------------------------------------------------------------


class TestMockTransform:
    def test_passthrough(self) -> None:
        txf = MockVideoTransformProvider()
        frame = _raw_frame()
        result = txf.transform(frame)
        assert result is frame
        assert txf.call_count == 1
        assert txf.last_frame is frame

    def test_name(self) -> None:
        assert MockVideoTransformProvider().name == "mock"

    def test_reset(self) -> None:
        txf = MockVideoTransformProvider()
        txf.transform(_raw_frame())
        txf.reset()
        assert txf.call_count == 0
        assert txf.last_frame is None


# ---------------------------------------------------------------------------
# VideoEffectTransform — requires cv2
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _require_cv2() -> None:
    pytest.importorskip("cv2", reason="OpenCV not installed")
    pytest.importorskip("numpy", reason="numpy not installed")


EFFECTS = [
    "grayscale",
    "sepia",
    "invert",
    "blur",
    "cartoon",
    "edges",
    "sketch",
    "pixelate",
]


class TestVideoEffectTransform:
    @pytest.fixture(autouse=True)
    def _skip_without_cv2(self, _require_cv2: None) -> None:
        pass

    @pytest.mark.parametrize("effect", EFFECTS)
    def test_effect_produces_valid_output(self, effect: str) -> None:
        from roomkit.video.pipeline.transform.effects import VideoEffectTransform

        txf = VideoEffectTransform(effect=effect)
        frame = _raw_frame(width=64, height=48)
        result = txf.transform(frame)

        assert result.codec == "raw_rgb24"
        assert result.width == 64
        assert result.height == 48
        expected_size = 64 * 48 * 3
        assert len(result.data) == expected_size
        assert result.metadata.get("transform") == effect

    @pytest.mark.parametrize("effect", EFFECTS)
    def test_effect_preserves_frame_metadata(self, effect: str) -> None:
        from roomkit.video.pipeline.transform.effects import VideoEffectTransform

        txf = VideoEffectTransform(effect=effect)
        frame = _raw_frame(seq=42)
        result = txf.transform(frame)
        assert result.sequence == 42
        assert result.timestamp_ms == 0.0

    def test_name_includes_effect(self) -> None:
        from roomkit.video.pipeline.transform.effects import VideoEffectTransform

        txf = VideoEffectTransform(effect="blur")
        assert txf.name == "effect:blur"

    def test_non_raw_frame_passthrough(self) -> None:
        """Encoded frames pass through unchanged."""
        from roomkit.video.pipeline.transform.effects import VideoEffectTransform

        txf = VideoEffectTransform(effect="grayscale")
        frame = _encoded_frame()
        result = txf.transform(frame)
        assert result is frame


class TestVideoEffectTransformValidation:
    @pytest.fixture(autouse=True)
    def _skip_without_cv2(self, _require_cv2: None) -> None:
        pass

    def test_invalid_effect_raises(self) -> None:
        from roomkit.video.pipeline.transform.effects import VideoEffectTransform

        with pytest.raises(ValueError, match="Unknown effect"):
            VideoEffectTransform(effect="hologram")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineWithTransforms:
    def test_transform_runs_in_pipeline(self) -> None:
        """Transforms run between resizer and filters."""
        txf = MockVideoTransformProvider()
        flt = MockVideoFilterProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(transforms=[txf], filters=[flt]))
        frame = _raw_frame()
        result = pipeline.process_inbound("s1", frame)

        assert result is frame
        assert txf.call_count == 1
        assert flt.call_count == 1

    def test_multiple_transforms_chained(self) -> None:
        """Multiple transforms are applied in order."""
        txf1 = MockVideoTransformProvider()
        txf2 = MockVideoTransformProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(transforms=[txf1, txf2]))
        frame = _raw_frame()
        pipeline.process_inbound("s1", frame)

        assert txf1.call_count == 1
        assert txf2.call_count == 1

    def test_transform_on_encoded_frame(self) -> None:
        """Encoded frames still pass through transforms (transform decides)."""
        txf = MockVideoTransformProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(transforms=[txf]))
        frame = _encoded_frame()
        result = pipeline.process_inbound("s1", frame)
        assert result is frame
        # Mock transform passes through regardless of codec
        assert txf.call_count == 1

    def test_reset_resets_transforms(self) -> None:
        txf = MockVideoTransformProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(transforms=[txf]))
        pipeline.process_inbound("s1", _raw_frame())
        assert txf.call_count == 1
        pipeline.reset("s1")
        assert txf.call_count == 0

    def test_close_closes_transforms(self) -> None:
        """close() should not raise when transforms are configured."""
        txf = MockVideoTransformProvider()
        pipeline = VideoPipeline(VideoPipelineConfig(transforms=[txf]))
        pipeline.close()


class TestPipelineConfigWithTransforms:
    def test_default_empty(self) -> None:
        config = VideoPipelineConfig()
        assert config.transforms == []

    def test_transforms_field_order(self) -> None:
        """Transforms field exists alongside other fields."""
        txf = MockVideoTransformProvider()
        flt = MockVideoFilterProvider()
        config = VideoPipelineConfig(transforms=[txf], filters=[flt])
        assert config.transforms == [txf]
        assert config.filters == [flt]
