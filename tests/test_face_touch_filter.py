"""Tests for face touch detection filter, mock, and pipeline integration."""

from __future__ import annotations

import pytest

from roomkit.video.events import VideoDetectionEvent
from roomkit.video.pipeline import (
    FilterContext,
    FilterEvent,
    VideoPipeline,
    VideoPipelineConfig,
)
from roomkit.video.pipeline.filter.mediapipe_face_touch import (
    _SENSITIVITY_PARAMS,
    FaceTouchConfig,
    FaceTouchFilter,
    FaceTouchSensitivity,
    FaceZone,
)
from roomkit.video.pipeline.filter.mock_face_touch import MockFaceTouchFilter
from roomkit.video.video_frame import VideoFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_frame(seq: int = 0, width: int = 640, height: int = 480) -> VideoFrame:
    """Create a raw RGB24 frame for testing."""
    data = b"\x00" * (width * height * 3)
    return VideoFrame(
        data=data,
        codec="raw_rgb24",
        width=width,
        height=height,
        sequence=seq,
    )


def _detection_event(
    zone: str = "left_cheek",
    hand: str = "right",
    confidence: float = 0.85,
) -> VideoDetectionEvent:
    """Create a test VideoDetectionEvent."""
    return VideoDetectionEvent(
        kind="face_touch",
        labels=[zone],
        confidence=confidence,
        metadata={"zone": zone, "hand": hand},
    )


# ---------------------------------------------------------------------------
# FaceTouchConfig tests
# ---------------------------------------------------------------------------


class TestFaceTouchConfig:
    def test_default_config(self) -> None:
        config = FaceTouchConfig()
        assert config.sensitivity == FaceTouchSensitivity.MEDIUM
        assert config.every_n_frames == 3
        assert FaceZone.LEFT_CHEEK in config.zones
        assert FaceZone.FOREHEAD not in config.zones

    def test_sensitivity_presets_resolve(self) -> None:
        for sensitivity in FaceTouchSensitivity:
            config = FaceTouchConfig(sensitivity=sensitivity)
            params = _SENSITIVITY_PARAMS[sensitivity]
            assert config.resolved_touch_distance == params["touch_distance_threshold"]
            assert config.resolved_confirmation == params["confirmation_frames"]
            assert config.resolved_cooldown == params["cooldown_frames"]
            assert config.resolved_z_depth == params["z_depth_threshold"]

    def test_explicit_overrides_take_precedence(self) -> None:
        config = FaceTouchConfig(
            sensitivity=FaceTouchSensitivity.LOW,
            touch_distance_threshold=0.99,
            confirmation_frames=10,
        )
        assert config.resolved_touch_distance == 0.99
        assert config.resolved_confirmation == 10
        # Non-overridden params still use preset
        assert (
            config.resolved_cooldown
            == _SENSITIVITY_PARAMS[FaceTouchSensitivity.LOW]["cooldown_frames"]
        )

    def test_custom_zones(self) -> None:
        config = FaceTouchConfig(zones=frozenset({FaceZone.FOREHEAD, FaceZone.CHIN}))
        assert FaceZone.FOREHEAD in config.zones
        assert FaceZone.LEFT_CHEEK not in config.zones


# ---------------------------------------------------------------------------
# FaceTouchFilter construction tests
# ---------------------------------------------------------------------------


class TestFaceTouchFilterConstruction:
    def test_default_construction(self) -> None:
        f = FaceTouchFilter()
        assert f.name == "face_touch"

    def test_invalid_every_n_frames(self) -> None:
        with pytest.raises(ValueError, match="every_n_frames must be >= 1"):
            FaceTouchFilter(FaceTouchConfig(every_n_frames=0))

    def test_reset_clears_state(self) -> None:
        f = FaceTouchFilter()
        f._frame_count = 42
        f._logged_first = True
        f._sessions["test"] = object()  # type: ignore[assignment]
        f.reset()
        assert f._frame_count == 0
        assert not f._logged_first
        assert len(f._sessions) == 0

    def test_close_without_models(self) -> None:
        """Close should not error if models were never loaded."""
        f = FaceTouchFilter()
        f.close()  # should be a no-op

    def test_every_n_frames_throttling(self) -> None:
        """Verify filter skips frames based on every_n_frames."""
        f = FaceTouchFilter(FaceTouchConfig(every_n_frames=3))
        ctx = FilterContext()

        # Track which frames would trigger _detect via frame_count
        detected_frames = []

        def spy_detect(frame: VideoFrame, context: FilterContext) -> None:
            detected_frames.append(f._frame_count)
            # Don't actually run MediaPipe
            return

        f._detect = spy_detect  # type: ignore[assignment]

        for i in range(9):
            f.filter(_raw_frame(seq=i), ctx)

        # With every_n_frames=3, detection runs at frame_count 1, 4, 7
        assert detected_frames == [1, 4, 7]

    def test_every_n_frames_one_runs_every_frame(self) -> None:
        """With every_n_frames=1, detection runs on every frame."""
        f = FaceTouchFilter(FaceTouchConfig(every_n_frames=1))
        ctx = FilterContext()

        detected_frames: list[int] = []

        def spy_detect(frame: VideoFrame, context: FilterContext) -> None:
            detected_frames.append(f._frame_count)

        f._detect = spy_detect  # type: ignore[assignment]

        for i in range(5):
            f.filter(_raw_frame(seq=i), ctx)

        assert detected_frames == [1, 2, 3, 4, 5]

    def test_throttle_resets_after_reset(self) -> None:
        """After reset(), throttle counter restarts from zero."""
        f = FaceTouchFilter(FaceTouchConfig(every_n_frames=3))
        ctx = FilterContext()

        detected_frames: list[int] = []

        def spy_detect(frame: VideoFrame, context: FilterContext) -> None:
            detected_frames.append(f._frame_count)

        f._detect = spy_detect  # type: ignore[assignment]

        # Process 3 frames: should detect at frame_count 1 only
        for i in range(3):
            f.filter(_raw_frame(seq=i), ctx)
        assert detected_frames == [1]

        # Reset and process again — should detect at frame_count 1 again
        f.reset()
        f._detect = spy_detect  # type: ignore[assignment]
        detected_frames.clear()

        for i in range(3):
            f.filter(_raw_frame(seq=i + 10), ctx)
        assert detected_frames == [1]


# ---------------------------------------------------------------------------
# MockFaceTouchFilter tests
# ---------------------------------------------------------------------------


class TestMockFaceTouchFilter:
    def test_name(self) -> None:
        mock = MockFaceTouchFilter()
        assert mock.name == "mock_face_touch"

    def test_emits_at_configured_sequences(self) -> None:
        event = _detection_event()
        mock = MockFaceTouchFilter(events_at={3: [event], 7: [event]})
        ctx = FilterContext()

        for seq in range(10):
            frame = _raw_frame(seq=seq)
            result = mock.filter(frame, ctx)
            assert result is frame  # never modifies frame

        # Should have emitted 2 events
        assert len(ctx.events) == 2
        assert all(e.kind == "face_touch" for e in ctx.events)

    def test_no_events_without_configuration(self) -> None:
        mock = MockFaceTouchFilter()
        ctx = FilterContext()
        mock.filter(_raw_frame(seq=0), ctx)
        assert len(ctx.events) == 0

    def test_multiple_events_at_same_frame(self) -> None:
        e1 = _detection_event(zone="left_cheek")
        e2 = _detection_event(zone="right_cheek")
        mock = MockFaceTouchFilter(events_at={5: [e1, e2]})
        ctx = FilterContext()

        for seq in range(6):
            mock.filter(_raw_frame(seq=seq), ctx)

        assert len(ctx.events) == 2
        assert ctx.events[0].data.labels == ["left_cheek"]
        assert ctx.events[1].data.labels == ["right_cheek"]

    def test_reset(self) -> None:
        mock = MockFaceTouchFilter()
        mock._frame_count = 10
        mock.reset()
        assert mock._frame_count == 0


# ---------------------------------------------------------------------------
# FilterContext events field tests
# ---------------------------------------------------------------------------


class TestFilterContextEvents:
    def test_events_default_empty(self) -> None:
        ctx = FilterContext()
        assert ctx.events == []

    def test_events_appendable(self) -> None:
        ctx = FilterContext()
        ctx.events.append(FilterEvent(kind="test", data=42))
        assert len(ctx.events) == 1
        assert ctx.events[0].kind == "test"
        assert ctx.events[0].data == 42

    def test_events_independent_per_context(self) -> None:
        ctx1 = FilterContext()
        ctx2 = FilterContext()
        ctx1.events.append(FilterEvent(kind="a"))
        assert len(ctx2.events) == 0


# ---------------------------------------------------------------------------
# VideoPipeline drain_events integration tests
# ---------------------------------------------------------------------------


class TestPipelineDrainEvents:
    def test_drain_empty_without_events(self) -> None:
        pipeline = VideoPipeline(VideoPipelineConfig())
        events = pipeline.drain_events("session-1")
        assert events == []

    def test_drain_returns_and_clears_events(self) -> None:
        event = _detection_event()
        mock = MockFaceTouchFilter(events_at={0: [event]})
        pipeline = VideoPipeline(VideoPipelineConfig(filters=[mock]))

        frame = _raw_frame(seq=0)
        pipeline.process_inbound("session-1", frame)

        events = pipeline.drain_events("session-1")
        assert len(events) == 1
        assert events[0].kind == "face_touch"
        assert events[0].data.labels == ["left_cheek"]

        # Second drain should return empty
        assert pipeline.drain_events("session-1") == []

    def test_drain_unknown_session(self) -> None:
        pipeline = VideoPipeline(VideoPipelineConfig(filters=[MockFaceTouchFilter()]))
        assert pipeline.drain_events("nonexistent") == []

    def test_events_accumulate_across_frames(self) -> None:
        e1 = _detection_event(zone="left_cheek")
        e2 = _detection_event(zone="chin")
        mock = MockFaceTouchFilter(events_at={0: [e1], 2: [e2]})
        pipeline = VideoPipeline(VideoPipelineConfig(filters=[mock]))

        for seq in range(3):
            pipeline.process_inbound("s1", _raw_frame(seq=seq))

        events = pipeline.drain_events("s1")
        assert len(events) == 2

    def test_reset_clears_events(self) -> None:
        event = _detection_event()
        mock = MockFaceTouchFilter(events_at={0: [event]})
        pipeline = VideoPipeline(VideoPipelineConfig(filters=[mock]))

        pipeline.process_inbound("s1", _raw_frame(seq=0))
        pipeline.reset("s1")

        assert pipeline.drain_events("s1") == []


# ---------------------------------------------------------------------------
# VideoDetectionEvent model tests
# ---------------------------------------------------------------------------


class TestVideoDetectionEvent:
    def test_construction(self) -> None:
        event = VideoDetectionEvent(
            kind="face_touch",
            labels=["left_cheek"],
            confidence=0.9,
            metadata={"zone": "left_cheek", "hand": "right", "distance": 0.03},
            frame_sequence=42,
        )
        assert event.kind == "face_touch"
        assert event.labels == ["left_cheek"]
        assert event.confidence == 0.9
        assert event.metadata["zone"] == "left_cheek"
        assert event.frame_sequence == 42
        assert event.timestamp is not None

    def test_defaults(self) -> None:
        event = VideoDetectionEvent(kind="object")
        assert event.labels == []
        assert event.confidence == 0.0
        assert event.metadata == {}
        assert event.frame_sequence == 0
