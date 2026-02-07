"""Tests for SherpaOnnxVADProvider."""

from __future__ import annotations

import importlib
import struct
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.vad.base import VADEventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    amplitude: int = 0,
    n_samples: int = 320,
    sample_rate: int = 16000,
) -> AudioFrame:
    """Create an AudioFrame filled with a constant int16 value."""
    data = struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))
    return AudioFrame(data=data, sample_rate=sample_rate)


def _silence(n_samples: int = 320, sample_rate: int = 16000) -> AudioFrame:
    return _make_frame(amplitude=0, n_samples=n_samples, sample_rate=sample_rate)


def _speech(
    amplitude: int = 1000,
    n_samples: int = 320,
    sample_rate: int = 16000,
) -> AudioFrame:
    return _make_frame(amplitude=amplitude, n_samples=n_samples, sample_rate=sample_rate)


def _mock_sherpa_module() -> MagicMock:
    """Create a MagicMock that stands in for the sherpa_onnx module."""
    return MagicMock()


def _make_provider(
    sherpa_mock: MagicMock,
    detector_mock: MagicMock,
    **config_kwargs: Any,
) -> Any:
    """Create a SherpaOnnxVADProvider with mocked sherpa_onnx."""
    sherpa_mock.VoiceActivityDetector.return_value = detector_mock
    with patch.dict("sys.modules", {"sherpa_onnx": sherpa_mock}):
        import roomkit.voice.pipeline.vad.sherpa_onnx as vad_mod

        importlib.reload(vad_mod)
        from roomkit.voice.pipeline.vad.sherpa_onnx import (
            SherpaOnnxVADConfig,
            SherpaOnnxVADProvider,
        )

        cfg = SherpaOnnxVADConfig(**config_kwargs)
        provider = SherpaOnnxVADProvider(cfg)
        provider._sherpa = sherpa_mock
        return provider


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------


class TestBasicTransitions:
    def test_silence_produces_no_events(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = False
        detector.empty.return_value = True

        vad = _make_provider(sherpa, detector)

        for _ in range(20):
            assert vad.process(_silence()) is None

    def test_speech_start(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = True
        detector.empty.return_value = True

        vad = _make_provider(sherpa, detector)
        event = vad.process(_speech())

        assert event is not None
        assert event.type == VADEventType.SPEECH_START
        assert event.confidence == 1.0

    def test_speech_end_after_silence(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True

        # 1 speech frame, then silence
        speech_flags = [True] + [True] * 5 + [False] * 10
        detector.is_speech_detected.side_effect = speech_flags

        vad = _make_provider(
            sherpa,
            detector,
            silence_threshold_ms=100,
            min_speech_duration_ms=0,
            speech_pad_ms=0,
        )

        # Speech start
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START

        # Continue speaking
        for _ in range(5):
            assert vad.process(_speech()) is None

        # Silence until SPEECH_END
        end_event = None
        for _ in range(10):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        assert end_event.type == VADEventType.SPEECH_END
        assert end_event.audio_bytes is not None
        assert len(end_event.audio_bytes) > 0
        assert end_event.duration_ms is not None
        assert end_event.duration_ms > 0


# ---------------------------------------------------------------------------
# Audio accumulation
# ---------------------------------------------------------------------------


class TestAudioAccumulation:
    def test_accumulated_audio_contains_speech_frames(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True

        # 4 speech frames, then enough silence
        speech_flags = [True] * 4 + [False] * 10
        detector.is_speech_detected.side_effect = speech_flags

        vad = _make_provider(
            sherpa,
            detector,
            silence_threshold_ms=60,
            min_speech_duration_ms=0,
            speech_pad_ms=0,
        )

        # Start speech
        vad.process(_speech())
        # 3 more speech frames
        for _ in range(3):
            vad.process(_speech())

        # Silence until end
        end_event = None
        for _ in range(10):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        # At least 4 speech frames worth of data
        assert len(end_event.audio_bytes) >= 320 * 2 * 4


# ---------------------------------------------------------------------------
# Min speech duration filtering
# ---------------------------------------------------------------------------


class TestMinSpeechDuration:
    def test_short_speech_discarded(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True

        # 1 speech frame then silence
        speech_flags = [True] + [False] * 20
        detector.is_speech_detected.side_effect = speech_flags

        vad = _make_provider(
            sherpa,
            detector,
            silence_threshold_ms=40,
            min_speech_duration_ms=500,  # much longer than 1 frame
            speech_pad_ms=0,
        )

        # Single speech frame → SPEECH_START
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START

        # Immediate silence → no SPEECH_END (too short)
        events = []
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                events.append(ev)

        assert all(e.type != VADEventType.SPEECH_END for e in events)

    def test_long_enough_speech_emitted(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True

        # 11 speech frames then silence
        speech_flags = [True] * 11 + [False] * 20
        detector.is_speech_detected.side_effect = speech_flags

        vad = _make_provider(
            sherpa,
            detector,
            silence_threshold_ms=60,
            min_speech_duration_ms=100,  # 5 frames at 20ms
            speech_pad_ms=0,
        )

        # Start
        vad.process(_speech())
        # 10 more frames (220ms total > 100ms min)
        for _ in range(10):
            vad.process(_speech())

        end_event = None
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        assert end_event.type == VADEventType.SPEECH_END


# ---------------------------------------------------------------------------
# Pre-roll buffer
# ---------------------------------------------------------------------------


class TestPreRoll:
    def test_pre_roll_included_in_audio(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True

        # 10 silence frames, then 1 speech, then silence to end
        speech_flags = [False] * 10 + [True] + [False] * 20
        detector.is_speech_detected.side_effect = speech_flags

        vad = _make_provider(
            sherpa,
            detector,
            silence_threshold_ms=60,
            min_speech_duration_ms=0,
            speech_pad_ms=100,  # ~5 frames at 20ms
        )

        # Feed 10 silence frames (go into pre-roll)
        for _ in range(10):
            vad.process(_silence())

        # Now speech
        vad.process(_speech())

        # End with silence
        end_event = None
        for _ in range(20):
            ev = vad.process(_silence())
            if ev is not None:
                end_event = ev
                break

        assert end_event is not None
        # Audio should include pre-roll + speech + some silence
        assert len(end_event.audio_bytes) > 320 * 2  # more than just speech frame


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True
        detector.is_speech_detected.return_value = True

        vad = _make_provider(sherpa, detector)

        # Start speaking
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START

        # Reset mid-speech
        vad.reset()
        detector.reset.assert_called_once()

        # Should be back to idle — next speech triggers SPEECH_START again
        event = vad.process(_speech())
        assert event is not None
        assert event.type == VADEventType.SPEECH_START


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_flushes_detector(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True
        detector.is_speech_detected.return_value = False

        vad = _make_provider(sherpa, detector)
        # Trigger lazy init
        vad.process(_silence())

        vad.close()
        detector.flush.assert_called_once()
        assert vad._detector is None


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        vad = _make_provider(sherpa, detector)
        assert vad.name == "SherpaOnnxVAD"


# ---------------------------------------------------------------------------
# Multiple utterances
# ---------------------------------------------------------------------------


class TestMultipleUtterances:
    def test_two_utterances(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True

        # 320 samples at 16kHz = 20ms per frame
        # silence_threshold_ms=60 → need 3 silence frames to trigger SPEECH_END
        #
        # First utterance:  6 speech + 3 silence (SPEECH_END on 3rd)
        # Gap:              we break after SPEECH_END, no extra frames consumed
        # Second utterance: 6 speech + 3 silence (SPEECH_END on 3rd)
        speech_flags = (
            [True] * 6
            + [False] * 3  # first utterance: 9 flags
            + [True] * 6
            + [False] * 3  # second utterance: 9 flags
        )
        detector.is_speech_detected.side_effect = speech_flags

        vad = _make_provider(
            sherpa,
            detector,
            silence_threshold_ms=60,
            min_speech_duration_ms=0,
            speech_pad_ms=0,
        )

        events = []

        # First utterance
        events.append(vad.process(_speech()))
        for _ in range(5):
            vad.process(_speech())
        for _ in range(10):
            ev = vad.process(_silence())
            if ev is not None:
                events.append(ev)
                break

        # Second utterance
        events.append(vad.process(_speech()))
        for _ in range(5):
            vad.process(_speech())
        for _ in range(10):
            ev = vad.process(_silence())
            if ev is not None:
                events.append(ev)
                break

        types = [e.type for e in events if e is not None]
        assert types == [
            VADEventType.SPEECH_START,
            VADEventType.SPEECH_END,
            VADEventType.SPEECH_START,
            VADEventType.SPEECH_END,
        ]


# ---------------------------------------------------------------------------
# Lazy init
# ---------------------------------------------------------------------------


class TestLazyInit:
    def test_detector_not_created_until_first_process(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True
        detector.is_speech_detected.return_value = False

        vad = _make_provider(sherpa, detector)

        # Detector should not be created yet
        assert vad._detector is None

        # After first process() it should be created
        vad.process(_silence())
        assert vad._detector is not None


# ---------------------------------------------------------------------------
# Model type configuration
# ---------------------------------------------------------------------------


class TestModelType:
    def test_ten_vad_config(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True
        detector.is_speech_detected.return_value = False

        vad = _make_provider(
            sherpa,
            detector,
            model="/path/to/ten.onnx",
            model_type="ten",
            threshold=0.6,
            max_speech_duration=15.0,
        )

        # Trigger lazy init
        vad.process(_silence())

        # Check VadModelConfig was created and ten_vad was configured
        vad_model_config = sherpa.VadModelConfig.return_value
        assert vad_model_config.ten_vad.model == "/path/to/ten.onnx"
        assert vad_model_config.ten_vad.threshold == 0.6
        assert vad_model_config.ten_vad.max_speech_duration == 15.0
        assert vad_model_config.sample_rate == 16000

    def test_silero_vad_config(self) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True
        detector.is_speech_detected.return_value = False

        vad = _make_provider(
            sherpa,
            detector,
            model="/path/to/silero.onnx",
            model_type="silero",
            threshold=0.4,
            max_speech_duration=30.0,
        )

        # Trigger lazy init
        vad.process(_silence())

        # Check VadModelConfig was created and silero_vad was configured
        vad_model_config = sherpa.VadModelConfig.return_value
        assert vad_model_config.silero_vad.model == "/path/to/silero.onnx"
        assert vad_model_config.silero_vad.threshold == 0.4
        assert vad_model_config.silero_vad.max_speech_duration == 30.0
        assert vad_model_config.sample_rate == 16000


# ---------------------------------------------------------------------------
# Import error
# ---------------------------------------------------------------------------


class TestImportError:
    def test_import_error_when_sherpa_not_installed(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": None}):
            import roomkit.voice.pipeline.vad.sherpa_onnx as vad_mod

            importlib.reload(vad_mod)

            from roomkit.voice.pipeline.vad.sherpa_onnx import (
                SherpaOnnxVADConfig,
                SherpaOnnxVADProvider,
            )

            with pytest.raises(ImportError, match="sherpa-onnx is required"):
                SherpaOnnxVADProvider(SherpaOnnxVADConfig())


# ---------------------------------------------------------------------------
# PCM conversion helper
# ---------------------------------------------------------------------------


class TestPcmConversion:
    def test_silence(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.vad.sherpa_onnx as vad_mod

            importlib.reload(vad_mod)
            from roomkit.voice.pipeline.vad.sherpa_onnx import _pcm_s16le_to_float32

            pcm = struct.pack("<1h", 0)
            result = _pcm_s16le_to_float32(pcm)
            assert len(result) == 1
            assert abs(result[0]) < 1e-6

    def test_max_positive(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.vad.sherpa_onnx as vad_mod

            importlib.reload(vad_mod)
            from roomkit.voice.pipeline.vad.sherpa_onnx import _pcm_s16le_to_float32

            pcm = struct.pack("<1h", 32767)
            result = _pcm_s16le_to_float32(pcm)
            assert abs(result[0] - 1.0) < 0.001

    def test_max_negative(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.vad.sherpa_onnx as vad_mod

            importlib.reload(vad_mod)
            from roomkit.voice.pipeline.vad.sherpa_onnx import _pcm_s16le_to_float32

            pcm = struct.pack("<1h", -32768)
            result = _pcm_s16le_to_float32(pcm)
            assert abs(result[0] - (-1.0)) < 0.001


# ---------------------------------------------------------------------------
# Segment draining
# ---------------------------------------------------------------------------


class TestSegmentDraining:
    def test_completed_segments_are_drained(self) -> None:
        """Verify that completed segments are popped to prevent unbounded memory."""
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = False
        # Simulate: first call to empty() returns False (has segment),
        # pop() is called, second call returns True (no more segments)
        detector.empty.side_effect = [False, True]

        vad = _make_provider(sherpa, detector)
        vad.process(_silence())

        detector.pop.assert_called_once()
