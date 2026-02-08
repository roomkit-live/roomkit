"""Tests for VAD debug logging in SherpaOnnxVADProvider and EnergyVADProvider."""

from __future__ import annotations

import importlib
import logging
import struct
from typing import Any
from unittest.mock import MagicMock, patch

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.vad.energy import EnergyVADProvider

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
    return MagicMock()


def _make_sherpa_provider(
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
# SherpaOnnxVADProvider debug logging
# ---------------------------------------------------------------------------


class TestSherpaOnnxDebugLogging:
    def test_logs_summary_after_50_frames(self, caplog: Any) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = False
        detector.empty.return_value = True

        vad = _make_sherpa_provider(sherpa, detector)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.sherpa_onnx"):
            for _ in range(50):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "state=idle" in vad_messages[0]
        assert "is_speech=0/50" in vad_messages[0]

    def test_no_log_before_50_frames(self, caplog: Any) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = False
        detector.empty.return_value = True

        vad = _make_sherpa_provider(sherpa, detector)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.sherpa_onnx"):
            for _ in range(49):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 0

    def test_speech_count_tracked(self, caplog: Any) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True

        # 30 speech frames then 20 silence frames = 50 total
        speech_flags = [True] * 30 + [False] * 20
        detector.is_speech_detected.side_effect = speech_flags

        vad = _make_sherpa_provider(sherpa, detector, silence_threshold_ms=10000, speech_pad_ms=0)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.sherpa_onnx"):
            for i in range(50):
                frame = _speech() if i < 30 else _silence()
                vad.process(frame)

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "is_speech=30/50" in vad_messages[0]

    def test_rms_values_in_summary(self, caplog: Any) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = False
        detector.empty.return_value = True

        vad = _make_sherpa_provider(sherpa, detector)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.sherpa_onnx"):
            for _ in range(50):
                vad.process(_speech(amplitude=500))

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "rms_avg=500" in vad_messages[0]
        assert "rms_max=500" in vad_messages[0]

    def test_reset_clears_debug_counters(self, caplog: Any) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = False
        detector.empty.return_value = True

        vad = _make_sherpa_provider(sherpa, detector)

        # Feed 30 frames then reset
        for _ in range(30):
            vad.process(_silence())
        vad.reset()

        # Now feed 50 more — should log exactly once (not carry over from before)
        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.sherpa_onnx"):
            for _ in range(50):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "is_speech=0/50" in vad_messages[0]

    def test_speaking_state_shown(self, caplog: Any) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.empty.return_value = True
        # All speech so we stay in speaking state
        detector.is_speech_detected.return_value = True

        vad = _make_sherpa_provider(sherpa, detector, silence_threshold_ms=10000, speech_pad_ms=0)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.sherpa_onnx"):
            for _ in range(50):
                vad.process(_speech())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "state=speaking" in vad_messages[0]

    def test_multiple_summaries(self, caplog: Any) -> None:
        sherpa = _mock_sherpa_module()
        detector = MagicMock()
        detector.is_speech_detected.return_value = False
        detector.empty.return_value = True

        vad = _make_sherpa_provider(sherpa, detector)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.sherpa_onnx"):
            for _ in range(120):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 2


# ---------------------------------------------------------------------------
# EnergyVADProvider debug logging
# ---------------------------------------------------------------------------


class TestEnergyDebugLogging:
    def test_logs_summary_after_50_frames(self, caplog: Any) -> None:
        vad = EnergyVADProvider(energy_threshold=300)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.energy"):
            for _ in range(50):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "state=idle" in vad_messages[0]
        assert "is_speech=0/50" in vad_messages[0]

    def test_no_log_before_50_frames(self, caplog: Any) -> None:
        vad = EnergyVADProvider(energy_threshold=300)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.energy"):
            for _ in range(49):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 0

    def test_speech_count_tracked(self, caplog: Any) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=10000,
            speech_pad_ms=0,
        )

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.energy"):
            # 30 speech frames (amplitude 1000 > threshold 300)
            for _ in range(30):
                vad.process(_speech(amplitude=1000))
            # 20 silence frames
            for _ in range(20):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "is_speech=30/50" in vad_messages[0]

    def test_rms_values_in_summary(self, caplog: Any) -> None:
        vad = EnergyVADProvider(energy_threshold=300)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.energy"):
            for _ in range(50):
                vad.process(_speech(amplitude=500))

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "rms_avg=500" in vad_messages[0]
        assert "rms_max=500" in vad_messages[0]

    def test_reset_clears_debug_counters(self, caplog: Any) -> None:
        vad = EnergyVADProvider(energy_threshold=300)

        # Feed 30 frames then reset
        for _ in range(30):
            vad.process(_silence())
        vad.reset()

        # Now feed 50 more
        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.energy"):
            for _ in range(50):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "is_speech=0/50" in vad_messages[0]

    def test_speaking_state_shown(self, caplog: Any) -> None:
        vad = EnergyVADProvider(
            energy_threshold=300,
            silence_threshold_ms=10000,
            speech_pad_ms=0,
        )

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.energy"):
            for _ in range(50):
                vad.process(_speech(amplitude=1000))

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 1
        assert "state=speaking" in vad_messages[0]

    def test_multiple_summaries(self, caplog: Any) -> None:
        vad = EnergyVADProvider(energy_threshold=300)

        with caplog.at_level(logging.DEBUG, logger="roomkit.voice.pipeline.vad.energy"):
            for _ in range(120):
                vad.process(_silence())

        vad_messages = [r.message for r in caplog.records if "VAD: state=" in r.message]
        assert len(vad_messages) == 2
