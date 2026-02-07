"""Tests for WavFileRecorder."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.base import VoiceSession
from roomkit.voice.pipeline.recorder.base import (
    RecordingChannelMode,
    RecordingConfig,
    RecordingMode,
    RecordingTrigger,
)
from roomkit.voice.pipeline.recorder.wav import WavFileRecorder


def _frame(data: bytes = b"\x00\x00", sample_rate: int = 16000) -> AudioFrame:
    return AudioFrame(data=data, sample_rate=sample_rate, channels=1, sample_width=2)


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="c1")


def _pcm_silence(num_samples: int, sample_width: int = 2) -> bytes:
    """Generate silence PCM bytes."""
    return b"\x00" * (num_samples * sample_width)


def _pcm_tone(num_samples: int, value: int = 1000, sample_width: int = 2) -> bytes:
    """Generate constant-value PCM bytes (16-bit signed)."""
    fmt = "<h" if sample_width == 2 else "<b"
    return b"".join(struct.pack(fmt, value) for _ in range(num_samples))


class TestWavFileRecorderBasic:
    def test_name(self) -> None:
        recorder = WavFileRecorder()
        assert recorder.name == "WavFileRecorder"

    def test_start_returns_handle(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(storage=str(tmp_path))
        handle = recorder.start(_session(), config)

        assert handle.id
        assert handle.session_id == "s1"
        assert handle.state == "recording"
        assert handle.started_at is not None
        assert handle.path

    def test_stop_without_data_returns_empty_result(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(storage=str(tmp_path))
        handle = recorder.start(_session(), config)
        result = recorder.stop(handle)

        assert result.id == handle.id
        assert result.duration_seconds == 0.0
        assert result.urls == []

    def test_stop_unknown_handle(self) -> None:
        from roomkit.voice.pipeline.recorder.base import RecordingHandle

        recorder = WavFileRecorder()
        handle = RecordingHandle(id="nonexistent", session_id="s1")
        result = recorder.stop(handle)
        assert result.id == "nonexistent"
        assert result.urls == []

    def test_file_naming_contains_session_id(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(storage=str(tmp_path))
        handle = recorder.start(_session("my-session"), config)
        assert "my-session" in handle.path

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "sub" / "dir"
        recorder = WavFileRecorder()
        config = RecordingConfig(storage=str(out_dir))
        recorder.start(_session(), config)
        assert out_dir.exists()

    def test_default_storage_uses_tempdir(self) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig()
        handle = recorder.start(_session(), config)
        # Should not raise, path should be set
        assert handle.path
        recorder.stop(handle)


class TestMixedMode:
    def test_inbound_only(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        data = _pcm_tone(100, value=500)
        recorder.tap_inbound(handle, _frame(data))
        result = recorder.stop(handle)

        assert len(result.urls) == 1
        assert result.duration_seconds > 0
        assert result.size_bytes > 0
        assert result.mode == RecordingChannelMode.MIXED

        # Verify WAV is readable
        with wave.open(result.urls[0], "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 16000
            assert w.getnframes() == 100

    def test_mixed_averages_both_directions(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        inbound = _pcm_tone(10, value=1000)
        outbound = _pcm_tone(10, value=500)
        recorder.tap_inbound(handle, _frame(inbound))
        recorder.tap_outbound(handle, _frame(outbound))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            frames = w.readframes(10)

        # Average of 1000 and 500 = 750
        for i in range(10):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 750

    def test_mixed_pads_shorter_buffer(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        inbound = _pcm_tone(20, value=1000)
        outbound = _pcm_tone(10, value=500)
        recorder.tap_inbound(handle, _frame(inbound))
        recorder.tap_outbound(handle, _frame(outbound))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            assert w.getnframes() == 20
            frames = w.readframes(20)

        # First 10 samples: (1000 + 500) / 2 = 750
        for i in range(10):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 750

        # Last 10 samples: (1000 + 0) / 2 = 500
        for i in range(10, 20):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 500


class TestSeparateMode:
    def test_creates_two_files(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.SEPARATE,
        )
        handle = recorder.start(_session(), config)

        data = _pcm_tone(50)
        recorder.tap_inbound(handle, _frame(data))
        recorder.tap_outbound(handle, _frame(data))
        result = recorder.stop(handle)

        assert len(result.urls) == 2
        assert any("inbound" in u for u in result.urls)
        assert any("outbound" in u for u in result.urls)
        assert result.mode == RecordingChannelMode.SEPARATE

        for url in result.urls:
            with wave.open(url, "rb") as w:
                assert w.getnchannels() == 1
                assert w.getnframes() == 50

    def test_inbound_only_creates_one_file(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.SEPARATE,
            mode=RecordingMode.INBOUND_ONLY,
        )
        handle = recorder.start(_session(), config)

        recorder.tap_inbound(handle, _frame(_pcm_tone(50)))
        recorder.tap_outbound(handle, _frame(_pcm_tone(50)))  # should be ignored
        result = recorder.stop(handle)

        assert len(result.urls) == 1
        assert "inbound" in result.urls[0]


class TestStereoMode:
    def test_stereo_wav_has_two_channels(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.STEREO,
        )
        handle = recorder.start(_session(), config)

        inbound = _pcm_tone(10, value=1000)
        outbound = _pcm_tone(10, value=500)
        recorder.tap_inbound(handle, _frame(inbound))
        recorder.tap_outbound(handle, _frame(outbound))
        result = recorder.stop(handle)

        assert len(result.urls) == 1
        assert result.mode == RecordingChannelMode.STEREO

        with wave.open(result.urls[0], "rb") as w:
            assert w.getnchannels() == 2
            assert w.getnframes() == 10
            frames = w.readframes(10)

        # Verify interleaving: L=inbound, R=outbound
        for i in range(10):
            left = struct.unpack_from("<h", frames, i * 4)[0]
            right = struct.unpack_from("<h", frames, i * 4 + 2)[0]
            assert left == 1000
            assert right == 500

    def test_stereo_pads_shorter_buffer(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.STEREO,
        )
        handle = recorder.start(_session(), config)

        recorder.tap_inbound(handle, _frame(_pcm_tone(20, value=100)))
        recorder.tap_outbound(handle, _frame(_pcm_tone(10, value=200)))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            assert w.getnframes() == 20
            frames = w.readframes(20)

        # Last 10 right-channel samples should be 0 (silence padding)
        for i in range(10, 20):
            right = struct.unpack_from("<h", frames, i * 4 + 2)[0]
            assert right == 0


class TestRecordingMode:
    def test_inbound_only_ignores_outbound(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            mode=RecordingMode.INBOUND_ONLY,
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        recorder.tap_inbound(handle, _frame(_pcm_tone(50, value=100)))
        recorder.tap_outbound(handle, _frame(_pcm_tone(50, value=200)))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            frames = w.readframes(50)

        # Should only have inbound data (value=100), outbound was ignored
        # In MIXED mode with only inbound, the outbound buffer is empty (zeros)
        # so mix = (100 + 0) / 2 = 50
        for i in range(50):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 50

    def test_outbound_only_ignores_inbound(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            mode=RecordingMode.OUTBOUND_ONLY,
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        recorder.tap_inbound(handle, _frame(_pcm_tone(50, value=100)))
        recorder.tap_outbound(handle, _frame(_pcm_tone(50, value=200)))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            frames = w.readframes(50)

        # Only outbound (200), inbound ignored → mix = (0 + 200) / 2 = 100
        for i in range(50):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 100


class TestSpeechOnlyWarning:
    def test_speech_only_trigger_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            trigger=RecordingTrigger.SPEECH_ONLY,
        )
        with caplog.at_level("WARNING", logger="roomkit.voice.pipeline.recorder.wav"):
            recorder.start(_session(), config)

        assert "SPEECH_ONLY" in caplog.text


class TestDuration:
    def test_duration_calculation(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        # 16000 samples at 16kHz = 1.0 second
        data = _pcm_silence(16000)
        recorder.tap_inbound(handle, _frame(data))
        result = recorder.stop(handle)

        assert result.duration_seconds == pytest.approx(1.0)

    def test_duration_separate_mode(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.SEPARATE,
        )
        handle = recorder.start(_session(), config)

        # Inbound: 16000 samples = 1s; outbound: 8000 samples = 0.5s
        recorder.tap_inbound(handle, _frame(_pcm_silence(16000)))
        recorder.tap_outbound(handle, _frame(_pcm_silence(8000)))
        result = recorder.stop(handle)

        # Duration should be the longer one
        assert result.duration_seconds == pytest.approx(1.0)


class TestResetAndClose:
    def test_reset_stops_active_sessions(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.SEPARATE,
        )
        handle = recorder.start(_session(), config)
        recorder.tap_inbound(handle, _frame(_pcm_tone(50)))
        recorder.reset()

        # Session should be gone
        assert handle.id not in recorder._sessions

    def test_close_stops_all(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(storage=str(tmp_path))
        recorder.start(_session("s1"), config)
        recorder.start(_session("s2"), config)
        recorder.close()
        assert len(recorder._sessions) == 0


class TestImport:
    def test_import_from_pipeline(self) -> None:
        from roomkit.voice.pipeline import WavFileRecorder as W

        assert W is WavFileRecorder

    def test_import_from_voice(self) -> None:
        from roomkit.voice import WavFileRecorder as W

        assert W is WavFileRecorder
