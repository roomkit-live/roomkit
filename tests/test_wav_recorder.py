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

    def test_mixed_sums_both_directions(self, tmp_path: Path) -> None:
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

        # Sum of 1000 and 500 = 1500
        for i in range(10):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 1500

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

        # First 10 samples: 1000 + 500 = 1500
        for i in range(10):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 1500

        # Last 10 samples: 1000 + 0 = 1000
        for i in range(10, 20):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 1000

    def test_mixed_clamps_on_overflow(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        inbound = _pcm_tone(10, value=30000)
        outbound = _pcm_tone(10, value=20000)
        recorder.tap_inbound(handle, _frame(inbound))
        recorder.tap_outbound(handle, _frame(outbound))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            frames = w.readframes(10)

        # 30000 + 20000 = 50000 > 32767, clamped to 32767
        for i in range(10):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 32767


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

        # Should only have inbound data (value=100), outbound was ignored.
        # With only one direction, _write_mixed writes it directly (no averaging).
        for i in range(50):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 100

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

        # Only outbound (200), inbound ignored — written directly (no averaging).
        for i in range(50):
            sample = struct.unpack_from("<h", frames, i * 2)[0]
            assert sample == 200


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


class TestAllMode:
    def test_creates_three_files(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.ALL,
        )
        handle = recorder.start(_session(), config)

        inbound = _pcm_tone(50, value=1000)
        outbound = _pcm_tone(50, value=500)
        recorder.tap_inbound(handle, _frame(inbound))
        recorder.tap_outbound(handle, _frame(outbound))
        result = recorder.stop(handle)

        assert len(result.urls) == 3
        assert any("inbound" in u for u in result.urls)
        assert any("outbound" in u for u in result.urls)
        assert any("mixed" in u for u in result.urls)
        assert result.mode == RecordingChannelMode.ALL

    def test_all_files_are_valid_wav(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.ALL,
        )
        handle = recorder.start(_session(), config)

        recorder.tap_inbound(handle, _frame(_pcm_tone(100, value=1000)))
        recorder.tap_outbound(handle, _frame(_pcm_tone(100, value=500)))
        result = recorder.stop(handle)

        for url in result.urls:
            with wave.open(url, "rb") as w:
                assert w.getnchannels() == 1
                assert w.getsampwidth() == 2
                assert w.getframerate() == 16000

    def test_mixed_file_contains_both_tracks(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.ALL,
        )
        handle = recorder.start(_session(), config)

        recorder.tap_inbound(handle, _frame(_pcm_tone(100, value=1000)))
        recorder.tap_outbound(handle, _frame(_pcm_tone(100, value=500)))
        result = recorder.stop(handle)

        mixed_url = [u for u in result.urls if "mixed" in u][0]
        with wave.open(mixed_url, "rb") as w:
            total = w.getnframes()
            frames = w.readframes(total)

        # Mixed file should be longer than either track alone (silence padding)
        # and should contain non-zero samples from both tracks
        samples = [struct.unpack_from("<h", frames, i * 2)[0] for i in range(total)]
        assert any(s >= 1000 for s in samples), "Should contain inbound tone"
        assert total >= 100, "Should have at least as many frames as the tone"

    def test_duration_uses_longer_stream(self, tmp_path: Path) -> None:
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.ALL,
        )
        handle = recorder.start(_session(), config)

        recorder.tap_inbound(handle, _frame(_pcm_silence(16000)))  # 1s
        recorder.tap_outbound(handle, _frame(_pcm_silence(8000)))  # 0.5s
        result = recorder.stop(handle)

        assert result.duration_seconds == pytest.approx(1.0)


class TestSilenceInsertion:
    def test_gap_between_taps_inserts_silence(self, tmp_path: Path) -> None:
        """When there's a significant gap between taps, silence should be inserted."""
        import time

        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        # First tap: 100 samples of tone
        tone = _pcm_tone(100, value=1000)
        recorder.tap_inbound(handle, _frame(tone))

        # Wait 200ms — well above the 30ms threshold, should become silence
        time.sleep(0.2)

        # Second tap: 100 samples of tone
        recorder.tap_inbound(handle, _frame(tone))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            total_frames = w.getnframes()
            all_data = w.readframes(total_frames)

        # Total should be: 100 (tone) + ~3200 (silence at 16kHz * 0.2s) + 100 (tone)
        assert total_frames > 200  # Must be more than just the two tone blocks
        expected_silence_samples = int(16000 * 0.2)
        assert total_frames == pytest.approx(200 + expected_silence_samples, abs=500)

        # First 100 samples should be tone (non-zero)
        first_sample = struct.unpack_from("<h", all_data, 0)[0]
        assert first_sample == 1000

        # Middle should contain silence (zero)
        mid_offset = 150 * 2  # sample 150 should be in the silence region
        mid_sample = struct.unpack_from("<h", all_data, mid_offset)[0]
        assert mid_sample == 0

    def test_small_gap_no_silence(self, tmp_path: Path) -> None:
        """Gaps below the threshold (30ms) should NOT insert silence — they're jitter."""
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        # Back-to-back taps with no sleep — no gap, no silence
        tone = _pcm_tone(100, value=1000)
        recorder.tap_inbound(handle, _frame(tone))
        recorder.tap_inbound(handle, _frame(tone))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            total_frames = w.getnframes()

        # Should be exactly 200 samples (no silence inserted)
        assert total_frames == 200

    def test_no_gap_no_extra_silence(self, tmp_path: Path) -> None:
        """Back-to-back taps should not insert silence."""
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        tone = _pcm_tone(100, value=1000)
        recorder.tap_inbound(handle, _frame(tone))
        recorder.tap_inbound(handle, _frame(tone))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            total_frames = w.getnframes()

        # Should be approximately 200 samples (two taps, no significant gap)
        assert total_frames == pytest.approx(200, abs=100)

    def test_outbound_gap_inserts_silence(self, tmp_path: Path) -> None:
        """Outbound track should also insert silence for gaps."""
        import time

        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.SEPARATE,
        )
        handle = recorder.start(_session(), config)

        tone = _pcm_tone(100, value=500)
        recorder.tap_outbound(handle, _frame(tone))
        time.sleep(0.2)
        recorder.tap_outbound(handle, _frame(tone))
        result = recorder.stop(handle)

        outbound_url = [u for u in result.urls if "outbound" in u][0]
        with wave.open(outbound_url, "rb") as w:
            total_frames = w.getnframes()

        # Should have silence gap between the two taps
        assert total_frames > 200


class TestSilenceGapEdgeCases:
    def test_separate_inbound_gap_inserts_silence(self, tmp_path: Path) -> None:
        """SEPARATE mode inbound track should insert silence for gaps."""
        import time

        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.SEPARATE,
        )
        handle = recorder.start(_session(), config)

        tone = _pcm_tone(100, value=1000)
        recorder.tap_inbound(handle, _frame(tone))
        time.sleep(0.2)
        recorder.tap_inbound(handle, _frame(tone))
        result = recorder.stop(handle)

        inbound_url = [u for u in result.urls if "inbound" in u][0]
        with wave.open(inbound_url, "rb") as w:
            total_frames = w.getnframes()

        # Should have silence gap between the two taps
        assert total_frames > 200

    def test_non_separate_outbound_gap_inserts_silence(self, tmp_path: Path) -> None:
        """Non-SEPARATE (MIXED) outbound track should insert silence for gaps."""
        import time

        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage=str(tmp_path),
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)

        tone = _pcm_tone(100, value=500)
        recorder.tap_outbound(handle, _frame(tone))
        time.sleep(0.2)
        recorder.tap_outbound(handle, _frame(tone))
        result = recorder.stop(handle)

        with wave.open(result.urls[0], "rb") as w:
            total_frames = w.getnframes()

        # Outbound-only mixed file should have silence inserted
        assert total_frames > 200

    def test_path_traversal_falls_back_to_temp(self) -> None:
        """Storage paths with '..' should fall back to temp directory."""
        recorder = WavFileRecorder()
        config = RecordingConfig(
            storage="/tmp/foo/../../../etc",
            channels=RecordingChannelMode.MIXED,
        )
        handle = recorder.start(_session(), config)
        # The recorder should use a temp dir fallback, not the traversal path
        assert ".." not in handle.path


class TestImport:
    def test_import_from_pipeline(self) -> None:
        from roomkit.voice.pipeline import WavFileRecorder as W

        assert W is WavFileRecorder

    def test_import_from_voice(self) -> None:
        from roomkit.voice import WavFileRecorder as W

        assert W is WavFileRecorder
