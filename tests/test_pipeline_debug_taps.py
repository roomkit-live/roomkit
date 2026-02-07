"""Tests for PipelineDebugTaps (RFC §12.3.15)."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.pipeline.debug_taps import (
    ALL_STAGES,
    DebugTapSession,
    PipelineDebugTaps,
    _DebugWavWriter,
)
from roomkit.voice.pipeline.engine import AudioPipeline


def _make_frame(
    n_samples: int = 160,
    sample_rate: int = 16000,
    value: int = 1000,
) -> AudioFrame:
    """Create a test AudioFrame with PCM S16LE data."""
    data = struct.pack(f"<{n_samples}h", *([value] * n_samples))
    return AudioFrame(
        data=data,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
    )


# ---------------------------------------------------------------------------
# PipelineDebugTaps config
# ---------------------------------------------------------------------------


class TestPipelineDebugTapsConfig:
    def test_defaults(self) -> None:
        cfg = PipelineDebugTaps()
        assert cfg.output_dir == ""
        assert cfg.stages == ["all"]
        assert cfg.session_scoped is True

    def test_custom_stages(self) -> None:
        cfg = PipelineDebugTaps(
            output_dir="/tmp/debug",
            stages=["raw", "post_denoiser"],
        )
        assert cfg.output_dir == "/tmp/debug"
        assert cfg.stages == ["raw", "post_denoiser"]


# ---------------------------------------------------------------------------
# _DebugWavWriter
# ---------------------------------------------------------------------------


class TestDebugWavWriter:
    def test_lazy_open(self, tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        writer = _DebugWavWriter(path)
        assert not path.exists()
        writer.write(_make_frame())
        assert path.exists()
        writer.close()

    def test_bytes_written(self, tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        writer = _DebugWavWriter(path)
        frame = _make_frame(n_samples=160)
        writer.write(frame)
        assert writer.bytes_written == 320  # 160 samples * 2 bytes
        writer.close()

    def test_empty_data_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        writer = _DebugWavWriter(path)
        writer.write(AudioFrame(data=b""))
        assert writer.bytes_written == 0
        assert not path.exists()
        writer.close()

    def test_write_raw(self, tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        writer = _DebugWavWriter(path)
        data = struct.pack("<10h", *([500] * 10))
        writer.write_raw(data, sample_rate=16000)
        assert writer.bytes_written == 20
        writer.close()

        # Verify WAV file is valid
        with wave.open(str(path), "rb") as f:
            assert f.getnchannels() == 1
            assert f.getsampwidth() == 2
            assert f.getframerate() == 16000
            assert f.getnframes() == 10

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "test.wav"
        writer = _DebugWavWriter(path)
        writer.write(_make_frame())
        assert path.exists()
        writer.close()

    def test_double_close(self, tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        writer = _DebugWavWriter(path)
        writer.write(_make_frame())
        writer.close()
        writer.close()  # should not raise


# ---------------------------------------------------------------------------
# DebugTapSession
# ---------------------------------------------------------------------------


class TestDebugTapSession:
    def test_all_stages_enabled(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(output_dir=str(tmp_path))
        session = DebugTapSession(cfg, "sess-1")
        assert session._enabled_stages == set(ALL_STAGES)
        session.close()

    def test_filtered_stages(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["raw", "post_denoiser"],
        )
        session = DebugTapSession(cfg, "sess-1")
        assert session._enabled_stages == {"raw", "post_denoiser"}
        session.close()

    def test_unknown_stage_ignored(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["raw", "nonexistent"],
        )
        session = DebugTapSession(cfg, "sess-1")
        assert session._enabled_stages == {"raw"}
        session.close()

    def test_tap_writes_wav(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["post_denoiser"],
        )
        session = DebugTapSession(cfg, "sess-1")
        frame = _make_frame(n_samples=320)
        session.tap("post_denoiser", frame)
        session.close()

        wav_path = tmp_path / "sess-1_04_post_denoiser.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as f:
            assert f.getnframes() == 320
            assert f.getframerate() == 16000

    def test_tap_disabled_stage_does_nothing(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["raw"],
        )
        session = DebugTapSession(cfg, "sess-1")
        session.tap("post_denoiser", _make_frame())
        session.close()

        # No post_denoiser file should exist
        assert not (tmp_path / "sess-1_04_post_denoiser.wav").exists()

    def test_session_scoped_false(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["raw"],
            session_scoped=False,
        )
        session = DebugTapSession(cfg, "sess-1")
        session.tap("raw", _make_frame())
        session.close()

        assert (tmp_path / "01_raw.wav").exists()
        assert not (tmp_path / "sess-1_01_raw.wav").exists()

    def test_tap_vad_speech_creates_segment_files(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(output_dir=str(tmp_path))
        session = DebugTapSession(cfg, "sess-1")

        speech1 = struct.pack("<100h", *([500] * 100))
        speech2 = struct.pack("<200h", *([600] * 200))
        session.tap_vad_speech(speech1)
        session.tap_vad_speech(speech2)
        session.close()

        seg1 = tmp_path / "sess-1_05_post_vad_speech_001.wav"
        seg2 = tmp_path / "sess-1_05_post_vad_speech_002.wav"
        assert seg1.exists()
        assert seg2.exists()

        with wave.open(str(seg1), "rb") as f:
            assert f.getnframes() == 100
        with wave.open(str(seg2), "rb") as f:
            assert f.getnframes() == 200

    def test_tap_vad_speech_disabled(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["raw"],
        )
        session = DebugTapSession(cfg, "sess-1")
        session.tap_vad_speech(b"\x00" * 100)
        session.close()

        # No VAD speech files
        files = list(tmp_path.glob("*vad*"))
        assert files == []

    def test_total_bytes_written(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["raw", "post_denoiser"],
        )
        session = DebugTapSession(cfg, "sess-1")
        frame = _make_frame(n_samples=100)  # 200 bytes
        session.tap("raw", frame)
        session.tap("post_denoiser", frame)
        assert session.total_bytes_written == 400
        session.close()

    def test_multiple_frames_append(self, tmp_path: Path) -> None:
        cfg = PipelineDebugTaps(
            output_dir=str(tmp_path),
            stages=["raw"],
        )
        session = DebugTapSession(cfg, "sess-1")
        for _ in range(5):
            session.tap("raw", _make_frame(n_samples=160))
        session.close()

        wav_path = tmp_path / "sess-1_01_raw.wav"
        with wave.open(str(wav_path), "rb") as f:
            assert f.getnframes() == 800  # 5 * 160


# ---------------------------------------------------------------------------
# Integration with AudioPipeline engine
# ---------------------------------------------------------------------------


class TestDebugTapsEngineIntegration:
    def _make_pipeline(self, tmp_path: Path, stages: list[str] | None = None) -> AudioPipeline:
        from roomkit.voice.pipeline.config import AudioPipelineConfig

        cfg = AudioPipelineConfig(
            debug_taps=PipelineDebugTaps(
                output_dir=str(tmp_path),
                stages=stages or ["all"],
            ),
        )
        return AudioPipeline(cfg)

    def _mock_session(self, session_id: str = "test-session") -> MagicMock:
        session = MagicMock()
        session.id = session_id
        return session

    def test_session_lifecycle_creates_and_closes(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline(tmp_path)
        session = self._mock_session()

        pipeline.on_session_active(session)
        assert session.id in pipeline._debug_tap_sessions

        pipeline.on_session_ended(session)
        assert session.id not in pipeline._debug_tap_sessions

    def test_inbound_creates_raw_file(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline(tmp_path, stages=["raw"])
        session = self._mock_session()

        pipeline.on_session_active(session)
        pipeline.process_inbound(session, _make_frame(n_samples=160))
        pipeline.on_session_ended(session)

        wav_path = tmp_path / "test-session_01_raw.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as f:
            assert f.getnframes() == 160

    def test_outbound_creates_files(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline(tmp_path, stages=["outbound_raw", "outbound_final"])
        session = self._mock_session()

        pipeline.on_session_active(session)
        pipeline.process_outbound(session, _make_frame(n_samples=160))
        pipeline.on_session_ended(session)

        assert (tmp_path / "test-session_06_outbound_raw.wav").exists()
        assert (tmp_path / "test-session_07_outbound_final.wav").exists()

    def test_no_debug_taps_no_overhead(self, tmp_path: Path) -> None:
        from roomkit.voice.pipeline.config import AudioPipelineConfig

        pipeline = AudioPipeline(AudioPipelineConfig())
        session = self._mock_session()

        pipeline.on_session_active(session)
        assert session.id not in pipeline._debug_tap_sessions

        # Should not raise
        pipeline.process_inbound(session, _make_frame())
        pipeline.process_outbound(session, _make_frame())
        pipeline.on_session_ended(session)

    def test_reset_closes_debug_taps(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline(tmp_path)
        session = self._mock_session()

        pipeline.on_session_active(session)
        assert session.id in pipeline._debug_tap_sessions

        pipeline.reset()
        assert len(pipeline._debug_tap_sessions) == 0

    def test_all_inbound_stages_captured(self, tmp_path: Path) -> None:
        """With all stages enabled and no providers, raw/post_* should all get data."""
        pipeline = self._make_pipeline(tmp_path)
        session = self._mock_session()

        pipeline.on_session_active(session)
        for _ in range(3):
            pipeline.process_inbound(session, _make_frame())
        pipeline.on_session_ended(session)

        # Without AEC/AGC/denoiser, the taps still fire (same frame passed through)
        for stage in ["raw", "post_aec", "post_agc", "post_denoiser"]:
            _, label = {
                "raw": ("01", "raw"),
                "post_aec": ("02", "post_aec"),
                "post_agc": ("03", "post_agc"),
                "post_denoiser": ("04", "post_denoiser"),
            }[stage]
            prefix = {"raw": "01", "post_aec": "02", "post_agc": "03", "post_denoiser": "04"}[
                stage
            ]
            wav_path = tmp_path / f"test-session_{prefix}_{label}.wav"
            assert wav_path.exists(), f"Missing {wav_path.name}"
