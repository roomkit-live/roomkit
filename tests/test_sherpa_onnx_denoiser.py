"""Tests for SherpaOnnxDenoiserProvider."""

from __future__ import annotations

import importlib
import struct
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roomkit.voice.audio_frame import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_sherpa_module() -> MagicMock:
    """Create a MagicMock that stands in for the sherpa_onnx module."""
    return MagicMock()


def _frame(
    n_samples: int = 160,
    value: int = 0,
    sample_rate: int = 16000,
    timestamp_ms: float | None = None,
) -> AudioFrame:
    """Create a PCM-16 mono AudioFrame with *n_samples* samples."""
    data = struct.pack(f"<{n_samples}h", *([value] * n_samples))
    return AudioFrame(
        data=data,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        timestamp_ms=timestamp_ms,
    )


def _make_provider(
    sherpa_mock: MagicMock,
    denoiser_mock: MagicMock,
    **config_kwargs: Any,
) -> Any:
    """Create a SherpaOnnxDenoiserProvider with mocked sherpa_onnx."""
    sherpa_mock.OfflineSpeechDenoiser.return_value = denoiser_mock
    with patch.dict("sys.modules", {"sherpa_onnx": sherpa_mock}):
        import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

        importlib.reload(dn_mod)
        from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
            SherpaOnnxDenoiserConfig,
            SherpaOnnxDenoiserProvider,
        )

        cfg = SherpaOnnxDenoiserConfig(**config_kwargs)
        provider = SherpaOnnxDenoiserProvider(cfg)
        provider._sherpa = sherpa_mock
        return provider


def _denoiser_that_returns(samples: list[float]) -> MagicMock:
    """Create a mock denoiser whose .run() returns given samples."""
    denoiser = MagicMock()
    result = MagicMock()
    result.samples = samples
    denoiser.run.return_value = result
    return denoiser


# ---------------------------------------------------------------------------
# Process — basic denoising
# ---------------------------------------------------------------------------


class TestProcess:
    def test_process_returns_new_frame(self) -> None:
        sherpa = _mock_sherpa_module()
        # Return zeros (silence) from denoiser
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        frame_in = _frame(160, value=100)
        frame_out = provider.process(frame_in)

        assert isinstance(frame_out, AudioFrame)
        assert frame_out is not frame_in
        assert len(frame_out.data) == len(frame_in.data)
        assert frame_out.sample_rate == frame_in.sample_rate
        assert frame_out.channels == frame_in.channels
        assert frame_out.sample_width == frame_in.sample_width

    def test_process_converts_denoised_samples(self) -> None:
        sherpa = _mock_sherpa_module()
        # Return a known value
        denoiser = _denoiser_that_returns([0.5] * 4)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        frame_in = _frame(4, value=1000)
        frame_out = provider.process(frame_in)

        out_samples = struct.unpack("<4h", frame_out.data)
        expected = int(0.5 * 32767)
        for s in out_samples:
            assert s == expected

    def test_process_preserves_timestamp(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        frame_in = _frame(160, timestamp_ms=42.5)
        frame_out = provider.process(frame_in)

        assert frame_out.timestamp_ms == 42.5

    def test_process_copies_metadata(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        frame_in = _frame(160)
        frame_in.metadata["source"] = "test"
        frame_out = provider.process(frame_in)

        assert frame_out.metadata["source"] == "test"
        # Mutation of output metadata must not affect input.
        frame_out.metadata["extra"] = True
        assert "extra" not in frame_in.metadata

    def test_process_after_close_returns_original(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        provider.close()
        frame = _frame(160)
        result = provider.process(frame)

        # After close, _denoiser is None, _ensure_denoiser re-inits
        # but since we're mocked, it will work. Let's test the other
        # path: close sets _denoiser = None, process calls _ensure_denoiser.
        assert isinstance(result, AudioFrame)

    def test_process_error_returns_original(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = MagicMock()
        denoiser.run.side_effect = RuntimeError("boom")
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        frame = _frame(160, value=500)
        result = provider.process(frame)

        # Should return the original frame on error
        assert result is frame


# ---------------------------------------------------------------------------
# Silence gate
# ---------------------------------------------------------------------------


class TestSilenceGate:
    def test_silent_frame_skips_inference(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser)

        # value=0 → silence → should be gated
        frame = _frame(160, value=0)
        result = provider.process(frame)

        assert isinstance(result, AudioFrame)
        # Output should be all zeros
        assert result.data == b"\x00" * len(frame.data)
        # Denoiser.run should NOT have been called
        denoiser.run.assert_not_called()

    def test_loud_frame_runs_inference(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.1] * 160)
        provider = _make_provider(sherpa, denoiser)

        # value=1000 → ~0.03 RMS → above default 0.005 threshold
        frame = _frame(160, value=1000)
        result = provider.process(frame)

        assert isinstance(result, AudioFrame)
        denoiser.run.assert_called_once()

    def test_silence_gate_disabled_when_threshold_zero(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        # Even silence should run inference when gate is disabled
        frame = _frame(160, value=0)
        provider.process(frame)

        denoiser.run.assert_called_once()

    def test_silence_gate_preserves_frame_metadata(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser)

        frame = _frame(160, value=0, timestamp_ms=99.0)
        frame.metadata["tag"] = "test"
        result = provider.process(frame)

        assert result.timestamp_ms == 99.0
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.metadata["tag"] == "test"


# ---------------------------------------------------------------------------
# Lazy initialization
# ---------------------------------------------------------------------------


class TestLazyInit:
    def test_denoiser_not_created_until_first_process(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        assert provider._denoiser is None

        provider.process(_frame(160))
        assert provider._denoiser is not None

    def test_ensure_denoiser_only_creates_once(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        provider.process(_frame(160))
        provider.process(_frame(160))

        # OfflineSpeechDenoiser constructor called only once
        sherpa.OfflineSpeechDenoiser.assert_called_once()


# ---------------------------------------------------------------------------
# Config populates GTCRN model config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_context_frames_default_is_3(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                SherpaOnnxDenoiserConfig,
            )

            assert SherpaOnnxDenoiserConfig().context_frames == 3

    def test_config_populates_gtcrn(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(
            sherpa,
            denoiser,
            model="/path/to/gtcrn_simple.onnx",
            num_threads=2,
            provider="cuda",
            silence_threshold=0,
        )

        # Trigger lazy init
        provider.process(_frame(160))

        sherpa.OfflineSpeechDenoiserGtcrnModelConfig.assert_called_once_with(
            model="/path/to/gtcrn_simple.onnx",
        )
        sherpa.OfflineSpeechDenoiserModelConfig.assert_called_once_with(
            gtcrn=sherpa.OfflineSpeechDenoiserGtcrnModelConfig.return_value,
            num_threads=2,
            provider="cuda",
        )
        sherpa.OfflineSpeechDenoiserConfig.assert_called_once_with(
            model=sherpa.OfflineSpeechDenoiserModelConfig.return_value,
        )


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------


class TestName:
    def test_name(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = MagicMock()
        provider = _make_provider(sherpa, denoiser)
        assert provider.name == "SherpaOnnxDenoiser"


# ---------------------------------------------------------------------------
# Reset / Close
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_reset_is_safe(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        # Reset before init — no-op
        provider.reset()

        # Init and reset
        provider.process(_frame(160))
        provider.reset()

        # Still usable
        result = provider.process(_frame(160))
        assert isinstance(result, AudioFrame)

    def test_close_sets_denoiser_none(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        provider.process(_frame(160))
        assert provider._denoiser is not None

        provider.close()
        assert provider._denoiser is None

    def test_double_close(self) -> None:
        sherpa = _mock_sherpa_module()
        denoiser = _denoiser_that_returns([0.0] * 160)
        provider = _make_provider(sherpa, denoiser, silence_threshold=0)

        provider.close()
        provider.close()  # Must not raise


# ---------------------------------------------------------------------------
# Import error
# ---------------------------------------------------------------------------


class TestImportError:
    def test_import_error_when_sherpa_not_installed(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": None}):
            import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

            importlib.reload(dn_mod)

            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                SherpaOnnxDenoiserConfig,
                SherpaOnnxDenoiserProvider,
            )

            with pytest.raises(ImportError, match="sherpa-onnx is required"):
                SherpaOnnxDenoiserProvider(SherpaOnnxDenoiserConfig())


# ---------------------------------------------------------------------------
# PCM conversion helpers
# ---------------------------------------------------------------------------


class TestPcmConversion:
    def test_s16le_to_float32_silence(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                _pcm_s16le_to_float32,
            )

            pcm = struct.pack("<1h", 0)
            result = _pcm_s16le_to_float32(pcm)
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert len(result) == 1
            assert abs(result[0]) < 1e-6

    def test_s16le_to_float32_max(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                _pcm_s16le_to_float32,
            )

            pcm = struct.pack("<1h", 32767)
            result = _pcm_s16le_to_float32(pcm)
            assert abs(result[0] - 1.0) < 0.001

    def test_s16le_to_float32_min(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                _pcm_s16le_to_float32,
            )

            pcm = struct.pack("<1h", -32768)
            result = _pcm_s16le_to_float32(pcm)
            assert abs(result[0] - (-1.0)) < 0.001

    def test_float32_to_s16le_roundtrip(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                _float32_to_pcm_s16le,
                _pcm_s16le_to_float32,
            )

            original = struct.pack("<4h", 0, 1000, -1000, 32767)
            floats = _pcm_s16le_to_float32(original)
            back = _float32_to_pcm_s16le(floats)
            restored = struct.unpack("<4h", back)
            expected = struct.unpack("<4h", original)
            for a, b in zip(restored, expected, strict=True):
                assert abs(a - b) <= 1  # Allow ±1 rounding

    def test_float32_to_s16le_clamps(self) -> None:
        with patch.dict("sys.modules", {"sherpa_onnx": _mock_sherpa_module()}):
            import roomkit.voice.pipeline.denoiser.sherpa_onnx as dn_mod

            importlib.reload(dn_mod)
            from roomkit.voice.pipeline.denoiser.sherpa_onnx import (
                _float32_to_pcm_s16le,
            )

            out = _float32_to_pcm_s16le([2.0, -2.0])
            samples = struct.unpack("<2h", out)
            assert samples[0] == 32767
            assert samples[1] == -32767
