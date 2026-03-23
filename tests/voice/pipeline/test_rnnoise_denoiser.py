"""Tests for the RNNoise denoiser provider."""

from __future__ import annotations

import ctypes
import importlib
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.audio_frame import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_librnnoise():
    """Build a fake librnnoise CDLL."""
    lib = MagicMock(spec=ctypes.CDLL)
    lib.rnnoise_get_frame_size = MagicMock(return_value=480)
    lib.rnnoise_create = MagicMock(return_value=0xBEEF)
    lib.rnnoise_destroy = MagicMock()
    lib.rnnoise_process_frame = MagicMock(return_value=ctypes.c_float(0.9))
    return lib


def _make_provider(mock_lib, **kwargs):
    """Reload module and construct RNNoiseDenoiserProvider with librnnoise mocked."""
    with (
        patch("ctypes.util.find_library", return_value="/fake/librnnoise.so"),
        patch("ctypes.CDLL", return_value=mock_lib),
    ):
        import roomkit.voice.pipeline.denoiser.rnnoise as rnnoise_mod

        rnnoise_mod._lib = None  # reset module cache
        importlib.reload(rnnoise_mod)
        return rnnoise_mod.RNNoiseDenoiserProvider(**kwargs), rnnoise_mod


def _make_frame(n_samples: int, sample_rate: int = 16000) -> AudioFrame:
    return AudioFrame(
        data=b"\x01\x00" * n_samples,
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRNNoiseDenoiserProviderConstructor:
    def test_default_16khz(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib)

        assert provider.name == "rnnoise"
        assert provider._sample_rate == 16000
        assert provider._resample_factor == 3
        assert provider._input_frame_samples == 160

    def test_48khz(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=48000)

        assert provider._resample_factor == 1
        assert provider._input_frame_samples == 480

    def test_24khz(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=24000)

        assert provider._resample_factor == 2
        assert provider._input_frame_samples == 240


class TestRNNoiseDenoiserSampleRateValidation:
    def test_invalid_sample_rate(self):
        mock_lib = _make_mock_librnnoise()
        with pytest.raises(ValueError, match="divides evenly into 48000"):
            _make_provider(mock_lib, sample_rate=44100)

    def test_another_invalid_rate(self):
        mock_lib = _make_mock_librnnoise()
        with pytest.raises(ValueError, match="divides evenly into 48000"):
            _make_provider(mock_lib, sample_rate=22050)


class TestRNNoiseDenoiserProviderProcess:
    def test_process_one_chunk(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        # 160 samples = one chunk at 16kHz
        frame = _make_frame(n_samples=160, sample_rate=16000)
        result = provider.process(frame)

        assert result.sample_rate == 16000
        assert len(result.data) == len(frame.data)
        mock_lib.rnnoise_process_frame.assert_called_once()

    def test_process_wrong_size_passthrough(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        # 100 samples is not a multiple of 160
        frame = _make_frame(n_samples=100, sample_rate=16000)
        result = provider.process(frame)
        assert result is frame
        mock_lib.rnnoise_process_frame.assert_not_called()


class TestRNNoiseDenoiserProviderReset:
    def test_reset_recreates_state(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib)

        provider.reset()

        mock_lib.rnnoise_destroy.assert_called_once()
        assert mock_lib.rnnoise_create.call_count == 2  # init + reset


class TestRNNoiseDenoiserProviderClose:
    def test_close_destroys_state(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib)

        provider.close()
        assert provider._state is None
        mock_lib.rnnoise_destroy.assert_called_once()

    def test_close_idempotent(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib)

        provider.close()
        provider.close()
        mock_lib.rnnoise_destroy.assert_called_once()
