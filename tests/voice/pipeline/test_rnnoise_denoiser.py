"""Tests for the RNNoise denoiser provider."""

from __future__ import annotations

import ctypes
import importlib
import struct
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

    def process_frame(_state, output, input_):
        for index in range(480):
            output[index] = input_[index]
        return ctypes.c_float(0.9)

    lib.rnnoise_process_frame = MagicMock(side_effect=process_frame)
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
        with pytest.raises(ValueError, match="16000, 24000, or 48000"):
            _make_provider(mock_lib, sample_rate=44100)

    def test_another_invalid_rate(self):
        mock_lib = _make_mock_librnnoise()
        with pytest.raises(ValueError, match="16000, 24000, or 48000"):
            _make_provider(mock_lib, sample_rate=22050)

    def test_unadvertised_divisor_is_rejected(self):
        mock_lib = _make_mock_librnnoise()
        with pytest.raises(ValueError, match="16000, 24000, or 48000"):
            _make_provider(mock_lib, sample_rate=8000)


class TestRNNoiseDenoiserProviderProcess:
    def test_process_one_chunk(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        # 160 samples = one chunk at 16kHz
        frame = _make_frame(n_samples=160, sample_rate=16000)
        result = provider.process(frame, "s1")

        assert result.sample_rate == 16000
        assert len(result.data) == len(frame.data)
        mock_lib.rnnoise_process_frame.assert_called_once()

    def test_process_irregular_size_enters_fixed_delay_chunking(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        # 100 samples is not a multiple of 160
        frame = _make_frame(n_samples=100, sample_rate=16000)
        result = provider.process(frame, "s1")
        assert result is not frame
        assert len(result.data) == len(frame.data)
        assert result.data == b"\x00" * len(frame.data)
        mock_lib.rnnoise_process_frame.assert_not_called()

    def test_irregular_chunks_preserve_timeline_without_duplication(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        first = provider.process(AudioFrame(struct.pack("<80h", *([100] * 80)), 16000, 1, 2), "s1")
        second = provider.process(
            AudioFrame(struct.pack("<80h", *([200] * 80)), 16000, 1, 2), "s1"
        )
        third = provider.process(
            AudioFrame(struct.pack("<160h", *([300] * 160)), 16000, 1, 2), "s1"
        )

        assert len(first.data) == 160
        assert len(second.data) == 160
        assert len(third.data) == 320
        assert first.data + second.data == b"\x00" * 320
        delayed = struct.unpack("<160h", third.data)
        assert delayed[:79] == (100,) * 79
        assert 100 < delayed[79] < 200
        assert delayed[80:] == (200,) * 80
        assert mock_lib.rnnoise_process_frame.call_count == 2

    @pytest.mark.parametrize(
        "frame",
        [
            AudioFrame(b"\x00\x00" * 160, 8000, 1, 2),
            AudioFrame(b"\x00\x00" * 320, 16000, 2, 2),
            AudioFrame(b"\x00" * 160, 16000, 1, 1),
        ],
    )
    def test_mismatched_pcm_format_is_bypassed(self, frame: AudioFrame):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        assert provider.process(frame, "s1") is frame
        assert provider._streams == {}
        mock_lib.rnnoise_process_frame.assert_not_called()

    def test_native_failure_returns_original_and_clears_chunking(self):
        mock_lib = _make_mock_librnnoise()
        mock_lib.rnnoise_process_frame.side_effect = RuntimeError("boom")
        provider, _ = _make_provider(mock_lib, sample_rate=16000)
        frame = _make_frame(160)

        result = provider.process(frame, "s1")

        assert result is frame
        state = provider._streams["s1"]
        assert state.input_buffer == bytearray()
        assert state.output_buffer == bytearray()
        assert state.chunking is False


class TestRNNoiseDenoiserProviderStreams:
    def test_each_stream_gets_its_own_native_state(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        provider.process(_make_frame(n_samples=160, sample_rate=16000), "alice")
        provider.process(_make_frame(n_samples=160, sample_rate=16000), "bob")

        assert mock_lib.rnnoise_create.call_count == 2
        assert set(provider._streams) == {"alice", "bob"}

    def test_repeat_frames_reuse_one_state_per_stream(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        for _ in range(3):
            provider.process(_make_frame(n_samples=160, sample_rate=16000), "alice")

        mock_lib.rnnoise_create.assert_called_once()


class TestRNNoiseDenoiserProviderReset:
    def test_reset_destroys_only_that_stream(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        provider.process(_make_frame(n_samples=160, sample_rate=16000), "alice")
        provider.process(_make_frame(n_samples=160, sample_rate=16000), "bob")

        provider.reset("alice")

        mock_lib.rnnoise_destroy.assert_called_once()
        assert set(provider._streams) == {"bob"}

    def test_reset_unknown_stream_is_a_noop(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib)

        provider.reset("never-seen")
        mock_lib.rnnoise_destroy.assert_not_called()


class TestRNNoiseDenoiserProviderClose:
    def test_close_destroys_every_stream(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)

        provider.process(_make_frame(n_samples=160, sample_rate=16000), "alice")
        provider.process(_make_frame(n_samples=160, sample_rate=16000), "bob")

        provider.close()
        assert provider._streams == {}
        assert mock_lib.rnnoise_destroy.call_count == 2

    def test_close_idempotent(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)
        provider.process(_make_frame(n_samples=160, sample_rate=16000), "s1")

        provider.close()
        provider.close()
        mock_lib.rnnoise_destroy.assert_called_once()

    def test_process_after_close_does_not_resurrect_native_state(self):
        mock_lib = _make_mock_librnnoise()
        provider, _ = _make_provider(mock_lib, sample_rate=16000)
        provider.close()
        frame = _make_frame(160)

        assert provider.process(frame, "s1") is frame
        mock_lib.rnnoise_create.assert_not_called()
        assert provider._streams == {}
