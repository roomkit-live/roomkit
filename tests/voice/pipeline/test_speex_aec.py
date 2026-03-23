"""Tests for the SpeexDSP AEC provider."""

from __future__ import annotations

import ctypes
import importlib
from unittest.mock import MagicMock, patch

from roomkit.voice.audio_frame import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_SIZE = 320  # 20ms at 16kHz
_FRAME_BYTES = _FRAME_SIZE * 2


def _make_mock_speexdsp():
    """Build a fake libspeexdsp CDLL."""
    lib = MagicMock(spec=ctypes.CDLL)
    lib.speex_echo_state_init = MagicMock(return_value=0xDEAD)
    lib.speex_echo_state_destroy = MagicMock()
    lib.speex_echo_state_reset = MagicMock()
    lib.speex_echo_playback = MagicMock()
    lib.speex_echo_capture = MagicMock()
    lib.speex_echo_cancellation = MagicMock()
    lib.speex_echo_ctl = MagicMock(return_value=0)
    return lib


def _make_provider(mock_lib, **kwargs):
    """Reload module and construct SpeexAECProvider with ctypes mocked."""
    with (
        patch("ctypes.util.find_library", return_value="/fake/libspeexdsp.so"),
        patch("ctypes.CDLL", return_value=mock_lib),
    ):
        import roomkit.voice.pipeline.aec.speex as speex_mod

        speex_mod._lib = None  # reset module cache
        importlib.reload(speex_mod)
        return speex_mod.SpeexAECProvider(**kwargs), speex_mod


def _make_frame(n_bytes: int = _FRAME_BYTES) -> AudioFrame:
    return AudioFrame(
        data=b"\x01\x00" * (n_bytes // 2),
        sample_rate=16000,
        channels=1,
        sample_width=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpeexAECProviderConstructor:
    def test_defaults(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        assert provider.name == "speex_aec"
        assert provider._frame_size == 320
        assert provider._filter_length == 3200
        mock_lib.speex_echo_state_init.assert_called_with(320, 3200)

    def test_custom_params(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(
            mock_lib,
            frame_size=160,
            filter_length=1600,
            sample_rate=8000,
        )

        assert provider._frame_size == 160
        assert provider._sample_rate == 8000


class TestSpeexAECProviderProcess:
    def test_process_produces_output(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        frame = _make_frame()
        result = provider.process(frame)

        assert result.sample_rate == frame.sample_rate
        assert len(result.data) == len(frame.data)
        mock_lib.speex_echo_capture.assert_called_once()

    def test_process_frame_size_mismatch(self):
        """Wrong frame size -> passthrough."""
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        frame = _make_frame(n_bytes=160)
        result = provider.process(frame)
        assert result is frame
        mock_lib.speex_echo_capture.assert_not_called()


class TestSpeexAECProviderFeedReference:
    def test_feed_reference(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        frame = _make_frame()
        provider.feed_reference(frame)
        mock_lib.speex_echo_playback.assert_called_once()

    def test_feed_reference_size_mismatch(self):
        """Wrong frame size -> ignored."""
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        frame = _make_frame(n_bytes=160)
        provider.feed_reference(frame)
        mock_lib.speex_echo_playback.assert_not_called()


class TestSpeexAECProviderReset:
    def test_reset_calls_state_reset(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        provider.reset()
        mock_lib.speex_echo_state_reset.assert_called_once()


class TestSpeexAECProviderClose:
    def test_close_destroys_state(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)
        assert provider._state is not None

        provider.close()
        assert provider._state is None
        mock_lib.speex_echo_state_destroy.assert_called_once()

    def test_close_idempotent(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        provider.close()
        provider.close()
        mock_lib.speex_echo_state_destroy.assert_called_once()
