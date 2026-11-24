"""Tests for the SpeexDSP AEC provider."""

from __future__ import annotations

import ctypes
import logging
import struct
from unittest.mock import MagicMock, patch

from roomkit.voice.audio_frame import AudioFrame
from tests.private_module import load_private_copy

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
    """Construct SpeexAECProvider from a fresh module copy, ctypes mocked."""
    with (
        patch("ctypes.util.find_library", return_value="/fake/libspeexdsp.so"),
        patch("ctypes.CDLL", return_value=mock_lib),
    ):
        speex_mod = load_private_copy("roomkit.voice.pipeline.aec.speex", {})
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

        # Echo state is per stream and built on first use, not at construction.
        mock_lib.speex_echo_state_init.assert_not_called()
        provider.process(_make_frame(), "s1")
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
        result = provider.process(frame, "s1")

        assert result.sample_rate == frame.sample_rate
        assert len(result.data) == len(frame.data)
        assert result.metadata["echo_cancelled"] is True
        mock_lib.speex_echo_capture.assert_called_once()

    def test_debug_stats_report_the_interval_rms(self, caplog):
        """The DEBUG energy sums read the samples as signed little-endian PCM16."""
        mock_lib = _make_mock_speexdsp()  # its capture leaves the output silent
        provider, _ = _make_provider(mock_lib)
        frame = AudioFrame(
            data=struct.pack("<h", -1000) * _FRAME_SIZE, sample_rate=16000, channels=1
        )

        with caplog.at_level(logging.DEBUG):
            for _ in range(50):  # one log interval
                provider.process(frame, "s1")

        stats = [r.getMessage() for r in caplog.records if "[AEC stats]" in r.getMessage()]
        assert len(stats) == 1 and "in_rms=1000 out_rms=0 " in stats[0]

    def test_process_frame_size_mismatch(self):
        """Wrong frame size -> passthrough."""
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        frame = _make_frame(n_bytes=160)
        result = provider.process(frame, "s1")
        assert result is frame
        mock_lib.speex_echo_capture.assert_not_called()

    def test_mismatched_pcm_format_passes_through(self, caplog):
        """A frame in another format is never read as the stage's own: it
        passes on unchanged, with one warning per format (RFC §12.3.1)."""
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)
        frames = [
            AudioFrame(data=b"\x01\x00" * 320, sample_rate=8000, channels=1, sample_width=2)
            for _ in range(2)
        ]

        with caplog.at_level(logging.WARNING):
            results = [provider.process(frame, "s1") for frame in frames]

        assert results[0] is frames[0] and results[1] is frames[1]
        mock_lib.speex_echo_capture.assert_not_called()
        warnings = [r for r in caplog.records if "format mismatch" in r.getMessage()]
        assert len(warnings) == 1


class TestSpeexAECProviderFeedReference:
    def test_feed_reference(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        frame = _make_frame()
        provider.feed_reference(frame, "s1")
        mock_lib.speex_echo_playback.assert_called_once()

    def test_feed_reference_size_mismatch(self):
        """Wrong frame size -> ignored."""
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        frame = _make_frame(n_bytes=160)
        provider.feed_reference(frame, "s1")
        mock_lib.speex_echo_playback.assert_not_called()

    def test_mismatched_reference_format_is_ignored(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)
        frame = AudioFrame(
            data=b"\x01\x00" * 320,
            sample_rate=16000,
            channels=2,
            sample_width=2,
        )

        provider.feed_reference(frame, "s1")

        mock_lib.speex_echo_playback.assert_not_called()


class TestSpeexAECProviderStreams:
    def test_each_stream_gets_its_own_echo_state(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")

        assert mock_lib.speex_echo_state_init.call_count == 2
        assert set(provider._streams) == {"alice", "bob"}

    def test_repeat_frames_reuse_one_state_per_stream(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        for _ in range(3):
            provider.process(_make_frame(), "alice")

        mock_lib.speex_echo_state_init.assert_called_once()

    def test_reference_reaches_only_its_own_stream(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")
        provider.feed_reference(_make_frame(), "alice")

        # Alice saw the reference; Bob's canceller must not have.
        assert provider._streams["alice"].playback_fed is True
        assert provider._streams["bob"].playback_fed is False


class TestSpeexAECProviderReset:
    def test_reset_destroys_only_that_stream(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")

        provider.reset("alice")

        mock_lib.speex_echo_state_destroy.assert_called_once()
        assert set(provider._streams) == {"bob"}

    def test_reset_unknown_stream_is_a_noop(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        provider.reset("never-seen")
        mock_lib.speex_echo_state_destroy.assert_not_called()


class TestSpeexAECProviderClose:
    def test_close_destroys_every_stream(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")

        provider.close()

        assert mock_lib.speex_echo_state_destroy.call_count == 2
        assert provider._streams == {}

    def test_close_idempotent(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)
        provider.process(_make_frame(), "s1")

        provider.close()
        provider.close()
        mock_lib.speex_echo_state_destroy.assert_called_once()

    def test_process_after_close_does_not_resurrect_native_state(self):
        mock_lib = _make_mock_speexdsp()
        provider, _ = _make_provider(mock_lib)
        provider.close()
        frame = _make_frame()

        result = provider.process(frame, "s1")

        assert result is frame
        mock_lib.speex_echo_state_init.assert_not_called()
