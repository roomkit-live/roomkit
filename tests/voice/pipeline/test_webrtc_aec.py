"""Tests for the WebRTC AEC provider."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from roomkit.voice.audio_frame import AudioFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_aec_module():
    """Build a fake aec_audio_processing module with AudioProcessor."""
    processor = MagicMock()
    processor.process_stream = MagicMock(side_effect=lambda data: data)
    processor.process_reverse_stream = MagicMock()
    processor.set_stream_format = MagicMock()
    processor.set_reverse_stream_format = MagicMock()
    processor.set_stream_delay = MagicMock()

    ap_cls = MagicMock(return_value=processor)
    mod = SimpleNamespace(AudioProcessor=ap_cls)
    return mod, ap_cls, processor


def _make_provider(mock_mod, **kwargs):
    """Reload module and construct WebRTCAECProvider inside active patch."""
    with patch.dict(sys.modules, {"aec_audio_processing": mock_mod}):
        import roomkit.voice.pipeline.aec.webrtc as webrtc_mod

        importlib.reload(webrtc_mod)
        return webrtc_mod.WebRTCAECProvider(**kwargs), webrtc_mod


def _make_frame(n_bytes: int = 320, sample_rate: int = 16000) -> AudioFrame:
    return AudioFrame(
        data=b"\x01\x00" * (n_bytes // 2),
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebRTCAECProviderConstructor:
    def test_defaults(self):
        mock_mod, ap_cls, processor = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)

        assert provider.name == "webrtc_aec3"

        # Processors are per stream and built on first use.
        ap_cls.assert_not_called()
        provider.set_active(True)
        provider.process(_make_frame(), "s1")
        ap_cls.assert_called_once_with(enable_aec=True, enable_ns=False, enable_agc=False)

    def test_custom_params(self):
        mock_mod, ap_cls, processor = _make_mock_aec_module()
        provider, _ = _make_provider(
            mock_mod,
            sample_rate=48000,
            channels=1,
            stream_delay_ms=50,
            enable_ns=True,
            enable_agc=True,
        )
        assert provider._sample_rate == 48000
        assert provider._stream_delay_ms == 50

        provider.set_active(True)
        provider.process(_make_frame(n_bytes=960, sample_rate=48000), "s1")
        processor.set_stream_delay.assert_called_once_with(50)


class TestWebRTCAECProviderProcess:
    def test_process_bypass_by_default(self):
        """Starts in bypass mode — process() returns frame unchanged."""
        mock_mod, _, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)

        frame = _make_frame()
        result = provider.process(frame, "s1")
        assert result is frame

    def test_process_active(self):
        """When activated, process() passes frames through the AP."""
        mock_mod, _, processor = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)

        # 10ms frame at 16kHz = 160 samples = 320 bytes
        frame = _make_frame(n_bytes=320)
        result = provider.process(frame, "s1")
        assert result.sample_rate == frame.sample_rate
        processor.process_stream.assert_called()

    def test_process_energy_diagnostics(self):
        """Energy totals are exact integer sums of squares (passthrough mock)."""
        mock_mod, _, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)

        # 160 samples of amplitude 1000 → energy = 160 * 1000²
        frame = AudioFrame(
            data=(1000).to_bytes(2, "little", signed=True) * 160,
            sample_rate=16000,
            channels=1,
            sample_width=2,
        )
        provider.process(frame, "s1")

        st = provider._streams["s1"]
        assert st.total_in_energy == 160 * 1000**2
        # Mock AP is passthrough, so output energy matches input exactly.
        assert st.total_out_energy == st.total_in_energy


class TestWebRTCAECProviderFeedReference:
    def test_feed_reference(self):
        mock_mod, _, processor = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)

        frame = _make_frame(n_bytes=320)
        provider.feed_reference(frame, "s1")
        processor.process_reverse_stream.assert_called()


class TestWebRTCAECProviderSetActive:
    def test_toggle(self):
        mock_mod, _, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)

        assert provider._bypass is True

        provider.set_active(True)
        assert provider._bypass is False

        provider.set_active(False)
        assert provider._bypass is True


class TestWebRTCAECProviderStreams:
    def test_each_stream_gets_its_own_processor(self):
        mock_mod, ap_cls, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")

        assert ap_cls.call_count == 2
        assert set(provider._streams) == {"alice", "bob"}

    def test_streams_do_not_share_counters(self):
        mock_mod, _, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)

        for _ in range(3):
            provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")

        assert provider._streams["alice"].process_count == 3
        assert provider._streams["bob"].process_count == 1

    def test_reference_reaches_only_its_own_stream(self):
        mock_mod, _, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")
        provider.feed_reference(_make_frame(), "alice")

        assert provider._streams["alice"].ref_fed_count == 1
        assert provider._streams["bob"].ref_fed_count == 0


class TestWebRTCAECProviderReset:
    def test_reset_drops_only_that_stream(self):
        mock_mod, ap_cls, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")

        provider.reset("alice")
        assert set(provider._streams) == {"bob"}

        # The next frame builds a fresh processor — that is what clears the
        # stale adaptive filter after barge-in.
        provider.process(_make_frame(), "alice")
        assert ap_cls.call_count == 3
        assert provider._streams["alice"].process_count == 1


class TestWebRTCAECProviderClose:
    def test_close_releases_every_stream(self):
        mock_mod, _, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)

        provider.process(_make_frame(), "alice")
        provider.process(_make_frame(), "bob")

        provider.close()
        assert provider._streams == {}

    def test_process_after_close_passes_through(self):
        mock_mod, _, _ = _make_mock_aec_module()
        provider, _ = _make_provider(mock_mod)
        provider.set_active(True)
        provider.close()

        frame = _make_frame()
        assert provider.process(frame, "s1") is frame
        assert provider._streams == {}
