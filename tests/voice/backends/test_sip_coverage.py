"""Tests for SIP voice backend — codec constants and AudioStats coverage."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from roomkit.voice.backends._sip_types import CODEC_INFO, PT_G722, PT_PCMA, PT_PCMU, AudioStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_aiosipua():
    """Build fake aiosipua + aiosipua.rtp_bridge modules."""
    rtp_bridge = SimpleNamespace(
        CallSession=MagicMock(),
    )
    aiosipua = SimpleNamespace(
        UdpSipTransport=MagicMock(),
        SipUAS=MagicMock(),
        SipUAC=MagicMock(),
        rtp_bridge=rtp_bridge,
    )
    return aiosipua, rtp_bridge


def _get_sip_module(aiosipua_mod, rtp_bridge_mod):
    """Reload the SIP module with mocks active. Caller must keep patch active."""
    import roomkit.voice.backends.sip as sip_mod

    importlib.reload(sip_mod)
    return sip_mod


# ---------------------------------------------------------------------------
# Tests — codec mapping constants
# ---------------------------------------------------------------------------


class TestCodecConstants:
    def test_payload_type_values(self):
        assert PT_PCMU == 0
        assert PT_PCMA == 8
        assert PT_G722 == 9

    def test_codec_info_entries(self):
        info = CODEC_INFO

        name, rtp_clock, audio_rate = info[PT_PCMU]
        assert name == "PCMU"
        assert rtp_clock == 8000
        assert audio_rate == 8000

        name, rtp_clock, audio_rate = info[PT_PCMA]
        assert name == "PCMA"
        assert rtp_clock == 8000
        assert audio_rate == 8000

        # G.722: RTP clock 8000, audio rate 16000 per RFC 3551
        name, rtp_clock, audio_rate = info[PT_G722]
        assert name == "G722"
        assert rtp_clock == 8000
        assert audio_rate == 16000


# ---------------------------------------------------------------------------
# Tests — AudioStats
# ---------------------------------------------------------------------------


class TestAudioStats:
    def test_initial_values(self):
        stats = AudioStats()
        assert stats.inbound_packets == 0
        assert stats.inbound_bytes == 0
        assert stats.inbound_first_ts == 0.0
        assert stats.inbound_last_ts == 0.0
        assert stats.inbound_gaps == 0
        assert stats.inbound_max_gap_ms == 0.0
        assert stats.outbound_frames == 0
        assert stats.outbound_bytes == 0
        assert stats.outbound_first_ts == 0.0
        assert stats.outbound_last_ts == 0.0
        assert stats.outbound_max_burst == 0
        assert stats.outbound_calls == 0

    def test_sync_from_rtp_receiver_report(self):
        stats = AudioStats()
        stats.sync_from_rtp({})
        assert stats.has_remote_report is False

        # concealed syncs even before any RR arrives
        stats.sync_from_rtp({"concealed_frames": 3})
        assert stats.concealed_frames == 3
        assert stats.has_remote_report is False

        stats.sync_from_rtp(
            {
                "concealed_frames": 5,
                "remote_packets_lost": 12,
                "remote_fraction_lost": 26,
                "remote_jitter": 80,
            }
        )
        assert stats.has_remote_report is True
        assert stats.remote_packets_lost == 12
        assert stats.remote_fraction_lost == 26
        assert stats.remote_jitter_units == 80

        # RR values survive a later sync from a dict without RR keys
        stats.sync_from_rtp({"concealed_frames": 6})
        assert stats.concealed_frames == 6
        assert stats.remote_packets_lost == 12
        assert stats.has_remote_report is True

    def test_counter_mutation(self):
        stats = AudioStats()
        stats.inbound_packets = 42
        stats.outbound_frames = 10
        stats.inbound_max_gap_ms = 15.5
        assert stats.inbound_packets == 42
        assert stats.outbound_frames == 10
        assert stats.inbound_max_gap_ms == 15.5


# ---------------------------------------------------------------------------
# Tests — SIPVoiceBackend constructor and properties
# ---------------------------------------------------------------------------


class TestSIPVoiceBackendConstructor:
    def test_name_and_capabilities(self):
        from roomkit.voice.base import VoiceCapability

        aiosipua, rtp_bridge = _make_mock_aiosipua()
        with patch.dict(sys.modules, {"aiosipua": aiosipua, "aiosipua.rtp_bridge": rtp_bridge}):
            sip_mod = _get_sip_module(aiosipua, rtp_bridge)
            backend = sip_mod.SIPVoiceBackend()

        assert backend.name == "SIP"
        assert VoiceCapability.DTMF_SIGNALING in backend.capabilities
        assert VoiceCapability.INTERRUPTION in backend.capabilities

    def test_custom_params(self):
        aiosipua, rtp_bridge = _make_mock_aiosipua()
        with patch.dict(sys.modules, {"aiosipua": aiosipua, "aiosipua.rtp_bridge": rtp_bridge}):
            sip_mod = _get_sip_module(aiosipua, rtp_bridge)
            backend = sip_mod.SIPVoiceBackend(
                local_sip_addr=("127.0.0.1", 5061),
                rtp_port_start=30000,
                rtp_port_end=31000,
                supported_codecs=[0, 8],
            )

        assert backend._local_sip_addr == ("127.0.0.1", 5061)
        assert backend._supported_codecs == [0, 8]
