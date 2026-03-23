"""Tests for the RTP voice backend."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.voice.base import VoiceCapability, VoiceSessionState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_aiortp():
    """Build a fake aiortp module with an RTPSession.create async factory."""
    rtp_session = MagicMock()
    rtp_session.close = AsyncMock()
    rtp_session.on_audio = None
    rtp_session.on_dtmf = None

    rtp_session_cls = MagicMock()
    rtp_session_cls.create = AsyncMock(return_value=rtp_session)

    mod = SimpleNamespace(RTPSession=rtp_session_cls)
    return mod, rtp_session


def _load_module(mock_mod):
    """Reload the rtp module with aiortp mocked. Caller must keep patch active."""
    import roomkit.voice.backends.rtp as rtp_mod

    importlib.reload(rtp_mod)
    return rtp_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRTPVoiceBackendConstructor:
    def test_defaults(self):
        mock_mod, _ = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend()

        assert backend.name == "RTP"
        assert VoiceCapability.DTMF_SIGNALING in backend.capabilities
        assert VoiceCapability.INTERRUPTION in backend.capabilities

    def test_custom_params(self):
        mock_mod, _ = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(
                local_addr=("127.0.0.1", 5000),
                remote_addr=("10.0.0.1", 6000),
                payload_type=8,
                clock_rate=16000,
            )

        assert backend._payload_type == 8
        assert backend._clock_rate == 16000


class TestRTPVoiceBackendConnect:
    async def test_connect_creates_session(self):
        mock_mod, rtp_session = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(remote_addr=("10.0.0.1", 20000))
            session = await backend.connect("room-1", "user-1", "voice-1")

        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.state == VoiceSessionState.ACTIVE
        assert backend.get_session(session.id) is session

    async def test_connect_requires_remote_addr(self):
        mock_mod, _ = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend()
            with pytest.raises(ValueError, match="remote_addr"):
                await backend.connect("room-1", "user-1", "voice-1")


class TestRTPVoiceBackendDisconnect:
    async def test_disconnect_cleans_up(self):
        mock_mod, rtp_session = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(remote_addr=("10.0.0.1", 20000))
            session = await backend.connect("room-1", "user-1", "voice-1")
            await backend.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        assert backend.get_session(session.id) is None
        rtp_session.close.assert_awaited_once()


class TestRTPVoiceBackendCallbacks:
    async def test_on_audio_received(self):
        mock_mod, rtp_session = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(remote_addr=("10.0.0.1", 20000))

            received = []
            backend.on_audio_received(lambda sess, frame: received.append((sess, frame)))
            session = await backend.connect("room-1", "user-1", "voice-1")

        # The RTP session mock's on_audio was set by connect()
        handler = rtp_session.on_audio
        assert handler is not None
        handler(b"\x00" * 320, 0)
        assert len(received) == 1
        assert received[0][0] is session

    async def test_on_dtmf_received(self):
        mock_mod, rtp_session = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(remote_addr=("10.0.0.1", 20000))

            dtmf_events = []
            backend.on_dtmf_received(lambda sess, evt: dtmf_events.append((sess, evt)))
            _session = await backend.connect("room-1", "user-1", "voice-1")

        handler = rtp_session.on_dtmf
        assert handler is not None
        # duration in RTP timestamp units (8000 Hz -> 1280 = 160ms)
        handler("5", 1280)
        assert len(dtmf_events) == 1
        assert dtmf_events[0][1].digit == "5"

    async def test_on_session_ready(self):
        mock_mod, _ = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(remote_addr=("10.0.0.1", 20000))

            ready = []
            backend.on_session_ready(lambda s: ready.append(s))
            session = await backend.connect("room-1", "user-1", "voice-1")

        assert len(ready) == 1
        assert ready[0] is session


class TestRTPVoiceBackendClose:
    async def test_close_disconnects_all(self):
        mock_mod, rtp_session = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(remote_addr=("10.0.0.1", 20000))

            s1 = await backend.connect("room-1", "user-1", "voice-1")
            s2 = await backend.connect("room-2", "user-2", "voice-2")
            await backend.close()

        assert s1.state == VoiceSessionState.ENDED
        assert s2.state == VoiceSessionState.ENDED
        assert backend.list_sessions("room-1") == []

    async def test_list_sessions(self):
        mock_mod, _ = _make_mock_aiortp()
        with patch.dict(sys.modules, {"aiortp": mock_mod}):
            rtp_mod = _load_module(mock_mod)
            backend = rtp_mod.RTPVoiceBackend(remote_addr=("10.0.0.1", 20000))

            await backend.connect("room-A", "u1", "v1")
            await backend.connect("room-A", "u2", "v2")
            await backend.connect("room-B", "u3", "v3")

        assert len(backend.list_sessions("room-A")) == 2
        assert len(backend.list_sessions("room-B")) == 1
