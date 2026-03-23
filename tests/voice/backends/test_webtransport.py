"""Tests for the WebTransport voice backend."""

from __future__ import annotations

import importlib
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from roomkit.voice.base import VoiceCapability, VoiceSessionState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER_STRUCT = struct.Struct("<H")


def _make_mock_aioquic():
    """Build a fake aioquic module tree."""
    mod = SimpleNamespace(
        asyncio=SimpleNamespace(
            serve=AsyncMock(),
            QuicConnectionProtocol=type("QuicConnectionProtocol", (), {}),
        ),
        quic=SimpleNamespace(
            configuration=SimpleNamespace(
                QuicConfiguration=MagicMock(),
            ),
        ),
        h3=SimpleNamespace(
            connection=SimpleNamespace(
                H3Connection=MagicMock(),
            ),
            events=SimpleNamespace(
                DatagramReceived=type("DatagramReceived", (), {}),
                HeadersReceived=type("HeadersReceived", (), {}),
                WebTransportStreamDataReceived=type("WebTransportStreamDataReceived", (), {}),
            ),
        ),
    )
    return mod


def _import_backend():
    """Import WebTransportBackend (aioquic only needed at start(), not __init__)."""
    import roomkit.voice.backends.webtransport as wt_mod

    importlib.reload(wt_mod)
    return wt_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebTransportBackendConstructor:
    def test_defaults(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        assert backend.name == "webtransport"
        assert VoiceCapability.INTERRUPTION in backend.capabilities
        assert backend._port == 4433
        assert backend._path == "/audio"

    def test_custom_params(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend(
            host="127.0.0.1",
            port=5555,
            input_sample_rate=48000,
            output_sample_rate=24000,
            path="/voice",
        )
        assert backend._host == "127.0.0.1"
        assert backend._port == 5555
        assert backend._input_sample_rate == 48000
        assert backend._output_sample_rate == 24000
        assert backend._path == "/voice"


class TestDatagramHeaderEncoding:
    def test_16khz_header(self):
        """16000 Hz → 160 as uint16 LE."""
        encoded = _HEADER_STRUCT.pack(16000 // 100)
        assert len(encoded) == 2
        assert _HEADER_STRUCT.unpack(encoded)[0] == 160

    def test_48khz_header(self):
        """48000 Hz → 480 as uint16 LE."""
        encoded = _HEADER_STRUCT.pack(48000 // 100)
        assert _HEADER_STRUCT.unpack(encoded)[0] == 480

    def test_roundtrip(self):
        for rate in (8000, 16000, 24000, 44100, 48000):
            encoded = _HEADER_STRUCT.pack(rate // 100)
            decoded = _HEADER_STRUCT.unpack(encoded)[0] * 100
            assert decoded == rate


class TestWebTransportBackendConnect:
    async def test_connect_creates_session(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        session = await backend.connect("room-1", "user-1", "voice-1")
        assert session.room_id == "room-1"
        assert session.state == VoiceSessionState.ACTIVE
        assert backend.get_session(session.id) is session

    async def test_disconnect_cleans_up(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.disconnect(session)

        assert session.state == VoiceSessionState.ENDED
        assert backend.get_session(session.id) is None

    async def test_list_sessions(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        await backend.connect("room-A", "u1", "v1")
        await backend.connect("room-A", "u2", "v2")
        await backend.connect("room-B", "u3", "v3")

        assert len(backend.list_sessions("room-A")) == 2
        assert len(backend.list_sessions("room-B")) == 1


class TestWebTransportBackendCallbacks:
    def test_on_audio_received(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        received = []
        backend.on_audio_received(lambda s, f: received.append(f))
        assert backend._audio_received_callback is not None

    def test_on_session_ready(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        backend.on_session_ready(lambda s: None)
        assert len(backend._session_ready_callbacks) == 1

    def test_on_client_disconnected(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        backend.on_client_disconnected(lambda s: None)
        assert len(backend._disconnect_callbacks) == 1


class TestWebTransportBackendClose:
    async def test_close_disconnects_all(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        s1 = await backend.connect("room-1", "u1", "v1")
        s2 = await backend.connect("room-2", "u2", "v2")

        await backend.close()

        assert s1.state == VoiceSessionState.ENDED
        assert s2.state == VoiceSessionState.ENDED


class TestHandleDatagram:
    async def test_inbound_datagram_fires_callback(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        received = []
        backend.on_audio_received(lambda s, f: received.append((s, f)))

        session = await backend.connect("room-1", "u1", "v1")

        # Wire up the stream mapping manually (normally done by _on_client_connect)
        stream_id = 42
        backend._stream_sessions[stream_id] = session.id
        protocol = MagicMock()
        backend._session_transports[session.id] = (protocol, stream_id)

        # Build a datagram: 2-byte header + PCM data
        header = _HEADER_STRUCT.pack(16000 // 100)
        pcm = b"\x00" * 320
        backend._handle_datagram(protocol, stream_id, header + pcm)

        assert len(received) == 1
        assert received[0][0] is session
        assert received[0][1].data == pcm

    async def test_short_datagram_ignored(self):
        wt_mod = _import_backend()
        backend = wt_mod.WebTransportBackend()

        received = []
        backend.on_audio_received(lambda s, f: received.append(f))

        # Only header, no audio — should be silently ignored
        backend._handle_datagram(MagicMock(), 0, b"\xa0\x00")
        assert len(received) == 0
