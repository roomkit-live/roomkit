"""Tests for WebTransportBackend."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.webtransport import (
    _HEADER_SIZE,
    _HEADER_STRUCT,
    WebTransportBackend,
)
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession, VoiceSessionState

# ---------------------------------------------------------------------------
# Datagram header encoding/decoding
# ---------------------------------------------------------------------------


class TestDatagramProtocol:
    def test_header_encodes_sample_rate(self) -> None:
        """Sample rate 16000 encodes as 160 in the 2-byte header."""
        header = _HEADER_STRUCT.pack(16000 // 100)
        assert len(header) == _HEADER_SIZE
        assert _HEADER_STRUCT.unpack(header)[0] == 160

    def test_header_encodes_48000(self) -> None:
        """Sample rate 48000 encodes as 480."""
        header = _HEADER_STRUCT.pack(48000 // 100)
        assert _HEADER_STRUCT.unpack(header)[0] == 480

    def test_header_encodes_8000(self) -> None:
        """Sample rate 8000 encodes as 80."""
        header = _HEADER_STRUCT.pack(8000 // 100)
        assert _HEADER_STRUCT.unpack(header)[0] == 80

    def test_header_roundtrip(self) -> None:
        """Encode then decode produces original value."""
        for rate in (8000, 16000, 24000, 44100, 48000):
            encoded = rate // 100
            packed = _HEADER_STRUCT.pack(encoded)
            decoded = _HEADER_STRUCT.unpack(packed)[0] * 100
            assert decoded == rate


# ---------------------------------------------------------------------------
# Backend construction and properties
# ---------------------------------------------------------------------------


class TestWebTransportBackendProperties:
    def test_name(self) -> None:
        backend = WebTransportBackend()
        assert backend.name == "webtransport"

    def test_capabilities(self) -> None:
        backend = WebTransportBackend()
        assert backend.capabilities == VoiceCapability.INTERRUPTION

    def test_default_config(self) -> None:
        backend = WebTransportBackend()
        assert backend._host == "0.0.0.0"
        assert backend._port == 4433
        assert backend._path == "/audio"
        assert backend._input_sample_rate == 16000
        assert backend._output_sample_rate == 16000

    def test_custom_config(self) -> None:
        backend = WebTransportBackend(
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


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestWebTransportSessions:
    async def test_connect_creates_session(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1",
            participant_id="user1",
            channel_id="voice",
        )
        assert isinstance(session, VoiceSession)
        assert session.room_id == "room1"
        assert session.participant_id == "user1"
        assert session.state == VoiceSessionState.ACTIVE
        assert session.metadata["transport"] == "webtransport"

    async def test_connect_stores_session(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1",
            participant_id="user1",
            channel_id="voice",
        )
        assert backend.get_session(session.id) is session

    async def test_disconnect_removes_session(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1",
            participant_id="user1",
            channel_id="voice",
        )
        await backend.disconnect(session)
        assert backend.get_session(session.id) is None
        assert session.state == VoiceSessionState.ENDED

    async def test_disconnect_cleans_up_transport_mapping(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1",
            participant_id="user1",
            channel_id="voice",
        )
        # Simulate stream mapping
        backend._stream_sessions[42] = session.id
        backend._session_transports[session.id] = (MagicMock(), 42)

        await backend.disconnect(session)
        assert 42 not in backend._stream_sessions
        assert session.id not in backend._session_transports

    async def test_list_sessions_by_room(self) -> None:
        backend = WebTransportBackend()
        s1 = await backend.connect(room_id="room1", participant_id="u1", channel_id="v")
        s2 = await backend.connect(room_id="room1", participant_id="u2", channel_id="v")
        s3 = await backend.connect(room_id="room2", participant_id="u3", channel_id="v")

        room1_sessions = backend.list_sessions("room1")
        assert len(room1_sessions) == 2
        assert s1 in room1_sessions
        assert s2 in room1_sessions
        assert s3 not in room1_sessions

    async def test_get_session_returns_none_for_unknown(self) -> None:
        backend = WebTransportBackend()
        assert backend.get_session("nonexistent") is None

    async def test_connect_with_metadata(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1",
            participant_id="user1",
            channel_id="voice",
            metadata={"custom_key": "custom_value"},
        )
        assert session.metadata["custom_key"] == "custom_value"
        assert session.metadata["transport"] == "webtransport"
        assert session.metadata["input_sample_rate"] == 16000


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


class TestWebTransportCallbacks:
    def test_on_audio_received_registers(self) -> None:
        backend = WebTransportBackend()
        cb = MagicMock()
        backend.on_audio_received(cb)
        assert backend._audio_received_callback is cb

    def test_on_session_ready_registers(self) -> None:
        backend = WebTransportBackend()
        cb = MagicMock()
        backend.on_session_ready(cb)
        assert cb in backend._session_ready_callbacks

    def test_on_client_disconnected_registers(self) -> None:
        backend = WebTransportBackend()
        cb = MagicMock()
        backend.on_client_disconnected(cb)
        assert cb in backend._disconnect_callbacks

    def test_set_session_factory(self) -> None:
        backend = WebTransportBackend()
        factory = MagicMock()
        backend.set_session_factory(factory)
        assert backend._session_factory is factory


# ---------------------------------------------------------------------------
# Inbound datagram handling
# ---------------------------------------------------------------------------


class TestInboundDatagrams:
    async def test_handle_datagram_fires_callback(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        # Set up stream mapping
        stream_id = 10
        backend._stream_sessions[stream_id] = session.id

        received_frames: list[tuple[VoiceSession, AudioFrame]] = []
        backend.on_audio_received(lambda s, f: received_frames.append((s, f)))

        # Build datagram: 2-byte header (16000Hz) + PCM data
        pcm_data = struct.pack("<4h", 100, -200, 300, -400)
        header = _HEADER_STRUCT.pack(16000 // 100)
        datagram = header + pcm_data

        protocol = MagicMock()
        backend._handle_datagram(protocol, stream_id, datagram)

        assert len(received_frames) == 1
        recv_session, recv_frame = received_frames[0]
        assert recv_session is session
        assert recv_frame.data == pcm_data
        assert recv_frame.sample_rate == 16000
        assert recv_frame.channels == 1
        assert recv_frame.sample_width == 2

    def test_handle_datagram_ignores_too_short(self) -> None:
        backend = WebTransportBackend()
        received: list = []
        backend.on_audio_received(lambda s, f: received.append((s, f)))

        backend._handle_datagram(MagicMock(), 10, b"\x00")
        assert len(received) == 0

    def test_handle_datagram_ignores_header_only(self) -> None:
        """A datagram with only the 2-byte header (no audio) is dropped."""
        backend = WebTransportBackend()
        received: list = []
        backend.on_audio_received(lambda s, f: received.append((s, f)))

        header = _HEADER_STRUCT.pack(160)
        backend._handle_datagram(MagicMock(), 10, header)
        assert len(received) == 0

    def test_handle_datagram_ignores_unknown_stream(self) -> None:
        backend = WebTransportBackend()
        received: list = []
        backend.on_audio_received(lambda s, f: received.append((s, f)))

        header = _HEADER_STRUCT.pack(160)
        backend._handle_datagram(MagicMock(), 999, header + b"\x00\x00")
        assert len(received) == 0

    async def test_handle_datagram_decodes_sample_rate(self) -> None:
        """Verify various sample rates are decoded correctly from the header."""
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        backend._stream_sessions[5] = session.id

        received_rates: list[int] = []
        backend.on_audio_received(lambda s, f: received_rates.append(f.sample_rate))

        for rate in (8000, 16000, 24000, 48000):
            header = _HEADER_STRUCT.pack(rate // 100)
            backend._handle_datagram(MagicMock(), 5, header + b"\x00\x00")

        assert received_rates == [8000, 16000, 24000, 48000]


# ---------------------------------------------------------------------------
# Outbound audio send
# ---------------------------------------------------------------------------


class TestOutboundAudio:
    async def test_send_datagram_sends_to_protocol(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        protocol = MagicMock()
        stream_id = 7
        backend._session_transports[session.id] = (protocol, stream_id)

        pcm_data = b"\x01\x00\x02\x00"
        backend._send_datagram(session, pcm_data, 16000)

        protocol.send_datagram.assert_called_once()
        args = protocol.send_datagram.call_args
        assert args[0][0] == stream_id
        sent_data = args[0][1]
        # First 2 bytes are the header
        assert sent_data[:_HEADER_SIZE] == _HEADER_STRUCT.pack(160)
        assert sent_data[_HEADER_SIZE:] == pcm_data

    async def test_send_datagram_no_transport_is_silent(self) -> None:
        """Sending to a session with no transport mapping should not raise."""
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        # No transport registered — should silently skip
        backend._send_datagram(session, b"\x00\x00", 16000)

    async def test_send_audio_sync_schedules_on_loop(self) -> None:
        """send_audio_sync uses call_soon_threadsafe for thread safety."""
        import asyncio

        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        protocol = MagicMock()
        backend._session_transports[session.id] = (protocol, 3)
        backend._loop = asyncio.get_running_loop()

        chunk = AudioChunk(data=b"\x01\x00", sample_rate=16000)
        backend.send_audio_sync(session, chunk)

        # call_soon_threadsafe schedules but doesn't call immediately in
        # the same iteration — give the loop a tick
        await asyncio.sleep(0)
        protocol.send_datagram.assert_called_once()

    async def test_send_audio_sync_no_transport_is_silent(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        # No transport registered — should not raise
        chunk = AudioChunk(data=b"\x01\x00", sample_rate=16000)
        backend.send_audio_sync(session, chunk)

    def test_send_audio_sync_no_loop_is_silent(self) -> None:
        backend = WebTransportBackend()
        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="p1",
            channel_id="v",
            state=VoiceSessionState.ACTIVE,
        )
        backend._session_transports["s1"] = (MagicMock(), 3)
        backend._loop = None
        # Should not raise
        chunk = AudioChunk(data=b"\x01\x00", sample_rate=16000)
        backend.send_audio_sync(session, chunk)

    async def test_send_audio_bytes(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        protocol = MagicMock()
        backend._session_transports[session.id] = (protocol, 3)

        await backend.send_audio(session, b"\x01\x00\x02\x00")
        protocol.send_datagram.assert_called_once()

    async def test_send_audio_async_iterator(self) -> None:
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        protocol = MagicMock()
        backend._session_transports[session.id] = (protocol, 3)

        async def audio_gen():
            yield AudioChunk(data=b"\x01\x00", sample_rate=16000)
            yield AudioChunk(data=b"\x02\x00", sample_rate=24000)

        await backend.send_audio(session, audio_gen())
        assert protocol.send_datagram.call_count == 2

    async def test_send_datagram_protocol_error_is_logged(self) -> None:
        """Protocol errors during send should not propagate."""
        backend = WebTransportBackend()
        session = await backend.connect(
            room_id="room1", participant_id="user1", channel_id="voice"
        )
        protocol = MagicMock()
        protocol.send_datagram.side_effect = RuntimeError("connection lost")
        backend._session_transports[session.id] = (protocol, 3)

        # Should not raise
        backend._send_datagram(session, b"\x00\x00", 16000)

    def test_interrupt_is_noop(self) -> None:
        backend = WebTransportBackend()
        session = VoiceSession(
            id="s1",
            room_id="r1",
            participant_id="p1",
            channel_id="v",
            state=VoiceSessionState.ACTIVE,
        )
        # Should not raise
        backend.interrupt(session)


# ---------------------------------------------------------------------------
# Client connect/disconnect lifecycle
# ---------------------------------------------------------------------------


class TestClientLifecycle:
    async def test_on_client_connect_default_factory(self) -> None:
        backend = WebTransportBackend()
        protocol = MagicMock()
        stream_id = 20

        accepted = await backend._on_client_connect(protocol, stream_id, "/audio")
        assert accepted is True
        assert stream_id in backend._stream_sessions
        session_id = backend._stream_sessions[stream_id]
        assert session_id in backend._session_transports

    async def test_on_client_connect_wrong_path_rejected(self) -> None:
        backend = WebTransportBackend()
        protocol = MagicMock()

        accepted = await backend._on_client_connect(protocol, 20, "/wrong")
        assert accepted is False
        assert 20 not in backend._stream_sessions

    async def test_on_client_connect_fires_session_ready(self) -> None:
        backend = WebTransportBackend()
        ready_sessions: list[VoiceSession] = []
        backend.on_session_ready(lambda s: ready_sessions.append(s))

        accepted = await backend._on_client_connect(MagicMock(), 30, "/audio")
        assert accepted is True
        assert len(ready_sessions) == 1
        assert ready_sessions[0].metadata["transport"] == "webtransport"

    async def test_on_client_connect_with_custom_factory(self) -> None:
        backend = WebTransportBackend()

        custom_session = await backend.connect(
            room_id="custom", participant_id="p1", channel_id="v"
        )
        backend.set_session_factory(lambda conn_id: custom_session)

        accepted = await backend._on_client_connect(MagicMock(), 40, "/audio")
        assert accepted is True
        assert backend._stream_sessions[40] == custom_session.id

    async def test_on_client_connect_with_async_factory(self) -> None:
        backend = WebTransportBackend()

        custom_session = await backend.connect(
            room_id="custom", participant_id="p1", channel_id="v"
        )

        async def async_factory(conn_id: str) -> VoiceSession:
            return custom_session

        backend.set_session_factory(async_factory)

        accepted = await backend._on_client_connect(MagicMock(), 50, "/audio")
        assert accepted is True
        assert backend._stream_sessions[50] == custom_session.id

    async def test_on_client_disconnect_fires_callbacks(self) -> None:
        backend = WebTransportBackend()
        disconnected: list[VoiceSession] = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        # First connect
        await backend._on_client_connect(MagicMock(), 60, "/audio")
        assert 60 in backend._stream_sessions
        session_id = backend._stream_sessions[60]

        # Then disconnect
        await backend._on_client_disconnect(60)
        assert len(disconnected) == 1
        assert disconnected[0].state == VoiceSessionState.ENDED
        assert 60 not in backend._stream_sessions
        # Session should be removed from _sessions
        assert backend.get_session(session_id) is None

    async def test_on_client_disconnect_unknown_stream(self) -> None:
        """Disconnecting an unknown stream should be a no-op."""
        backend = WebTransportBackend()
        disconnected: list = []
        backend.on_client_disconnected(lambda s: disconnected.append(s))

        await backend._on_client_disconnect(999)
        assert len(disconnected) == 0


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    async def test_start_requires_aioquic(self) -> None:
        """start() should raise ImportError when aioquic is not available."""
        backend = WebTransportBackend()
        # We don't install aioquic in test env, so this should fail gracefully
        try:
            await backend.start()
            # If aioquic IS installed, just close the server
            await backend.close()
        except ImportError:
            pass  # Expected when aioquic is not installed

    async def test_close_without_start(self) -> None:
        """close() should be safe to call even if never started."""
        backend = WebTransportBackend()
        await backend.close()  # Should not raise

    async def test_close_disconnects_all_sessions(self) -> None:
        backend = WebTransportBackend()
        s1 = await backend.connect(room_id="r1", participant_id="p1", channel_id="v")
        s2 = await backend.connect(room_id="r1", participant_id="p2", channel_id="v")

        await backend.close()
        assert s1.state == VoiceSessionState.ENDED
        assert s2.state == VoiceSessionState.ENDED
        assert len(backend._sessions) == 0
