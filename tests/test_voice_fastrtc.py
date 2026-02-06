"""Tests for FastRTCVoiceBackend."""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend, _pcm16_to_mulaw
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSession, VoiceSessionState

# ---------------------------------------------------------------------------
# mu-law encoder tests
# ---------------------------------------------------------------------------


class TestMuLawEncoder:
    def test_silence_encodes_to_known_value(self) -> None:
        """PCM silence (0x0000) should encode to mu-law 0xFF."""
        pcm = struct.pack("<h", 0)
        result = _pcm16_to_mulaw(pcm)
        assert len(result) == 1
        # Mu-law encoding of 0 is 0xFF (positive zero)
        assert result[0] == 0xFF

    def test_output_length_matches_sample_count(self) -> None:
        """Each 16-bit sample produces exactly one mu-law byte."""
        n_samples = 100
        pcm = struct.pack(f"<{n_samples}h", *range(n_samples))
        result = _pcm16_to_mulaw(pcm)
        assert len(result) == n_samples

    def test_positive_and_negative_differ_in_sign_bit(self) -> None:
        """Positive and negative samples of same magnitude differ only in sign bit (0x80)."""
        pos_pcm = struct.pack("<h", 1000)
        neg_pcm = struct.pack("<h", -1000)
        pos_result = _pcm16_to_mulaw(pos_pcm)[0]
        neg_result = _pcm16_to_mulaw(neg_pcm)[0]
        # Sign bit is 0x80: positive samples have it set, negative don't
        assert (pos_result & 0x80) != (neg_result & 0x80)
        assert (pos_result & 0x7F) == (neg_result & 0x7F)

    def test_clipping_at_max(self) -> None:
        """Samples at max int16 should be clipped (not crash)."""
        pcm = struct.pack("<h", 32767)
        result = _pcm16_to_mulaw(pcm)
        assert len(result) == 1

    def test_clipping_at_min(self) -> None:
        """Samples at min int16 should be clipped (not crash)."""
        pcm = struct.pack("<h", -32768)
        result = _pcm16_to_mulaw(pcm)
        assert len(result) == 1

    def test_empty_input(self) -> None:
        """Empty PCM data should produce empty output."""
        result = _pcm16_to_mulaw(b"")
        assert result == b""

    def test_multi_sample_encoding(self) -> None:
        """Encode multiple samples and verify output length."""
        samples = [0, 100, -100, 1000, -1000, 32767, -32768]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = _pcm16_to_mulaw(pcm)
        assert len(result) == len(samples)

    def test_louder_signal_produces_lower_mulaw_magnitude(self) -> None:
        """Mu-law is companded: louder signals map to lower byte values (after inversion)."""
        quiet_pcm = struct.pack("<h", 100)
        loud_pcm = struct.pack("<h", 20000)
        quiet_val = _pcm16_to_mulaw(quiet_pcm)[0] & 0x7F
        loud_val = _pcm16_to_mulaw(loud_pcm)[0] & 0x7F
        # Lower mu-law values (ignoring sign) = louder signal
        assert loud_val < quiet_val


# ---------------------------------------------------------------------------
# FastRTCVoiceBackend tests
# ---------------------------------------------------------------------------


class TestFastRTCVoiceBackendProperties:
    def test_name(self) -> None:
        backend = FastRTCVoiceBackend()
        assert backend.name == "FastRTC"

    def test_capabilities_none(self) -> None:
        backend = FastRTCVoiceBackend()
        assert backend.capabilities == VoiceCapability.NONE

    def test_custom_sample_rates(self) -> None:
        backend = FastRTCVoiceBackend(input_sample_rate=16000, output_sample_rate=8000)
        assert backend._input_sample_rate == 16000
        assert backend._output_sample_rate == 8000


class TestFastRTCSessionManagement:
    async def test_connect_creates_session(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.channel_id == "voice-1"
        assert session.state == VoiceSessionState.ACTIVE

    async def test_connect_includes_sample_rates_in_metadata(self) -> None:
        backend = FastRTCVoiceBackend(input_sample_rate=48000, output_sample_rate=24000)
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert session.metadata["input_sample_rate"] == 48000
        assert session.metadata["output_sample_rate"] == 24000

    async def test_connect_merges_custom_metadata(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect(
            "room-1", "user-1", "voice-1", metadata={"custom": "value"}
        )
        assert session.metadata["custom"] == "value"
        assert "input_sample_rate" in session.metadata

    async def test_disconnect_ends_session(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.disconnect(session)
        assert session.state == VoiceSessionState.ENDED
        assert backend.get_session(session.id) is None

    async def test_get_session(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        found = backend.get_session(session.id)
        assert found is not None
        assert found.id == session.id

    async def test_get_session_not_found(self) -> None:
        backend = FastRTCVoiceBackend()
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions_by_room(self) -> None:
        backend = FastRTCVoiceBackend()
        await backend.connect("room-1", "user-1", "voice-1")
        await backend.connect("room-1", "user-2", "voice-1")
        await backend.connect("room-2", "user-3", "voice-1")

        room1 = backend.list_sessions("room-1")
        assert len(room1) == 2

        room2 = backend.list_sessions("room-2")
        assert len(room2) == 1

    async def test_close_disconnects_all_sessions(self) -> None:
        backend = FastRTCVoiceBackend()
        s1 = await backend.connect("room-1", "user-1", "voice-1")
        s2 = await backend.connect("room-1", "user-2", "voice-1")
        await backend.close()

        assert s1.state == VoiceSessionState.ENDED
        assert s2.state == VoiceSessionState.ENDED
        assert backend.get_session(s1.id) is None
        assert backend.get_session(s2.id) is None


class TestFastRTCOnAudioReceived:
    """Tests for on_audio_received and _handle_audio_frame."""

    async def test_on_audio_received_registers_callback(self) -> None:
        backend = FastRTCVoiceBackend()
        callback = MagicMock()
        backend.on_audio_received(callback)
        assert backend._audio_received_callback is callback

    async def test_handle_audio_frame_fires_callback(self) -> None:
        numpy = pytest.importorskip("numpy")
        backend = FastRTCVoiceBackend()
        frames: list[tuple[str, AudioFrame]] = []

        def callback(session: VoiceSession, frame: AudioFrame) -> None:
            frames.append((session.id, frame))

        backend.on_audio_received(callback)
        session = await backend.connect("room-1", "user-1", "voice-1")
        backend._register_websocket("ws-1", session.id, MagicMock())

        audio_data = numpy.array([100, 200, 300], dtype=numpy.int16)
        backend._handle_audio_frame("ws-1", audio_data, 16000)

        assert len(frames) == 1
        assert frames[0][0] == session.id
        assert frames[0][1].data == audio_data.tobytes()
        assert frames[0][1].sample_rate == 16000
        assert frames[0][1].channels == 1

    async def test_handle_audio_frame_converts_float_to_int16(self) -> None:
        numpy = pytest.importorskip("numpy")
        backend = FastRTCVoiceBackend()
        frames: list[tuple[str, AudioFrame]] = []

        def callback(session: VoiceSession, frame: AudioFrame) -> None:
            frames.append((session.id, frame))

        backend.on_audio_received(callback)
        session = await backend.connect("room-1", "user-1", "voice-1")
        backend._register_websocket("ws-1", session.id, MagicMock())

        float_audio = numpy.array([0.5, -0.5], dtype=numpy.float32)
        backend._handle_audio_frame("ws-1", float_audio, 16000)

        assert len(frames) == 1
        expected = numpy.array([16383, -16383], dtype=numpy.int16).tobytes()
        assert frames[0][1].data == expected

    async def test_handle_audio_frame_flattens_multichannel(self) -> None:
        numpy = pytest.importorskip("numpy")
        backend = FastRTCVoiceBackend()
        frames: list[tuple[str, AudioFrame]] = []

        def callback(session: VoiceSession, frame: AudioFrame) -> None:
            frames.append((session.id, frame))

        backend.on_audio_received(callback)
        session = await backend.connect("room-1", "user-1", "voice-1")
        backend._register_websocket("ws-1", session.id, MagicMock())

        multichannel = numpy.array([[100, 200], [300, 400]], dtype=numpy.int16)
        backend._handle_audio_frame("ws-1", multichannel, 16000)

        assert len(frames) == 1
        expected = numpy.array([100, 200, 300, 400], dtype=numpy.int16).tobytes()
        assert frames[0][1].data == expected

    async def test_handle_audio_frame_no_callback(self) -> None:
        """No crash if _handle_audio_frame fires without a registered callback."""
        numpy = pytest.importorskip("numpy")
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        backend._register_websocket("ws-1", session.id, MagicMock())
        # Should not raise
        audio_data = numpy.array([100], dtype=numpy.int16)
        backend._handle_audio_frame("ws-1", audio_data, 16000)

    async def test_handle_audio_frame_unknown_websocket(self) -> None:
        """No crash when websocket ID is not mapped to a session."""
        numpy = pytest.importorskip("numpy")
        backend = FastRTCVoiceBackend()
        callback = MagicMock()
        backend.on_audio_received(callback)
        # No session registered for "ws-unknown"
        audio_data = numpy.array([100], dtype=numpy.int16)
        backend._handle_audio_frame("ws-unknown", audio_data, 16000)
        callback.assert_not_called()


class TestFastRTCWebSocketRegistration:
    async def test_register_websocket(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        ws = MagicMock()

        backend._register_websocket("ws-123", session.id, ws)
        assert backend._websockets[session.id] is ws
        assert session.metadata["websocket_id"] == "ws-123"

    async def test_find_session_by_websocket_id(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        backend._register_websocket("ws-123", session.id, MagicMock())

        found = backend._find_session_by_websocket_id("ws-123")
        assert found is not None
        assert found.id == session.id

    async def test_find_session_by_websocket_id_not_found(self) -> None:
        backend = FastRTCVoiceBackend()
        assert backend._find_session_by_websocket_id("nonexistent") is None

    async def test_disconnect_removes_websocket(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        backend._register_websocket("ws-1", session.id, MagicMock())
        await backend.disconnect(session)
        assert session.id not in backend._websockets


class TestFastRTCSendAudio:
    async def test_send_audio_bytes_no_websocket(self) -> None:
        """send_audio should not crash when no WebSocket is registered."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        # Should not raise
        await backend.send_audio(session, b"\x00\x00")

    async def test_send_audio_bytes_sends_mulaw(self) -> None:
        """send_audio with bytes should convert to mu-law and send via WebSocket."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        ws = AsyncMock()
        backend._register_websocket("ws-1", session.id, ws)

        pcm = struct.pack("<h", 1000)  # one sample
        await backend.send_audio(session, pcm)

        ws.send_json.assert_called_once()
        call_args = ws.send_json.call_args[0][0]
        assert call_args["event"] == "media"
        assert "payload" in call_args["media"]

    async def test_send_audio_stream(self) -> None:
        """send_audio with async iterator should stream chunks."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        ws = AsyncMock()
        backend._register_websocket("ws-1", session.id, ws)

        async def audio_gen():
            yield AudioChunk(data=struct.pack("<h", 100))
            yield AudioChunk(data=struct.pack("<h", 200))

        await backend.send_audio(session, audio_gen())
        assert ws.send_json.call_count == 2


class TestFastRTCSendTranscription:
    async def test_send_transcription_with_websocket(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        ws = AsyncMock()
        backend._register_websocket("ws-1", session.id, ws)

        await backend.send_transcription(session, "Hello", "user")

        ws.send_json.assert_called_once()
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "transcription"
        assert call_args["data"]["text"] == "Hello"
        assert call_args["data"]["role"] == "user"

    async def test_send_transcription_without_websocket(self) -> None:
        """send_transcription should not crash without a WebSocket."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        # Should not raise
        await backend.send_transcription(session, "Hello", "user")

    async def test_send_transcription_default_role(self) -> None:
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        ws = AsyncMock()
        backend._register_websocket("ws-1", session.id, ws)

        await backend.send_transcription(session, "Hello")

        call_args = ws.send_json.call_args[0][0]
        assert call_args["data"]["role"] == "user"


class TestFastRTCResolveWebSocket:
    """Test _resolve_websocket fallback to Stream.connections."""

    async def test_resolve_from_explicit_registry(self) -> None:
        """When websocket is registered explicitly, _resolve_websocket finds it."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect("room-1", "user-1", "voice-1")
        mock_ws = AsyncMock()
        backend._register_websocket("ws-1", session.id, mock_ws)

        resolved = backend._resolve_websocket(session)
        assert resolved is mock_ws

    async def test_resolve_falls_back_to_stream_connections(self) -> None:
        """When no explicit registration, falls back to Stream.connections."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect(
            "room-1",
            "user-1",
            "voice-1",
            metadata={"websocket_id": "ws-test-123"},
        )

        assert session.id not in backend._websockets

        mock_ws = AsyncMock()
        mock_handler = MagicMock()
        mock_handler.websocket = mock_ws
        mock_stream = MagicMock()
        mock_stream.connections = {"ws-test-123": [mock_handler]}
        backend._stream = mock_stream

        resolved = backend._resolve_websocket(session)
        assert resolved is mock_ws
        assert backend._websockets[session.id] is mock_ws

    async def test_resolve_returns_none_when_no_connection(self) -> None:
        """Returns None when no websocket exists anywhere."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect(
            "room-1",
            "user-1",
            "voice-1",
            metadata={"websocket_id": "ws-missing"},
        )

        mock_stream = MagicMock()
        mock_stream.connections = {}
        backend._stream = mock_stream

        resolved = backend._resolve_websocket(session)
        assert resolved is None

    async def test_send_audio_uses_resolved_websocket(self) -> None:
        """send_audio should work via Stream.connections fallback."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect(
            "room-1",
            "user-1",
            "voice-1",
            metadata={"websocket_id": "ws-fallback"},
        )

        mock_ws = AsyncMock()
        mock_handler = MagicMock()
        mock_handler.websocket = mock_ws
        mock_stream = MagicMock()
        mock_stream.connections = {"ws-fallback": [mock_handler]}
        backend._stream = mock_stream

        pcm = struct.pack("<h", 1000)
        await backend.send_audio(session, pcm)
        mock_ws.send_json.assert_called_once()

    async def test_send_transcription_uses_resolved_websocket(self) -> None:
        """send_transcription should work via Stream.connections fallback."""
        backend = FastRTCVoiceBackend()
        session = await backend.connect(
            "room-1",
            "user-1",
            "voice-1",
            metadata={"websocket_id": "ws-fallback"},
        )

        mock_ws = AsyncMock()
        mock_handler = MagicMock()
        mock_handler.websocket = mock_ws
        mock_stream = MagicMock()
        mock_stream.connections = {"ws-fallback": [mock_handler]}
        backend._stream = mock_stream

        await backend.send_transcription(session, "Hello", "assistant")
        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "transcription"
        assert call_args["data"]["text"] == "Hello"


class TestFastRTCLazyLoader:
    def test_get_fastrtc_backend_returns_class(self) -> None:
        from roomkit.voice import get_fastrtc_backend

        cls = get_fastrtc_backend()
        assert cls is FastRTCVoiceBackend

    def test_get_mount_fastrtc_voice_returns_function(self) -> None:
        from roomkit.voice import get_mount_fastrtc_voice
        from roomkit.voice.backends.fastrtc import mount_fastrtc_voice

        fn = get_mount_fastrtc_voice()
        assert fn is mount_fastrtc_voice
