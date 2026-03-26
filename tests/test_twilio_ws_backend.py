"""Tests for TwilioWebSocketBackend."""

from __future__ import annotations

import asyncio
import base64
import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends._mulaw import pcm16_to_mulaw
from roomkit.voice.backends.twilio_ws import TWILIO_SAMPLE_RATE, TwilioWebSocketBackend
from roomkit.voice.base import AudioChunk, VoiceSession


def _make_mulaw_payload(num_samples: int = 160) -> str:
    """Create a base64-encoded mu-law payload (8 kHz silence)."""
    pcm = b"\x00\x00" * num_samples
    mulaw = pcm16_to_mulaw(pcm)
    return base64.b64encode(mulaw).decode("ascii")


def _make_pcm(num_samples: int = 160, value: int = 0) -> bytes:
    """Create 16-bit mono PCM data."""
    return struct.pack(f"<{num_samples}h", *([value] * num_samples))


class TestConstruction:
    def test_name(self) -> None:
        backend = TwilioWebSocketBackend()
        assert backend.name == "twilio-ws"

    def test_default_output_rate(self) -> None:
        backend = TwilioWebSocketBackend()
        assert backend._output_sample_rate == 24000

    def test_custom_output_rate(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=16000)
        assert backend._output_sample_rate == 16000

    def test_same_rate_resampler_is_passthrough(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=TWILIO_SAMPLE_RATE)
        data = b"\x01\x02\x03\x04"
        assert backend._resample_inbound(data) == data


class TestSessionLifecycle:
    async def test_connect_creates_session(self) -> None:
        backend = TwilioWebSocketBackend()
        session = await backend.connect("room1", "user1", "ch1")

        assert session.room_id == "room1"
        assert session.participant_id == "user1"
        assert session.channel_id == "ch1"
        assert session.id == "twilio-room1-user1"

    async def test_connect_with_metadata(self) -> None:
        backend = TwilioWebSocketBackend()
        session = await backend.connect("r", "u", "c", metadata={"key": "val"})
        assert session.metadata == {"key": "val"}

    async def test_get_session(self) -> None:
        backend = TwilioWebSocketBackend()
        session = await backend.connect("r1", "u1", "c1")
        assert backend.get_session(session.id) is session
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions(self) -> None:
        backend = TwilioWebSocketBackend()
        s1 = await backend.connect("room1", "u1", "c1")
        s2 = await backend.connect("room1", "u2", "c1")
        await backend.connect("room2", "u3", "c1")

        room1_sessions = backend.list_sessions("room1")
        assert len(room1_sessions) == 2
        assert s1 in room1_sessions
        assert s2 in room1_sessions
        assert len(backend.list_sessions("room2")) == 1

    async def test_disconnect_removes_session(self) -> None:
        backend = TwilioWebSocketBackend()
        session = await backend.connect("r1", "u1", "c1")
        await backend.disconnect(session)
        assert backend.get_session(session.id) is None

    async def test_disconnect_fires_callback(self) -> None:
        backend = TwilioWebSocketBackend()
        cb = MagicMock()
        backend.on_client_disconnected(cb)
        session = await backend.connect("r", "u", "c")
        await backend.disconnect(session)
        cb.assert_called_once_with(session)

    async def test_disconnect_clears_state(self) -> None:
        backend = TwilioWebSocketBackend()
        ws = AsyncMock()
        backend.bind_websocket(ws)

        session = await backend.connect("r", "u", "c")
        await backend.disconnect(session)

        assert backend._websocket is None
        assert backend._write_queue is None
        assert backend._websocket is None


class TestCallbacks:
    def test_on_audio_received(self) -> None:
        backend = TwilioWebSocketBackend()
        cb = MagicMock()
        backend.on_audio_received(cb)
        assert backend._audio_received_cb is cb

    def test_on_session_ready(self) -> None:
        backend = TwilioWebSocketBackend()
        cb = MagicMock()
        backend.on_session_ready(cb)
        assert backend._session_ready_cb is cb

    def test_on_client_disconnected(self) -> None:
        backend = TwilioWebSocketBackend()
        cb = MagicMock()
        backend.on_client_disconnected(cb)
        assert backend._transport_disconnect_cb is cb

    def test_notify_session_ready(self) -> None:
        backend = TwilioWebSocketBackend()
        cb = MagicMock()
        backend.on_session_ready(cb)
        session = VoiceSession(id="s1", room_id="r", participant_id="p", channel_id="c")
        backend.notify_session_ready(session)
        cb.assert_called_once_with(session)

    def test_notify_session_ready_no_callback(self) -> None:
        backend = TwilioWebSocketBackend()
        session = VoiceSession(id="s1", room_id="r", participant_id="p", channel_id="c")
        backend.notify_session_ready(session)  # should not raise


class TestInboundAudio:
    async def test_feed_twilio_audio_fires_callback(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=TWILIO_SAMPLE_RATE)
        frames: list[AudioFrame] = []
        backend.on_audio_received(lambda session, frame: frames.append(frame))

        session = await backend.connect("r", "u", "c")
        payload = _make_mulaw_payload(160)
        await backend.feed_twilio_audio(session, payload)

        assert len(frames) == 1
        assert frames[0].sample_rate == TWILIO_SAMPLE_RATE
        assert frames[0].channels == 1
        assert frames[0].sample_width == 2

    async def test_feed_twilio_audio_resamples(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=16000)
        frames: list[AudioFrame] = []
        backend.on_audio_received(lambda session, frame: frames.append(frame))

        session = await backend.connect("r", "u", "c")
        payload = _make_mulaw_payload(160)
        await backend.feed_twilio_audio(session, payload)

        # With resampling, the output rate should be the pipeline rate
        assert len(frames) >= 1
        assert frames[0].sample_rate == 16000

    async def test_feed_without_callback_no_error(self) -> None:
        backend = TwilioWebSocketBackend()
        session = await backend.connect("r", "u", "c")
        await backend.feed_twilio_audio(session, _make_mulaw_payload())


class TestOutboundAudio:
    async def test_send_audio_bytes_no_websocket(self) -> None:
        backend = TwilioWebSocketBackend()
        session = await backend.connect("r", "u", "c")
        # Should not raise when no websocket bound
        await backend.send_audio(session, _make_pcm(160))

    async def test_send_audio_bytes_via_queue(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=TWILIO_SAMPLE_RATE)
        ws = AsyncMock()
        backend.bind_websocket(ws)

        session = await backend.connect("r", "u", "c")
        await backend.send_audio(session, _make_pcm(160, value=1000))
        # Give the writer task time to process
        await asyncio.sleep(0.05)

        assert ws.send_json.call_count == 1
        msg = ws.send_json.call_args[0][0]
        assert msg["event"] == "media"
        assert msg["conversation_id"] == "r"
        assert "payload" in msg["media"]

    async def test_send_audio_resamples_to_8khz(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=16000)
        ws = AsyncMock()
        backend.bind_websocket(ws)

        session = await backend.connect("r", "u", "c")
        pcm_16k = _make_pcm(320)  # 320 samples at 16kHz = 20ms
        await backend.send_audio(session, pcm_16k)
        await asyncio.sleep(0.05)

        assert ws.send_json.call_count == 1
        # Decode the payload back to verify it's mu-law
        msg = ws.send_json.call_args[0][0]
        mulaw_data = base64.b64decode(msg["media"]["payload"])
        # Should be roughly 160 samples (20ms at 8kHz)
        assert len(mulaw_data) == pytest.approx(160, abs=20)

    async def test_send_audio_stream(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=TWILIO_SAMPLE_RATE)
        ws = AsyncMock()
        backend.bind_websocket(ws)

        async def _chunks():
            yield AudioChunk(data=_make_pcm(160, value=500))
            yield AudioChunk(data=_make_pcm(160, value=500))

        session = await backend.connect("r", "u", "c")
        await backend.send_audio(session, _chunks())
        await asyncio.sleep(0.05)

        assert ws.send_json.call_count == 2

    async def test_send_without_writer_uses_direct_send(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=TWILIO_SAMPLE_RATE)
        ws = AsyncMock()
        # Set websocket directly without bind (no writer task)
        backend._websocket = ws
        session = await backend.connect("r", "u", "c")
        await backend.send_audio(session, _make_pcm(160, value=100))
        ws.send_json.assert_called_once()

    async def test_send_empty_pcm_skipped(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=TWILIO_SAMPLE_RATE)
        ws = AsyncMock()
        backend.bind_websocket(ws)

        session = await backend.connect("r", "u", "c")
        await backend._send_mulaw_frame(session, b"")
        await asyncio.sleep(0.05)

        ws.send_json.assert_not_called()


class TestWriterTask:
    async def test_bind_websocket_creates_writer(self) -> None:
        backend = TwilioWebSocketBackend()
        ws = AsyncMock()
        backend.bind_websocket(ws)

        assert backend._write_queue is not None
        assert backend._writer_task is not None

    async def test_disconnect_stops_writer(self) -> None:
        backend = TwilioWebSocketBackend()
        ws = AsyncMock()
        backend.bind_websocket(ws)

        session = await backend.connect("r", "u", "c")
        await backend.disconnect(session)

        assert backend._writer_task is None
        assert backend._write_queue is None

    async def test_writer_handles_websocket_error(self) -> None:
        backend = TwilioWebSocketBackend(output_sample_rate=TWILIO_SAMPLE_RATE)
        ws = AsyncMock()
        ws.send_json.side_effect = ConnectionError("closed")
        backend.bind_websocket(ws)

        session = await backend.connect("r", "u", "c")
        await backend.send_audio(session, _make_pcm(160, value=100))
        # Writer should handle the error gracefully
        await asyncio.sleep(0.05)

        # Disconnect should not raise even though writer already exited
        await backend.disconnect(session)


class TestResampler:
    def test_build_passthrough_at_same_rate(self) -> None:
        resampler = TwilioWebSocketBackend._build_resampler(8000, 8000)
        data = b"\x01\x02\x03\x04"
        assert resampler(data) is data

    def test_build_upsample(self) -> None:
        resampler = TwilioWebSocketBackend._build_resampler(8000, 16000)
        pcm_8k = _make_pcm(160)  # 20ms at 8kHz
        result = resampler(pcm_8k)
        # Upsampled to 16kHz: ~320 samples = 640 bytes
        assert len(result) == pytest.approx(640, abs=20)

    def test_build_downsample(self) -> None:
        resampler = TwilioWebSocketBackend._build_resampler(16000, 8000)
        pcm_16k = _make_pcm(320)  # 20ms at 16kHz
        result = resampler(pcm_16k)
        # Downsampled to 8kHz: ~160 samples = 320 bytes
        assert len(result) == pytest.approx(320, abs=20)
