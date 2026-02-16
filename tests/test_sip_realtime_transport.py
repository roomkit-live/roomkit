"""Tests for SIPRealtimeTransport — bridges SIP audio to realtime providers."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.sip_transport import SIPRealtimeTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAudioFrame:
    """Minimal AudioFrame-like object."""

    data: bytes
    sample_rate: int = 8000
    channels: int = 1
    sample_width: int = 2


@dataclass
class FakeVoiceSession:
    """Minimal VoiceSession-like object."""

    id: str = "sip-session-1"
    room_id: str = "room-1"
    participant_id: str = "caller-1"
    channel_id: str = "voice"
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_sip_backend() -> MagicMock:
    """Create a mock SIPVoiceBackend."""
    backend = MagicMock()
    backend._audio_received_callback = None
    backend._codec_rates = {}  # real dict so .get() returns int, not MagicMock
    backend.on_audio_received = MagicMock(
        side_effect=lambda cb: setattr(backend, "_audio_received_callback", cb)
    )
    backend.send_audio = AsyncMock()
    backend.cancel_audio = AsyncMock(return_value=True)
    backend.end_of_response = MagicMock()
    return backend


def _make_rt_session(session_id: str = "rt-1", room_id: str = "room-1") -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id=room_id,
        participant_id="caller-1",
        channel_id="realtime-voice",
        state=VoiceSessionState.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Transport tests
# ---------------------------------------------------------------------------


class TestSIPRealtimeTransportInit:
    def test_properties(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        assert transport.name == "SIPRealtimeTransport"

    def test_wires_audio_callback(self) -> None:
        backend = _make_sip_backend()
        SIPRealtimeTransport(backend)
        backend.on_audio_received.assert_called_once()


class TestAccept:
    @pytest.mark.asyncio
    async def test_accept_stores_mappings(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        rt_session = _make_rt_session()
        voice_session = FakeVoiceSession()

        await transport.accept(rt_session, voice_session)

        assert transport._voice_sessions[rt_session.id] is voice_session
        assert transport._rt_sessions[rt_session.id] is rt_session
        assert transport._voice_to_rt[voice_session.id] == rt_session.id

        await transport.disconnect(rt_session)


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        rt_session = _make_rt_session()
        voice_session = FakeVoiceSession()
        await transport.accept(rt_session, voice_session)
        await transport.disconnect(rt_session)

        assert rt_session.id not in transport._voice_sessions
        assert rt_session.id not in transport._rt_sessions
        assert voice_session.id not in transport._voice_to_rt

    @pytest.mark.asyncio
    async def test_disconnect_unknown_session(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        rt_session = _make_rt_session("unknown")
        await transport.disconnect(rt_session)  # Should not raise


class TestSendAudio:
    @pytest.mark.asyncio
    async def test_send_audio_delegates_to_backend(self) -> None:
        """send_audio() delegates directly to backend.send_audio()."""
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        rt_session = _make_rt_session()
        voice_session = FakeVoiceSession()
        await transport.accept(rt_session, voice_session)

        audio = struct.pack("<160h", *([500] * 160))
        await transport.send_audio(rt_session, audio)

        backend.send_audio.assert_awaited_once_with(voice_session, audio)

        await transport.disconnect(rt_session)

    @pytest.mark.asyncio
    async def test_send_audio_unknown_session_noop(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        rt_session = _make_rt_session("unknown")
        await transport.send_audio(rt_session, b"\x00" * 100)
        backend.send_audio.assert_not_awaited()


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_send_message_is_noop(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        rt_session = _make_rt_session()
        await transport.send_message(rt_session, {"type": "transcription", "text": "hello"})


class TestInboundAudio:
    @pytest.mark.asyncio
    async def test_sip_audio_forwarded_directly(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        rt_session = _make_rt_session()
        voice_session = FakeVoiceSession()
        await transport.accept(rt_session, voice_session)

        received: list[tuple[str, bytes]] = []

        def on_audio(session: VoiceSession, audio: bytes) -> None:
            received.append((session.id, audio))

        transport.on_audio_received(on_audio)

        # Simulate SIP backend delivering 10ms of 8kHz audio (80 samples)
        raw_audio = struct.pack("<80h", *([500] * 80))
        frame = FakeAudioFrame(data=raw_audio)
        transport._on_sip_audio(voice_session, frame)

        assert len(received) == 1
        assert received[0][0] == rt_session.id
        # Audio is passed through without resampling
        assert received[0][1] == raw_audio

    @pytest.mark.asyncio
    async def test_sip_audio_unknown_session_ignored(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        received: list[Any] = []
        transport.on_audio_received(lambda s, a: received.append(1))

        frame = FakeAudioFrame(data=struct.pack("<80h", *([0] * 80)))
        transport._on_sip_audio(FakeVoiceSession(id="unknown"), frame)

        assert len(received) == 0


class TestClose:
    @pytest.mark.asyncio
    async def test_close_disconnects_all(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        rt1 = _make_rt_session("rt-1")
        rt2 = _make_rt_session("rt-2")
        vs1 = FakeVoiceSession(id="sip-1")
        vs2 = FakeVoiceSession(id="sip-2")

        await transport.accept(rt1, vs1)
        await transport.accept(rt2, vs2)

        await transport.close()

        assert len(transport._rt_sessions) == 0
        assert len(transport._voice_sessions) == 0
        assert len(transport._voice_to_rt) == 0


class TestDelegation:
    @pytest.mark.asyncio
    async def test_interrupt_delegates_to_cancel_audio(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        rt_session = _make_rt_session()
        voice_session = FakeVoiceSession()
        await transport.accept(rt_session, voice_session)

        transport.interrupt(rt_session)
        await asyncio.sleep(0.05)

        backend.cancel_audio.assert_awaited_once_with(voice_session)

        await transport.disconnect(rt_session)

    @pytest.mark.asyncio
    async def test_end_of_response_delegates_to_backend(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)

        rt_session = _make_rt_session()
        voice_session = FakeVoiceSession()
        await transport.accept(rt_session, voice_session)

        transport.end_of_response(rt_session)
        backend.end_of_response.assert_called_once_with(voice_session)

        await transport.disconnect(rt_session)

    @pytest.mark.asyncio
    async def test_end_of_response_unknown_session_noop(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        rt_session = _make_rt_session("unknown")
        transport.end_of_response(rt_session)
        backend.end_of_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupt_unknown_session_noop(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        rt_session = _make_rt_session("unknown")
        transport.interrupt(rt_session)
        await asyncio.sleep(0.05)
        backend.cancel_audio.assert_not_awaited()


class TestCallbackRegistration:
    def test_on_audio_received(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        cb = MagicMock()
        transport.on_audio_received(cb)
        assert cb in transport._audio_callbacks

    def test_on_client_disconnected(self) -> None:
        backend = _make_sip_backend()
        transport = SIPRealtimeTransport(backend)
        cb = MagicMock()
        transport.on_client_disconnected(cb)
        assert cb in transport._disconnect_callbacks
