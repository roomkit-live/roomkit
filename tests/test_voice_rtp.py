"""Tests for the RTP voice backend."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("aiortp")

from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.rtp import RTPVoiceBackend
from roomkit.voice.base import AudioChunk, VoiceCapability, VoiceSessionState
from roomkit.voice.pipeline.dtmf.base import DTMFEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_rtp_session() -> MagicMock:
    """Return a mock aiortp.RTPSession with the expected interface."""
    session = MagicMock()
    session.on_audio = None
    session.on_dtmf = None
    session.send_audio_pcm = MagicMock()
    session.close = AsyncMock()
    return session


async def _chunks_from_bytes(data: bytes, chunk_size: int = 320) -> Any:
    """Yield AudioChunks from a bytes buffer."""
    offset = 0
    while offset < len(data):
        yield AudioChunk(data=data[offset : offset + chunk_size], sample_rate=8000)
        offset += chunk_size
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRTPVoiceBackend:
    """Tests for RTPVoiceBackend."""

    @pytest.fixture
    def mock_rtp_session(self) -> MagicMock:
        return _make_mock_rtp_session()

    @pytest.fixture
    async def backend(self, mock_rtp_session: MagicMock) -> RTPVoiceBackend:
        """Return a backend with mocked aiortp.RTPSession.create."""
        with patch("roomkit.voice.backends.rtp._import_aiortp") as mock_import:
            mock_mod = MagicMock()
            mock_mod.RTPSession.create = AsyncMock(return_value=mock_rtp_session)
            mock_import.return_value = mock_mod

            backend = RTPVoiceBackend(
                local_addr=("127.0.0.1", 10000),
                remote_addr=("192.168.1.1", 20000),
                payload_type=0,
                clock_rate=8000,
            )
        return backend

    # -- connect / disconnect -------------------------------------------------

    async def test_connect_creates_session(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")

        assert session.room_id == "room-1"
        assert session.participant_id == "user-1"
        assert session.channel_id == "voice-1"
        assert session.state == VoiceSessionState.ACTIVE
        assert session.metadata["backend"] == "rtp"
        assert session.metadata["remote_addr"] == ("192.168.1.1", 20000)

    async def test_connect_sets_rtp_callbacks(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        await backend.connect("room-1", "user-1", "voice-1")

        # Callbacks are set on the mock RTPSession
        assert mock_rtp_session.on_audio is not None
        assert mock_rtp_session.on_dtmf is not None

    async def test_connect_remote_addr_from_metadata(self, mock_rtp_session: MagicMock) -> None:
        with patch("roomkit.voice.backends.rtp._import_aiortp") as mock_import:
            mock_mod = MagicMock()
            mock_mod.RTPSession.create = AsyncMock(return_value=mock_rtp_session)
            mock_import.return_value = mock_mod

            backend = RTPVoiceBackend(
                local_addr=("127.0.0.1", 10000),
                # No remote_addr at init
            )

        session = await backend.connect(
            "room-1",
            "user-1",
            "voice-1",
            metadata={"remote_addr": ("10.0.0.1", 5060)},
        )
        assert session.metadata["remote_addr"] == ("10.0.0.1", 5060)

    async def test_connect_no_remote_addr_raises(self) -> None:
        with patch("roomkit.voice.backends.rtp._import_aiortp") as mock_import:
            mock_mod = MagicMock()
            mock_import.return_value = mock_mod

            backend = RTPVoiceBackend(local_addr=("127.0.0.1", 10000))

        with pytest.raises(ValueError, match="remote_addr must be provided"):
            await backend.connect("room-1", "user-1", "voice-1")

    async def test_disconnect_closes_rtp_session(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")
        await backend.disconnect(session)

        mock_rtp_session.close.assert_awaited_once()
        assert session.state == VoiceSessionState.ENDED
        assert backend.get_session(session.id) is None

    async def test_close_disconnects_all(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        s1 = await backend.connect("room-1", "user-1", "voice-1")
        s2 = await backend.connect("room-1", "user-2", "voice-2")

        await backend.close()

        assert s1.state == VoiceSessionState.ENDED
        assert s2.state == VoiceSessionState.ENDED

    # -- session queries -------------------------------------------------------

    async def test_get_session(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert backend.get_session(session.id) is session
        assert backend.get_session("nonexistent") is None

    async def test_list_sessions(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        s1 = await backend.connect("room-1", "user-1", "voice-1")
        await backend.connect("room-2", "user-2", "voice-2")

        room1_sessions = backend.list_sessions("room-1")
        assert len(room1_sessions) == 1
        assert room1_sessions[0].id == s1.id

    # -- capabilities ----------------------------------------------------------

    def test_capabilities(self, backend: RTPVoiceBackend) -> None:
        assert VoiceCapability.DTMF_SIGNALING in backend.capabilities
        assert VoiceCapability.INTERRUPTION in backend.capabilities

    def test_name(self, backend: RTPVoiceBackend) -> None:
        assert backend.name == "RTP"

    # -- inbound audio ---------------------------------------------------------

    async def test_inbound_audio_fires_callback(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        received: list[tuple[Any, AudioFrame]] = []
        backend.on_audio_received(lambda s, f: received.append((s, f)))

        session = await backend.connect("room-1", "user-1", "voice-1")

        # Simulate aiortp delivering decoded PCM
        on_audio = mock_rtp_session.on_audio
        pcm_data = b"\x00\x01" * 160  # 160 samples
        on_audio(pcm_data, 0)

        assert len(received) == 1
        sess, frame = received[0]
        assert sess.id == session.id
        assert frame.data == pcm_data
        assert frame.sample_rate == 8000
        assert frame.channels == 1
        assert frame.sample_width == 2

    async def test_inbound_audio_no_callback_is_noop(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        # Don't register a callback
        await backend.connect("room-1", "user-1", "voice-1")
        on_audio = mock_rtp_session.on_audio
        # Should not raise
        on_audio(b"\x00\x01" * 160, 0)

    # -- inbound DTMF ----------------------------------------------------------

    async def test_dtmf_fires_callback(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        received: list[tuple[Any, DTMFEvent]] = []
        backend.on_dtmf_received(lambda s, e: received.append((s, e)))

        session = await backend.connect("room-1", "user-1", "voice-1")

        # Simulate aiortp delivering DTMF: digit "5", duration 1280 (timestamp units)
        on_dtmf = mock_rtp_session.on_dtmf
        on_dtmf("5", 1280)

        assert len(received) == 1
        sess, event = received[0]
        assert sess.id == session.id
        assert event.digit == "5"
        # 1280 / 8000 * 1000 = 160ms
        assert event.duration_ms == 160.0
        assert event.confidence == 1.0

    async def test_dtmf_multiple_callbacks(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        calls_a: list[str] = []
        calls_b: list[str] = []
        backend.on_dtmf_received(lambda s, e: calls_a.append(e.digit))
        backend.on_dtmf_received(lambda s, e: calls_b.append(e.digit))

        await backend.connect("room-1", "user-1", "voice-1")
        mock_rtp_session.on_dtmf("#", 800)

        assert calls_a == ["#"]
        assert calls_b == ["#"]

    # -- outbound audio (bytes) ------------------------------------------------

    async def test_send_audio_bytes(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")

        # 20ms frame at 8kHz = 160 samples = 320 bytes
        pcm = b"\x00\x01" * 160
        await backend.send_audio(session, pcm)

        mock_rtp_session.send_audio_pcm.assert_called_once_with(pcm, 0)

    async def test_send_audio_bytes_multiple_frames(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")

        # Two 20ms frames = 640 bytes
        pcm = b"\x00\x01" * 320
        await backend.send_audio(session, pcm)

        calls = mock_rtp_session.send_audio_pcm.call_args_list
        assert len(calls) == 2
        # First frame: timestamp 0, second: timestamp 160
        assert calls[0].args == (pcm[:320], 0)
        assert calls[1].args == (pcm[320:], 160)

    # -- outbound audio (streaming) --------------------------------------------

    async def test_send_audio_stream(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")

        # Stream one 20ms chunk
        pcm = b"\x00\x01" * 160

        await backend.send_audio(session, _chunks_from_bytes(pcm))

        mock_rtp_session.send_audio_pcm.assert_called_once_with(pcm, 0)

    # -- cancel / is_playing ---------------------------------------------------

    async def test_cancel_audio(
        self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock
    ) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")

        # Not playing -> cancel returns False
        assert await backend.cancel_audio(session) is False

    async def test_is_playing(self, backend: RTPVoiceBackend, mock_rtp_session: MagicMock) -> None:
        session = await backend.connect("room-1", "user-1", "voice-1")
        assert backend.is_playing(session) is False
