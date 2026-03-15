"""Tests for PersonaPlex Realtime provider."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from roomkit.voice.base import VoiceSession, VoiceSessionState

# Protocol constants (mirror provider)
_MSG_HANDSHAKE = 0x00
_MSG_AUDIO = 0x01
_MSG_TEXT = 0x02
_MSG_CONTROL = 0x03
_MSG_ERROR = 0x05

_CTRL_START = 0x00
_CTRL_PAUSE = 0x02


# -- Mock WebSocket --


class MockWebSocket:
    """Mock WebSocket that yields predefined binary messages."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.sent: list[bytes] = []
        self.closed = False

    def push(self, msg: bytes) -> None:
        self._queue.put_nowait(msg)

    def push_close(self) -> None:
        self._queue.put_nowait(None)

    async def recv(self) -> bytes:
        msg = await self._queue.get()
        if msg is None:
            raise ConnectionError("closed")
        return msg

    async def send(self, data: bytes | str) -> None:
        if isinstance(data, str):
            data = data.encode()
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> MockWebSocket:
        return self

    async def __anext__(self) -> bytes:
        msg = await self._queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


# -- Helpers --


def _make_session(session_id: str = "test-sess") -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id="test-room",
        participant_id="test-user",
        channel_id="test-channel",
    )


@dataclass
class _MockOpusWriter:
    """Fake Opus encoder — append_pcm returns encoded bytes directly."""

    def append_pcm(self, pcm: Any) -> bytes:
        return pcm.tobytes()

    def read_bytes(self) -> bytes:
        return b""


@dataclass
class _MockOpusReader:
    """Fake Opus decoder — append_bytes returns decoded PCM directly."""

    def append_bytes(self, data: bytes) -> np.ndarray:
        if not data:
            return np.array([], dtype=np.float32)
        # Treat stored bytes as float32
        return np.frombuffer(data, dtype=np.float32).copy()

    def read_pcm(self) -> np.ndarray:
        return np.array([], dtype=np.float32)


# -- Fixtures --


@pytest.fixture
def mock_ws() -> MockWebSocket:
    return MockWebSocket()


@pytest.fixture
def mock_sphn() -> MagicMock:
    m = MagicMock()
    m.OpusStreamWriter = lambda *_a, **_kw: _MockOpusWriter()
    m.OpusStreamReader = lambda *_a, **_kw: _MockOpusReader()
    return m


@pytest.fixture
def _patch_deps(mock_ws: MockWebSocket, mock_sphn: MagicMock):
    """Patch websockets and sphn for all PersonaPlex tests."""
    mock_ws_mod = MagicMock()
    mock_ws_mod.connect = AsyncMock(return_value=mock_ws)
    with patch.dict(sys.modules, {"websockets": mock_ws_mod, "sphn": mock_sphn}):
        yield


@pytest.fixture
def provider(_patch_deps: None):
    from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider

    return PersonaPlexRealtimeProvider(
        server_url="wss://test:8998/api/chat",
        response_end_timeout=0.15,
    )


# -- Tests --


async def test_connect_sets_active(provider, mock_ws: MockWebSocket) -> None:
    """connect() waits for handshake and sets session ACTIVE."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))
    mock_ws.push_close()

    session = _make_session()
    await provider.connect(session, system_prompt="Hello", voice="NATM0.pt")

    assert session.state == VoiceSessionState.ACTIVE
    assert session.provider_session_id == session.id
    # Should have sent start control
    assert bytes([_MSG_CONTROL, _CTRL_START]) in mock_ws.sent
    await provider.disconnect(session)


async def test_connect_bad_handshake(provider, mock_ws: MockWebSocket) -> None:
    """connect() raises on missing handshake."""
    mock_ws.push(bytes([_MSG_TEXT]) + b"wrong")

    session = _make_session()
    with pytest.raises(ConnectionError, match="handshake"):
        await provider.connect(session)


async def test_send_audio_encodes_and_sends(provider, mock_ws: MockWebSocket) -> None:
    """send_audio() encodes PCM to Opus and sends with 0x01 prefix."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))
    mock_ws.push_close()

    session = _make_session()
    await provider.connect(session)

    pcm = np.zeros(480, dtype=np.int16).tobytes()
    await provider.send_audio(session, pcm)

    # Find audio messages (prefixed with 0x01)
    audio_msgs = [m for m in mock_ws.sent if m and m[0] == _MSG_AUDIO]
    assert len(audio_msgs) == 1
    assert len(audio_msgs[0]) > 1  # prefix + payload
    await provider.disconnect(session)


async def test_receive_audio_fires_callback(provider, mock_ws: MockWebSocket) -> None:
    """Receiving audio (0x01) fires on_audio callback with PCM bytes."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))

    audio_received: list[bytes] = []
    provider.on_audio(lambda _s, audio: audio_received.append(audio))

    session = _make_session()
    await provider.connect(session)

    # Push audio: prefix + float32 samples
    pcm_float = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock_ws.push(bytes([_MSG_AUDIO]) + pcm_float.tobytes())
    mock_ws.push_close()

    # Let receive loop process
    await asyncio.sleep(0.05)

    assert len(audio_received) == 1
    # Should be int16 PCM bytes
    decoded = np.frombuffer(audio_received[0], dtype=np.int16)
    assert len(decoded) == 3
    await provider.disconnect(session)


async def test_receive_text_fires_transcription(provider, mock_ws: MockWebSocket) -> None:
    """Receiving text tokens (0x02) fires transcription callback."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))

    transcriptions: list[tuple[str, str, bool]] = []
    provider.on_transcription(
        lambda _s, text, role, final: transcriptions.append((text, role, final))
    )

    session = _make_session()
    await provider.connect(session)

    mock_ws.push(bytes([_MSG_TEXT]) + b"Hello")
    mock_ws.push(bytes([_MSG_TEXT]) + b" world")
    mock_ws.push_close()

    await asyncio.sleep(0.05)

    # Should have partial transcriptions
    assert len(transcriptions) >= 2
    # First partial: "Hello"
    assert transcriptions[0] == ("Hello", "assistant", False)
    # Second partial: "Hello world"
    assert transcriptions[1] == ("Hello world", "assistant", False)
    await provider.disconnect(session)


async def test_receive_error_fires_callback(provider, mock_ws: MockWebSocket) -> None:
    """Receiving error (0x05) fires on_error callback."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))

    errors: list[tuple[str, str]] = []
    provider.on_error(lambda _s, code, msg: errors.append((code, msg)))

    session = _make_session()
    await provider.connect(session)

    mock_ws.push(bytes([_MSG_ERROR]) + b"something broke")
    mock_ws.push_close()

    await asyncio.sleep(0.05)

    assert len(errors) == 1
    assert errors[0] == ("server_error", "something broke")
    await provider.disconnect(session)


async def test_response_start_end(provider, mock_ws: MockWebSocket) -> None:
    """response_start fires on first audio, response_end after silence."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))

    events: list[str] = []
    provider.on_response_start(lambda _s: events.append("start"))
    provider.on_response_end(lambda _s: events.append("end"))

    session = _make_session()
    await provider.connect(session)

    pcm = np.array([0.5], dtype=np.float32)
    mock_ws.push(bytes([_MSG_AUDIO]) + pcm.tobytes())
    mock_ws.push_close()

    await asyncio.sleep(0.05)
    assert "start" in events

    # Wait for response_end timeout (0.15s configured in fixture)
    await asyncio.sleep(0.2)
    assert "end" in events
    await provider.disconnect(session)


async def test_interrupt_sends_pause(provider, mock_ws: MockWebSocket) -> None:
    """interrupt() sends control pause (0x03, 0x02)."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))
    mock_ws.push_close()

    session = _make_session()
    await provider.connect(session)
    await provider.interrupt(session)

    assert bytes([_MSG_CONTROL, _CTRL_PAUSE]) in mock_ws.sent
    await provider.disconnect(session)


async def test_disconnect_cleanup(provider, mock_ws: MockWebSocket) -> None:
    """disconnect() cancels tasks, closes WS, sets ENDED."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))
    mock_ws.push_close()

    session = _make_session()
    await provider.connect(session)
    await provider.disconnect(session)

    assert session.state == VoiceSessionState.ENDED
    assert mock_ws.closed


async def test_disconnect_flushes_transcription(provider, mock_ws: MockWebSocket) -> None:
    """disconnect() fires final transcription if text buffer has content."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))

    transcriptions: list[tuple[str, bool]] = []
    provider.on_transcription(lambda _s, text, _r, final: transcriptions.append((text, final)))

    session = _make_session()
    await provider.connect(session)

    mock_ws.push(bytes([_MSG_TEXT]) + b"partial")
    await asyncio.sleep(0.05)
    mock_ws.push_close()
    await asyncio.sleep(0.05)

    await provider.disconnect(session)

    # Should have partial + final
    finals = [(t, f) for t, f in transcriptions if f]
    assert len(finals) >= 1
    assert finals[-1][0] == "partial"


async def test_tools_and_temp_warn(provider, mock_ws: MockWebSocket) -> None:
    """Passing tools or temperature logs warnings but doesn't fail."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))
    mock_ws.push_close()

    session = _make_session()
    # Should not raise
    await provider.connect(
        session,
        tools=[{"type": "function", "name": "test"}],
        temperature=0.5,
    )
    await provider.disconnect(session)


async def test_send_event_control(provider, mock_ws: MockWebSocket) -> None:
    """send_event() can send raw control messages."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))
    mock_ws.push_close()

    session = _make_session()
    await provider.connect(session)

    await provider.send_event(session, {"type": "control", "action": 1})
    assert bytes([_MSG_CONTROL, 1]) in mock_ws.sent

    await provider.send_event(session, {"type": "ping"})
    assert bytes([0x06]) in mock_ws.sent
    await provider.disconnect(session)


async def test_close_disconnects_all(provider, mock_ws: MockWebSocket) -> None:
    """close() disconnects all active sessions."""
    mock_ws.push(bytes([_MSG_HANDSHAKE]))
    mock_ws.push_close()

    session = _make_session()
    await provider.connect(session)
    await provider.close()

    assert session.state == VoiceSessionState.ENDED
