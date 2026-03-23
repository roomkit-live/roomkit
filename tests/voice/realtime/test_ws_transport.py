"""Tests for WebSocketRealtimeTransport."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

from roomkit.voice.base import VoiceSession


def _make_session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(
        id=sid,
        room_id="room1",
        participant_id="p1",
        channel_id="ch1",
    )


def _load_module():
    """Import the transport module with websockets mocked."""
    fake_ws = SimpleNamespace(
        connect=lambda *a, **kw: None,
    )
    mods = {"websockets": fake_ws}
    with patch.dict(sys.modules, mods):
        import roomkit.voice.realtime.ws_transport as mod

        importlib.reload(mod)
        return mod


class TestWebSocketRealtimeTransport:
    def test_constructor_and_name(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        assert transport.name == "WebSocketRealtimeTransport"

    def test_default_audio_format(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        assert transport._audio_format == "base64_json"

    def test_binary_audio_format(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport(audio_format="binary")
        assert transport._audio_format == "binary"

    def test_callback_registration_audio(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()

        cb = lambda session, audio: None  # noqa: E731
        transport.on_audio_received(cb)
        assert cb in transport._audio_callbacks

    def test_callback_registration_disconnect(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()

        cb = lambda session: None  # noqa: E731
        transport.on_client_disconnected(cb)
        assert cb in transport._disconnect_callbacks

    async def test_disconnect_unknown_session_is_noop(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        session = _make_session("unknown")
        # Should not raise
        await transport.disconnect(session)

    async def test_close_empty_transport(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        await transport.close()

    async def test_send_audio_no_connection(self):
        mod = _load_module()
        transport = mod.WebSocketRealtimeTransport()
        session = _make_session("s1")
        # No websocket accepted, should be a no-op
        await transport.send_audio(session, b"\x00\x01\x02")

    async def test_constructor_with_auth(self):
        mod = _load_module()

        async def fake_auth(connection):
            return {"user_id": "u1"}

        transport = mod.WebSocketRealtimeTransport(authenticate=fake_auth)
        assert transport._authenticate is fake_auth
