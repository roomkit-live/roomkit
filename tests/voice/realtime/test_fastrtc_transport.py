"""Tests for FastRTCRealtimeTransport."""

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
    """Import the transport module with fastrtc and numpy mocked."""
    fake_np = SimpleNamespace(
        int16="int16",
        ndarray=type("ndarray", (), {}),
    )

    # AsyncStreamHandler needs to be a class that can be subclassed
    class FakeAsyncStreamHandler:
        def __init__(self, **kwargs):
            self.expected_layout = kwargs.get("expected_layout", "mono")
            self.output_sample_rate = kwargs.get("output_sample_rate", 24000)
            self.input_sample_rate = kwargs.get("input_sample_rate", 16000)
            self.channel = None

    fake_fastrtc = SimpleNamespace(
        AsyncStreamHandler=FakeAsyncStreamHandler,
        Stream=lambda **kw: SimpleNamespace(mount=lambda app, path: None),
    )

    fake_fastrtc_utils = SimpleNamespace(
        current_context=SimpleNamespace(get=lambda: None),
    )

    mods = {
        "numpy": fake_np,
        "fastrtc": fake_fastrtc,
        "fastrtc.utils": fake_fastrtc_utils,
    }
    with patch.dict(sys.modules, mods):
        import roomkit.voice.realtime.fastrtc_transport as mod

        importlib.reload(mod)
        return mod


class TestFastRTCRealtimeTransport:
    def test_constructor_and_name(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport()
        assert transport.name == "FastRTCRealtimeTransport"

    def test_constructor_with_custom_rates(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport(
            input_sample_rate=8000,
            output_sample_rate=16000,
        )
        assert transport._input_sample_rate == 8000
        assert transport._output_sample_rate == 16000

    def test_callback_registration_audio(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport()

        cb = lambda session, audio: None  # noqa: E731
        transport.on_audio_received(cb)
        assert cb in transport._audio_callbacks

    def test_callback_registration_disconnect(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport()

        cb = lambda session: None  # noqa: E731
        transport.on_client_disconnected(cb)
        assert cb in transport._disconnect_callbacks

    def test_callback_registration_connected(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport()

        cb = lambda webrtc_id: None  # noqa: E731
        transport.on_client_connected(cb)
        assert transport._connected_callback is cb

    async def test_accept_and_disconnect(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport()
        session = _make_session("s1")

        await transport.accept(session, "webrtc-123")
        assert session.id in transport._sessions
        assert transport._session_handlers[session.id] == "webrtc-123"

        await transport.disconnect(session)
        assert session.id not in transport._sessions
        assert session.id not in transport._session_handlers

    async def test_close_empty_transport(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport()
        await transport.close()

    async def test_close_with_sessions(self):
        mod = _load_module()
        transport = mod.FastRTCRealtimeTransport()
        session = _make_session("s1")
        await transport.accept(session, "webrtc-456")

        await transport.close()
        assert len(transport._sessions) == 0
