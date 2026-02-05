"""Tests for FastRTCRealtimeTransport."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from roomkit.voice.realtime.base import RealtimeSession


def _make_session(
    session_id: str = "session-1",
    room_id: str = "room-1",
    participant_id: str = "user-1",
    channel_id: str = "voice",
) -> RealtimeSession:
    return RealtimeSession(
        id=session_id,
        room_id=room_id,
        participant_id=participant_id,
        channel_id=channel_id,
    )


@pytest.fixture
def transport():
    # Patch numpy import inside the module so we don't need it installed
    from roomkit.voice.realtime.fastrtc_transport import FastRTCRealtimeTransport

    return FastRTCRealtimeTransport(input_sample_rate=16000, output_sample_rate=24000)


@pytest.fixture
def handler(transport):
    """Create a _PassthroughHandler with mocked numpy."""
    from roomkit.voice.realtime.fastrtc_transport import _PassthroughHandler

    with patch("roomkit.voice.realtime.fastrtc_transport.asyncio"):
        pass  # Just ensure import works

    return _PassthroughHandler(
        transport,
        input_sample_rate=16000,
        output_sample_rate=24000,
    )


class TestFastRTCRealtimeTransport:
    async def test_name(self, transport) -> None:
        assert transport.name == "FastRTCRealtimeTransport"

    async def test_accept_maps_session_to_webrtc_id(self, transport) -> None:
        session = _make_session()
        webrtc_id = "webrtc-123"

        await transport.accept(session, webrtc_id)

        assert transport._sessions[session.id] is session
        assert transport._session_handlers[session.id] == webrtc_id
        assert transport._webrtc_sessions[webrtc_id] is session

    async def test_send_audio_queues_to_handler(self, transport, handler) -> None:
        session = _make_session()
        webrtc_id = "webrtc-123"

        # Register handler and accept session
        transport._register_handler(webrtc_id, handler)
        await transport.accept(session, webrtc_id)

        audio_data = b"\x00\x01" * 100
        await transport.send_audio(session, audio_data)

        # Check audio was queued
        queued = handler._audio_queue.get_nowait()
        assert queued == audio_data

    async def test_send_audio_no_handler_noop(self, transport) -> None:
        """send_audio with no handler registered should not raise."""
        session = _make_session()
        await transport.accept(session, "webrtc-missing")
        # Should not raise
        await transport.send_audio(session, b"\x00\x01")

    async def test_send_message_via_data_channel(self, transport, handler) -> None:
        session = _make_session()
        webrtc_id = "webrtc-123"

        # Set up a mock data channel
        handler.channel = MagicMock()
        handler.channel.send = MagicMock()

        transport._register_handler(webrtc_id, handler)
        await transport.accept(session, webrtc_id)

        message = {"type": "transcription", "text": "hello"}
        await transport.send_message(session, message)

        handler.channel.send.assert_called_once()
        import json

        sent_data = handler.channel.send.call_args[0][0]
        assert json.loads(sent_data) == message

    async def test_send_message_no_channel_noop(self, transport, handler) -> None:
        """send_message with no data channel should not raise."""
        session = _make_session()
        webrtc_id = "webrtc-123"

        handler.channel = None
        transport._register_handler(webrtc_id, handler)
        await transport.accept(session, webrtc_id)

        # Should not raise
        await transport.send_message(session, {"type": "test"})

    async def test_receive_fires_audio_callbacks(self, transport, handler) -> None:
        import numpy as np

        session = _make_session()
        webrtc_id = "webrtc-123"

        received: list[tuple[str, bytes]] = []

        def audio_cb(sess: RealtimeSession, audio: bytes) -> None:
            received.append((sess.id, audio))

        transport.on_audio_received(audio_cb)
        transport._register_handler(webrtc_id, handler)
        handler._webrtc_id = webrtc_id
        await transport.accept(session, webrtc_id)

        # Simulate receiving a frame
        audio_array = np.array([100, -200, 300], dtype=np.int16)
        frame = (16000, audio_array)
        await handler.receive(frame)

        assert len(received) == 1
        assert received[0][0] == session.id
        assert received[0][1] == audio_array.tobytes()

    async def test_receive_before_session_bound_is_noop(
        self, transport, handler
    ) -> None:
        """receive() before a session is accepted should be a no-op."""
        import numpy as np

        webrtc_id = "webrtc-123"
        transport._register_handler(webrtc_id, handler)
        handler._webrtc_id = webrtc_id

        received: list[tuple[str, bytes]] = []
        transport.on_audio_received(lambda s, a: received.append((s.id, a)))

        audio_array = np.array([100], dtype=np.int16)
        await handler.receive((16000, audio_array))

        # No session bound, so no callback fired
        assert len(received) == 0

    async def test_disconnect_cleans_up(self, transport, handler) -> None:
        session = _make_session()
        webrtc_id = "webrtc-123"

        transport._register_handler(webrtc_id, handler)
        await transport.accept(session, webrtc_id)

        await transport.disconnect(session)

        # Session mappings removed
        assert session.id not in transport._sessions
        assert session.id not in transport._session_handlers

        # Handler receives None sentinel
        sentinel = handler._audio_queue.get_nowait()
        assert sentinel is None

    async def test_on_client_connected_callback(self, transport) -> None:
        connected_ids: list[str] = []

        transport.on_client_connected(lambda wid: connected_ids.append(wid))

        # Simulate handler registration (which happens during start_up)
        handler = MagicMock()
        transport._register_handler("webrtc-abc", handler)

        assert connected_ids == ["webrtc-abc"]

    async def test_on_client_connected_async_callback(self, transport) -> None:
        connected_ids: list[str] = []

        async def on_connected(wid: str) -> None:
            connected_ids.append(wid)

        transport.on_client_connected(on_connected)

        handler = MagicMock()
        transport._register_handler("webrtc-def", handler)

        # Allow the ensure_future to run
        await asyncio.sleep(0.01)

        assert connected_ids == ["webrtc-def"]

    async def test_multiple_sessions(self, transport) -> None:
        """Multiple WebRTC connections should coexist independently."""
        from roomkit.voice.realtime.fastrtc_transport import _PassthroughHandler

        session1 = _make_session("session-1")
        session2 = _make_session("session-2", participant_id="user-2")

        handler1 = _PassthroughHandler(
            transport, input_sample_rate=16000, output_sample_rate=24000
        )
        handler2 = _PassthroughHandler(
            transport, input_sample_rate=16000, output_sample_rate=24000
        )

        transport._register_handler("webrtc-1", handler1)
        transport._register_handler("webrtc-2", handler2)
        await transport.accept(session1, "webrtc-1")
        await transport.accept(session2, "webrtc-2")

        # Send audio to each
        await transport.send_audio(session1, b"audio-1")
        await transport.send_audio(session2, b"audio-2")

        assert handler1._audio_queue.get_nowait() == b"audio-1"
        assert handler2._audio_queue.get_nowait() == b"audio-2"

        # Disconnect one doesn't affect the other
        await transport.disconnect(session1)
        assert session1.id not in transport._sessions
        assert session2.id in transport._sessions

    async def test_close_disconnects_all(self, transport) -> None:
        from roomkit.voice.realtime.fastrtc_transport import _PassthroughHandler

        session1 = _make_session("session-1")
        session2 = _make_session("session-2")

        handler1 = _PassthroughHandler(
            transport, input_sample_rate=16000, output_sample_rate=24000
        )
        handler2 = _PassthroughHandler(
            transport, input_sample_rate=16000, output_sample_rate=24000
        )

        transport._register_handler("webrtc-1", handler1)
        transport._register_handler("webrtc-2", handler2)
        await transport.accept(session1, "webrtc-1")
        await transport.accept(session2, "webrtc-2")

        await transport.close()

        assert len(transport._sessions) == 0
        assert len(transport._session_handlers) == 0

    async def test_unregister_handler_fires_disconnect_callbacks(
        self, transport
    ) -> None:
        session = _make_session()
        webrtc_id = "webrtc-123"

        disconnected: list[str] = []

        async def on_disconnect(sess: RealtimeSession) -> None:
            disconnected.append(sess.id)

        transport.on_client_disconnected(on_disconnect)

        handler = MagicMock()
        transport._register_handler(webrtc_id, handler)
        await transport.accept(session, webrtc_id)

        # Simulate handler shutdown (unregister)
        transport._unregister_handler(webrtc_id)

        # Allow ensure_future to run
        await asyncio.sleep(0.01)

        assert disconnected == [session.id]

    async def test_get_session_returns_none_for_unknown(self, transport) -> None:
        assert transport._get_session("unknown") is None
        assert transport._get_session(None) is None

    async def test_handler_copy(self, handler) -> None:
        """copy() should create a new handler with same config."""
        copied = handler.copy()

        assert copied is not handler
        assert copied.input_sample_rate == handler.input_sample_rate
        assert copied.output_sample_rate == handler.output_sample_rate
        assert copied._transport is handler._transport
        assert copied._session is None
        assert copied._webrtc_id is None

    async def test_handler_emit_timeout_returns_none(self, handler) -> None:
        """emit() should return None on timeout (empty queue)."""
        result = await handler.emit()
        assert result is None

    async def test_handler_emit_none_sentinel(self, handler) -> None:
        """emit() should return None when sentinel is queued."""
        handler._audio_queue.put_nowait(None)
        result = await handler.emit()
        assert result is None

    async def test_handler_emit_audio(self, handler) -> None:
        """emit() should return (sample_rate, ndarray) for queued audio."""
        import numpy as np

        pcm_bytes = np.array([100, -200, 300], dtype=np.int16).tobytes()
        handler._audio_queue.put_nowait(pcm_bytes)

        result = await handler.emit()
        assert result is not None
        rate, arr = result
        assert rate == 24000
        np.testing.assert_array_equal(arr, np.array([100, -200, 300], dtype=np.int16))

    async def test_handler_shutdown_unregisters(self, transport, handler) -> None:
        webrtc_id = "webrtc-123"
        transport._register_handler(webrtc_id, handler)
        handler._webrtc_id = webrtc_id

        assert webrtc_id in transport._handlers

        handler.shutdown()

        assert webrtc_id not in transport._handlers
