"""Audio and A/V mounts share connection registration and auth propagation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.video.backends.fastrtc import FastRTCVideoBackend
from roomkit.voice.auth import auth_context
from roomkit.voice.backends.fastrtc import FastRTCVoiceBackend
from roomkit.voice.base import VoiceSession


@pytest.mark.parametrize("backend_type", [FastRTCVoiceBackend, FastRTCVideoBackend])
@pytest.mark.parametrize("websocket_mode", [False, True])
async def test_factory_auth_registration_and_audio_forwarding(
    backend_type, websocket_mode: bool
) -> None:
    backend = backend_type()
    websocket = MagicMock() if websocket_mode else None
    metadata = {"user": "alice"}
    seen = []

    async def factory(connection_id: str) -> VoiceSession:
        seen.append(auth_context.get())
        return await backend.connect("r1", "alice", "voice")

    backend._session_factory = factory
    backend._webrtc_auth_meta["connection"] = metadata
    backend._handle_audio_frame = MagicMock()
    samples = object()
    token = auth_context.set({"parent": True})
    try:
        await backend._receive_audio(
            "connection", websocket, (16000, samples), metadata if websocket_mode else None
        )
        assert seen == [metadata]
        assert auth_context.get() == {"parent": True}
        session = backend._find_session_by_websocket_id("connection")
        assert session is not None
        if websocket_mode:
            assert backend._websockets[session.id] is websocket
        else:
            assert session.metadata["transport"] == "webrtc"
            assert "connection" in backend._emit_queues
            if isinstance(backend, FastRTCVideoBackend):
                assert "connection" in backend._video_emit_queues
        await backend._receive_audio("connection", websocket, (16000, samples), None)
        assert len(seen) == 1
        assert backend._handle_audio_frame.call_count == 2
        backend._handle_audio_frame.assert_called_with("connection", samples, 16000)
    finally:
        auth_context.reset(token)
        await backend.close()


@pytest.mark.parametrize("backend_type", [FastRTCVoiceBackend, FastRTCVideoBackend])
@pytest.mark.parametrize("raises", [False, True])
async def test_failed_session_creation_restores_auth_and_drops_audio(
    backend_type, raises: bool
) -> None:
    backend = backend_type()
    factory = (
        AsyncMock(side_effect=RuntimeError("refused")) if raises else AsyncMock(return_value=None)
    )
    backend._session_factory = factory
    backend._handle_audio_frame = MagicMock()
    token = auth_context.set({"parent": True})
    try:
        await backend._receive_audio("connection", None, (16000, object()), {"user": "alice"})
        assert auth_context.get() == {"parent": True}
        backend._handle_audio_frame.assert_not_called()
        assert backend._find_session_by_websocket_id("connection") is None
    finally:
        auth_context.reset(token)
        await backend.close()
