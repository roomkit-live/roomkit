"""Tests for pluggable transport authentication."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.voice.auth import AuthCallback, auth_context
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.ws_transport import WebSocketRealtimeTransport


def _make_session(session_id: str = "sess-1") -> VoiceSession:
    return VoiceSession(
        id=session_id,
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice",
        state=VoiceSessionState.ACTIVE,
    )


# -------------------------------------------------------------------------
# WebSocketRealtimeTransport auth tests
# -------------------------------------------------------------------------


async def test_ws_transport_auth_accept() -> None:
    """Successful auth should merge metadata into session."""

    async def auth_ok(conn: Any) -> dict[str, Any] | None:
        return {"user_id": "alice", "role": "admin"}

    transport = WebSocketRealtimeTransport(authenticate=auth_ok)

    session = _make_session()
    mock_ws = AsyncMock()
    # Simulate async iteration (empty — no messages)
    mock_ws.__aiter__ = MagicMock(return_value=iter([]))

    await transport.accept(session, mock_ws)

    assert session.metadata["user_id"] == "alice"
    assert session.metadata["role"] == "admin"
    assert session.id in transport._sessions

    await transport.close()


async def test_ws_transport_auth_reject() -> None:
    """Auth returning None should raise PermissionError."""

    async def auth_reject(conn: Any) -> dict[str, Any] | None:
        return None

    transport = WebSocketRealtimeTransport(authenticate=auth_reject)

    session = _make_session()
    mock_ws = AsyncMock()

    with pytest.raises(PermissionError, match="Authentication rejected"):
        await transport.accept(session, mock_ws)

    # Session should not be registered
    assert session.id not in transport._sessions

    await transport.close()


async def test_ws_transport_auth_error() -> None:
    """Auth raising an exception should raise PermissionError."""

    async def auth_error(conn: Any) -> dict[str, Any] | None:
        raise ValueError("bad token")

    transport = WebSocketRealtimeTransport(authenticate=auth_error)

    session = _make_session()
    mock_ws = AsyncMock()

    with pytest.raises(PermissionError, match="Authentication failed"):
        await transport.accept(session, mock_ws)

    assert session.id not in transport._sessions

    await transport.close()


async def test_ws_transport_no_auth() -> None:
    """Transport without auth should accept connections directly."""
    transport = WebSocketRealtimeTransport()

    session = _make_session()
    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=iter([]))

    await transport.accept(session, mock_ws)
    assert session.id in transport._sessions

    await transport.close()


# -------------------------------------------------------------------------
# audio_format tests
# -------------------------------------------------------------------------


async def test_ws_transport_send_audio_binary() -> None:
    """audio_format='binary' should send raw bytes, not base64 JSON."""
    transport = WebSocketRealtimeTransport(audio_format="binary")

    session = _make_session()
    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=iter([]))

    await transport.accept(session, mock_ws)

    audio = b"\x00\x01\x02\x03"
    await transport.send_audio(session, audio)

    mock_ws.send.assert_called_once_with(audio)

    await transport.close()


async def test_ws_transport_send_audio_base64_json() -> None:
    """audio_format='base64_json' (default) should send base64-encoded JSON."""
    import base64
    import json

    transport = WebSocketRealtimeTransport()

    session = _make_session()
    mock_ws = AsyncMock()
    mock_ws.__aiter__ = MagicMock(return_value=iter([]))

    await transport.accept(session, mock_ws)

    audio = b"\x00\x01\x02\x03"
    await transport.send_audio(session, audio)

    sent = mock_ws.send.call_args[0][0]
    parsed = json.loads(sent)
    assert parsed["type"] == "audio"
    assert base64.b64decode(parsed["data"]) == audio

    await transport.close()


# -------------------------------------------------------------------------
# AuthCallback type + auth_context tests
# -------------------------------------------------------------------------


async def test_auth_context_contextvar() -> None:
    """auth_context should default to None and be settable."""
    assert auth_context.get() is None

    token = auth_context.set({"user": "bob"})
    assert auth_context.get() == {"user": "bob"}

    auth_context.reset(token)
    assert auth_context.get() is None


async def test_auth_callback_type_alias() -> None:
    """AuthCallback type alias should accept an async callable."""

    async def my_auth(conn: Any) -> dict[str, Any] | None:
        return {"ok": True}

    cb: AuthCallback = my_auth
    result = await cb("fake_conn")
    assert result == {"ok": True}
