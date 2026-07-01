"""OpenAIRealtimeProvider.reconfigure: in-band ``session.update``, no teardown.

Tool Search / skill activation call ``reconfigure`` mid-conversation to expose
newly matched tools. The inherited base implementation reconnects the socket,
which on the OpenAI wire drops the conversation and the in-flight tool call.
These tests pin the override to a partial, in-band ``session.update``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState


@pytest.fixture
def provider() -> OpenAIRealtimeProvider:
    return OpenAIRealtimeProvider(api_key="sk-test", model="gpt-realtime-1.5")


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="u1",
        channel_id="v1",
        state=VoiceSessionState.CONNECTING,
    )


def _mock_ws() -> AsyncMock:
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.__aiter__ = MagicMock(return_value=iter([]))
    return ws


def _inject_ws(provider: OpenAIRealtimeProvider, session: VoiceSession, ws: AsyncMock) -> None:
    """Wire a live socket into provider internals without calling connect()."""
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE


def _sent_events(ws: AsyncMock) -> list[dict]:
    return [json.loads(call.args[0]) for call in ws.send.call_args_list]


async def test_reconfigure_sends_partial_session_update_in_band(
    provider: OpenAIRealtimeProvider, session: VoiceSession
) -> None:
    ws = _mock_ws()
    _inject_ws(provider, session, ws)

    tools = [
        {"type": "function", "name": "calendar", "description": "d", "parameters": {}},
        {"type": "function", "name": "find_tools", "description": "d", "parameters": {}},
    ]
    await provider.reconfigure(session, tools=tools, system_prompt="new prompt")

    events = _sent_events(ws)
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "session.update"
    assert evt["session"]["instructions"] == "new prompt"
    assert [t["name"] for t in evt["session"]["tools"]] == ["calendar", "find_tools"]

    # In-band: the live socket is preserved, never torn down.
    ws.close.assert_not_called()
    assert provider._connections[session.id] is ws


async def test_reconfigure_does_not_reconnect(
    provider: OpenAIRealtimeProvider, session: VoiceSession
) -> None:
    """The whole point: no disconnect()/connect() round-trip."""
    ws = _mock_ws()
    _inject_ws(provider, session, ws)

    provider.disconnect = AsyncMock()  # type: ignore[method-assign]
    provider.connect = AsyncMock()  # type: ignore[method-assign]

    await provider.reconfigure(session, tools=[{"name": "x", "parameters": {}}])

    provider.disconnect.assert_not_called()
    provider.connect.assert_not_called()


async def test_reconfigure_omits_unspecified_fields(
    provider: OpenAIRealtimeProvider, session: VoiceSession
) -> None:
    ws = _mock_ws()
    _inject_ws(provider, session, ws)

    # Explicit empty list clears tools; instructions left untouched (omitted).
    await provider.reconfigure(session, tools=[])

    evt = _sent_events(ws)[0]
    assert evt["session"]["tools"] == []
    assert "instructions" not in evt["session"]
    assert "audio" not in evt["session"]


async def test_reconfigure_noop_when_nothing_changes(
    provider: OpenAIRealtimeProvider, session: VoiceSession
) -> None:
    ws = _mock_ws()
    _inject_ws(provider, session, ws)

    await provider.reconfigure(session)  # all fields None

    ws.send.assert_not_called()


async def test_reconfigure_temperature_only_is_noop(
    provider: OpenAIRealtimeProvider, session: VoiceSession
) -> None:
    ws = _mock_ws()
    _inject_ws(provider, session, ws)

    await provider.reconfigure(session, temperature=0.8)  # ignored by GA API

    ws.send.assert_not_called()


async def test_reconfigure_voice_uses_ga_nesting(
    provider: OpenAIRealtimeProvider, session: VoiceSession
) -> None:
    ws = _mock_ws()
    _inject_ws(provider, session, ws)

    await provider.reconfigure(session, voice="marin")

    evt = _sent_events(ws)[0]
    assert evt["session"]["audio"]["output"]["voice"] == "marin"


async def test_reconfigure_without_connection_is_safe(
    provider: OpenAIRealtimeProvider, session: VoiceSession
) -> None:
    # No socket injected — must not raise.
    await provider.reconfigure(session, tools=[{"name": "x", "parameters": {}}])
