"""Strict instruction context is never compressed or silently reconnected."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.genai", reason="google-genai not installed")

from roomkit.providers.gemini.realtime import GeminiLiveProvider, _GeminiSessionState, _GoAwayError
from roomkit.voice.base import VoiceSession, VoiceSessionState


@pytest.fixture
def provider():
    return GeminiLiveProvider(api_key="test-key", model="gemini-3.1-flash-live-preview")


def seed(provider, *, strict=True, live=None):
    session = VoiceSession(
        id="strict", room_id="room", participant_id="person", channel_id="voice"
    )
    session.state = VoiceSessionState.ACTIVE
    state = _GeminiSessionState(
        session=session,
        live_session=live,
        resumption_handle="previous-handle",
        provider_config={"preserve_context": strict},
    )
    provider._sessions[session.id] = state
    return session, state


@pytest.mark.parametrize("strict", [True, False])
def test_config_compression_and_resumption_match_context_policy(provider, strict):
    config = provider._build_config(
        system_prompt="Binding instructions",
        voice=None,
        tools=None,
        temperature=None,
        provider_config={"preserve_context": strict},
    )
    assert provider.supports_context_preservation
    assert (config.context_window_compression is None) is strict
    assert (config.session_resumption is None) is strict


@pytest.mark.parametrize("failure", [ConnectionError("socket dropped"), _GoAwayError()])
async def test_strict_disconnect_stops_without_reconnect_or_replay(provider, failure):
    class LostConnection:
        async def receive(self):
            raise failure
            yield  # pragma: no cover

    session, state = seed(provider, live=LostConnection())
    state.queued_text_injections.append(("pending operation", "user", False))
    provider._reconnect = AsyncMock()
    errors = []
    provider.on_error(lambda sess, code, message: errors.append((sess.state, code, message)))
    await asyncio.wait_for(provider._receive_loop(session), 1)
    provider._reconnect.assert_not_awaited()
    assert session.state == VoiceSessionState.ENDED
    assert len(errors) == 1
    assert errors[0][0] == VoiceSessionState.ENDED
    assert errors[0][1] == "context_preservation_ended"
    assert "not been replayed" in errors[0][2]
    assert not state.queued_text_injections


async def test_direct_reconnect_also_refuses_uncertain_context(provider):
    session, _ = seed(provider)
    with pytest.raises(RuntimeError, match="uncertain context"):
        await provider._reconnect(session)
    assert session.state == VoiceSessionState.ENDED


async def test_reconfigure_cannot_disable_preservation_mid_session(provider):
    session, state = seed(provider)
    with pytest.raises(ValueError, match="preserving Gemini context"):
        await provider.reconfigure(session, provider_config={"preserve_context": False})
    assert state.provider_config["preserve_context"] is True
    assert session.state == VoiceSessionState.ACTIVE


async def test_unavailable_connection_does_not_claim_result_delivery(provider):
    session, _ = seed(provider)
    with pytest.raises(RuntimeError, match="active Gemini connection"):
        await provider.submit_tool_result(session, "activation", '{"instructions":"full body"}')


@pytest.mark.parametrize("strict", [True, False])
async def test_nonresumable_update_invalidates_old_handle(provider, strict):
    session, state = seed(provider, strict=strict)
    await provider._on_session_resumption(
        session,
        state,
        SimpleNamespace(resumable=False, new_handle=None),
    )
    assert state.resumption_handle is None


async def test_actual_sdk_response_keeps_full_instructions_and_reference(provider):
    live = SimpleNamespace(send_tool_response=AsyncMock())
    session, _ = seed(provider, live=live)
    import json

    payload = {"instructions": "Mandatory rule.\n" * 2500, "references": ["guide.md"]}
    await provider.submit_tool_result(session, "activation", json.dumps(payload))
    response = live.send_tool_response.call_args.kwargs["function_responses"][0]
    assert response.id == "activation"
    assert response.response == payload
