"""Adapters report submission, unsupported input, and uncertain SDK failures."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from roomkit import (
    Access,
    ChannelType,
    EventSource,
    HookExecution,
    HookTrigger,
    RoomEvent,
    TextContent,
    VoiceInjectionResult,
)
from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.voice.base import VoiceSession
from tests.test_proactive_delivery_voice import voice_room


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(id="s", room_id="r", channel_id="voice", participant_id="p")


@pytest.mark.parametrize("silent", [False, True])
async def test_elevenlabs_reports_only_completed_send(session, silent) -> None:
    provider = ElevenLabsRealtimeProvider(
        ElevenLabsRealtimeConfig(api_key="test", agent_id="agent")
    )
    missing = await provider.inject_text(session, "result", silent=silent)
    assert missing.status == "not_sent" and missing.retryable
    conversation = AsyncMock()
    provider._conversations[session.id] = conversation
    result = await provider.inject_text(session, "result", silent=silent)
    assert result.status == "sent"
    send = conversation.send_contextual_update if silent else conversation.send_user_message
    send.assert_awaited_once_with("result")
    send.side_effect = ConnectionError("confirmation lost")
    with pytest.raises(ConnectionError):
        await provider.inject_text(session, "another", silent=silent)


async def test_anam_does_not_report_swallowed_failure_as_sent(session) -> None:
    provider = AnamRealtimeProvider(AnamConfig(api_key="test", persona_id="persona"))
    missing = await provider.inject_text(session, "result")
    assert missing.status == "not_sent" and missing.retryable
    sdk = AsyncMock()
    provider._states[session.id] = SimpleNamespace(anam_session=sdk)
    result = await provider.inject_text(session, "result")
    assert result.status == "sent"
    sdk.send_message.assert_awaited_once_with("result")
    sdk.send_message.side_effect = ConnectionError("confirmation lost")
    failure = await provider.inject_text(session, "another")
    assert failure.status == "unknown" and not failure.retryable


@pytest.mark.parametrize("mode", ["muted", "read_only", "output_muted"])
@pytest.mark.parametrize("entry", ["proactive", "broadcast"])
async def test_anam_cannot_speak_when_silent_delivery_is_required(mode, entry) -> None:
    async with voice_room(1) as (kit, channel, mock, sessions):
        if mode == "read_only":
            await kit.set_access("r", "voice", Access.READ_ONLY)
        elif mode == "muted":
            await kit.mute("r", "voice")
        else:
            await kit.mute_output("r", "voice")
        provider = AnamRealtimeProvider(AnamConfig(api_key="test", persona_id="persona"))
        sdk = AsyncMock()
        provider._states[sessions[0].id] = SimpleNamespace(anam_session=sdk)
        with patch.object(mock, "inject_text", side_effect=provider.inject_text):
            if entry == "proactive":
                result = await kit.deliver(
                    "r",
                    "result",
                    channel_id="voice",
                    session_id=sessions[0].id,
                    idempotency_key="key",
                )
                assert result.status == "failed" and not result.error.retryable
                assert result.reason == "voice_silent_injection_unsupported"
            else:
                await channel.on_event(
                    broadcast_event(),
                    await kit.store.get_binding("r", "voice"),
                    await kit._build_context("r"),
                )
        sdk.send_message.assert_not_awaited()


def broadcast_event() -> RoomEvent:
    return RoomEvent(
        room_id="r",
        source=EventSource(channel_id="worker", channel_type=ChannelType.AI),
        content=TextContent(body="result"),
    )


@pytest.mark.parametrize("status", ["sent", "not_sent", "unknown", None])
async def test_broadcast_hook_requires_explicit_injection_acceptance(status) -> None:
    async with voice_room(1) as (kit, channel, provider, _):
        observed = []

        @kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, execution=HookExecution.ASYNC)
        async def injected(event, context):
            observed.append(event)

        result = VoiceInjectionResult(status=status) if status else None
        with patch.object(provider, "inject_text", return_value=result):
            await channel.on_event(
                broadcast_event(),
                await kit.store.get_binding("r", "voice"),
                await kit._build_context("r"),
            )
        await kit.hook_engine.drain_observers()
        assert len(observed) == (1 if status == "sent" else 0)


async def test_personaplex_unsupported_input_cannot_succeed(session) -> None:
    result = await PersonaPlexRealtimeProvider().inject_text(session, "result")
    assert result.status == "not_sent"
    assert result.reason == "voice_injection_unsupported" and not result.retryable
