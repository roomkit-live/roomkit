"""Tool results on the OpenAI Realtime wire: one continuation per response (RMK-279).

The model is asked to go on (``response.create``) once per response, when that
response is done and every call it emitted has its output (RFC §12.4). xAI
speaks the same wire and inherits the behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.openai.realtime_base import OpenAIRealtimeBase
from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState

_PROVIDERS: dict[str, Callable[[], OpenAIRealtimeBase]] = {
    "openai": lambda: OpenAIRealtimeProvider(api_key="sk-test"),
    "xai": lambda: XAIRealtimeProvider(XAIRealtimeConfig(api_key=SecretStr("xai-test"))),
}


@pytest.fixture(params=sorted(_PROVIDERS))
def provider(request: pytest.FixtureRequest) -> OpenAIRealtimeBase:
    return _PROVIDERS[request.param]()


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(id="s1", room_id="r1", participant_id="p1", channel_id="voice")


def _attach(provider: OpenAIRealtimeBase, session: VoiceSession) -> AsyncMock:
    ws = AsyncMock()
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE
    return ws


def _wire(ws: AsyncMock) -> list[str]:
    """What went out, as ``item(<call_id>)`` or the bare event type."""
    sent = []
    for call in ws.send.call_args_list:
        event = json.loads(call.args[0])
        if event["type"] == "conversation.item.create":
            sent.append(f"item({event['item'].get('call_id', 'text')})")
        else:
            sent.append(event["type"])
    return sent


async def _response_created(provider: OpenAIRealtimeBase, session: VoiceSession) -> None:
    await provider._handle_server_event(session, {"type": "response.created", "response": {}})


async def _call(provider: OpenAIRealtimeBase, session: VoiceSession, call_id: str) -> None:
    await provider._handle_server_event(
        session,
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "name": "lookup",
                "arguments": "{}",
                "status": "completed",
            },
        },
    )


async def _response_done(
    provider: OpenAIRealtimeBase, session: VoiceSession, status: str = "completed"
) -> None:
    await provider._handle_server_event(
        session, {"type": "response.done", "response": {"status": status}}
    )


async def _result(provider: OpenAIRealtimeBase, session: VoiceSession, call_id: str) -> None:
    await provider.submit_tool_result(session, call_id, '{"ok": true}')


class TestOneContinuationPerResponse:
    async def test_parallel_calls_answered_before_the_response_ends(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _result(provider, session, "call_b")
        assert _wire(ws) == ["item(call_a)", "item(call_b)"]

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "item(call_b)", "response.create"]

    async def test_parallel_calls_answered_after_the_response_ends(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _response_done(provider, session)
        await _result(provider, session, "call_b")
        assert _wire(ws) == ["item(call_b)"]

        await _result(provider, session, "call_a")

        assert _wire(ws) == ["item(call_b)", "item(call_a)", "response.create"]

    async def test_a_single_call_continues_after_its_result(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)

        await _result(provider, session, "call_a")

        assert _wire(ws) == ["item(call_a)", "response.create"]

    @pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
    async def test_a_response_ends_the_same_way_whatever_its_status(
        self, provider: OpenAIRealtimeBase, session: VoiceSession, status: str
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _response_done(provider, session, status)
        assert _wire(ws) == ["item(call_a)"]

        await _result(provider, session, "call_b")

        assert _wire(ws) == ["item(call_a)", "item(call_b)", "response.create"]

    async def test_a_result_submitted_inside_the_tool_call_callback(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)

        async def serve_at_once(sess: VoiceSession, call_id: str, name: str, args: dict) -> None:
            await provider.submit_tool_result(sess, call_id, '{"ok": true}')

        provider.on_tool_call(serve_at_once)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "response.create"]

    async def test_a_response_without_calls_asks_nothing(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _response_done(provider, session)

        assert _wire(ws) == []

    async def test_the_continuation_opens_a_fresh_count(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]
        await _response_done(provider, session)
        # The continuation calls a tool of its own
        await _response_created(provider, session)
        await _call(provider, session, "call_b")
        await _response_done(provider, session)
        await _result(provider, session, "call_b")

        assert _wire(ws) == ["item(call_a)", "response.create", "item(call_b)", "response.create"]


class TestAResultTheConversationHasLeft:
    async def test_it_waits_for_the_response_in_progress(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        # The user speaks again before call_a is answered
        await _response_created(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "response.create"]

    async def test_it_does_not_hold_the_new_response(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _response_done(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_c")
        await _response_done(provider, session)

        await _result(provider, session, "call_c")

        assert _wire(ws) == ["item(call_a)", "item(call_c)", "response.create"]

    async def test_it_continues_at_once_when_nothing_is_in_progress(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _result(provider, session, "call_a")
        await _response_done(provider, session)
        await _response_created(provider, session)
        await _response_done(provider, session)

        await _result(provider, session, "call_b")

        assert _wire(ws) == ["item(call_a)", "item(call_b)", "response.create"]

    async def test_it_joins_the_new_response_calls(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_c")
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]

        await _result(provider, session, "call_c")

        assert _wire(ws) == ["item(call_a)", "item(call_c)", "response.create"]


class TestAnIdIssuedAgainDuringTheSend:
    async def test_the_new_call_holds_its_response(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        """RMK-441: the id leaves the response's books before the send
        yields, as it leaves the open calls; a call issued under it during
        the send is a new call, which holds its response until answered."""
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "c1")
        await _response_done(provider, session)
        reissued = False

        async def send(payload: str) -> None:
            nonlocal reissued
            if json.loads(payload)["type"] == "conversation.item.create" and not reissued:
                reissued = True
                await _response_created(provider, session)
                await _call(provider, session, "c1")

        ws.send.side_effect = send
        await _result(provider, session, "c1")
        await _response_done(provider, session)

        assert _wire(ws) == ["item(c1)"]
        await _result(provider, session, "c1")

        assert _wire(ws) == ["item(c1)", "item(c1)", "response.create"]


class TestARequestNotYetBegun:
    async def test_a_result_in_between_waits_for_the_requested_response(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _call(provider, session, "call_b")
        await _response_done(provider, session)
        await _response_created(provider, session)  # the user spoke again
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)", "response.create"]

        # call_b's output lands before the requested response has begun
        await _result(provider, session, "call_b")
        assert _wire(ws) == ["item(call_a)", "response.create", "item(call_b)"]
        await _response_created(provider, session)
        assert _wire(ws) == ["item(call_a)", "response.create", "item(call_b)"]

        await _response_done(provider, session)

        assert _wire(ws) == [
            "item(call_a)",
            "response.create",
            "item(call_b)",
            "response.create",
        ]

    async def test_the_requested_response_owes_nothing_when_nothing_came_in_between(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        await _response_created(provider, session)

        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "response.create"]


async def _speech(provider: OpenAIRealtimeBase, session: VoiceSession, edge: str) -> None:
    """The server VAD's own speech boundary: ``started`` or ``stopped``."""
    await provider._handle_server_event(
        session, {"type": f"input_audio_buffer.speech_{edge}", "item_id": "item_1"}
    )


class TestTheCallerIsSpeaking:
    """RMK-288: a continuation never starts while the caller speaks (RFC §12.4)."""

    async def test_results_in_before_a_barge_in_wait_for_the_end_of_the_turn(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _result(provider, session, "call_a")
        await provider.send_activity_start(session)
        await provider.interrupt(session)
        await _response_done(provider, session, status="cancelled")
        assert _wire(ws) == ["item(call_a)", "response.cancel"]

        await provider.send_activity_end(session)

        assert _wire(ws) == [
            "item(call_a)",
            "response.cancel",
            "input_audio_buffer.commit",
            "response.create",
        ]

    async def test_a_result_that_lands_while_the_caller_speaks_waits(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await provider.send_activity_start(session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)"]

        await provider.send_activity_end(session)

        assert _wire(ws) == ["item(call_a)", "input_audio_buffer.commit", "response.create"]
        await _response_created(provider, session)
        await _response_done(provider, session)
        assert _wire(ws).count("response.create") == 1

    async def test_under_server_vad_the_servers_answer_covers_it(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _result(provider, session, "call_a")
        await _speech(provider, session, "started")
        await _response_done(provider, session, status="cancelled")
        await _speech(provider, session, "stopped")
        assert _wire(ws) == ["item(call_a)"]

        # The server's own response to the turn, then its end: nothing owed
        await _response_created(provider, session)
        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)"]

    async def test_under_server_vad_without_its_answers_the_provider_asks(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        provider._provider_configs[session.id] = {"create_response": False}
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _result(provider, session, "call_a")
        await _speech(provider, session, "started")
        await _response_done(provider, session, status="cancelled")
        assert _wire(ws) == ["item(call_a)"]

        await _speech(provider, session, "stopped")

        assert _wire(ws) == ["item(call_a)", "response.create"]


class TestOneRequestAtATime:
    """RMK-288: every emitter reads the same in-progress state (RFC §12.4)."""

    async def _requested_continuation(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> AsyncMock:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)", "response.create"]
        return ws

    async def test_an_injection_does_not_double_a_requested_continuation(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = await self._requested_continuation(provider, session)

        await provider.inject_text(session, "Also say goodbye.", role="system")

        assert _wire(ws) == ["item(call_a)", "response.create", "item(text)"]

    async def test_the_end_of_a_turn_does_not_double_a_requested_continuation(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = await self._requested_continuation(provider, session)

        await provider.send_activity_end(session)

        assert _wire(ws) == ["item(call_a)", "response.create", "input_audio_buffer.commit"]

    async def test_two_injections_ask_once(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)

        await provider.inject_text(session, "Greet the caller.", role="system")
        await provider.inject_text(session, "Mention the offer.", role="system")

        assert _wire(ws) == ["item(text)", "response.create", "item(text)"]

    async def test_a_result_after_an_injection_request_is_owed(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await provider.inject_text(session, "Greet the caller.", role="system")
        await _result(provider, session, "call_a")
        await _response_created(provider, session)
        assert _wire(ws) == ["item(text)", "response.create", "item(call_a)"]

        await _response_done(provider, session)

        assert _wire(ws) == ["item(text)", "response.create", "item(call_a)", "response.create"]


class TestTheCallersTurnIsAnswered:
    """RMK-288: a caller's turn that met a response in progress is owed one,
    and a rejected request does not leave the session mute (RFC §12.4)."""

    async def test_a_turn_ending_during_a_response_is_answered_after_it(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await provider.send_activity_start(session)
        await provider.send_activity_end(session)
        assert _wire(ws) == ["input_audio_buffer.commit"]

        await _response_done(provider, session)

        assert _wire(ws) == ["input_audio_buffer.commit", "response.create"]
        await _response_created(provider, session)
        await _response_done(provider, session)
        assert _wire(ws).count("response.create") == 1

    async def test_a_continuation_sent_after_the_turn_covers_it(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _result(provider, session, "call_a")
        await provider.send_activity_start(session)
        await provider.send_activity_end(session)

        await _response_done(provider, session)
        await _response_created(provider, session)
        await _response_done(provider, session)

        assert _wire(ws) == ["item(call_a)", "input_audio_buffer.commit", "response.create"]

    async def test_a_turn_owed_while_the_caller_speaks_again_waits_for_that_turn(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await provider.send_activity_end(session)
        await provider.send_activity_start(session)

        await _response_done(provider, session)
        assert _wire(ws) == ["input_audio_buffer.commit"]

        await provider.send_activity_end(session)
        assert _wire(ws) == [
            "input_audio_buffer.commit",
            "input_audio_buffer.commit",
            "response.create",
        ]

    async def test_a_rejected_request_does_not_leave_the_session_mute(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await _result(provider, session, "call_a")
        assert _wire(ws) == ["item(call_a)", "response.create"]

        await provider._handle_server_event(
            session,
            {"type": "error", "error": {"code": "invalid_request_error", "message": "no"}},
        )
        await provider.inject_text(session, "Say goodbye.", role="system")

        assert _wire(ws) == ["item(call_a)", "response.create", "item(text)", "response.create"]


class TestAnEndedSession:
    @pytest.mark.parametrize("end", ["disconnect", "connection lost"])
    async def test_an_ended_session_keeps_no_open_calls(
        self, provider: OpenAIRealtimeBase, session: VoiceSession, end: str
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)

        if end == "disconnect":
            await provider.disconnect(session)
        else:
            await provider._retire_lost_connection(session, ws, "peer closed")

        assert session.id not in provider._pending_responses

    async def test_a_closed_session_sends_nothing(
        self, provider: OpenAIRealtimeBase, session: VoiceSession
    ) -> None:
        ws = _attach(provider, session)
        await _response_created(provider, session)
        await _call(provider, session, "call_a")
        await _response_done(provider, session)
        await provider.disconnect(session)
        sent_before = ws.send.await_count

        await _result(provider, session, "call_a")

        assert ws.send.await_count == sent_before
