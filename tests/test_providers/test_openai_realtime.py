"""Tests for OpenAIRealtimeProvider."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from roomkit.voice.base import VoiceSession, VoiceSessionState


def _make_session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(
        id=sid,
        room_id="room1",
        participant_id="p1",
        channel_id="ch1",
    )


def _load_provider():
    """Import the provider module with websockets mocked."""
    fake_ws = SimpleNamespace(
        connect=lambda *a, **kw: None,
    )
    mods = {"websockets": fake_ws}
    with patch.dict(sys.modules, mods):
        import roomkit.providers.openai.realtime as mod

        importlib.reload(mod)
        return mod


def _make_mock_ws() -> AsyncMock:
    """Build a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_connected_provider(mod, session: VoiceSession | None = None):
    """Create a provider with a mock WebSocket already connected."""
    provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
    session = session or _make_session()
    ws = _make_mock_ws()
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    session.state = VoiceSessionState.ACTIVE
    return provider, ws, session


class TestOpenAIRealtimeProvider:
    def test_constructor_and_name(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        assert provider.name == "OpenAIRealtimeProvider"

    def test_constructor_with_model_and_base_url(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(
            api_key="sk-test",
            model="gpt-realtime-2.0",
            base_url="wss://custom.api.com/v1/realtime",
        )
        assert provider._model == "gpt-realtime-2.0"
        assert provider._base_url == "wss://custom.api.com/v1/realtime"

    def test_is_responding_default(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        assert provider.is_responding("unknown-session") is False

    def test_callback_registration(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")

        audio_cb = lambda session, audio: None  # noqa: E731
        transcription_cb = lambda session, text, role, final: None  # noqa: E731
        speech_start_cb = lambda session: None  # noqa: E731
        speech_end_cb = lambda session: None  # noqa: E731
        tool_call_cb = lambda session, cid, name, args: None  # noqa: E731
        response_start_cb = lambda session: None  # noqa: E731
        response_end_cb = lambda session: None  # noqa: E731
        error_cb = lambda session, code, msg: None  # noqa: E731

        provider.on_audio(audio_cb)
        provider.on_transcription(transcription_cb)
        provider.on_speech_start(speech_start_cb)
        provider.on_speech_end(speech_end_cb)
        provider.on_tool_call(tool_call_cb)
        provider.on_response_start(response_start_cb)
        provider.on_response_end(response_end_cb)
        provider.on_error(error_cb)

        assert audio_cb in provider._audio_callbacks
        assert transcription_cb in provider._transcription_callbacks
        assert speech_start_cb in provider._speech_start_callbacks
        assert speech_end_cb in provider._speech_end_callbacks
        assert tool_call_cb in provider._tool_call_callbacks
        assert response_start_cb in provider._response_start_callbacks
        assert response_end_cb in provider._response_end_callbacks
        assert error_cb in provider._error_callbacks

    async def test_disconnect_unknown_session_is_noop(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session("unknown")
        # Should not raise
        await provider.disconnect(session)

    async def test_close_empty_provider(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        # Should not raise when no sessions exist
        await provider.close()

    def test_build_turn_detection_semantic_vad(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(
            "semantic_vad", {"eagerness": "high"}
        )
        assert result is not None
        assert result["type"] == "semantic_vad"
        assert result["eagerness"] == "high"

    def test_build_turn_detection_server_vad(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(
            "server_vad", {"threshold": 0.5, "silence_duration_ms": 500}
        )
        assert result is not None
        assert result["type"] == "server_vad"
        assert result["threshold"] == 0.5
        assert result["silence_duration_ms"] == 500

    def test_build_turn_detection_none(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(None, {})
        assert result is None

    def test_build_turn_detection_semantic_vad_with_all_options(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(
            "semantic_vad",
            {
                "eagerness": "low",
                "interrupt_response": True,
                "create_response": False,
            },
        )
        assert result["type"] == "semantic_vad"
        assert result["interrupt_response"] is True
        assert result["create_response"] is False

    def test_build_turn_detection_server_vad_with_all_options(self):
        mod = _load_provider()
        result = mod.OpenAIRealtimeProvider._build_turn_detection(
            "server_vad",
            {
                "threshold": 0.3,
                "silence_duration_ms": 1000,
                "prefix_padding_ms": 200,
                "interrupt_response": False,
                "create_response": True,
            },
        )
        assert result["prefix_padding_ms"] == 200
        assert result["interrupt_response"] is False
        assert result["create_response"] is True

    # ── connect() ───────────────────────────────────────────────

    async def test_connect_success(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()

        mock_ws = _make_mock_ws()

        # Make websockets.connect return a coroutine that resolves to mock_ws
        async def fake_connect(*args, **kwargs):
            return mock_ws

        with patch.dict(sys.modules, {"websockets": MagicMock(connect=fake_connect)}):
            await provider.connect(
                session,
                system_prompt="You are helpful.",
                voice="alloy",
                tools=[{"name": "get_weather", "description": "Get weather"}],
            )

        assert session.state == VoiceSessionState.ACTIVE
        assert session.provider_session_id == session.id
        assert session.id in provider._connections
        assert session.id in provider._receive_tasks

        # Clean up the task
        provider._receive_tasks[session.id].cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await provider._receive_tasks[session.id]

    async def test_connect_sends_session_update(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()

        mock_ws = _make_mock_ws()

        async def fake_connect(*args, **kwargs):
            return mock_ws

        with patch.dict(sys.modules, {"websockets": MagicMock(connect=fake_connect)}):
            await provider.connect(session, system_prompt="Test prompt")

        # The first send should be session.update
        sent = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert sent["type"] == "session.update"
        assert sent["session"]["instructions"] == "Test prompt"

        # Clean up
        provider._receive_tasks[session.id].cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await provider._receive_tasks[session.id]

    async def test_connect_temperature_warning(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()
        mock_ws = _make_mock_ws()

        async def fake_connect(*args, **kwargs):
            return mock_ws

        with patch.dict(sys.modules, {"websockets": MagicMock(connect=fake_connect)}):
            # Should log a warning about temperature being ignored
            await provider.connect(session, temperature=0.7)

        # Clean up
        provider._receive_tasks[session.id].cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await provider._receive_tasks[session.id]

    # ── send_audio() ────────────────────────────────────────────

    async def test_send_audio(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        audio_data = b"\x00\x01\x02\x03"
        await provider.send_audio(session, audio_data)

        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "input_audio_buffer.append"
        assert sent["audio"] == base64.b64encode(audio_data).decode("ascii")

    async def test_send_audio_no_connection(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()
        # Should return without error when no ws connection
        await provider.send_audio(session, b"\x00")

    # ── inject_text() ──────────────────────────────────────────

    async def test_inject_text_user(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        await provider.inject_text(session, "Hello, world!")

        assert ws.send.await_count == 2  # conversation.item.create + response.create
        first_msg = json.loads(ws.send.call_args_list[0][0][0])
        assert first_msg["type"] == "conversation.item.create"
        assert first_msg["item"]["role"] == "user"
        assert first_msg["item"]["content"][0]["text"] == "Hello, world!"

        second_msg = json.loads(ws.send.call_args_list[1][0][0])
        assert second_msg["type"] == "response.create"

    async def test_inject_text_system_role(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        await provider.inject_text(session, "System instruction", role="system")

        first_msg = json.loads(ws.send.call_args_list[0][0][0])
        assert first_msg["item"]["role"] == "system"

    async def test_inject_text_invalid_role_defaults_to_user(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        await provider.inject_text(session, "Test", role="assistant")

        first_msg = json.loads(ws.send.call_args_list[0][0][0])
        assert first_msg["item"]["role"] == "user"

    async def test_inject_text_silent(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        await provider.inject_text(session, "Silent context", silent=True)

        # Only conversation.item.create, NO response.create
        assert ws.send.await_count == 1
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "conversation.item.create"

    async def test_inject_text_skips_response_when_responding(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        # Mark session as already responding
        provider._responding.add(session.id)

        await provider.inject_text(session, "Additional context")

        # Should send conversation.item.create but NOT response.create
        assert ws.send.await_count == 1
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "conversation.item.create"

    async def test_inject_text_no_connection(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()
        # Should return without error
        await provider.inject_text(session, "No connection")

    # ── submit_tool_result() ────────────────────────────────────

    async def test_submit_tool_result(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        await provider.submit_tool_result(session, "call-123", '{"temperature": 72}')

        assert ws.send.await_count == 2
        first_msg = json.loads(ws.send.call_args_list[0][0][0])
        assert first_msg["type"] == "conversation.item.create"
        assert first_msg["item"]["type"] == "function_call_output"
        assert first_msg["item"]["call_id"] == "call-123"
        assert first_msg["item"]["output"] == '{"temperature": 72}'

        second_msg = json.loads(ws.send.call_args_list[1][0][0])
        assert second_msg["type"] == "response.create"

    async def test_submit_tool_result_no_connection(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()
        await provider.submit_tool_result(session, "call-1", "result")

    # ── interrupt() ─────────────────────────────────────────────

    async def test_interrupt(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        await provider.interrupt(session)

        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "response.cancel"

    async def test_interrupt_no_connection(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()
        await provider.interrupt(session)

    # ── send_event() ────────────────────────────────────────────

    async def test_send_event(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        custom_event = {"type": "custom.event", "data": "test"}
        await provider.send_event(session, custom_event)

        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "custom.event"

    async def test_send_event_no_connection(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()
        await provider.send_event(session, {"type": "test"})

    # ── disconnect() ────────────────────────────────────────────

    async def test_disconnect_cleans_up(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        # Add a receive task
        async def dummy():
            await asyncio.sleep(100)

        provider._receive_tasks[session.id] = asyncio.create_task(dummy())
        provider._responding.add(session.id)

        await provider.disconnect(session)

        assert session.id not in provider._connections
        assert session.id not in provider._sessions
        assert session.id not in provider._receive_tasks
        assert session.id not in provider._responding
        assert session.state == VoiceSessionState.ENDED

    # ── close() ─────────────────────────────────────────────────

    async def test_close_disconnects_all_sessions(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        await provider.close()
        assert len(provider._sessions) == 0
        assert session.state == VoiceSessionState.ENDED

    # ── _handle_server_event() ──────────────────────────────────

    async def test_handle_speech_started(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        started = []
        provider.on_speech_start(lambda s: started.append(s.id))

        await provider._handle_server_event(session, {"type": "input_audio_buffer.speech_started"})
        assert started == [session.id]

    async def test_handle_speech_stopped(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        stopped = []
        provider.on_speech_end(lambda s: stopped.append(s.id))

        await provider._handle_server_event(session, {"type": "input_audio_buffer.speech_stopped"})
        assert stopped == [session.id]

    async def test_handle_audio_delta(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        received_audio = []
        provider.on_audio(lambda s, audio: received_audio.append(audio))

        raw_audio = b"\x00\x01\x02"
        encoded = base64.b64encode(raw_audio).decode("ascii")

        await provider._handle_server_event(
            session, {"type": "response.output_audio.delta", "delta": encoded}
        )
        assert len(received_audio) == 1
        assert received_audio[0] == raw_audio

    async def test_handle_audio_delta_empty(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        received_audio = []
        provider.on_audio(lambda s, audio: received_audio.append(audio))

        await provider._handle_server_event(
            session, {"type": "response.output_audio.delta", "delta": ""}
        )
        assert received_audio == []

    async def test_handle_assistant_transcription_delta(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        await provider._handle_server_event(
            session,
            {"type": "response.output_audio_transcript.delta", "delta": "Hello"},
        )
        assert transcriptions == [("Hello", "assistant", False)]

    async def test_handle_user_transcription_completed(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        await provider._handle_server_event(
            session,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "What is the weather?",
            },
        )
        assert transcriptions == [("What is the weather?", "user", True)]

    async def test_handle_assistant_transcription_done(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        transcriptions = []
        provider.on_transcription(
            lambda s, text, role, final: transcriptions.append((text, role, final))
        )

        await provider._handle_server_event(
            session,
            {
                "type": "response.output_audio_transcript.done",
                "transcript": "The weather is sunny.",
            },
        )
        assert transcriptions == [("The weather is sunny.", "assistant", True)]

    async def test_handle_function_call_done(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        tool_calls = []
        provider.on_tool_call(lambda s, cid, name, args: tool_calls.append((cid, name, args)))

        await provider._handle_server_event(
            session,
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-1",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
        )
        assert tool_calls == [("call-1", "get_weather", {"city": "NYC"})]

    async def test_handle_function_call_invalid_json(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        tool_calls = []
        provider.on_tool_call(lambda s, cid, name, args: tool_calls.append((cid, name, args)))

        await provider._handle_server_event(
            session,
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-2",
                "name": "broken_tool",
                "arguments": "not-json",
            },
        )
        assert len(tool_calls) == 1
        assert tool_calls[0][2] == {"raw": "not-json"}

    async def test_handle_response_created(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        starts = []
        provider.on_response_start(lambda s: starts.append(s.id))

        await provider._handle_server_event(session, {"type": "response.created"})

        assert session.id in provider._responding
        assert starts == [session.id]

    async def test_handle_response_done_success(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        ends = []
        provider.on_response_end(lambda s: ends.append(s.id))
        provider._responding.add(session.id)

        await provider._handle_server_event(
            session,
            {
                "type": "response.done",
                "response": {"status": "completed", "usage": {}},
            },
        )

        assert session.id not in provider._responding
        assert ends == [session.id]

    async def test_handle_response_done_with_usage(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        provider._responding.add(session.id)

        await provider._handle_server_event(
            session,
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "input_token_details": {"cached_tokens": 20},
                        "output_token_details": {"text_tokens": 30},
                    },
                },
            },
        )
        assert session.id not in provider._responding

    async def test_handle_response_done_failed(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        errors = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))
        provider._responding.add(session.id)

        await provider._handle_server_event(
            session,
            {
                "type": "response.done",
                "response": {
                    "status": "failed",
                    "status_details": {
                        "error": {
                            "type": "server_error",
                            "code": "500",
                            "message": "Internal error",
                        }
                    },
                },
            },
        )
        assert len(errors) == 1
        assert errors[0] == ("500", "Internal error")

    async def test_handle_session_created(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)
        # Should not raise
        await provider._handle_server_event(
            session,
            {
                "type": "session.created",
                "session": {"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}},
            },
        )

    async def test_handle_session_updated(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)
        await provider._handle_server_event(
            session,
            {
                "type": "session.updated",
                "session": {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}},
            },
        )

    async def test_handle_audio_buffer_committed(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)
        await provider._handle_server_event(session, {"type": "input_audio_buffer.committed"})

    async def test_handle_error_event(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        errors = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        await provider._handle_server_event(
            session,
            {
                "type": "error",
                "error": {"code": "rate_limit", "message": "Too many requests"},
            },
        )
        assert errors == [("rate_limit", "Too many requests")]

    async def test_handle_unknown_event(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)
        # Unknown events should not raise
        await provider._handle_server_event(session, {"type": "some.unknown.event"})

    # ── _receive_loop() ─────────────────────────────────────────

    async def test_receive_loop_processes_messages(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        speech_starts = []
        provider.on_speech_start(lambda s: speech_starts.append(s.id))

        # Simulate ws yielding messages via async iteration
        messages = [
            json.dumps({"type": "input_audio_buffer.speech_started"}),
            json.dumps({"type": "input_audio_buffer.speech_stopped"}),
        ]

        async def fake_iter():
            for m in messages:
                yield m

        ws.__aiter__ = lambda self: fake_iter()

        await provider._receive_loop(session)
        assert len(speech_starts) == 1

    async def test_receive_loop_handles_invalid_json(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        async def fake_iter():
            yield "not-valid-json"
            yield json.dumps({"type": "input_audio_buffer.speech_started"})

        ws.__aiter__ = lambda self: fake_iter()

        # Should not raise; invalid JSON is logged and skipped
        await provider._receive_loop(session)

    async def test_receive_loop_handles_connection_closed(self):
        mod = _load_provider()
        provider, ws, session = _make_connected_provider(mod)

        errors = []
        provider.on_error(lambda s, code, msg: errors.append((code, msg)))

        async def fake_iter():
            raise ConnectionError("WebSocket closed")
            yield  # make it a generator  # pragma: no cover

        ws.__aiter__ = lambda self: fake_iter()

        await provider._receive_loop(session)
        assert session.state == VoiceSessionState.ENDED
        assert len(errors) == 1

    async def test_receive_loop_no_ws(self):
        mod = _load_provider()
        provider = mod.OpenAIRealtimeProvider(api_key="sk-test")
        session = _make_session()
        # No ws registered — should return immediately
        await provider._receive_loop(session)

    # ── Callback error handling ─────────────────────────────────

    async def test_callback_exception_is_caught(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        def bad_cb(s):
            raise ValueError("Callback error")

        provider.on_speech_start(bad_cb)

        # Should not raise
        await provider._handle_server_event(session, {"type": "input_audio_buffer.speech_started"})

    async def test_audio_callback_exception_is_caught(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        def bad_cb(s, audio):
            raise ValueError("Audio callback error")

        provider.on_audio(bad_cb)

        raw = base64.b64encode(b"\x00").decode()
        await provider._handle_server_event(
            session, {"type": "response.output_audio.delta", "delta": raw}
        )

    async def test_transcription_callback_exception_is_caught(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        def bad_cb(s, text, role, final):
            raise ValueError("Transcription callback error")

        provider.on_transcription(bad_cb)

        await provider._handle_server_event(
            session,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "test",
            },
        )

    async def test_tool_call_callback_exception_is_caught(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        def bad_cb(s, cid, name, args):
            raise ValueError("Tool callback error")

        provider.on_tool_call(bad_cb)

        await provider._handle_server_event(
            session,
            {
                "type": "response.function_call_arguments.done",
                "call_id": "c1",
                "name": "fn",
                "arguments": "{}",
            },
        )

    async def test_error_callback_exception_is_caught(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        def bad_cb(s, code, msg):
            raise ValueError("Error callback error")

        provider.on_error(bad_cb)

        await provider._handle_server_event(
            session,
            {"type": "error", "error": {"code": "test", "message": "test"}},
        )

    # ── Async callback support ──────────────────────────────────

    async def test_async_callbacks_are_awaited(self):
        mod = _load_provider()
        provider, _, session = _make_connected_provider(mod)

        called = []

        async def async_cb(s):
            called.append("async")

        provider.on_speech_start(async_cb)

        await provider._handle_server_event(session, {"type": "input_audio_buffer.speech_started"})
        assert called == ["async"]
