"""Prism wire behavior against a local WebSocket service."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime import MockRealtimeTransport, RimeRealtimeConfig, RimeRealtimeProvider
from tests.voice.rime_wire import control, prism_handshake, receive, send, server


def session(identifier: str = "s1") -> VoiceSession:
    return VoiceSession(id=identifier, room_id="r1", participant_id="p1", channel_id="voice")


def provider(url: str, **kwargs: Any) -> RimeRealtimeProvider:
    return RimeRealtimeProvider(
        RimeRealtimeConfig(api_key="test-key", websocket_url=url, **kwargs)
    )


async def test_events_preserve_audio_text_turns_late_transcripts_and_usage() -> None:
    finished = asyncio.Event()
    observed = []
    settings = []

    async def handler(ws: ServerConnection) -> None:
        settings.append(await prism_handshake(ws))
        await receive(ws)  # continuous input starts after setup
        for event in [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "conversation.item.input_audio_transcription.delta", "delta": "Hi"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "response.text.delta", "delta": "Hello"},
            {"type": "response.audio.delta", "delta": "AQACAA=="},
            {"type": "response.text.done", "text": "Hello."},
            {
                "type": "response.done",
                "response": {
                    "id": "response-1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                        "context_window": 65536,
                    },
                },
            },
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Hi!"},
        ]:
            await send(ws, **event)
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url), session()
        rt.on_audio(lambda _, data: observed.append(("audio", data)))
        rt.on_speech_start(lambda _: observed.append(("speech_start",)))
        rt.on_speech_end(lambda _: observed.append(("speech_end",)))
        rt.on_response_start(lambda _: observed.append(("response_start",)))
        rt.on_response_end(lambda _: observed.append(("response_end",)))

        def transcript(_: VoiceSession, text: str, role: str, final: bool) -> None:
            observed.append(("text", text, role, final))
            if role == "user" and final:
                finished.set()

        rt.on_transcription(transcript)
        try:
            await rt.connect(s, system_prompt="Be brief.", voice="aurora")
            await asyncio.wait_for(finished.wait(), 1)
            assert not rt.is_responding(s.id)
            assert s.last_usage == {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "context_window": 65536,
            }
        finally:
            await rt.close()
    assert settings[0]["voice"] == "aurora" and settings[0]["instructions"] == "Be brief."
    assert "temperature" not in settings[0] and "audio" not in settings[0]
    assert observed == [
        ("speech_start",),
        ("text", "Hi", "user", False),
        ("speech_end",),
        ("response_start",),
        ("text", "Hello", "assistant", False),
        ("audio", b"\x01\x00\x02\x00"),
        ("text", "Hello.", "assistant", True),
        ("response_end",),
        ("text", "Hi!", "user", True),
    ]
    assert s.state == VoiceSessionState.ENDED


@pytest.mark.parametrize("finish_first", [True, False])
async def test_tools_continue_once_after_parent_done_and_every_result_ack(
    finish_first: bool,
) -> None:
    completed = asyncio.Event()
    calls = []
    outputs = []
    continuations = []

    async def handler(ws: ServerConnection) -> None:
        settings = await prism_handshake(ws)
        assert settings["tools"][0] == {
            "type": "function",
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        }
        await receive(ws)
        await send(ws, type="response.created", response={"id": "parent"})
        for call_id in ["a", "b", "a"]:  # a duplicate must not execute twice
            await send(
                ws,
                type="response.function_call_arguments.done",
                response_id="parent",
                call_id=call_id,
                name="lookup",
                arguments='{"city":"Paris"}',
            )
        if finish_first:
            await send(ws, type="response.done", response={"id": "parent", "status": "completed"})
        for _ in range(2):
            event = await control(ws)
            assert event["type"] == "conversation.item.create"
            outputs.append(event["item"])
            await send(
                ws,
                type="conversation.item.created",
                prism_request_event_id=event["event_id"],
                item={"type": "function_call_output", "call_id": event["item"]["call_id"]},
            )
        if not finish_first:
            await send(ws, type="response.done", response={"id": "parent", "status": "completed"})
        continuations.append(await control(ws))
        await send(ws, type="response.created", response={"id": "child"})
        await send(ws, type="response.text.done", text="It closes at four.")
        await send(ws, type="response.done", response={"id": "child", "status": "completed"})
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url), session()

        async def tool(_: VoiceSession, call_id: str, name: str, args: Any) -> None:
            calls.append((call_id, name, args))
            await rt.submit_tool_result(s, call_id, '{"closes":"16:00"}')
            await rt.submit_tool_result(s, call_id, "duplicate result")

        rt.on_tool_call(tool)
        rt.on_transcription(lambda _, text, role, final: completed.set() if final else None)
        try:
            await rt.connect(
                s,
                tools=[
                    {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                        "tags": ["internal"],
                    }
                ],
            )
            await asyncio.wait_for(completed.wait(), 1)
        finally:
            await rt.close()
    assert sorted(call[0] for call in calls) == ["a", "b"]
    assert all(call[2] == {"city": "Paris"} for call in calls)
    assert len(outputs) == 2 and all(
        output["output"] == '{"closes":"16:00"}' for output in outputs
    )
    assert len(continuations) == 1
    assert continuations[0]["response"]["metadata"] == {
        "prism_cause": "tool_continuation",
        "prism_parent_response_id": "parent",
    }


@pytest.mark.parametrize(
    ("role", "silent"),
    [("user", False), ("system", False), ("assistant", False), ("user", True), ("system", True)],
)
async def test_text_injection_maps_intent_and_waits_for_acceptance(
    role: str, silent: bool
) -> None:
    commands = []

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        event = await control(ws)
        commands.append(event)
        if silent:
            await send(
                ws,
                type="conversation.item.created",
                prism_request_event_id=event["event_id"],
                item={"type": "message", "id": "history"},
            )
        else:
            await send(
                ws,
                type="response.created",
                prism_request_event_id=event["event_id"],
                response={"id": "reply"},
            )
            await send(ws, type="response.done", response={"id": "reply", "status": "completed"})
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url), session()
        try:
            await rt.connect(s)
            result = await rt.inject_text(s, "Hello.", role=role, silent=silent)
            assert result.status == "sent"
        finally:
            await rt.close()
    if silent:
        assert commands[0]["item"] == {"type": "message", "role": role, "text": "Hello."}
    else:
        response = commands[0]["response"]
        assert response["metadata"]["prism_cause"] == (
            "user_text" if role == "user" else "proactive"
        )
        assert ("prism_input_text" in response) == (role == "user")
        if role == "assistant":
            assert "Say this to the user" in response["prism_instruction"]


@pytest.mark.parametrize("refuse", [True, False])
async def test_injection_distinguishes_refusal_from_lost_ack(refuse: bool) -> None:
    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        event = await control(ws)
        if refuse:
            await send(
                ws,
                type="error",
                error={
                    "code": "typed_turn_busy",
                    "message": "Busy",
                    "event_id": event["event_id"],
                    "scope": "event",
                },
            )
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url, timeout_s=0.05), session()
        try:
            await rt.connect(s)
            result = await rt.inject_text(s, "Hello.")
            assert rt.is_responding(s.id) is not refuse
        finally:
            await rt.close()
    assert result.status == ("not_sent" if refuse else "unknown")
    assert result.retryable is refuse


async def test_close_cancels_an_unacknowledged_connection_setup() -> None:
    connected = asyncio.Event()

    async def handler(ws: ServerConnection) -> None:
        connected.set()
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url), session()
        connecting = asyncio.create_task(rt.connect(s))
        await asyncio.wait_for(connected.wait(), 1)
        await asyncio.wait_for(rt.close(), 1)
        assert connecting.cancelled()
        assert s.state == VoiceSessionState.ENDED


async def test_concurrent_close_waits_for_pending_tool_cleanup() -> None:
    called, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        await receive(ws)
        await send(
            ws,
            type="response.function_call_arguments.done",
            response_id="parent",
            call_id="call",
            name="lookup",
            arguments="{}",
        )
        await ws.wait_closed()

    async def abandoned(_: VoiceSession, ids: list[str]) -> None:
        closing.set()
        await release.wait()

    async with server(handler) as url:
        rt, s = provider(url), session()
        rt.on_tool_call(lambda *_: called.set())
        rt.on_tool_call_cancelled(abandoned)
        await rt.connect(s)
        await asyncio.wait_for(called.wait(), 1)
        first = asyncio.create_task(rt.disconnect(s))
        await asyncio.wait_for(closing.wait(), 1)
        second = asyncio.create_task(rt.close())
        try:
            await asyncio.sleep(0)
            assert not second.done()
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 1)
        assert s.state == VoiceSessionState.ENDED


async def test_refused_tool_result_abandons_remaining_calls_without_continuation() -> None:
    called = asyncio.Event()
    calls, abandoned, outputs = [], [], []

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        await receive(ws)
        for call_id in ("a", "b"):
            await send(
                ws,
                type="response.function_call_arguments.done",
                response_id="parent",
                call_id=call_id,
                name="lookup",
                arguments="{}",
            )
        event = await control(ws)
        outputs.append(event)
        await send(
            ws,
            type="error",
            error={
                "code": "invalid_event",
                "message": "Result refused",
                "scope": "event",
                "event_id": event["event_id"],
            },
        )
        await ws.wait_closed()

    def tool(_: VoiceSession, call_id: str, name: str, args: Any) -> None:
        calls.append(call_id)
        if len(calls) == 2:
            called.set()

    async with server(handler) as url:
        rt, s = provider(url), session()
        rt.on_tool_call(tool)
        rt.on_tool_call_cancelled(lambda _, ids: abandoned.extend(ids))
        try:
            await rt.connect(s)
            await asyncio.wait_for(called.wait(), 1)
            with pytest.raises(RuntimeError, match="Result refused"):
                await rt.submit_tool_result(s, "a", "result")
            await rt.submit_tool_result(s, "b", "too late")
        finally:
            await rt.close()
    assert abandoned == ["b"]
    assert len(outputs) == 1


async def test_existing_realtime_channel_routes_prism_audio_to_its_transport() -> None:
    started, finished = asyncio.Event(), asyncio.Event()

    class Transport(MockRealtimeTransport):
        async def send_audio(self, session: VoiceSession, audio: Any) -> None:
            await super().send_audio(session, audio)
            finished.set()

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        await started.wait()
        await send(ws, type="response.created", response={"id": "reply"})
        await send(ws, type="response.audio.delta", delta="AQACAA==")
        await send(ws, type="response.text.done", text="Hello.")
        await send(ws, type="response.done", response={"id": "reply", "status": "completed"})
        await ws.wait_closed()

    async with server(handler) as url:
        rt, transport, kit = provider(url), Transport(), RoomKit()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=rt,
            transport=transport,
            input_sample_rate=16000,
            output_sample_rate=24000,
        )
        kit.register_channel(channel)
        try:
            await kit.create_room(room_id="r1")
            await kit.attach_channel("r1", "voice")
            s = await channel.start_session("r1", "user", connection=None)
            started.set()
            await asyncio.wait_for(finished.wait(), 1)
            assert transport.sent_audio == [(s.id, b"\x01\x00\x02\x00")]
        finally:
            await kit.close()
        assert s.state == VoiceSessionState.ENDED


@pytest.mark.parametrize("fatal_event", [None, "error", "session.idle_timeout"])
async def test_fatal_connection_failure_ends_the_existing_channel_transport(
    fatal_event: str | None,
) -> None:
    started, disconnected = asyncio.Event(), asyncio.Event()

    class Transport(MockRealtimeTransport):
        async def disconnect(self, session: VoiceSession) -> None:
            await super().disconnect(session)
            disconnected.set()

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        await started.wait()
        if fatal_event:
            await send(
                ws,
                type=fatal_event,
                error={
                    "scope": "session",
                    "code": "unavailable",
                    "message": "Service stopped",
                },
            )
            await ws.wait_closed()

    async with server(handler) as url:
        kit, transport = RoomKit(), Transport()
        channel = RealtimeVoiceChannel("voice", provider=provider(url), transport=transport)
        kit.register_channel(channel)
        try:
            await kit.create_room(room_id="r1")
            await kit.attach_channel("r1", "voice")
            s = await channel.start_session("r1", "user", connection=None)
            started.set()
            await asyncio.wait_for(disconnected.wait(), 1)
            assert s.state == VoiceSessionState.ENDED
        finally:
            await kit.close()


async def test_interrupt_and_truncation_target_the_playing_message() -> None:
    audio_received = asyncio.Event()
    commands = []

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        await receive(ws)
        await send(ws, type="response.created", response={"id": "reply"})
        await send(
            ws,
            type="response.output_item.added",
            item={"id": "spoken", "type": "message", "role": "assistant"},
        )
        await send(ws, type="response.audio.delta", delta=base64.b64encode(b"\0" * 4800).decode())
        commands.append(await control(ws))
        commands.append(await control(ws))
        await send(
            ws,
            type="conversation.item.truncated",
            prism_request_event_id=commands[-1]["event_id"],
            item_id="spoken",
            audio_end_ms=50,
            content_index=0,
        )
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url), session()
        rt.on_audio(lambda *_: audio_received.set())
        try:
            await rt.connect(s)
            await asyncio.wait_for(audio_received.wait(), 1)
            await rt.interrupt(s)
            await rt.truncate_audio(s, 50)
        finally:
            await rt.close()
    assert commands[0] == {"type": "response.cancel", "response_id": "reply"}
    assert commands[1]["item_id"] == "spoken" and commands[1]["audio_end_ms"] == 50


async def test_close_abandons_pending_calls_once_and_drops_late_results() -> None:
    called = asyncio.Event()
    abandoned = []

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        await receive(ws)
        await send(ws, type="response.created", response={"id": "reply"})
        await send(
            ws,
            type="response.function_call_arguments.done",
            response_id="reply",
            call_id="a",
            name="slow",
            arguments="{}",
        )
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url), session()
        rt.on_tool_call(lambda *_: called.set())
        rt.on_tool_call_cancelled(lambda _, ids: abandoned.extend(ids))
        try:
            await rt.connect(s)
            await asyncio.wait_for(called.wait(), 1)
            await rt.disconnect(s)
            await rt.submit_tool_result(s, "a", "too late")
            await rt.disconnect(s)
        finally:
            await rt.close()
    assert abandoned == ["a"] and s.state == VoiceSessionState.ENDED


async def test_mid_session_reconfiguration_keeps_connection_and_other_session_settings() -> None:
    connected = []
    patches = []

    async def handler(ws: ServerConnection) -> None:
        settings = await prism_handshake(ws)
        connected.append(settings)
        if settings["instructions"] == "First":
            event = await control(ws)
            patches.append(event["session"])
            await send(
                ws,
                type="session.updated",
                prism_request_event_id=event["event_id"],
                session={**settings, **event["session"]},
            )
        await ws.wait_closed()

    async with server(handler) as url:
        rt, one, two = provider(url), session("one"), session("two")
        try:
            await rt.connect(one, system_prompt="First")
            await rt.connect(two, system_prompt="Second")
            await rt.reconfigure(one, system_prompt="Updated", tools=[], voice="brando")
            with pytest.raises(ValueError, match="fixed"):
                await rt.reconfigure(one, provider_config={"eagerness": "low"})
        finally:
            await rt.close()
    assert len(connected) == 2 and connected[1]["instructions"] == "Second"
    assert patches == [{"voice": "brando", "instructions": "Updated", "tools": []}]


async def test_audio_is_continuous_with_silence_before_and_after_capture() -> None:
    captured = asyncio.Event()
    idle = asyncio.Event()
    frames = []

    async def handler(ws: ServerConnection) -> None:
        await prism_handshake(ws)
        while not idle.is_set():
            event = await receive(ws)
            assert event["type"] == "input_audio_buffer.append"
            data = base64.b64decode(event["audio"])
            frames.append(data)
            if any(data):
                captured.set()
            elif captured.is_set():
                idle.set()
        await ws.wait_closed()

    async with server(handler) as url:
        rt, s = provider(url), session()
        try:
            await rt.connect(s)
            await rt.send_audio(s, b"\x01\x00" * 320)
            await asyncio.wait_for(idle.wait(), 1)
        finally:
            await rt.close()
    assert all(len(frame) == 640 for frame in frames)
    assert sum(frame.count(b"\x01\x00") for frame in frames) == 320
    assert frames[-1] == b"\0" * 640


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"server_vad": False}, "server_vad"),
        ({"input_sample_rate": 24000}, "16000"),
        ({"temperature": 0.8}, "temperature"),
        ({"provider_config": {"preserve_context": True}}, "Unsupported"),
    ],
)
async def test_unsupported_capabilities_are_refused_before_connect(
    config: dict[str, Any], match: str
) -> None:
    rt = provider("ws://127.0.0.1:1")
    with pytest.raises(ValueError, match=match):
        await rt.connect(session(), **config)
