"""OpenAILiveProvider speaks the Live API and the full-duplex contract (RFC §12.4.1).

A fake WebSocket stands in for the API: the tests push server events and read
the client events the provider sends. What they check is the contract the
channel relies on — one immutable ``session.start``, boundaries synthesized
from transcript deltas, the two delegation modes, bounded appends, no-op
interruption, graceful close, and usage that stays in seconds.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from roomkit.providers.openai.live import (
    HostedReasoning,
    IntegratorReasoning,
    OpenAILiveProvider,
)
from roomkit.providers.openai.live_events import (
    MAX_APPEND_TOKENS,
    build_audio_format,
    chunk_text,
    estimated_tokens,
    format_backend_tools,
    history_items,
)
from roomkit.voice.base import VoiceSession, VoiceSessionState

_EOF = object()
TOOL = {
    "name": "get_weather",
    "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    "strict": True,
    "tags": ["weather"],
}


class _FakeWS:
    """The API end of the socket: records what the client sends, replays events."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def push(self, event: dict[str, Any]) -> None:
        self._queue.put_nowait(json.dumps(event))

    def end(self) -> None:
        self._queue.put_nowait(_EOF)

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def close(self) -> None:
        self.closed = True
        self._queue.put_nowait(_EOF)

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is _EOF:
            raise StopAsyncIteration
        return item

    def types(self) -> list[str]:
        return [m["type"] for m in self.sent]

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m["type"] == event_type]


class _Recorder:
    """Every provider callback, in arrival order."""

    def __init__(self, provider: OpenAILiveProvider) -> None:
        self.audio: list[bytes] = []
        self.transcripts: list[tuple[str, str, bool]] = []
        self.speech: list[str] = []
        self.tools: list[tuple[str, str, dict[str, Any]]] = []
        self.responses: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self.delegations: list[tuple[str, str]] = []
        provider.on_audio(lambda s, a: self.audio.append(a))
        provider.on_transcription(lambda s, t, r, f: self.transcripts.append((t, r, f)))
        provider.on_speech_start(lambda s: self.speech.append("start"))
        provider.on_speech_end(lambda s: self.speech.append("end"))
        provider.on_tool_call(lambda s, c, n, a: self.tools.append((c, n, a)))
        provider.on_response_start(lambda s: self.responses.append("start"))
        provider.on_response_end(lambda s: self.responses.append("end"))
        provider.on_error(lambda s, c, m: self.errors.append((c, m)))
        provider.on_delegation(lambda s, d, t: self.delegations.append((d, t)))


def _provider(**overrides: Any) -> OpenAILiveProvider:
    kwargs: dict[str, Any] = {"api_key": "sk-test", "turn_gap_ms": 50, "close_timeout_s": 0.2}
    kwargs.update(overrides)
    return OpenAILiveProvider(**kwargs)


def _started() -> dict[str, Any]:
    return {"type": "session.started", "session": {"id": "sess_1", "expires_at": 1}}


async def _connect(
    provider: OpenAILiveProvider, session: VoiceSession, **kwargs: Any
) -> tuple[_FakeWS, AsyncMock]:
    ws = _FakeWS()
    ws.push(_started())
    connect = AsyncMock(return_value=ws)
    with patch("websockets.connect", connect):
        await provider.connect(session, **kwargs)
    return ws, connect


async def _settle() -> None:
    await asyncio.sleep(0.01)


def _response_event(inner: dict[str, Any], delegation_id: str | None = "d1") -> dict[str, Any]:
    return {"type": "response.event", "delegation_id": delegation_id, "event": inner}


def _function_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return _response_event(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            },
        }
    )


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="u1",
        channel_id="rt-1",
        state=VoiceSessionState.CONNECTING,
    )


class TestIdentity:
    def test_capabilities(self) -> None:
        provider = _provider()
        assert provider.name == "OpenAILiveProvider"
        assert provider.model_name == "gpt-live-1"
        assert provider.full_duplex is True
        assert provider.supports_mid_session_reconfigure is False
        assert isinstance(provider.delegation, IntegratorReasoning)
        assert "gpt-live-1" in {m.id for m in OpenAILiveProvider.available_models()}
        voices = {v.id for v in OpenAILiveProvider.available_voices()}
        assert {"marin", "cedar", "vesper", "bossa"} <= voices
        assert len(voices) == len(OpenAILiveProvider.available_voices())  # unique ids

    def test_rejects_bad_timing(self) -> None:
        with pytest.raises(ValueError, match="turn_gap_ms"):
            _provider(turn_gap_ms=0)
        with pytest.raises(ValueError, match="close_timeout_s"):
            _provider(close_timeout_s=-1)


class TestSessionStart:
    async def test_hosted_start_payload(self, session: VoiceSession) -> None:
        provider = _provider(
            delegation=HostedReasoning(
                model="gpt-5.6-terra",
                instructions="backend rules",
                reasoning_effort="low",
                max_output_tokens=200,
            )
        )
        ws, connect = await _connect(
            provider, session, system_prompt="front rules", voice="cedar", tools=[TOOL]
        )

        assert connect.call_args[0][0] == "wss://api.openai.com/v1/live/sessions"
        assert connect.call_args[1]["additional_headers"] == {"Authorization": "Bearer sk-test"}
        assert ws.types() == ["session.start"]
        cfg = ws.sent[0]["session"]
        assert cfg["model"] == "gpt-live-1"
        assert cfg["instructions"] == "front rules"
        assert cfg["audio"] == {
            "format": {"type": "audio/pcm", "rate": 24000},
            "output": {"voice": "cedar"},
        }
        responses = cfg["delegation"]["responses"]
        assert cfg["delegation"]["type"] == "responses"
        assert responses["model"] == "gpt-5.6-terra"
        assert responses["instructions"] == "backend rules"
        assert responses["reasoning"] == {"effort": "low"}
        assert responses["max_output_tokens"] == 200
        # Responses tool shape: strict and tags dropped, type added.
        assert responses["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": TOOL["parameters"],
            }
        ]
        assert session.state == VoiceSessionState.ACTIVE
        assert session.provider_session_id == "sess_1"
        assert provider.is_responding(session.id) is False

    async def test_integrator_start_keeps_tools_off_the_wire(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(provider, session, tools=[TOOL])

        cfg = ws.sent[0]["session"]
        assert cfg["delegation"] == {"type": "client"}
        assert "get_weather" not in json.dumps(cfg)

    async def test_history_seeds_the_session(self, session: VoiceSession) -> None:
        history = [
            {"role": "system", "text": "be brief"},
            {"role": "user", "text": "hi"},
            {"role": "assistant", "text": "hello"},
            {"role": "tool", "text": "ignored"},
        ]
        ws, _ = await _connect(
            provider := _provider(), session, provider_config={"history": history}
        )

        items = ws.sent[0]["session"]["input"]
        assert [(i["role"], i["content"][0]["type"]) for i in items] == [
            ("developer", "input_text"),
            ("user", "input_text"),
            ("assistant", "output_text"),
        ]
        assert provider.is_responding(session.id) is False

    async def test_sampling_and_vad_flags_have_no_wire_effect(self, session: VoiceSession) -> None:
        ws, _ = await _connect(_provider(), session, temperature=0.5, server_vad=False)
        assert "temperature" not in json.dumps(ws.sent[0])


class TestAudio:
    @pytest.mark.parametrize(
        ("rate", "codec", "wire"),
        [
            (16000, "pcm", {"type": "audio/pcm", "rate": 16000}),
            (24000, "pcm", {"type": "audio/pcm", "rate": 24000}),
            (8000, "pcmu", {"type": "audio/pcmu", "rate": 8000}),
            (8000, "pcma", {"type": "audio/pcma", "rate": 8000}),
        ],
    )
    async def test_one_format_serves_both_directions(
        self, session: VoiceSession, rate: int, codec: str, wire: dict[str, Any]
    ) -> None:
        ws, _ = await _connect(
            _provider(),
            session,
            input_sample_rate=rate,
            output_sample_rate=rate,
            provider_config={"codec": codec},
        )
        assert ws.sent[0]["session"]["audio"]["format"] == wire

    async def test_invalid_format_fails_before_connecting(self, session: VoiceSession) -> None:
        connect = AsyncMock()
        with patch("websockets.connect", connect):
            with pytest.raises(ValueError, match="codec"):
                await _provider().connect(session, output_sample_rate=8000)
            with pytest.raises(ValueError, match="44100"):
                await _provider().connect(session, output_sample_rate=44100)
        connect.assert_not_called()

    async def test_send_audio_resamples_input_to_the_session_rate(
        self, session: VoiceSession
    ) -> None:
        provider = _provider()
        ws, _ = await _connect(
            provider, session, input_sample_rate=16000, output_sample_rate=24000
        )
        await provider.send_audio(session, b"\x01\x00" * 160)  # 10 ms at 16 kHz

        appends = ws.of_type("session.input_audio.append")
        assert len(appends) == 1
        pcm = base64.b64decode(appends[0]["audio"])
        assert abs(len(pcm) - 480) <= 4  # 10 ms at 24 kHz

    async def test_send_audio_encodes_g711(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(
            provider,
            session,
            input_sample_rate=8000,
            output_sample_rate=8000,
            provider_config={"codec": "pcmu"},
        )
        await provider.send_audio(session, b"\x01\x00" * 160)

        payload = base64.b64decode(ws.of_type("session.input_audio.append")[0]["audio"])
        assert len(payload) == 160  # one byte per sample

    async def test_output_audio_reaches_the_callback_as_pcm(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(
            provider,
            session,
            input_sample_rate=8000,
            output_sample_rate=8000,
            provider_config={"codec": "pcmu"},
        )
        ws.push(
            {
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(b"\xff" * 160).decode(),
            }
        )
        await _settle()

        assert len(rec.audio) == 1
        assert len(rec.audio[0]) == 320  # decoded to PCM16

    async def test_send_audio_after_disconnect_is_dropped(self, session: VoiceSession) -> None:
        provider = _provider(close_timeout_s=0)
        ws, _ = await _connect(provider, session)
        await provider.disconnect(session)
        await provider.send_audio(session, b"\x01\x00" * 160)
        assert ws.of_type("session.input_audio.append") == []


class TestSynthesizedBoundaries:
    async def test_assistant_turn_opens_and_closes_a_response(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)

        ws.push({"type": "session.output_transcript.delta", "delta": "Hello"})
        ws.push({"type": "session.output_transcript.delta", "delta": " world"})
        await _settle()

        assert rec.responses == ["start"]
        assert rec.transcripts == [("Hello", "assistant", False), (" world", "assistant", False)]
        assert provider.is_responding(session.id) is True

        await asyncio.sleep(0.12)  # past the 50 ms gap

        assert rec.transcripts[-1] == ("Hello world", "assistant", True)
        assert rec.responses == ["start", "end"]
        assert provider.is_responding(session.id) is False

    async def test_user_turn_opens_and_closes_speech(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)

        ws.push({"type": "session.input_transcript.delta", "delta": "Is my"})
        ws.push({"type": "session.input_transcript.delta", "delta": " flight on time?"})
        await _settle()
        assert rec.speech == ["start"]
        assert [t for t in rec.transcripts if t[1] == "user"] == [
            ("Is my", "user", False),
            (" flight on time?", "user", False),
        ]

        await asyncio.sleep(0.12)
        assert rec.transcripts[-1] == ("Is my flight on time?", "user", True)
        assert rec.speech == ["start", "end"]
        assert rec.responses == []

    async def test_speakers_overlap_with_independent_gaps(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)

        ws.push({"type": "session.output_transcript.delta", "delta": "Let me check"})
        ws.push({"type": "session.input_transcript.delta", "delta": "actually"})
        await _settle()
        assert rec.responses == ["start"]
        assert rec.speech == ["start"]

        await asyncio.sleep(0.12)
        finals = [t for t in rec.transcripts if t[2]]
        assert ("Let me check", "assistant", True) in finals
        assert ("actually", "user", True) in finals

    async def test_empty_delta_is_ignored(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)
        ws.push({"type": "session.output_transcript.delta", "delta": ""})
        await _settle()
        assert rec.responses == []
        assert rec.transcripts == []


class TestDelegation:
    async def test_created_maps_wire_targets(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)

        ws.push(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "d1", "type": "delegation", "target": "responses"},
            }
        )
        ws.push(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "d2", "type": "delegation", "target": "client"},
            }
        )
        ws.push({"type": "session.delegation.created", "delegation": {"target": "client"}})
        await _settle()

        assert rec.delegations == [("d1", "hosted"), ("d2", "integrator")]

    async def test_hosted_tool_flow_resumes_after_the_last_result(
        self, session: VoiceSession
    ) -> None:
        provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session, tools=[TOOL])

        ws.push(_response_event({"type": "response.created"}))
        ws.push(_function_call("call_1", "get_weather", '{"city": "Paris"}'))
        ws.push(_function_call("call_2", "get_weather", '{"city": "Lyon"}'))
        ws.push(
            _response_event(
                {
                    "type": "response.completed",
                    "response": {
                        "model": "gpt-5.6-terra",
                        "usage": {
                            "input_tokens": 120,
                            "output_tokens": 30,
                            "input_tokens_details": {"cached_tokens": 100},
                            "output_tokens_details": {"reasoning_tokens": 10},
                        },
                    },
                }
            )
        )
        await _settle()

        assert rec.tools == [
            ("call_1", "get_weather", {"city": "Paris"}),
            ("call_2", "get_weather", {"city": "Lyon"}),
        ]
        assert ws.of_type("response.create") == []

        await provider.submit_tool_result(session, "call_1", '{"temp": 21}')
        assert ws.of_type("response.item.create")[-1]["item"] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"temp": 21}',
        }
        assert ws.of_type("response.create") == [], "one call still open"

        await provider.submit_tool_result(session, "call_2", '{"temp": 24}')
        assert len(ws.of_type("response.create")) == 1

        backend = session._last_usage["backend"]
        assert backend["model"] == "gpt-5.6-terra"
        assert (backend["input_tokens"], backend["output_tokens"]) == (120, 30)
        assert (backend["cached_tokens"], backend["reasoning_tokens"]) == (100, 10)
        assert "input_tokens" not in session._last_usage  # never the live model's

    async def test_results_before_completion_wait_for_it(self, session: VoiceSession) -> None:
        provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
        ws, _ = await _connect(provider, session, tools=[TOOL])

        ws.push(_response_event({"type": "response.created"}))
        ws.push(_function_call("call_1", "get_weather", "{}"))
        await _settle()
        await provider.submit_tool_result(session, "call_1", "{}")
        assert ws.of_type("response.create") == []

        ws.push(_response_event({"type": "response.completed", "response": {}}))
        await _settle()
        assert len(ws.of_type("response.create")) == 1

    async def test_text_only_response_needs_no_continuation(self, session: VoiceSession) -> None:
        provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
        ws, _ = await _connect(provider, session)

        ws.push(_response_event({"type": "response.created"}))
        ws.push(_response_event({"type": "response.completed", "response": {}}))
        await _settle()

        assert ws.of_type("response.create") == []
        assert provider._states[session.id].pending == {}

    async def test_unknown_call_id_is_dropped(self, session: VoiceSession) -> None:
        provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
        ws, _ = await _connect(provider, session)
        await provider.submit_tool_result(session, "nope", "{}")
        assert ws.types() == ["session.start"]

    async def test_unparseable_arguments_arrive_raw(self, session: VoiceSession) -> None:
        provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)
        ws.push(_function_call("call_1", "get_weather", "not json"))
        await _settle()
        assert rec.tools == [("call_1", "get_weather", {"raw": "not json"})]

    async def test_failed_backend_response_is_an_error(self, session: VoiceSession) -> None:
        provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)
        ws.push(
            _response_event(
                {"type": "response.failed", "response": {"error": {"message": "quota"}}}
            )
        )
        await _settle()
        assert rec.errors == [("response.failed", "quota")]

    async def test_delegation_output_maps_spoken_and_silent(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(provider, session)

        await provider.submit_delegation_output(session, "d1", "UA482 is cancelled.", spoken=True)
        await provider.submit_delegation_output(session, "d1", "Crew shortage.", spoken=False)

        assert ws.sent[-2] == {
            "type": "session.commentary.append",
            "delegation_id": "d1",
            "content": "UA482 is cancelled.",
        }
        assert ws.sent[-1] == {
            "type": "session.thinking.append",
            "delegation_id": "d1",
            "content": "Crew shortage.",
        }

    async def test_long_output_is_split_on_sentences(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(provider, session)
        text = " ".join(f"Sentence number {i} says something useful." for i in range(400))

        await provider.submit_delegation_output(session, "d1", text, spoken=True)

        appends = ws.of_type("session.commentary.append")
        assert len(appends) > 1
        assert all(estimated_tokens(a["content"]) <= MAX_APPEND_TOKENS for a in appends)
        assert all(a["delegation_id"] == "d1" for a in appends)
        assert " ".join(a["content"] for a in appends).split() == text.split()


class TestInjectText:
    async def test_roles_map_to_appends(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(provider, session)

        await provider.inject_text(session, "Be formal.", role="system")
        await provider.inject_text(session, "Greet the user.", role="user")
        await provider.inject_text(session, "The user is a VIP.", role="user", silent=True)

        assert ws.sent[1:] == [
            {
                "type": "session.instructions.append",
                "delegation_id": None,
                "content": "Be formal.",
            },
            {
                "type": "session.commentary.append",
                "delegation_id": None,
                "content": "Greet the user.",
            },
            {
                "type": "session.thinking.append",
                "delegation_id": None,
                "content": "The user is a VIP.",
            },
        ]


class TestNoOps:
    async def test_interrupt_and_truncate_send_nothing(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(provider, session)
        await provider.interrupt(session)
        await provider.truncate_audio(session, 320)
        assert ws.types() == ["session.start"]

    async def test_image_injection_is_unsupported(self, session: VoiceSession) -> None:
        with pytest.raises(NotImplementedError):
            await _provider().inject_image(session, b"png", "image/png")


class TestLifecycle:
    async def test_startup_error_fails_connect(self, session: VoiceSession) -> None:
        provider = _provider()
        ws = _FakeWS()
        ws.push(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "code": "bad_model", "message": "no"},
            }
        )
        with (
            patch("websockets.connect", AsyncMock(return_value=ws)),
            pytest.raises(ConnectionError, match="bad_model"),
        ):
            await provider.connect(session)

        assert session.state == VoiceSessionState.ENDED
        assert provider._states == {}
        assert ws.closed

    async def test_connection_loss_fires_error_and_ends(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)

        ws.end()
        await _settle()

        assert rec.errors and rec.errors[0][0] == "connection_closed"
        assert session.state == VoiceSessionState.ENDED
        assert provider._states == {}

    async def test_graceful_close_waits_for_session_closed(self, session: VoiceSession) -> None:
        provider = _provider(close_timeout_s=1.0)
        ws, _ = await _connect(provider, session)

        task = asyncio.create_task(provider.disconnect(session))
        await _settle()
        assert "session.close" in ws.types()
        assert not task.done()

        ws.push({"type": "session.closed", "reason": "client", "usage": {"seconds": 12.5}})
        await asyncio.wait_for(task, timeout=1.0)

        assert ws.closed
        assert session.state == VoiceSessionState.ENDED
        assert session._last_usage["live_seconds"] == 12.5

    async def test_close_timeout_still_disconnects(self, session: VoiceSession) -> None:
        provider = _provider(close_timeout_s=0.05)
        ws, _ = await _connect(provider, session)
        await asyncio.wait_for(provider.disconnect(session), timeout=1.0)
        assert ws.closed
        assert session.state == VoiceSessionState.ENDED

    async def test_disconnect_delivers_open_turn_finals(self, session: VoiceSession) -> None:
        provider = _provider(close_timeout_s=0)
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)
        ws.push({"type": "session.output_transcript.delta", "delta": "Half a sent"})
        await _settle()

        await provider.disconnect(session)

        assert ("Half a sent", "assistant", True) in rec.transcripts
        assert rec.responses == ["start", "end"]

    async def test_usage_is_seconds_not_tokens(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(provider, session)
        ws.push({"type": "session.usage.updated", "usage": {"seconds": 3.0}})
        await _settle()
        assert session._last_usage == {"live_seconds": 3.0}

    async def test_runtime_error_reaches_on_error(self, session: VoiceSession) -> None:
        provider = _provider()
        rec = _Recorder(provider)
        ws, _ = await _connect(provider, session)
        ws.push({"type": "error", "error": {"code": "rate_limited", "message": "slow down"}})
        await _settle()
        assert rec.errors == [("rate_limited", "slow down")]
        assert session.state == VoiceSessionState.ACTIVE

    async def test_close_disconnects_every_session(self, session: VoiceSession) -> None:
        provider = _provider(close_timeout_s=0)
        ws, _ = await _connect(provider, session)
        await provider.close()
        assert ws.closed
        assert provider._states == {}


class TestReconfigure:
    async def test_prompt_change_is_an_instructions_append(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, connect = await _connect(provider, session, system_prompt="v1")
        await provider.reconfigure(session, system_prompt="v2")
        assert ws.sent[-1] == {
            "type": "session.instructions.append",
            "delegation_id": None,
            "content": "v2",
        }
        assert connect.call_count == 1

    async def test_hosted_tool_change_is_a_session_update(self, session: VoiceSession) -> None:
        provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
        ws, _ = await _connect(provider, session, tools=[TOOL])
        other = {"name": "book_flight", "description": "Book", "parameters": {"type": "object"}}

        await provider.reconfigure(session, tools=[TOOL])  # unchanged
        assert ws.of_type("session.update") == []

        await provider.reconfigure(session, tools=[TOOL, other])
        update = ws.of_type("session.update")[0]["session"]
        assert [t["name"] for t in update["delegation"]["responses"]["tools"]] == [
            "get_weather",
            "book_flight",
        ]

    async def test_integrator_tool_change_stays_local(self, session: VoiceSession) -> None:
        provider = _provider()
        ws, _ = await _connect(provider, session, tools=[TOOL])
        await provider.reconfigure(session, tools=[])
        assert ws.types() == ["session.start"]

    async def test_voice_change_replaces_the_session(self, session: VoiceSession) -> None:
        provider = _provider(close_timeout_s=0)
        ws1, ws2 = _FakeWS(), _FakeWS()
        ws1.push(_started())
        ws2.push(_started())
        connect = AsyncMock(side_effect=[ws1, ws2])
        with patch("websockets.connect", connect):
            await provider.connect(
                session,
                system_prompt="keep me",
                voice="marin",
                input_sample_rate=16000,
                output_sample_rate=24000,
            )
            await provider.reconfigure(session, voice="cedar")

        assert connect.call_count == 2
        assert ws1.closed
        cfg = ws2.sent[0]["session"]
        assert cfg["audio"]["output"] == {"voice": "cedar"}
        assert cfg["audio"]["format"]["rate"] == 24000
        assert cfg["instructions"] == "keep me"
        assert session.state == VoiceSessionState.ACTIVE
        assert provider._states[session.id].ws is ws2


class TestHelpers:
    def test_chunk_text_edges(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("Short.") == ["Short."]
        long_word = "x" * 4000
        chunks = chunk_text(long_word, token_limit=100)
        assert len(chunks) > 1
        assert "".join(chunks) == long_word

    def test_estimated_tokens_counts_non_ascii_as_whole_tokens(self) -> None:
        assert estimated_tokens("abcd") == 1
        assert estimated_tokens("日本") == 2

    def test_build_audio_format_rejects_mismatches(self) -> None:
        with pytest.raises(ValueError, match="only for 8 kHz"):
            build_audio_format(24000, "pcmu")
        with pytest.raises(ValueError, match="pcmu"):
            build_audio_format(8000, "pcm")

    def test_backend_tools_pass_hosted_tools_through(self) -> None:
        assert format_backend_tools([{"type": "web_search"}]) == [{"type": "web_search"}]

    def test_history_keeps_the_most_recent_items(self) -> None:
        items = history_items([{"role": "user", "text": str(i)} for i in range(200)])
        assert len(items) == 128
        assert items[-1]["content"][0]["text"] == "199"
