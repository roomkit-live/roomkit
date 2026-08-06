"""Unit tests for DeepgramAgentProvider (Deepgram Voice Agent, speech-to-speech)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeTransport

_SETTINGS_APPLIED = json.dumps({"type": "SettingsApplied"})


class _FakeWS:
    """Async WebSocket double: replays inbound frames, records outbound ones."""

    def __init__(
        self, handshake: list[Any] | None = None, *, fail_with: Exception | None = None
    ) -> None:
        self.sent: list[Any] = []
        self.closed = False
        self._handshake = list(handshake if handshake is not None else [_SETTINGS_APPLIED])
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._fail_with = fail_with

    async def send(self, message: Any) -> None:
        self.sent.append(message)

    async def recv(self) -> Any:
        if self._handshake:
            return self._handshake.pop(0)
        return await self._queue.get()

    async def close(self) -> None:
        self.closed = True

    def push(self, message: Any) -> None:
        """Queue a frame for the provider's receive loop."""
        self._queue.put_nowait(message)

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> Any:
        if self._fail_with is not None:
            raise self._fail_with
        return await self._queue.get()

    # -- assertions helpers --

    @property
    def json_sent(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.sent if isinstance(m, str)]

    def last_of_type(self, mtype: str) -> dict[str, Any]:
        matches = [m for m in self.json_sent if m.get("type") == mtype]
        assert matches, f"no {mtype} message sent (got {[m.get('type') for m in self.json_sent]})"
        return matches[-1]


class _Recorder:
    """Callback double that records calls and lets a test await the next one."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._event = asyncio.Event()

    def __call__(self, *args: Any) -> None:
        self.calls.append(args)
        self._event.set()

    async def wait(self, timeout: float = 1.0) -> tuple[Any, ...]:
        await asyncio.wait_for(self._event.wait(), timeout)
        self._event.clear()
        return self.calls[-1]


@pytest.fixture
def config() -> DeepgramAgentConfig:
    return DeepgramAgentConfig(api_key=SecretStr("dg-test-key"))


@pytest.fixture
def provider(config: DeepgramAgentConfig) -> DeepgramAgentProvider:
    return DeepgramAgentProvider(config)


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(
        id="test-session-1",
        room_id="room-1",
        participant_id="user-1",
        channel_id="voice-1",
        state=VoiceSessionState.CONNECTING,
    )


async def _connect(
    provider: DeepgramAgentProvider,
    session: VoiceSession,
    ws: _FakeWS | None = None,
    **kwargs: Any,
) -> _FakeWS:
    """Connect the provider against a fake socket."""
    ws = ws or _FakeWS()
    with patch("websockets.connect", AsyncMock(return_value=ws)):
        await provider.connect(session, **kwargs)
    return ws


class TestConfig:
    def test_defaults(self) -> None:
        cfg = DeepgramAgentConfig(api_key=SecretStr("dg-key"))
        assert cfg.base_url == "wss://agent.deepgram.com/v1/agent/converse"
        assert cfg.listen_model == "nova-3"
        assert cfg.listen_version is None
        assert cfg.think_provider == "open_ai"
        assert cfg.think_model == "gpt-4o-mini"
        assert cfg.speak_model == "aura-2-thalia-en"
        assert cfg.keepalive_interval == 8.0


class TestProviderBasics:
    def test_name(self, provider: DeepgramAgentProvider) -> None:
        assert provider.name == "DeepgramAgentProvider"

    def test_init_with_api_key(self) -> None:
        p = DeepgramAgentProvider(api_key="dg-key-direct")
        assert p.name == "DeepgramAgentProvider"

    def test_init_requires_key_or_config(self) -> None:
        with pytest.raises(ValueError, match="Either config or api_key"):
            DeepgramAgentProvider()

    def test_available_voices(self) -> None:
        voices = DeepgramAgentProvider.available_voices()
        ids = [v.id for v in voices]
        assert "aura-2-thalia-en" in ids
        assert "aura-2-agathe-fr" in ids
        assert len(ids) == len(set(ids)), "voice ids must be unique"
        # Aura-1 is still resolvable but superseded.
        aura1 = next(v for v in voices if v.id == "aura-asteria-en")
        assert aura1.deprecated is True
        assert all(not v.deprecated for v in voices if v.id.startswith("aura-2-"))

    def test_is_responding_unknown_session(self, provider: DeepgramAgentProvider) -> None:
        assert provider.is_responding("nope") is False


class TestConnect:
    async def test_sends_settings_and_activates(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(
            provider,
            session,
            system_prompt="Be brief",
            voice="aura-2-hera-en",
            temperature=0.4,
            input_sample_rate=16000,
            output_sample_rate=24000,
        )

        settings = ws.last_of_type("Settings")
        assert settings["audio"]["input"] == {"encoding": "linear16", "sample_rate": 16000}
        assert settings["audio"]["output"] == {
            "encoding": "linear16",
            "sample_rate": 24000,
            "container": "none",
        }
        agent = settings["agent"]
        assert agent["listen"]["provider"] == {"type": "deepgram", "model": "nova-3"}
        assert agent["think"]["prompt"] == "Be brief"
        assert agent["think"]["provider"]["temperature"] == 0.4
        assert agent["speak"]["provider"]["model"] == "aura-2-hera-en"

        assert session.state == VoiceSessionState.ACTIVE
        assert session.provider_session_id == session.id

        await provider.disconnect(session)

    async def test_auth_header_and_url(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = _FakeWS()
        connect_mock = AsyncMock(return_value=ws)
        with patch("websockets.connect", connect_mock):
            await provider.connect(session)

        assert connect_mock.call_args[0][0] == "wss://agent.deepgram.com/v1/agent/converse"
        headers = connect_mock.call_args.kwargs["additional_headers"]
        assert headers["Authorization"] == "Token dg-test-key"

        await provider.disconnect(session)

    async def test_tools_are_projected_to_functions(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
                "tags": ["meteo"],  # RoomKit-only key — Deepgram rejects unknown fields
            },
            {"description": "nameless tools are skipped"},
        ]
        ws = await _connect(provider, session, tools=tools)

        functions = ws.last_of_type("Settings")["agent"]["think"]["functions"]
        assert functions == [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        await provider.disconnect(session)

    async def test_provider_config_overrides(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(
            provider,
            session,
            input_sample_rate=8000,
            output_sample_rate=8000,
            provider_config={
                "input_encoding": "mulaw",
                "output_encoding": "mulaw",
                "listen_model": "flux-general-en",
                "listen_version": "v2",
                "think_model": "gpt-4o",
                "greeting": "Bonjour !",
                "tags": ["sip"],
                "settings": {"agent": {"listen": {"provider": {"keyterms": ["RoomKit"]}}}},
            },
        )

        settings = ws.last_of_type("Settings")
        assert settings["audio"]["input"] == {"encoding": "mulaw", "sample_rate": 8000}
        assert settings["audio"]["output"]["encoding"] == "mulaw"
        assert settings["tags"] == ["sip"]
        listen = settings["agent"]["listen"]["provider"]
        assert listen["model"] == "flux-general-en"
        assert listen["version"] == "v2"
        # The escape hatch merges into the built payload rather than replacing it.
        assert listen["keyterms"] == ["RoomKit"]
        assert settings["agent"]["think"]["provider"]["model"] == "gpt-4o"
        assert settings["agent"]["greeting"] == "Bonjour !"

        await provider.disconnect(session)

    async def test_server_vad_false_warns(
        self,
        provider: DeepgramAgentProvider,
        session: VoiceSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="roomkit.providers.deepgram.realtime"):
            await _connect(provider, session, server_vad=False)

        assert "server_vad=False is not supported" in caplog.text
        await provider.disconnect(session)

    async def test_error_during_handshake_raises_and_cleans_up(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = _FakeWS(
            handshake=[json.dumps({"type": "Error", "code": "BAD_KEY", "description": "nope"})]
        )
        with (
            patch("websockets.connect", AsyncMock(return_value=ws)),
            pytest.raises(ConnectionError, match="BAD_KEY"),
        ):
            await provider.connect(session)

        assert ws.closed is True
        assert provider._states == {}
        assert session.state != VoiceSessionState.ACTIVE

    async def test_greeting_audio_before_ack_is_forwarded(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        audio = _Recorder()
        provider.on_audio(audio)
        ws = _FakeWS(handshake=[b"\x01\x02", _SETTINGS_APPLIED])

        with patch("websockets.connect", AsyncMock(return_value=ws)):
            await provider.connect(session)

        assert audio.calls == [(session, b"\x01\x02")]
        await provider.disconnect(session)


class TestInboundDispatch:
    async def test_binary_frame_fires_audio(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        audio = _Recorder()
        provider.on_audio(audio)
        ws = await _connect(provider, session)

        ws.push(b"pcm-bytes")
        assert await audio.wait() == (session, b"pcm-bytes")

        await provider.disconnect(session)

    async def test_assistant_transcript(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        transcription = _Recorder()
        speech_end = _Recorder()
        provider.on_transcription(transcription)
        provider.on_speech_end(speech_end)
        ws = await _connect(provider, session)

        ws.push(json.dumps({"type": "ConversationText", "role": "assistant", "content": "Salut"}))
        assert await transcription.wait() == (session, "Salut", "assistant", True)
        assert speech_end.calls == []

        await provider.disconnect(session)

    async def test_user_transcript_closes_the_turn(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        transcription = _Recorder()
        speech_end = _Recorder()
        provider.on_transcription(transcription)
        provider.on_speech_end(speech_end)
        ws = await _connect(provider, session)

        ws.push(json.dumps({"type": "ConversationText", "role": "user", "content": "Bonjour"}))
        assert await transcription.wait() == (session, "Bonjour", "user", True)
        # Deepgram has no speech-stopped event; the user's transcript ends the turn.
        assert speech_end.calls == [(session,)]

        await provider.disconnect(session)

    async def test_user_started_speaking(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        speech_start = _Recorder()
        provider.on_speech_start(speech_start)
        ws = await _connect(provider, session)
        provider._states[session.id].responding = True

        ws.push(json.dumps({"type": "UserStartedSpeaking"}))
        assert await speech_start.wait() == (session,)
        assert provider.is_responding(session.id) is False

        await provider.disconnect(session)

    async def test_first_audio_frame_opens_the_turn(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        """The channel arms echo cancellation on response_start.

        If audio reaches the speaker first, the AEC is still bypassed and the
        agent hears itself — so the first frame must open the turn even when
        AgentStartedSpeaking never arrives.
        """
        start = _Recorder()
        provider.on_response_start(start)
        ws = await _connect(provider, session)

        ws.push(b"first-frame")
        assert await start.wait() == (session,)
        assert provider.is_responding(session.id) is True

        # Idempotent: the rest of the turn does not re-open it.
        ws.push(b"second-frame")
        ws.push(json.dumps({"type": "AgentStartedSpeaking"}))
        ws.push(json.dumps({"type": "AgentAudioDone"}))
        await asyncio.sleep(0.05)
        assert len(start.calls) == 1

        await provider.disconnect(session)

    async def test_turn_reopens_after_audio_done(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        start = _Recorder()
        end = _Recorder()
        provider.on_response_start(start)
        provider.on_response_end(end)
        ws = await _connect(provider, session)

        ws.push(b"turn-1")
        await start.wait()
        ws.push(json.dumps({"type": "AgentAudioDone"}))
        await end.wait()

        ws.push(b"turn-2")
        await start.wait()
        assert len(start.calls) == 2

        await provider.disconnect(session)

    async def test_response_lifecycle(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        start = _Recorder()
        end = _Recorder()
        provider.on_response_start(start)
        provider.on_response_end(end)
        ws = await _connect(provider, session)

        ws.push(json.dumps({"type": "AgentStartedSpeaking"}))
        await start.wait()
        assert provider.is_responding(session.id) is True

        ws.push(json.dumps({"type": "AgentAudioDone"}))
        await end.wait()
        assert provider.is_responding(session.id) is False

        await provider.disconnect(session)

    async def test_function_call_request_parses_arguments(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        tool_call = _Recorder()
        provider.on_tool_call(tool_call)
        ws = await _connect(provider, session)

        ws.push(
            json.dumps(
                {
                    "type": "FunctionCallRequest",
                    "functions": [
                        {"id": "srv", "name": "billing", "arguments": "{}", "client_side": False},
                        {
                            "id": "fc_1",
                            "name": "get_weather",
                            "arguments": '{"location": "Montr\\u00e9al"}',
                            "client_side": True,
                        },
                    ],
                }
            )
        )

        call = await tool_call.wait()
        # Server-side calls are Deepgram's to run — only the client-side one surfaces.
        assert tool_call.calls == [call]
        assert call == (session, "fc_1", "get_weather", {"location": "Montréal"})
        assert provider._states[session.id].pending_calls["fc_1"] == "get_weather"

        await provider.disconnect(session)

    async def test_unparseable_arguments_fall_back_to_empty(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        tool_call = _Recorder()
        provider.on_tool_call(tool_call)
        ws = await _connect(provider, session)

        ws.push(
            json.dumps(
                {
                    "type": "FunctionCallRequest",
                    "functions": [{"id": "fc_2", "name": "boom", "arguments": "not-json"}],
                }
            )
        )

        assert await tool_call.wait() == (session, "fc_2", "boom", {})
        await provider.disconnect(session)

    @pytest.mark.parametrize("etype", ["Error", "Warning"])
    async def test_diagnostics_fire_on_error(
        self, provider: DeepgramAgentProvider, session: VoiceSession, etype: str
    ) -> None:
        error = _Recorder()
        provider.on_error(error)
        ws = await _connect(provider, session)

        ws.push(json.dumps({"type": etype, "code": "SOME_CODE", "description": "details"}))
        assert await error.wait() == (session, "SOME_CODE", "details")

        await provider.disconnect(session)

    async def test_undecodable_frame_is_ignored(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        error = _Recorder()
        provider.on_error(error)
        audio = _Recorder()
        provider.on_audio(audio)
        ws = await _connect(provider, session)

        ws.push("<not json>")
        ws.push(b"still-flowing")
        assert await audio.wait() == (session, b"still-flowing")
        assert error.calls == []

        await provider.disconnect(session)


class TestOutbound:
    async def test_send_audio_is_binary(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        await provider.send_audio(session, b"\x00\x01")
        assert ws.sent[-1] == b"\x00\x01"
        await provider.disconnect(session)

    async def test_send_audio_without_session_is_noop(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        await provider.send_audio(session, b"\x00")  # must not raise

    async def test_inject_user_message(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        await provider.inject_text(session, "Quelle heure est-il ?")
        assert ws.last_of_type("InjectUserMessage")["content"] == "Quelle heure est-il ?"
        await provider.disconnect(session)

    async def test_inject_agent_message(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        await provider.inject_text(session, "Bonjour !", role="assistant")
        assert ws.last_of_type("InjectAgentMessage")["content"] == "Bonjour !"
        await provider.disconnect(session)

    async def test_silent_injection_appends_to_prompt(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session, system_prompt="Tu es concis.")
        await provider.inject_text(session, "L'appelant est un client VIP.", silent=True)

        prompt = ws.last_of_type("UpdatePrompt")["prompt"]
        # Appended, not replaced — the original instructions must survive.
        assert prompt == "Tu es concis.\n\nL'appelant est un client VIP."
        assert provider._states[session.id].think["prompt"] == prompt

        await provider.disconnect(session)

    async def test_submit_tool_result_recovers_the_name(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        provider._states[session.id].pending_calls["fc_9"] = "get_weather"

        await provider.submit_tool_result(session, "fc_9", '{"temp": 21}')

        response = ws.last_of_type("FunctionCallResponse")
        assert response == {
            "type": "FunctionCallResponse",
            "id": "fc_9",
            "name": "get_weather",
            "content": '{"temp": 21}',
        }
        assert "fc_9" not in provider._states[session.id].pending_calls

        await provider.disconnect(session)

    async def test_interrupt_sends_nothing(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        provider._states[session.id].responding = True
        sent_before = len(ws.sent)

        await provider.interrupt(session)

        # The protocol has no client-side interrupt: state only.
        assert len(ws.sent) == sent_before
        assert provider.is_responding(session.id) is False

        await provider.disconnect(session)

    async def test_reconfigure_updates_in_place(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(
            provider,
            session,
            system_prompt="Agent 1",
            tools=[{"name": "old_tool", "description": "old"}],
        )

        await provider.reconfigure(
            session,
            system_prompt="Agent 2",
            voice="aura-2-zeus-en",
            tools=[{"name": "new_tool", "description": "new"}],
        )

        think = ws.last_of_type("UpdateThink")["think"]
        assert think["prompt"] == "Agent 2"
        assert [f["name"] for f in think["functions"]] == ["new_tool"]
        assert ws.last_of_type("UpdateSpeak")["speak"]["provider"]["model"] == "aura-2-zeus-en"

        # Same socket, still active: the conversation context is not thrown away.
        assert ws.closed is False
        assert session.state == VoiceSessionState.ACTIVE
        assert provider._states[session.id].ws is ws

        await provider.disconnect(session)

    async def test_reconfigure_keeps_tools_when_not_restated(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(
            provider, session, system_prompt="A", tools=[{"name": "keep_me", "description": "d"}]
        )

        await provider.reconfigure(session, system_prompt="B")

        think = ws.last_of_type("UpdateThink")["think"]
        assert think["prompt"] == "B"
        assert [f["name"] for f in think["functions"]] == ["keep_me"]

        await provider.disconnect(session)

    async def test_send_event_passthrough(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        await provider.send_event(session, {"type": "UpdateListen", "listen": {}})
        assert ws.last_of_type("UpdateListen") == {"type": "UpdateListen", "listen": {}}
        await provider.disconnect(session)


class TestLifecycle:
    async def test_keepalive_is_sent(self, session: VoiceSession) -> None:
        provider = DeepgramAgentProvider(
            DeepgramAgentConfig(api_key=SecretStr("dg"), keepalive_interval=0.01)
        )
        ws = await _connect(provider, session)

        await asyncio.sleep(0.1)
        assert any(m.get("type") == "KeepAlive" for m in ws.json_sent)

        await provider.disconnect(session)

    async def test_disconnect_closes_and_cancels(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        state = provider._states[session.id]

        await provider.disconnect(session)

        assert ws.closed is True
        assert session.state == VoiceSessionState.ENDED
        assert provider._states == {}
        assert state.receive_task is not None and state.receive_task.done()
        assert state.keepalive_task is not None and state.keepalive_task.done()

    async def test_disconnect_is_idempotent(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        await _connect(provider, session)
        await provider.disconnect(session)
        await provider.disconnect(session)  # must not raise

    async def test_close_disconnects_every_session(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        ws = await _connect(provider, session)
        await provider.close()
        assert ws.closed is True
        assert provider._states == {}

    async def test_missing_websockets_raises_import_error(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        with (
            patch.dict("sys.modules", {"websockets": None}),
            pytest.raises(ImportError, match="realtime-deepgram"),
        ):
            await provider.connect(session)

    async def test_receive_loop_reports_a_broken_socket(
        self, provider: DeepgramAgentProvider, session: VoiceSession
    ) -> None:
        error = _Recorder()
        provider.on_error(error)
        ws = _FakeWS(fail_with=OSError("socket died"))

        with patch("websockets.connect", AsyncMock(return_value=ws)):
            await provider.connect(session)

        assert await error.wait() == (session, "connection_error", "socket died")
        await provider.disconnect(session)


class TestChannelIntegration:
    """End-to-end through a real RealtimeVoiceChannel, only the socket faked."""

    async def test_tool_call_round_trip(self) -> None:
        """Tools declared on the channel reach Deepgram, results come back.

        Deepgram needs no dashboard registration, unlike ElevenLabs: the schema
        travels in Settings, so declaring the tool on the channel is the whole
        setup.
        """
        calls: list[tuple[str, dict[str, Any]]] = []

        async def handler(name: str, arguments: dict[str, Any]) -> str:
            calls.append((name, arguments))
            return json.dumps({"temperature_c": 21})

        provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key=SecretStr("dg")))
        channel = RealtimeVoiceChannel(
            "rt-dg",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[
                {
                    "name": "get_weather",
                    "description": "Current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            tool_handler=handler,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-dg")

        ws = _FakeWS()
        with patch("websockets.connect", AsyncMock(return_value=ws)):
            session = await channel.start_session(room.id, "user-1", "fake-ws")

        functions = ws.last_of_type("Settings")["agent"]["think"]["functions"]
        assert [f["name"] for f in functions] == ["get_weather"]
        assert functions[0]["parameters"]["required"] == ["city"]

        ws.push(
            json.dumps(
                {
                    "type": "FunctionCallRequest",
                    "functions": [
                        {
                            "id": "fc_1",
                            "name": "get_weather",
                            "arguments": '{"city": "Montreal"}',
                            "client_side": True,
                        }
                    ],
                }
            )
        )

        for _ in range(100):
            await asyncio.sleep(0.02)
            if any(m.get("type") == "FunctionCallResponse" for m in ws.json_sent):
                break

        assert calls == [("get_weather", {"city": "Montreal"})]
        response = ws.last_of_type("FunctionCallResponse")
        assert response["id"] == "fc_1"
        assert response["name"] == "get_weather"
        assert json.loads(response["content"]) == {"temperature_c": 21}

        await channel.end_session(session)
