"""What every realtime provider owes its tool calls (RFC §12.4, RMK-299).

One scenario per provider, each on its own fake transport:

- a call the provider abandons (Gemini's cancellation, ElevenLabs' wait timing
  out, GPT-Live's restart, a session's connection lost or closed) is reported
  to the channel once;
- a failed call's result travels as an error where the protocol can say so;
- the tasks a session lives on run in a context of their own;
- a provider whose model calls no tool is declared none;
- Gemini Live tells the model, once, that a call it could not parse did not run,
  and flags a failed call's result under ``error`` as Gemini text does;
- a call whose arguments do not read, which reaches the channel as the model's
  text (``test_realtime_call_arguments``), is refused by the channel and by a
  conference (RMK-375);
- a tool name the endpoint refuses fails when the session's tools are declared,
  and a name no vendor accepts when the tool is given (RMK-375).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from roomkit import ConferenceChannel, ConferenceRealtimeConfig, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.conference.mock import MockConferenceBackend
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.providers.ai.base import ProviderError
from roomkit.providers.ai.tool_calls import MALFORMED_CALL_NUDGE
from roomkit.providers.anam.config import AnamConfig
from roomkit.providers.anam.realtime import AnamRealtimeProvider
from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.providers.deepgram.settings import build_settings, patch_think
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.openai.live_config import HostedReasoning
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.providers.personaplex.realtime import PersonaPlexRealtimeProvider
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_openai_live import (
    TOOL,
    _FakeWS,
    _function_call,
    _provider,
    _started,
)
from tests.test_openai_live import _connect as live_connect
from tests.test_providers import test_gemini_realtime as gemini_tests
from tests.test_providers.test_gemini_realtime import (
    _blocking_call_state,
    _load_provider,
    _make_session,
)
from tests.test_realtime_deepgram import _connect as deepgram_connect
from tests.test_realtime_elevenlabs import _FakeAsyncConversation, _install_fake_sdk

Told = list[list[str]]
_ELEVENLABS = ElevenLabsRealtimeConfig(api_key="xi-test", agent_id="agent")


def _session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="u1",
        channel_id="rt",
        state=VoiceSessionState.CONNECTING,
    )


def _told(provider: Any) -> Told:
    """What the provider reports through ``on_tool_call_cancelled``."""
    told: Told = []
    provider.on_tool_call_cancelled(lambda s, ids: told.append(list(ids)))
    return told


# -- An abandoned call is reported to the channel, once -----------------------


async def _gemini_cancels() -> tuple[Told, list[str]]:
    provider, session, _state, _live = _blocking_call_state()
    told = _told(provider)

    cancellation = SimpleNamespace(tool_call_cancellation=SimpleNamespace(ids=["call-1"]))
    await provider._handle_server_response(session, cancellation)

    return told, ["call-1"]


async def _elevenlabs_times_out() -> tuple[Told, list[str]]:
    provider = ElevenLabsRealtimeProvider(_ELEVENLABS.model_copy(update={"tool_timeout_s": 0.01}))
    session = _session()
    told = _told(provider)
    handler = provider._make_tool_handler(session, "get_weather")

    with pytest.raises(RuntimeError, match="did not return"):
        await handler({"tool_call_id": "c1"})

    return told, ["c1"]


async def _gpt_live_restarts() -> tuple[Told, list[str]]:
    provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"), close_timeout_s=0)
    session = _session()
    told = _told(provider)
    first, second = _FakeWS(), _FakeWS()
    first.push(_started())
    second.push(_started())

    with patch("websockets.connect", AsyncMock(side_effect=[first, second])):
        await provider.connect(session, tools=[TOOL])
        first.push(_function_call("call_1", "get_weather", "{}"))
        await asyncio.sleep(0.01)
        await provider.reconfigure(session, voice="cedar")

    return told, ["call_1"]


class _ClosedForGoodError(Exception):
    """A Live socket closed with a code no reconnect recovers from."""

    code = 1008


async def _raising(exc: Exception) -> Any:
    raise exc
    yield  # an async generator, as the Live session's receive() is


async def _gemini_closes_for_good() -> tuple[Told, list[str]]:
    provider, session, _state, live = _blocking_call_state()
    session.state = VoiceSessionState.ACTIVE
    told = _told(provider)
    live.receive = MagicMock(return_value=_raising(_ClosedForGoodError("policy violation")))

    await provider._receive_loop(session)

    return told, ["call-1"]


async def _gemini_cannot_preserve_its_context() -> tuple[Told, list[str]]:
    provider, session, state, _live = _blocking_call_state()
    told = _told(provider)

    await provider._end_preserved_context(session, state)

    return told, ["call-1"]


async def _gemini_disconnects() -> tuple[Told, list[str]]:
    provider, session, _state, _live = _blocking_call_state()
    told = _told(provider)

    await provider.disconnect(session)

    return told, ["call-1"]


def _elevenlabs_waiting_on(call_id: str) -> tuple[ElevenLabsRealtimeProvider, VoiceSession, Told]:
    provider = ElevenLabsRealtimeProvider(_ELEVENLABS)
    session = _session()
    provider._sessions[session.id] = session
    provider._book_tool_call(session, call_id, asyncio.get_running_loop().create_future())
    return provider, session, _told(provider)


async def _elevenlabs_disconnects() -> tuple[Told, list[str]]:
    # A handoff reconnects through here: the base reconfigure.
    provider, session, told = _elevenlabs_waiting_on("c1")

    await provider.disconnect(session)

    return told, ["c1"]


async def _elevenlabs_conversation_fails() -> tuple[Told, list[str]]:
    provider, session, told = _elevenlabs_waiting_on("c1")

    await provider._fail_session(session, "session_ended", "ended by the service")

    return told, ["c1"]


async def _gpt_live_loses_its_connection() -> tuple[Told, list[str]]:
    provider = _provider(delegation=HostedReasoning(model="gpt-5.6-terra"))
    session = _session()
    told = _told(provider)
    ws = _FakeWS()
    ws.push(_started())
    with patch("websockets.connect", AsyncMock(return_value=ws)):
        await provider.connect(session, tools=[TOOL])

    ws.push(_function_call("call_1", "get_weather", "{}"))
    ws.end()
    await until(lambda: bool(told))

    return told, ["call_1"]


async def _deepgram_loses_its_connection() -> tuple[Told, list[str]]:
    provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key=SecretStr("dg-key")))
    session = _session()
    told = _told(provider)
    ws = await deepgram_connect(provider, session)
    call = {"id": "fc_1", "name": "get_weather", "arguments": "{}", "client_side": True}

    ws.push(json.dumps({"type": "FunctionCallRequest", "functions": [call]}))
    ws.finish()
    await until(lambda: bool(told))

    return told, ["fc_1"]


async def _openai_loses_its_connection() -> tuple[Told, list[str]]:
    provider = OpenAIRealtimeProvider(api_key="sk-test")
    session = _session()
    session.state = VoiceSessionState.ACTIVE
    ws = AsyncMock()
    provider._connections[session.id] = ws
    provider._sessions[session.id] = session
    told = _told(provider)
    for call_id in ("c1", "c2"):
        item = {"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"}
        await provider._on_output_item_done(session, {"item": item})
    await provider.submit_tool_result(session, "c2", "{}")

    await provider._discard_connection(session, ws, error_message="connection lost")

    return told, ["c1"]  # c2 was answered


_ABANDONS: dict[str, Callable[[], Awaitable[tuple[Told, list[str]]]]] = {
    "gemini-cancellation": _gemini_cancels,
    "gemini-closed-for-good": _gemini_closes_for_good,
    "gemini-context-not-preserved": _gemini_cannot_preserve_its_context,
    "gemini-disconnect": _gemini_disconnects,
    "elevenlabs-timeout": _elevenlabs_times_out,
    "elevenlabs-disconnect": _elevenlabs_disconnects,
    "elevenlabs-conversation-failure": _elevenlabs_conversation_fails,
    "gpt-live-restart": _gpt_live_restarts,
    "gpt-live-connection-lost": _gpt_live_loses_its_connection,
    "deepgram-connection-lost": _deepgram_loses_its_connection,
    "openai-connection-lost": _openai_loses_its_connection,
}


@pytest.mark.parametrize("scenario", list(_ABANDONS.values()), ids=list(_ABANDONS))
async def test_an_abandoned_call_is_reported_once(
    scenario: Callable[[], Awaitable[tuple[Told, list[str]]]],
) -> None:
    told, call_ids = await scenario()

    assert told == [call_ids]


# -- A failed call's result travels as an error where the protocol can say so -


async def test_elevenlabs_sends_a_failed_result_as_an_error() -> None:
    provider = ElevenLabsRealtimeProvider(_ELEVENLABS)
    session = _session()
    error = '{"error": "Tool \'get_weather\' is not permitted"}'
    provider.on_tool_call(
        lambda s, call_id, name, args: asyncio.ensure_future(
            provider.submit_tool_error(s, call_id, error)
        )
    )
    handler = provider._make_tool_handler(session, "get_weather")

    # The SDK sends a raising handler's text with ``is_error`` set.
    with pytest.raises(Exception) as raised:
        await handler({"tool_call_id": "c1"})

    assert str(raised.value) == error


@pytest.mark.parametrize(
    ("submit", "result", "response"),
    [
        ("submit_tool_error", "Guests cannot book rooms.", {"error": "Guests cannot book rooms."}),
        (
            "submit_tool_result",
            '{"error": "no flights that day", "flights": []}',
            {"result": '{"error": "no flights that day", "flights": []}'},
        ),
    ],
    ids=["failed-call-flagged", "served-body-with-an-error-key-not-flagged"],
)
async def test_gemini_live_flags_a_failed_call_as_gemini_text_does(
    submit: str, result: str, response: dict[str, str]
) -> None:
    """The flag follows how the call ended, never the result's text (RMK-375)."""
    provider, session, _state, live = _blocking_call_state(tool="book_room")

    await getattr(provider, submit)(session, "call-1", result)

    [sent] = live.send_tool_response.await_args.kwargs["function_responses"]
    assert sent.name == "book_room"
    assert sent.response == response


async def test_a_protocol_without_errors_sends_the_failed_result_as_a_result() -> None:
    provider = MockRealtimeProvider()
    session = _session()

    await provider.submit_tool_error(session, "c1", '{"error": "no"}')

    assert provider.tool_results == [(session.id, "c1", '{"error": "no"}')]


class _ErrorAware(MockRealtimeProvider):
    """Records which of the two submissions the channel chose."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    async def submit_tool_error(self, session: VoiceSession, call_id: str, result: str) -> None:
        self.errors.append(call_id)
        await super().submit_tool_error(session, call_id, result)


async def test_the_channel_submits_a_failed_call_as_an_error() -> None:
    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    provider = _ErrorAware()
    tools = [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=lookup,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_tool_call(session, "ok", "lookup", {})
    await provider.simulate_tool_call(session, "refused", "undeclared", {})
    await until(lambda: len(provider.tool_results) == 2)

    assert provider.errors == ["refused"]
    await kit.close()


async def test_a_conference_submits_a_failed_call_as_an_error() -> None:
    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        raise RuntimeError("backend down")

    provider = _ErrorAware()
    config = ConferenceRealtimeConfig(
        provider=provider, tools=[{"name": "x"}], tool_handler=handler
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "call-1", "x", {})
    await until(lambda: bool(provider.tool_results))

    assert provider.errors == ["call-1"]
    await kit.close()


# -- The tasks a session lives on run in a context of their own ---------------

_CALLER: contextvars.ContextVar[str | None] = contextvars.ContextVar("caller", default=None)


async def test_a_gpt_live_receive_loop_does_not_inherit_its_starters_context() -> None:
    provider = _provider()
    session = _session()
    ws = _FakeWS()
    ws.push(_started())
    _CALLER.set("the handler's call")

    with patch("websockets.connect", AsyncMock(return_value=ws)):
        await provider.connect(session)

    task = provider._states[session.id].receive_task
    assert task is not None
    assert task.get_context().get(_CALLER) is None
    await provider.disconnect(session)


async def test_the_elevenlabs_sdk_session_starts_in_a_context_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SDK creates the conversation's receive task inside start_session.
    _install_fake_sdk(monkeypatch)
    seen: list[str | None] = []

    async def start(conversation: _FakeAsyncConversation) -> None:
        seen.append(_CALLER.get())
        conversation.started.set()

    monkeypatch.setattr(_FakeAsyncConversation, "start_session", start)
    provider = ElevenLabsRealtimeProvider(_ELEVENLABS)
    session = _session()
    _CALLER.set("the handler's call")

    connect = asyncio.create_task(provider.connect(session))
    await until(lambda: bool(_FakeAsyncConversation.instances))
    conversation = _FakeAsyncConversation.instances[0]
    await conversation.started.wait()
    await conversation.audio_interface.start(AsyncMock())
    await connect

    assert seen == [None]
    await provider.disconnect(session)


async def test_a_session_task_runs_in_a_context_of_its_own() -> None:
    seen: list[str | None] = []

    async def loop() -> None:
        seen.append(_CALLER.get())

    _CALLER.set("the handler's call")
    await MockRealtimeProvider._session_task(loop(), name="t")

    assert seen == [None]


# -- A provider whose model calls no tool is declared none --------------------


@pytest.mark.parametrize(
    ("build", "supports"),
    [
        (lambda: AnamRealtimeProvider(AnamConfig(api_key="k", persona_id="p")), False),
        (PersonaPlexRealtimeProvider, False),
        (lambda: ElevenLabsRealtimeProvider(_ELEVENLABS), True),
        (MockRealtimeProvider, True),
    ],
    ids=["anam", "personaplex", "elevenlabs", "mock"],
)
def test_supports_tools_says_whether_the_model_calls_tools(
    build: Callable[[], Any], supports: bool
) -> None:
    assert build().supports_tools is supports


class _Toolless(MockRealtimeProvider):
    @property
    def supports_tools(self) -> bool:
        return False


async def test_a_toolless_provider_is_declared_no_tool(caplog: pytest.LogCaptureFixture) -> None:
    provider = _Toolless()
    tools = [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]
    with caplog.at_level(logging.WARNING, logger="roomkit.channels.tools"):
        channel = RealtimeVoiceChannel(
            "rt",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=tools,
            tool_search=True,
            system_prompt="Be brief.",
        )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")

    await channel.start_session("r1", "u1", "ws")

    connect = next(c for c in provider.calls if c.method == "connect")
    assert not connect.args.get("tools")
    # Nor a Tool Search preamble naming tools it cannot call.
    assert connect.args.get("system_prompt") == "Be brief."
    assert any("cannot call tools" in r.getMessage() for r in caplog.records)
    await kit.close()


# -- Gemini Live tells the model a call it could not parse did not run --------


async def test_gemini_tells_the_model_once_until_the_user_speaks_again() -> None:
    mod = _load_provider()
    provider = mod.GeminiLiveProvider(api_key="test-key", model="gemini-3.8-live")
    session = _make_session()
    state = mod._GeminiSessionState(session=session)
    provider._sessions[session.id] = state
    provider.inject_text = AsyncMock()  # type: ignore[method-assign]
    malformed = gemini_tests.TestGeminiLiveProvider._content(
        turn_complete=True, turn_complete_reason=SimpleNamespace(name="MALFORMED_FUNCTION_CALL")
    )

    await provider._handle_server_response(session, malformed)
    await provider._handle_server_response(session, malformed)
    assert provider.inject_text.await_count == 1
    assert provider.inject_text.await_args.args[1] == MALFORMED_CALL_NUDGE

    await provider._on_voice_activity(
        session, state, SimpleNamespace(voice_activity_type="ACTIVITY_START")
    )
    await provider._handle_server_response(session, malformed)
    assert provider.inject_text.await_count == 2


# -- A call whose arguments do not read reaches the channel as the model's text -

_CUT = '{"amount": 1000, "to": "acc'
"""A call cut mid-arguments: text that reads as no object."""


class _Observed:
    """The ON_TOOL_CALL events the observers receive."""

    def __init__(self, kit: RoomKit) -> None:
        self.events: list[Any] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
        async def observe(event: Any, ctx: Any) -> None:
            self.events.append(event)


def _refused_unread(provider: _ErrorAware) -> None:
    """The one result the provider got: the unreadable refusal, as an error."""
    [(_session_id, call_id, result)] = provider.tool_results
    body = json.loads(result)
    assert body["error"] == "Tool call arguments unreadable"
    assert body["tool"] == "transfer"
    assert provider.errors == [call_id]


async def test_the_channel_refuses_a_call_that_arrives_as_text() -> None:
    ran: list[dict[str, Any]] = []

    async def transfer(name: str, arguments: dict[str, Any]) -> str:
        ran.append(arguments)
        return "ok"

    provider = _ErrorAware()
    tools = [{"name": "transfer", "description": "d", "parameters": {"type": "object"}}]
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=transfer,
    )
    kit = RoomKit()
    observed = _Observed(kit)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")

    await provider.simulate_tool_call(session, "c1", "transfer", _CUT)
    await until(lambda: bool(provider.tool_results) and bool(observed.events))

    assert ran == []
    _refused_unread(provider)
    [event] = observed.events
    assert event.is_error
    assert event.arguments == {"raw": _CUT}
    await kit.close()


async def test_a_conference_refuses_a_call_that_arrives_as_text() -> None:
    ran: list[dict[str, Any]] = []

    async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        ran.append(arguments)
        return "ok"

    provider = _ErrorAware()
    config = ConferenceRealtimeConfig(
        provider=provider, tools=[{"name": "transfer"}], tool_handler=handler
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    observed = _Observed(kit)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "c1", "transfer", _CUT)
    await until(lambda: bool(provider.tool_results) and bool(observed.events))

    assert ran == []
    _refused_unread(provider)
    assert observed.events[0].is_error
    await kit.close()


# -- A tool name the endpoint refuses fails when the tools are declared -------

_DOTTED = {"name": "crm.lookup", "description": "d", "parameters": {"type": "object"}}
_GATEWAY = "wss://gateway.example/v1/live/sessions"


def _deepgram_settings(tools: list[dict[str, Any]], **pc: Any) -> dict[str, Any]:
    return build_settings(
        DeepgramAgentConfig(api_key=SecretStr("dg-key")),
        system_prompt=None,
        voice=None,
        tools=tools,
        temperature=None,
        input_sample_rate=16000,
        output_sample_rate=24000,
        pc=pc,
    )


def _deepgram_switched_to_open_ai(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """A Gemini think block, switched to OpenAI's without new tools."""
    think = _deepgram_settings(tools, think_provider="google")["agent"]["think"]
    return patch_think(
        think, system_prompt=None, tools=None, temperature=None, pc={"think_provider": "open_ai"}
    )


def _gpt_live_hosted(**kwargs: Any) -> Any:
    return _provider(delegation=HostedReasoning(model="gpt-5.6-terra"), **kwargs)


@pytest.mark.parametrize(
    "declare",
    [
        lambda tools: OpenAIRealtimeProvider(api_key="sk")._format_session_tools(tools),
        lambda tools: _gpt_live_hosted()._check_tool_names(tools),
        _deepgram_settings,
        _deepgram_switched_to_open_ai,
    ],
    ids=[
        "openai-realtime",
        "gpt-live-hosted",
        "deepgram-open-ai-think",
        "deepgram-switched-think",
    ],
)
def test_a_name_the_endpoint_refuses_fails_at_declaration(
    declare: Callable[[list[dict[str, Any]]], Any],
) -> None:
    with pytest.raises(ProviderError, match="crm.lookup"):
        declare([_DOTTED])


@pytest.mark.parametrize(
    "declare",
    [
        lambda tools: OpenAIRealtimeProvider(
            api_key="sk", base_url="wss://proxy.example/v1/realtime"
        )._format_session_tools(tools),
        lambda tools: _gpt_live_hosted(base_url=_GATEWAY)._check_tool_names(tools),
        lambda tools: _provider()._check_tool_names(tools),
        lambda tools: XAIRealtimeProvider(api_key="xai")._format_session_tools(tools),
        lambda tools: _deepgram_settings(tools, think_provider="google"),
        lambda tools: _deepgram_settings(
            tools, think_provider="open_ai", think_endpoint={"url": "https://llm.example"}
        ),
        lambda tools: _deepgram_settings(
            tools, settings={"agent": {"think": {"endpoint": {"url": "https://llm.example"}}}}
        ),
    ],
    ids=[
        "behind-base-url",
        "gpt-live-behind-base-url",
        "gpt-live-integrator",
        "xai-accepts-any",
        "deepgram-google-think",
        "deepgram-custom-think",
        "deepgram-endpoint-through-settings",
    ],
)
def test_an_endpoint_whose_rule_admits_the_name_or_is_unknown_declares_it(
    declare: Callable[[list[dict[str, Any]]], Any],
) -> None:
    declare([_DOTTED])


async def _openai_realtime_connects(connect: AsyncMock) -> None:
    with patch("websockets.connect", connect):
        await OpenAIRealtimeProvider(api_key="sk").connect(
            _session(), tools=[_DOTTED], input_sample_rate=24000
        )


async def _gpt_live_connects(connect: AsyncMock) -> None:
    with patch("websockets.connect", connect):
        await _gpt_live_hosted().connect(_session(), tools=[_DOTTED])


async def _deepgram_connects(connect: AsyncMock) -> None:
    provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key=SecretStr("dg-key")))
    with patch("websockets.connect", connect):
        await provider.connect(_session(), tools=[_DOTTED])


@pytest.mark.parametrize(
    "connects",
    [_openai_realtime_connects, _gpt_live_connects, _deepgram_connects],
    ids=["openai-realtime", "gpt-live-hosted", "deepgram"],
)
async def test_a_refused_name_fails_the_session_before_a_socket_opens(
    connects: Callable[[AsyncMock], Awaitable[None]],
) -> None:
    connect = AsyncMock()

    with pytest.raises(ProviderError, match="crm.lookup"):
        await connects(connect)

    connect.assert_not_called()


async def test_a_refused_name_leaves_a_gpt_live_session_as_it_was() -> None:
    provider = _gpt_live_hosted()
    session = _session()
    ws, _ = await live_connect(provider, session, tools=[TOOL], system_prompt="old")
    sent = list(ws.sent)

    with pytest.raises(ProviderError, match="crm.lookup"):
        await provider.reconfigure(session, system_prompt="new", tools=[TOOL, _DOTTED])
    with pytest.raises(ProviderError, match="crm.lookup"):
        await provider.reconfigure(session, voice="cedar", tools=[_DOTTED])

    assert ws.sent == sent
    assert not ws.closed
    assert provider._states[session.id].system_prompt == "old"
    await provider.disconnect(session)


@pytest.mark.parametrize("name", ["look up", "", "café"])
def test_a_name_no_vendor_accepts_is_refused_when_given(name: str) -> None:
    tools = [{"name": name, "description": "d", "parameters": {"type": "object"}}]
    with pytest.raises(ValueError, match="accepted by no provider"):
        RealtimeVoiceChannel(
            "rt", provider=MockRealtimeProvider(), transport=MockRealtimeTransport(), tools=tools
        )
    channel = RealtimeVoiceChannel(
        "rt", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
    )
    with pytest.raises(ValueError, match="accepted by no provider"):
        channel.configure(tools=tools)
    realtime = ConferenceRealtimeConfig(
        provider=MockRealtimeProvider(), tools=tools, tool_handler=AsyncMock()
    )
    with pytest.raises(ValueError, match="accepted by no provider"):
        ConferenceChannel("conf", backend=MockConferenceBackend(), realtime=realtime)


async def test_a_name_no_vendor_accepts_is_refused_with_a_session() -> None:
    """Tools given with a session, or to reconfigure one, are given too."""
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    unnamable = [{"name": "look up", "description": "d", "parameters": {"type": "object"}}]

    with pytest.raises(ValueError, match="accepted by no provider"):
        await channel.start_session("r1", "u1", "ws", metadata={"tools": unnamable})
    session = await channel.start_session("r1", "u1", "ws")
    with pytest.raises(ValueError, match="accepted by no provider"):
        await channel.reconfigure_session(session, tools=unnamable)

    assert [c.method for c in provider.calls].count("reconfigure") == 0
    await kit.close()
