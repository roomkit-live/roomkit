"""One book of open calls on every realtime provider (RMK-502, RFC §12.4).

A result submitted for a call the provider abandoned, or never issued, goes
out on none of them; a call issued again under an abandoned id is a new call,
answered; the abandonment is reported once. Each provider issues the call
from its own wire (its inbound event); the abandonment is the provider's own
path where it has one on the wire (Gemini's cancellation), else the step its
connection's end runs.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.providers.openai.live_config import HostedReasoning
from roomkit.providers.openai.realtime import OpenAIRealtimeProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from tests.conference.test_conference_realtime import until
from tests.test_openai_live import _FakeWS as LiveWS
from tests.test_openai_live import _function_call, _provider, _started
from tests.test_providers.test_gemini_realtime import _load_provider, _make_mock_live_session
from tests.test_realtime_deepgram import _connect as deepgram_connect
from tests.test_realtime_elevenlabs import _FakeAsyncConversation, _install_fake_sdk

TOOL = {"name": "lookup", "description": "Look up", "parameters": {"type": "object"}}


def _session(state: VoiceSessionState = VoiceSessionState.ACTIVE) -> VoiceSession:
    return VoiceSession(id="s1", room_id="r1", participant_id="u1", channel_id="rt", state=state)


class _Gemini:
    async def start(self) -> Any:
        mod = _load_provider()
        self.provider = mod.GeminiLiveProvider(api_key="k", model="gemini-3.8-live")
        self.session = _session()
        self.live = _make_mock_live_session()
        self.state = mod._GeminiSessionState(session=self.session, live_session=self.live)
        self.provider._sessions[self.session.id] = self.state
        return self.provider

    async def issue(self, call_id: str) -> None:
        call = SimpleNamespace(id=call_id, name="lookup", args={})
        message = SimpleNamespace(function_calls=[call])
        await self.provider._on_tool_call(self.session, self.state, message)

    async def abandon(self, call_id: str) -> None:
        cancellation = SimpleNamespace(tool_call_cancellation=SimpleNamespace(ids=[call_id]))
        await self.provider._handle_server_response(self.session, cancellation)

    async def sent(self, call_id: str) -> bool:
        self.live.send_tool_response.reset_mock()
        await self.provider.submit_tool_result(self.session, call_id, '{"ok": true}')
        return self.live.send_tool_response.await_count > 0

    async def reconnect(self) -> None:
        self.live = _make_mock_live_session()
        self.provider._open_live_session = AsyncMock(return_value=(None, self.live))
        self.provider._start_receive_loop = MagicMock()
        await self.provider.connect(self.session, tools=[TOOL])

    async def stop(self) -> None:
        return None


class _OpenAIRealtime:
    async def start(self) -> Any:
        self.provider = OpenAIRealtimeProvider(api_key="sk-test")
        self.session = _session()
        self.ws = AsyncMock()
        self.provider._connections[self.session.id] = self.ws
        self.provider._sessions[self.session.id] = self.session
        return self.provider

    async def issue(self, call_id: str) -> None:
        item = {"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"}
        await self.provider._on_output_item_done(self.session, {"item": item})

    async def abandon(self, call_id: str) -> None:
        await self.provider._abandon_open_calls(self.session)

    async def sent(self, call_id: str) -> bool:
        self.ws.send.reset_mock()
        await self.provider.submit_tool_result(self.session, call_id, '{"ok": true}')
        return any(
            json.loads(c.args[0]).get("item", {}).get("call_id") == call_id
            for c in self.ws.send.await_args_list
        )

    async def reconnect(self) -> None:
        self.ws = AsyncMock()
        self.session.renegotiate()
        with patch("websockets.connect", AsyncMock(return_value=self.ws)):
            await self.provider.connect(
                self.session, tools=[TOOL], input_sample_rate=24000, output_sample_rate=24000
            )

    async def stop(self) -> None:
        return None


class _GPTLive:
    async def start(self) -> Any:
        self.provider = _provider(
            delegation=HostedReasoning(model="gpt-5.6-terra"), close_timeout_s=0
        )
        self.session = _session(VoiceSessionState.CONNECTING)
        self.ws = LiveWS()
        self.ws.push(_started())
        self.issued = 0
        self.provider.on_tool_call(lambda *args: setattr(self, "issued", self.issued + 1))
        with patch("websockets.connect", AsyncMock(return_value=self.ws)):
            await self.provider.connect(self.session, tools=[TOOL])
        return self.provider

    async def issue(self, call_id: str) -> None:
        before = self.issued
        self.ws.push(_function_call(call_id, "lookup", "{}"))
        await until(lambda: self.issued > before)

    async def abandon(self, call_id: str) -> None:
        await self.provider._abandon_open_calls(self.provider._states[self.session.id])

    async def sent(self, call_id: str) -> bool:
        before = len(self.ws.sent)
        await self.provider.submit_tool_result(self.session, call_id, '{"ok": true}')
        return any(call_id in str(frame) for frame in self.ws.sent[before:])

    async def reconnect(self) -> None:
        self.ws = LiveWS()
        self.ws.push(_started())
        self.session.renegotiate()
        with patch("websockets.connect", AsyncMock(return_value=self.ws)):
            await self.provider.connect(self.session, tools=[TOOL])

    async def stop(self) -> None:
        await self.provider.disconnect(self.session)


class _Deepgram:
    async def start(self) -> Any:
        self.provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key=SecretStr("dg-key")))
        self.session = _session(VoiceSessionState.CONNECTING)
        self.issued = 0
        self.provider.on_tool_call(lambda *args: setattr(self, "issued", self.issued + 1))
        self.ws = await deepgram_connect(self.provider, self.session)
        self.old_ws: list[Any] = []
        return self.provider

    async def issue(self, call_id: str) -> None:
        before = self.issued
        function = {"id": call_id, "name": "lookup", "arguments": "{}", "client_side": True}
        self.ws.push(json.dumps({"type": "FunctionCallRequest", "functions": [function]}))
        await until(lambda: self.issued > before)

    async def abandon(self, call_id: str) -> None:
        # The Voice Agent protocol abandons on the connection's end only: the
        # book's own abandonment, as _finalize_session runs it.
        await self.provider._abandon_open_tool_calls(self.session, [call_id])

    async def sent(self, call_id: str) -> bool:
        before = len(self.ws.sent)
        await self.provider.submit_tool_result(self.session, call_id, '{"ok": true}')
        return any(call_id in str(frame) for frame in self.ws.sent[before:])

    async def reconnect(self) -> None:
        self.old_ws.append(self.ws)
        self.session.renegotiate()
        self.ws = await deepgram_connect(self.provider, self.session)

    async def stop(self) -> None:
        for ws in [*self.old_ws, self.ws]:
            ws.finish()
        await asyncio.sleep(0.05)


class _ElevenLabs:
    async def start(self) -> Any:
        config = ElevenLabsRealtimeConfig(api_key="xi-test", agent_id="agent", tool_timeout_s=5)
        self.provider = ElevenLabsRealtimeProvider(config)
        self.session = _session()
        self.provider._sessions[self.session.id] = self.session
        self.handler = self.provider._make_tool_handler(self.session, "lookup")
        self.waiting: dict[str, asyncio.Future[Any]] = {}
        return self.provider

    async def issue(self, call_id: str) -> None:
        self.waiting[call_id] = asyncio.ensure_future(self.handler({"tool_call_id": call_id}))
        await until(lambda: self.provider._holds_tool_call(self.session, call_id))

    async def abandon(self, call_id: str) -> None:
        ids = self.provider._reject_pending_tools(self.session, "ended")
        await self.provider._abandon_tool_calls(self.session, ids)

    async def sent(self, call_id: str) -> bool:
        """The SDK sends what the waiting handler returns: sent when one
        returns the result."""
        await self.provider.submit_tool_result(self.session, call_id, '{"ok": true}')
        task = self.waiting.get(call_id)
        if task is None:
            return False
        await asyncio.wait([task], timeout=1)
        return task.done() and not task.cancelled() and task.exception() is None

    async def reconnect(self) -> None:
        """Connect again through the fake SDK the provider's own tests use."""
        self.patches = pytest.MonkeyPatch()
        _install_fake_sdk(self.patches)
        self.session.renegotiate()
        connecting = asyncio.create_task(self.provider.connect(self.session, tools=[TOOL]))
        await until(lambda: bool(_FakeAsyncConversation.instances))
        conversation = _FakeAsyncConversation.instances[-1]
        await conversation.started.wait()
        await conversation.audio_interface.start(AsyncMock())
        await connecting

    async def stop(self) -> None:
        for task in self.waiting.values():
            task.cancel()
        await asyncio.gather(*self.waiting.values(), return_exceptions=True)
        if getattr(self, "patches", None) is not None:
            await self.provider.disconnect(self.session)
            self.patches.undo()


PROVIDERS = {
    "gemini": _Gemini,
    "openai-realtime": _OpenAIRealtime,
    "gpt-live": _GPTLive,
    "deepgram": _Deepgram,
    "elevenlabs": _ElevenLabs,
}


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_a_result_for_an_abandoned_or_unknown_id_goes_out_on_no_provider(name: str) -> None:
    driver = PROVIDERS[name]()
    provider = await driver.start()
    abandoned: list[list[str]] = []
    provider.on_tool_call_cancelled(lambda session, ids: abandoned.append(list(ids)))

    await driver.issue("c1")
    await driver.abandon("c1")
    late = await driver.sent("c1")
    unknown = await driver.sent("never")
    await driver.issue("c1")
    reissued = await driver.sent("c1")
    await driver.stop()

    assert (late, unknown, reissued) == (False, False, True)
    assert abandoned[0] == ["c1"]
    assert sum(ids.count("c1") for ids in abandoned) == 1


async def test_a_cancellation_naming_a_call_twice_abandons_it_once() -> None:
    """A server may name a call twice in one cancellation: it is abandoned
    and reported once, and the receive loop never sees an error."""
    driver = _Gemini()
    provider = await driver.start()
    abandoned: list[list[str]] = []
    provider.on_tool_call_cancelled(lambda session, ids: abandoned.append(list(ids)))
    await driver.issue("c1")

    cancellation = SimpleNamespace(tool_call_cancellation=SimpleNamespace(ids=["c1", "c1"]))
    await provider._handle_server_response(driver.session, cancellation)

    assert abandoned == [["c1"]]


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_a_second_connect_under_a_live_session_abandons_the_old_connection_s_calls(
    name: str,
) -> None:
    """Connected again under the same session id, on every provider: the call
    the old connection issued is abandoned and reported once, and its result
    goes out on neither connection (RFC §12.4)."""
    driver = PROVIDERS[name]()
    provider = await driver.start()
    abandoned: list[list[str]] = []
    provider.on_tool_call_cancelled(lambda session, ids: abandoned.append(list(ids)))
    await driver.issue("old-1")

    await driver.reconnect()
    late = await driver.sent("old-1")
    await driver.stop()

    assert abandoned == [["old-1"]]
    assert late is False
