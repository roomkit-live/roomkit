"""Every realtime call reaches the channel, and every refusal at its entry
takes the path of any call (RMK-442, RFC §12.4).

A provider hands on a call that named no tool, which the channel refuses
before the gate and answers under its id. A call no result can name (no id,
an id in flight) is refused on the normal path: the session's end is read
first, a ``call_tool`` transport unwrapped, the transcription barrier waited
for, and nothing is sent. The ElevenLabs SDK cases live beside its canaries
(``tests/test_providers/test_elevenlabs_sdk_patch.py``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import SecretStr

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.deepgram.config import DeepgramAgentConfig
from roomkit.providers.deepgram.realtime import DeepgramAgentProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.test_openai_live import _connect as live_connect
from tests.test_openai_live import _function_call, _response_event
from tests.test_openai_live import _provider as live_provider
from tests.test_providers.test_openai_realtime_tool_results import _PROVIDERS
from tests.test_providers.test_openai_realtime_tool_results import _attach as openai_attach
from tests.test_realtime_deepgram import _connect as deepgram_connect

NAMELESS = "Tool call named no tool"


def _session() -> VoiceSession:
    return VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="p1",
        channel_id="rt",
        state=VoiceSessionState.CONNECTING,
    )


async def test_deepgram_hands_on_a_call_that_named_no_tool() -> None:
    provider = DeepgramAgentProvider(DeepgramAgentConfig(api_key=SecretStr("k")))
    session = _session()
    heard: list[Any] = []
    provider.on_tool_call(lambda *a: heard.append(a[1:3]))
    ws = await deepgram_connect(provider, session)
    ws.push(json.dumps({"type": "FunctionCallRequest", "functions": [{"id": "fc1"}]}))
    await asyncio.sleep(0.1)

    assert heard == [("fc1", "")]
    assert provider._holds_tool_call(session, "fc1")
    await provider.disconnect(session)


async def test_gpt_live_hands_on_a_call_that_named_no_tool() -> None:
    provider = live_provider()
    session = _session()
    heard: list[Any] = []
    provider.on_tool_call(lambda *a: heard.append(a[1:]))
    ws, _ = await live_connect(provider, session)
    ws.push(_response_event({"type": "response.created"}))
    ws.push(_function_call("c1", "", "{}"))
    await asyncio.sleep(0.1)

    assert heard == [("c1", "", {})]
    assert provider._holds_tool_call(session, "c1")
    await provider.disconnect(session)


@pytest.mark.parametrize("vendor", sorted(_PROVIDERS))
async def test_openai_realtime_hands_on_explicit_nulls_as_strings(vendor: str) -> None:
    """``"call_id": null`` and ``"name": null`` reach the channel as ``""``:
    a call without an id, naming no tool, which it refuses."""
    provider = _PROVIDERS[vendor]()
    session = _session()
    openai_attach(provider, session)
    heard: list[Any] = []
    provider.on_tool_call(lambda *a: heard.append(a[1:3]))

    await provider._handle_server_event(
        session,
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": None,
                "name": None,
                "arguments": "{}",
                "status": "completed",
            },
        },
    )

    assert heard == [("", "")]


async def _channel(
    **kwargs: Any,
) -> tuple[RoomKit, RealtimeVoiceChannel, MockRealtimeProvider, Any, list[Any]]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt", provider=provider, transport=MockRealtimeTransport(), **kwargs
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    seen: list[Any] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        seen.append((event.tool_call_id, event.name, event.is_error, event.cancelled))

    session = await channel.start_session("r1", "u", "ws")
    return kit, channel, provider, session, seen


@pytest.mark.parametrize("name", ["", None])
async def test_a_call_that_named_no_tool_is_refused_under_its_id(name: str | None) -> None:
    """Every provider hands such a call on with ``""``; ``None`` reads alike."""
    kit, _, provider, session, seen = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}], tool_handler=lambda *a: "ok"
    )

    await provider.simulate_tool_call(session, "c1", name, {})  # type: ignore[arg-type]
    for _ in range(100):
        if provider.tool_results:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    [(_, call_id, body)] = provider.tool_results
    assert call_id == "c1" and json.loads(body)["error"] == NAMELESS
    assert seen == [("c1", "", True, False)]


async def test_an_entry_refusal_waits_behind_the_barrier_under_the_wrapped_tool() -> None:
    names = [f"t{i}" for i in range(30)]
    kit, channel, provider, session, seen = await _channel(
        tools=[{"name": n, "description": n, "parameters": {"type": "object"}} for n in names],
        tool_handler=lambda *a: asyncio.sleep(0.2, "ok"),
        tool_search=True,
    )
    assert channel._tool_search_support is not None
    channel._tool_search_support.uses_call_tool = True
    wrapped = {"name": "t1", "arguments_json": "{}"}
    barrier = channel._transcription_order_locks.setdefault(session.id, asyncio.Lock())
    await barrier.acquire()

    await provider.simulate_tool_call(session, "d1", "call_tool", wrapped)
    await provider.simulate_tool_call(session, "d1", "call_tool", wrapped)
    await provider.simulate_tool_call(session, "", "call_tool", wrapped)
    await asyncio.sleep(0.1)
    reported_while_held = list(seen)
    barrier.release()
    # The served call's result may go out after its report: wait for both,
    # the one result this scenario sends (the refused calls send none).
    for _ in range(250):
        if len(seen) == 3 and len(provider.tool_results) == 1:
            break
        await asyncio.sleep(0.02)
    await kit.close()

    assert reported_while_held == []
    assert sorted(seen) == sorted(
        [("d1", "t1", True, False), ("", "t1", True, False), ("d1", "t1", False, False)]
    )
    # Only the call the id names was answered.
    assert [result[1] for result in provider.tool_results] == ["d1"]


async def test_an_id_less_call_on_an_ended_session_is_cancelled() -> None:
    kit, channel, provider, session, seen = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}], tool_handler=lambda *a: "ok"
    )
    await channel.end_session(session)

    await provider.simulate_tool_call(session, "", "t1", {})
    await asyncio.sleep(0.1)
    await kit.close()

    assert seen == [("", "t1", True, True)]
    assert provider.tool_results == []


@pytest.mark.parametrize(("name", "arguments"), [("t1", "not json"), ("", {})])
async def test_an_unreadable_call_on_an_ended_session_is_cancelled(
    name: str, arguments: Any
) -> None:
    """The session's end is read before the call: whoever issued it is gone,
    and it is cancelled rather than refused, nameless or unreadable alike."""
    kit, channel, provider, session, seen = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}], tool_handler=lambda *a: "ok"
    )
    await channel.end_session(session)

    await provider.simulate_tool_call(session, "c1", name, arguments)
    await asyncio.sleep(0.1)
    await kit.close()

    assert seen == [("c1", name, True, True)]
    assert provider.tool_results == []


async def test_an_id_less_refusal_names_the_tool_call_tool_carries() -> None:
    names = [f"t{i}" for i in range(30)]
    kit, channel, provider, session, _ = await _channel(
        tools=[{"name": n, "description": n, "parameters": {"type": "object"}} for n in names],
        tool_handler=lambda *a: "ok",
        tool_search=True,
    )
    assert channel._tool_search_support is not None
    channel._tool_search_support.uses_call_tool = True
    bodies: list[str] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="bodies")
    async def record(event: Any, ctx: Any) -> None:
        bodies.append(json.loads(str(event.result))["error"])

    wrapped = {"name": "t1", "arguments_json": "{}"}
    await provider.simulate_tool_call(session, "", "call_tool", wrapped)
    await provider.simulate_tool_call(session, "", "", {})
    for _ in range(100):
        if len(bodies) == 2:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert sorted(bodies) == [
        "A tool call came without an id",
        "Tool call 't1' came without an id",
    ]


async def test_a_duplicate_s_observer_reconnecting_abandons_the_first_call() -> None:
    """The duplicate serves no call's context: a reconnect its observer causes
    orphans the first call under the id, which is abandoned, nothing sent."""
    holder: dict[str, Any] = {}
    kit, _, provider, session, seen = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}],
        tool_handler=lambda *a: asyncio.sleep(0.3, "ok"),
    )

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="reconnects")
    async def reconnects(event: Any, ctx: Any) -> None:
        if event.is_error and not event.cancelled:
            await holder["provider"].simulate_tool_call_cancellation(session, ["d1"])

    holder["provider"] = provider
    await provider.simulate_tool_call(session, "d1", "t1", {})
    await asyncio.sleep(0.02)
    await provider.simulate_tool_call(session, "d1", "t1", {})
    for _ in range(100):
        if len(seen) == 2:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.4)
    await kit.close()

    assert seen == [("d1", "t1", True, False), ("d1", "t1", True, True)]
    assert provider.tool_results == []


async def test_closing_reports_the_calls_waiting_behind_the_barrier() -> None:
    """A duplicate and an id-less call still behind the transcription barrier
    when the channel closes are reported once each, cancelled."""
    kit, channel, provider, session, seen = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}], tool_handler=lambda *a: "ok"
    )
    barrier = channel._transcription_order_locks.setdefault(session.id, asyncio.Lock())
    await barrier.acquire()

    await provider.simulate_tool_call(session, "d1", "t1", {})
    await provider.simulate_tool_call(session, "d1", "t1", {})
    await provider.simulate_tool_call(session, "", "t1", {})
    await asyncio.sleep(0.05)
    await kit.close()
    await asyncio.sleep(0.05)

    assert sorted(seen) == [
        ("", "t1", True, True),
        ("d1", "t1", True, True),
        ("d1", "t1", True, True),
    ]


async def test_a_call_on_an_ended_session_leaves_no_barrier_behind() -> None:
    kit, channel, provider, session, _ = await _channel(
        tools=[{"name": "t1", "parameters": {"type": "object"}}], tool_handler=lambda *a: "ok"
    )
    await channel.end_session(session)

    await provider.simulate_tool_call(session, "c1", "t1", {})
    await provider.simulate_tool_call(session, "", "t1", {})
    await asyncio.sleep(0.1)

    assert session.id not in channel._transcription_order_locks
    await kit.close()
