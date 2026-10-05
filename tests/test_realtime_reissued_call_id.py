"""An id names its realtime call until the call's result goes out (RMK-441,
RFC §12.4).

A call issued under the id after that is a new call to the channel and to the
provider alike, and gets its own answer, even while the first call is still
finishing its report. Before, the provider freed the id with the result and
the channel only once its task ended: a call re-issued in between was refused
as a duplicate by the one and booked by the other, and never answered.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from roomkit import ConferenceRealtimeConfig, HookExecution, HookTrigger, RoomKit
from roomkit.channels._realtime_tool_calls import RealtimeToolCall, ToolCallBook
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.providers.elevenlabs.config import ElevenLabsRealtimeConfig
from roomkit.providers.elevenlabs.realtime import ElevenLabsRealtimeProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.test_realtime_elevenlabs import _FakeAsyncConversation, _install_fake_sdk

TOOLS = [{"name": "lookup", "description": "look up", "parameters": {"type": "object"}}]


def _call(session: Any, call_id: str = "c1") -> RealtimeToolCall:
    return RealtimeToolCall(session, call_id, "lookup", {})


class TestTheBook:
    def test_an_id_is_refused_while_its_result_has_not_gone_out(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        first = _call(session)

        assert book.open(first)
        assert not book.open(_call(session))

    def test_an_id_is_free_once_its_result_went_out(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        first, second = _call(session), _call(session)
        book.open(first)
        first.delivered = True

        assert book.open(second)
        assert book.get("s1", "c1") is second
        assert book.holds(first) and book.holds(second)

    def test_closing_the_first_call_keeps_the_second(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        first, second = _call(session), _call(session)
        book.open(first)
        first.delivered = True
        book.open(second)

        book.close(first)

        assert not book.holds(first) and book.get("s1", "c1") is second

    def test_the_session_s_end_takes_both(self) -> None:
        book, session = ToolCallBook(), SimpleNamespace(id="s1")
        first, second = _call(session), _call(session)
        book.open(first)
        first.delivered = True
        book.open(second)

        assert book.take("s1") == [first, second]
        assert not book.busy("s1")


def _slow_reports(kit: RoomKit) -> asyncio.Event:
    """Hold every ON_TOOL_CALL report on its context: the window between a
    result going out and the call's task ending."""
    gate = asyncio.Event()
    original = kit._hook_context

    async def slow(room_id: str, trigger: Any, **kwargs: Any) -> Any:
        if trigger == HookTrigger.ON_TOOL_CALL:
            await gate.wait()
        return await original(room_id, trigger, **kwargs)

    kit._hook_context = slow  # type: ignore[method-assign]
    return gate


async def test_a_session_answers_an_id_reissued_while_its_report_runs() -> None:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=AsyncMock(return_value="found"),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    observed: list[str] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        observed.append(str(event.result))

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    gate = _slow_reports(kit)

    await provider.simulate_tool_call(session, "c1", "ghost", {})
    await until(lambda: len(provider.tool_results) == 1)
    await provider.simulate_tool_call(session, "c1", "ghost", {})
    await until(lambda: len(provider.tool_results) == 2)
    gate.set()
    await until(lambda: len(observed) == 2)
    await kit.close()

    assert [result[1] for result in provider.tool_results] == ["c1", "c1"]
    assert len(observed) == 2 and all("not declared" in body for body in observed)


async def _lookup_session(
    provider: MockRealtimeProvider, handler: Any
) -> tuple[RoomKit, RealtimeVoiceChannel, Any]:
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    return kit, channel, await channel.start_session("r1", "u", "ws")


class _ReissuingProvider(MockRealtimeProvider):
    """Frees the id with the result, then the send yields while the vendor
    issues the id again."""

    def __init__(self) -> None:
        super().__init__()
        self.reissued = False

    async def submit_tool_result(self, session: Any, call_id: str, result: str) -> None:
        await super().submit_tool_result(session, call_id, result)
        if not self.reissued:
            self.reissued = True
            await self.simulate_tool_call(session, call_id, "lookup", {})
            await asyncio.sleep(0.02)


async def test_an_id_reissued_while_the_result_is_sent_is_answered() -> None:
    """The call counts delivered before its result is sent: a call the vendor
    issues under the id during the send is a new call, answered."""
    provider = _ReissuingProvider()
    kit, _, session = await _lookup_session(provider, AsyncMock(return_value="found"))
    observed: list[str] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        observed.append(str(event.result))

    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await until(lambda: len(observed) == 2 and len(provider.tool_results) == 2)
    await kit.close()

    assert [(result[1], result[2]) for result in provider.tool_results] == [
        ("c1", "found"),
        ("c1", "found"),
    ]
    assert observed == ["found", "found"]


async def test_a_cancellation_names_the_call_reissued_under_the_id() -> None:
    """The provider cancels the id while its first call's report runs and
    the second call works: the second call is the one abandoned."""
    provider = MockRealtimeProvider()
    never = asyncio.Event()

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        await never.wait()
        return "late"

    kit, _, session = await _lookup_session(provider, handler)
    observed: list[tuple[str, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        observed.append((event.name, event.cancelled))

    gate = _slow_reports(kit)
    await provider.simulate_tool_call(session, "c1", "ghost", {})
    await until(lambda: len(provider.tool_results) == 1)
    await provider.simulate_tool_call(session, "c1", "lookup", {})
    await asyncio.sleep(0.02)
    await provider.simulate_tool_call_cancellation(session, ["c1"])
    gate.set()
    await until(lambda: len(observed) == 2)
    await kit.close()

    assert sorted(observed) == [("ghost", False), ("lookup", True)]
    assert len(provider.tool_results) == 1


async def test_a_reconnect_the_first_call_caused_abandons_the_reissued_call() -> None:
    """The first call's own follow-up orphans the id the vendor issued again:
    the call it now names is another one, abandoned as any other, not spared
    as the follow-up's own call."""
    provider = MockRealtimeProvider()
    never = asyncio.Event()

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        await never.wait()
        return "late"

    kit, _, session = await _lookup_session(provider, handler)
    observed: list[tuple[str, bool]] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        observed.append((event.name, event.cancelled))
        if event.name == "ghost":
            # In the first call's context: the vendor issues c1 again, then
            # a reconnect this call caused orphans what the provider holds.
            await provider.simulate_tool_call(session, "c1", "lookup", {})
            await asyncio.sleep(0.01)
            await provider._abandon_tool_calls(session, ["c1"])

    await provider.simulate_tool_call(session, "c1", "ghost", {})
    await until(lambda: len(observed) == 2)
    await kit.close()

    assert observed == [("ghost", False), ("lookup", True)]
    assert len(provider.tool_results) == 1


async def test_a_conference_answers_an_id_reissued_while_its_report_runs() -> None:
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(provider=provider, tools=TOOLS, tool_handler=lambda *a: "ok")
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    gate = _slow_reports(kit)

    await provider.simulate_tool_call(session, "c1", "ghost", {})
    await until(lambda: len(provider.tool_results) == 1)
    await provider.simulate_tool_call(session, "c1", "ghost", {})
    await until(lambda: len(provider.tool_results) == 2)
    gate.set()
    await kit.close()

    assert [result[1] for result in provider.tool_results] == ["c1", "c1"]


async def test_elevenlabs_answers_an_id_reissued_while_its_report_runs(
    monkeypatch: Any,
) -> None:
    _install_fake_sdk(monkeypatch)
    fake = _FakeAsyncConversation
    provider = ElevenLabsRealtimeProvider(ElevenLabsRealtimeConfig(api_key="k", agent_id="a"))
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=TOOLS,
        tool_handler=AsyncMock(return_value="found"),
        input_sample_rate=16000,
        output_sample_rate=16000,
    )
    kit = RoomKit()
    kit.register_channel(channel)

    # An observer, or no report is made and the window never opens.
    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: Any, ctx: Any) -> None:
        pass

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    gate = _slow_reports(kit)
    start = asyncio.create_task(channel.start_session("r1", "u", "ws"))
    while not fake.instances:
        await asyncio.sleep(0)
    conversation = fake.instances[-1]
    await conversation.started.wait()
    await conversation.audio_interface.start(AsyncMock())
    session = await start
    tools = conversation.client_tools

    first = asyncio.create_task(tools.handle("ghost", {"tool_call_id": "c1"}))
    await asyncio.wait([first], timeout=1)
    second = asyncio.create_task(tools.handle("ghost", {"tool_call_id": "c1"}))
    done, _ = await asyncio.wait([second], timeout=1)
    gate.set()
    await kit.close()

    # The SDK hands a refusal back as the handler's error: answered, both.
    assert first.done() and second in done
    assert not provider._has_open_tool_calls(session)
    for task in (first, second):
        assert "not declared" in json.loads(str(task.exception()))["error"]
