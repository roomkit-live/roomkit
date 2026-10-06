"""Speech-to-speech composition: sessions, transcripts, tools (RFC §12.10.12).

The provider hears a mix and speaks on the bot track. These tests cover the
boundary contracts around that: what a configuration refuses, when a session
is established and what its failure costs, whose words are kept, and how a
tool call is answered. The voice path — turns, barge-in, terminal chunks —
has its own file.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import (
    ConferenceRealtimeConfig,
    MockConferenceBackend,
    RoomKit,
)
from roomkit.channels._conference_tools import MAX_RESULT_CHARS
from roomkit.channels.base import Channel
from roomkit.channels.conference import ConferenceChannel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType, HookExecution, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult
from roomkit.models.tool_call import ToolCallEvent
from roomkit.tools.policy import ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider
from roomkit.voice.tts.mock import MockTTSProvider

ROOM = "room-1"


class _Source(Channel):
    """A channel that only exists to originate events of a given type."""

    def __init__(self, channel_id: str, channel_type: ChannelType) -> None:
        super().__init__(channel_id)
        self._type = channel_type

    @property
    def channel_type(self) -> ChannelType:
        return self._type

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self._type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    """Wait until a predicate holds, rather than towards when it might."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0)


async def realtime_kit(
    *,
    provider: MockRealtimeProvider | None = None,
    config: ConferenceRealtimeConfig | None = None,
    backend: MockConferenceBackend | None = None,
    source_type: ChannelType = ChannelType.AI,
    **channel_kwargs: object,
) -> tuple[RoomKit, ConferenceChannel, MockConferenceBackend, MockRealtimeProvider]:
    provider = provider or MockRealtimeProvider()
    config = config or ConferenceRealtimeConfig(provider=provider)
    backend = backend or MockConferenceBackend()
    channel = ConferenceChannel("conf", backend=backend, realtime=config, **channel_kwargs)  # type: ignore[arg-type]
    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(_Source("src", source_type))
    await kit.create_room(ROOM)
    await kit.attach_channel(ROOM, "conf")
    await kit.attach_channel(ROOM, "src")
    return kit, channel, backend, provider


class _RefusingProvider(MockRealtimeProvider):
    """Connects never: the provider is down."""

    def __init__(self) -> None:
        super().__init__()
        self.connect_attempts = 0

    async def connect(self, session, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.connect_attempts += 1
        raise RuntimeError("provider down")


class TestConfigurationRefusals:
    async def test_tts_and_realtime_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                tts=MockTTSProvider(),
                realtime=ConferenceRealtimeConfig(provider=MockRealtimeProvider()),
            )

    async def test_an_e2ee_conference_refuses_a_realtime_provider(self) -> None:
        with pytest.raises(ValueError, match="encrypted"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                realtime=ConferenceRealtimeConfig(provider=MockRealtimeProvider()),
                e2ee=True,
            )

    async def test_tools_without_a_handler_are_refused(self) -> None:
        with pytest.raises(ValueError, match="tool_handler"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                realtime=ConferenceRealtimeConfig(
                    provider=MockRealtimeProvider(),
                    tools=[{"name": "lookup"}],
                ),
            )

    async def test_two_tools_under_one_name_are_refused(self) -> None:
        """Declared once and served by the other, the model would call one
        tool's schema on the other's server (RFC §21.1)."""
        tools = [
            {"name": "lookup", "description": "server A", "parameters": {}},
            {"name": "lookup", "description": "server B", "parameters": {}},
        ]
        with pytest.raises(ValueError, match="'lookup' is given twice"):
            ConferenceChannel(
                "conf",
                backend=MockConferenceBackend(),
                realtime=ConferenceRealtimeConfig(
                    provider=MockRealtimeProvider(), tools=tools, tool_handler=AsyncMock()
                ),
            )


class TestSessionLifecycle:
    async def test_nothing_connects_before_a_need(self) -> None:
        _, channel, _, provider = await realtime_kit()

        assert channel._realtime.session_for(ROOM) is None
        assert all(call.method != "connect" for call in provider.calls)

    async def test_the_session_carries_the_configuration(self) -> None:
        provider = MockRealtimeProvider()
        _, channel, backend, _ = await realtime_kit(
            provider=provider,
            config=ConferenceRealtimeConfig(
                provider=provider,
                system_prompt="Be brief.",
                voice="verse",
                input_sample_rate=24000,
                server_vad=False,
            ),
        )

        session = await channel._realtime.ensure_session(ROOM)

        assert session is not None
        assert session.participant_id == "roomkit"
        connect = next(call for call in provider.calls if call.method == "connect")
        assert connect.args["system_prompt"] == "Be brief."
        assert connect.args["voice"] == "verse"
        assert connect.args["input_sample_rate"] == 24000
        assert connect.args["server_vad"] is False
        assert backend.bots, "the session's first need joins the bot"

    async def test_the_mixed_path_connects_and_feeds_the_provider(self) -> None:
        """End to end: a participant speaks, the lanes feed the mixer, the
        mixer establishes the session and the provider hears the window."""
        from tests.conference.lane_audio import say

        _, channel, backend, provider = await realtime_kit()
        await backend.simulate_participant_joined(ROOM, "p-alice")
        track = await backend.simulate_track_published(ROOM, "p-alice")

        await say(backend, track)
        await until(lambda: bool(provider.sent_audio))

        assert channel._realtime.session_for(ROOM) is not None
        assert sum(1 for call in provider.calls if call.method == "connect") == 1

    async def test_a_connect_failure_fails_nothing_and_cools_down(self) -> None:
        provider = _RefusingProvider()
        _, channel, backend, _ = await realtime_kit(
            provider=provider, config=ConferenceRealtimeConfig(provider=provider)
        )

        assert await channel._realtime.ensure_session(ROOM) is None
        assert await channel._realtime.ensure_session(ROOM) is None

        assert provider.connect_attempts == 1, "the cooldown holds retries off"
        assert backend.bots, "the join itself stood: only the provider failed"

    async def test_a_detach_disconnects_the_session(self) -> None:
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await kit.detach_channel(ROOM, "conf")
        from tests.conference.test_conference_races import _settle

        await _settle(channel)

        assert any(call.method == "disconnect" for call in provider.calls)
        assert channel._realtime.session_for(ROOM) is None

    async def test_a_lost_bot_session_takes_the_provider_session_with_it(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        bot = backend.bots[0]

        await backend.simulate_bot_disconnected(bot)

        assert channel._realtime.session_for(ROOM) is None
        await until(lambda: any(call.method == "disconnect" for call in provider.calls))


class TestTranscription:
    async def test_user_transcriptions_are_discarded(self) -> None:
        """The provider heard a mix; its user-side transcript names nobody and
        is not stored (RFC 12.10.12) — the lanes' STT is the attributed path."""
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "who said this?", role="user")

        events = await kit.store.list_events(ROOM)
        assert all("who said this?" not in str(event.content) for event in events)

    async def test_assistant_finals_become_room_events(self) -> None:
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "Bonjour la salle.", role="assistant")

        events = await kit.store.list_events(ROOM)
        spoken = [e for e in events if isinstance(e.content, TextContent)]
        assert [e.content.body for e in spoken] == ["Bonjour la salle."]
        (event,) = spoken
        assert event.source.channel_type is ChannelType.CONFERENCE
        assert event.source.participant_id is None
        assert event.metadata["role"] == "assistant"

    async def test_assistant_finals_are_not_respoken_or_reinjected(self) -> None:
        _, channel, backend, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "Bonjour la salle.", role="assistant")

        assert backend.published_audio == []
        assert provider.injected_texts == []

    async def test_an_answer_is_one_deeper_than_what_the_model_heard(self) -> None:
        """RFC 12.10.12: after an injected event its depth plus one, after the
        room's people spoke 1, so a chain through the provider ends at the limit."""
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await kit.send_event(ROOM, "src", TextContent(body="agent says"), chain_depth=2)
        await provider.simulate_transcription(session, "after the agent", role="assistant")
        await provider.simulate_transcription(session, "someone spoke", role="user")
        await provider.simulate_transcription(session, "after the people", role="assistant")

        depths = {
            e.content.body: e.chain_depth
            for e in await kit.store.list_events(ROOM)
            if isinstance(e.content, TextContent)
        }
        assert (depths["after the agent"], depths["after the people"]) == (3, 1)

    async def test_assistant_partials_are_not_stored(self) -> None:
        kit, channel, _, provider = await realtime_kit()
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_transcription(session, "Bonj", role="assistant", is_final=False)

        events = await kit.store.list_events(ROOM)
        assert all("Bonj" not in str(event.content) for event in events)


class TestDeliver:
    async def test_an_ai_text_event_is_injected_not_synthesized(self) -> None:
        kit, channel, backend, provider = await realtime_kit()

        await kit.send_event(ROOM, "src", TextContent(body="résume la réunion"))

        assert [(text, role) for _, text, role in provider.injected_texts] == [
            ("résume la réunion", "system")
        ]
        assert backend.published_audio == []

    async def test_non_ai_text_is_not_injected_by_default(self) -> None:
        kit, _, _, provider = await realtime_kit(source_type=ChannelType.SMS)

        await kit.send_event(ROOM, "src", TextContent(body="un SMS qui passe"))

        assert provider.injected_texts == []

    async def test_speak_text_events_injects_non_ai_text(self) -> None:
        kit, _, _, provider = await realtime_kit(
            source_type=ChannelType.SMS, speak_text_events=True
        )

        await kit.send_event(ROOM, "src", TextContent(body="un SMS qui passe"))

        assert [text for _, text, _ in provider.injected_texts] == ["un SMS qui passe"]


class TestToolCalls:
    async def test_a_tool_call_is_answered_through_the_handler(self) -> None:
        seen: list[tuple[str, str, dict[str, object]]] = []

        async def handler(room_id: str, name: str, arguments: dict) -> str:  # type: ignore[type-arg]
            seen.append((room_id, name, arguments))
            return '{"weather": "sunny"}'

        provider = MockRealtimeProvider()
        _, channel, _, _ = await realtime_kit(
            provider=provider,
            config=ConferenceRealtimeConfig(
                provider=provider,
                tools=[{"name": "get_weather"}],
                tool_handler=handler,
            ),
        )
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "get_weather", {"city": "QC"})
        await until(lambda: bool(provider.tool_results))

        assert seen == [(ROOM, "get_weather", {"city": "QC"})]
        assert provider.tool_results == [(session.id, "call-1", '{"weather": "sunny"}')]

    async def test_a_failing_handler_submits_an_error_result(self) -> None:
        async def handler(room_id: str, name: str, arguments: dict) -> str:  # type: ignore[type-arg]
            raise RuntimeError("backend down")

        provider = MockRealtimeProvider()
        _, channel, _, _ = await realtime_kit(
            provider=provider,
            config=ConferenceRealtimeConfig(
                provider=provider, tools=[{"name": "x"}], tool_handler=handler
            ),
        )
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "x", {})
        await until(lambda: bool(provider.tool_results))

        (_, call_id, result) = provider.tool_results[0]
        assert call_id == "call-1"
        # The exception is logged; the model reads that the tool failed.
        assert json.loads(result) == {"error": "Tool 'x' failed (RuntimeError)"}

    async def test_a_call_with_no_handler_configured_still_gets_an_answer(self) -> None:
        provider = MockRealtimeProvider()
        _, channel, _, _ = await realtime_kit(provider=provider)
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "surprise", {})
        await until(lambda: bool(provider.tool_results))

        (_, _, result) = provider.tool_results[0]
        assert json.loads(result) == {"error": "No handler for tool surprise"}


_LOOKUP = {
    "name": "lookup",
    "description": "Look a customer up",
    "parameters": {
        "type": "object",
        "properties": {"email": {"type": "string"}},
        "required": ["email"],
    },
}


async def _gated_kit(
    handler: Callable[..., Awaitable[str]],
) -> tuple[RoomKit, ConferenceChannel, MockRealtimeProvider, list[ToolCallEvent]]:
    """A conference whose provider declares ``lookup``, with an ASYNC observer."""
    provider = MockRealtimeProvider()
    kit, channel, _, _ = await realtime_kit(
        provider=provider,
        config=ConferenceRealtimeConfig(provider=provider, tools=[_LOOKUP], tool_handler=handler),
    )
    observed: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
        observed.append(event)

    return kit, channel, provider, observed


async def _call(
    channel: ConferenceChannel,
    provider: MockRealtimeProvider,
    observed: list[ToolCallEvent],
    name: str,
    arguments: dict[str, Any],
) -> Any:
    """Issue one call; the model's answer, once ON_TOOL_CALL's observers saw it."""
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None
    await provider.simulate_tool_call(session, "call-1", name, arguments)
    await until(lambda: bool(provider.tool_results) and bool(observed))
    return json.loads(provider.tool_results[0][2])


async def _never_run(room_id: str, tool: str, args: dict[str, Any]) -> str:
    raise AssertionError(f"the handler ran {tool}")


async def _found(room_id: str, tool: str, args: dict[str, Any]) -> str:
    return '{"ssn": "123-45-6789"}'


class TestToolCallGate:
    """A conference's tool calls pass the tool gate (RFC 12.10.12)."""

    @pytest.mark.parametrize(
        ("name", "arguments", "error"),
        [
            ("delete_everything", {}, "Tool 'delete_everything' is not declared."),
            ("lookup", {"email": 3}, "Invalid arguments for 'lookup'"),
        ],
    )
    async def test_a_refused_call_never_reaches_the_handler(
        self, name: str, arguments: dict[str, Any], error: str
    ) -> None:
        kit, channel, provider, observed = await _gated_kit(_never_run)

        result = await _call(channel, provider, observed, name, arguments)

        assert error in result["error"]
        assert [(e.name, e.is_error) for e in observed] == [(name, True)]
        await kit.close()

    async def test_a_refusal_is_reported_once_it_is_on_the_wire(self) -> None:
        kit, channel, provider, observed = await _gated_kit(_never_run)
        sent_when_observed: list[int] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="order")
        async def order(event: ToolCallEvent, ctx: RoomContext) -> None:
            sent_when_observed.append(len(provider.tool_results))

        await _call(channel, provider, observed, "delete_everything", {})

        assert sent_when_observed == [1]
        await kit.close()

    async def test_on_tool_call_rewrites_and_observers_see_the_final_result(self) -> None:
        kit, channel, provider, observed = await _gated_kit(_found)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="redact")
        async def redact(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"result": '{"ssn": "[REDACTED]"}'})

        result = await _call(channel, provider, observed, "lookup", {"email": "a@b.example"})

        assert result == {"ssn": "[REDACTED]"}
        assert [e.result for e in observed] == ['{"ssn": "[REDACTED]"}']
        await kit.close()

    async def test_a_host_tool_under_an_exempt_name_is_governed(self) -> None:
        """RMK-294: a conference serves no tool of its own, so no name escapes
        its policy (RFC §21.1)."""
        provider = MockRealtimeProvider()
        ran: list[str] = []

        async def host(room_id: str, tool: str, args: dict[str, Any]) -> str:
            ran.append(tool)
            return "host ran"

        kit, channel, _, _ = await realtime_kit(
            provider=provider,
            config=ConferenceRealtimeConfig(
                provider=provider,
                tools=[{"name": "list_tools", "description": "host tool", "parameters": {}}],
                tool_handler=host,
                tool_policy=ToolPolicy(allow=["search_*"]),
            ),
        )
        observed: list[ToolCallEvent] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
        async def audit(event: ToolCallEvent, ctx: RoomContext) -> None:
            observed.append(event)

        result = await _call(channel, provider, observed, "list_tools", {})

        assert ran == []
        assert "error" in result
        await kit.close()

    async def test_a_hook_that_clears_the_result_withholds_it(self) -> None:
        """RMK-292: the conference reads a cleared result as every channel does."""
        kit, channel, provider, observed = await _gated_kit(_found)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="clear")
        async def clear(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult(action="allow", metadata={"result": None})

        result = await _call(channel, provider, observed, "lookup", {"email": "a@b.example"})

        assert result is None  # the model read JSON null
        assert [e.result for e in observed] == ["null"]
        await kit.close()

    async def test_a_block_withholds_the_result_and_is_observed(self) -> None:
        kit, channel, provider, observed = await _gated_kit(_found)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="withhold")
        async def withhold(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("restricted")

        result = await _call(channel, provider, observed, "lookup", {"email": "a@b.example"})

        assert result == {"error": "restricted"}
        assert [e.is_error for e in observed] == [True]
        await kit.close()

    async def test_before_tool_use_denies_before_the_handler(self) -> None:
        kit, channel, provider, observed = await _gated_kit(_never_run)

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
        async def deny(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("not here")

        result = await _call(channel, provider, observed, "lookup", {"email": "a@b.example"})

        assert result["error"] == "not here"  # the BLOCK's reason (RFC §9.3)
        await kit.close()

    async def test_arguments_before_tool_use_edited_in_place_are_validated_again(self) -> None:
        kit, channel, provider, observed = await _gated_kit(_never_run)

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="edit")
        async def edit(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
            event.arguments["email"] = 3
            return HookResult.allow()

        result = await _call(channel, provider, observed, "lookup", {"email": "a@b.example"})

        assert "Invalid rewritten arguments for 'lookup'" in result["error"]
        await kit.close()

    async def test_the_result_is_bounded(self) -> None:
        async def handler(room_id: str, tool: str, args: dict[str, Any]) -> str:
            return "x" * 100_000

        kit, channel, provider, _ = await _gated_kit(handler)
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "lookup", {"email": "a@b.example"})
        await until(lambda: bool(provider.tool_results))

        result = provider.tool_results[0][2]
        assert len(result) <= MAX_RESULT_CHARS
        assert "truncated" in result
        await kit.close()

    async def test_an_abandoned_call_interrupts_its_handler(self) -> None:
        started, interrupted, released = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def handler(room_id: str, tool: str, args: dict[str, Any]) -> str:
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                interrupted.set()
                raise
            return "{}"

        kit, channel, provider, observed = await _gated_kit(handler)
        reported: list[dict[str, Any]] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="slow")
        async def slow(event: ToolCallEvent, ctx: RoomContext) -> None:
            await released.wait()

        @kit.on("tool_call")
        async def framework_event(event: Any) -> None:
            reported.append(event.data)

        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None
        await provider.simulate_tool_call(session, "call-1", "lookup", {"email": "a@b.example"})
        await asyncio.wait_for(started.wait(), 2)
        # A slow audit hook does not hold up the provider's cancellation.
        await asyncio.wait_for(provider.simulate_tool_call_cancellation(session, ["call-1"]), 1)
        await asyncio.wait_for(interrupted.wait(), 2)
        released.set()
        await until(lambda: bool(observed) and bool(reported))

        assert provider.tool_results == []
        assert [(e.cancelled, e.is_error) for e in observed] == [(True, True)]
        assert reported[0]["cancelled"] is True
        await kit.close()

    async def test_a_cancellation_after_the_outcome_adds_no_second_one(self) -> None:
        kit, channel, provider, observed = await _gated_kit(_found)
        sending, release = asyncio.Event(), asyncio.Event()
        submit = provider.submit_tool_result

        async def slow_submit(session: Any, call_id: str, result: str) -> None:
            sending.set()
            await release.wait()
            await submit(session, call_id, result)

        provider.submit_tool_result = slow_submit  # type: ignore[method-assign]
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "call-1", "lookup", {"email": "a@b.example"})
        await asyncio.wait_for(sending.wait(), 2)
        await provider.simulate_tool_call_cancellation(session, ["call-1"])
        release.set()
        await until(lambda: bool(provider.tool_results))
        await asyncio.sleep(0.05)

        assert [(e.is_error, e.cancelled) for e in observed] == [(False, False)]
        await kit.close()


class TestDisclosure:
    async def test_info_reports_the_composition(self) -> None:
        _, channel, _, _ = await realtime_kit()

        info = channel.info()

        assert info["realtime_configured"] is True
        assert info["realtime_provider"] == "MockRealtimeProvider"
        assert info["rooms"][ROOM]["realtime_active"] is False

    async def test_a_connected_session_reads_active(self) -> None:
        _, channel, _, _ = await realtime_kit()
        await channel._realtime.ensure_session(ROOM)

        assert channel.info()["rooms"][ROOM]["realtime_active"] is True
