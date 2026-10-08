"""A ``find_tools`` reveal counts once its call is served, on every door
(RMK-447, RFC §6.4, §9.3).

An ON_TOOL_CALL hook judges a Tool Search call before the model reads its
result, as any other call: a block reveals nothing (not this turn, not the
next), a replacement is what the model reads, and a search that finds nothing
keeps the reveal window as it was. A realtime session declares the matches
only after the call's result went out.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit, ToolCallEvent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelType
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockCall, MockRealtimeProvider, MockRealtimeTransport
from tests.conftest import make_event
from tests.test_toolset_edges import (
    SPOTIFY,
    _call,
    _calling,
    _declared,
    _FixedProvider,
    _Recorder,
    _schema,
    _session,
    _tool,
)
from tests.tool_loop_modes import respond

FOUND = {"spotify_play", "spotify_search"}


def _judge_search(kit: RoomKit, verdict: str) -> list[ToolCallEvent]:
    """Block or replace every Tool Search call; the observers' reports."""

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="judge")
    async def judge(event: ToolCallEvent, ctx: Any) -> HookResult:
        if event.name not in ("find_tools", "list_tools"):
            return HookResult.allow()
        if verdict == "block":
            return HookResult.block("search is not allowed here")
        return HookResult.modify(replace(event, result='{"matches": []}'))

    seen: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.append(event)

    return seen


async def _text_kit(responses: list[AIResponse]) -> tuple[RoomKit, AIChannel, MockAIProvider]:
    provider = MockAIProvider(ai_responses=responses)
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[_tool(n) for n in SPOTIFY],
        tool_handler=_Recorder(),
        tool_search=True,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")
    return kit, channel, provider


async def _text_turn(kit: RoomKit, channel: AIChannel) -> None:
    binding = ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI)
    event = make_event(room_id="r1", body="go", channel_id="sms1")
    await respond(channel, event, binding, await kit._build_context("r1"))


def _round_declares(provider: MockAIProvider, index: int) -> set[str]:
    return {t.name for t in provider.calls[index].tools or [] if not t.defer_loading}


class TestText:
    async def test_a_blocked_search_reveals_nothing_this_turn_or_the_next(self) -> None:
        kit, channel, provider = await _text_kit(
            [
                _calling("find_tools", query="spotify"),
                AIResponse(content="done"),
                AIResponse(content="next turn"),
            ]
        )
        _judge_search(kit, "block")

        await _text_turn(kit, channel)
        await _text_turn(kit, channel)

        [answer] = [
            json.loads(str(part.result))
            for message in provider.calls[1].messages
            if message.role == "tool"
            for part in message.content
        ]
        assert answer == {"error": "search is not allowed here"}
        assert not FOUND & _round_declares(provider, 1)
        assert not FOUND & _round_declares(provider, 2)
        assert not FOUND & channel._tool_usage.tool_names("r1")
        await kit.close()

    async def test_a_served_search_reveals_for_the_next_round_and_turn(self) -> None:
        kit, channel, provider = await _text_kit(
            [
                _calling("find_tools", query="spotify"),
                AIResponse(content="done"),
                AIResponse(content="next turn"),
            ]
        )

        await _text_turn(kit, channel)
        await _text_turn(kit, channel)

        assert _round_declares(provider, 1) >= FOUND
        assert _round_declares(provider, 2) >= FOUND
        await kit.close()

    async def test_a_replaced_search_still_reveals_its_matches(self) -> None:
        """Served is served: a hook's replacement is what the model reads, and
        the matches are revealed, as a served activation opens its gates."""
        kit, channel, provider = await _text_kit(
            [_calling("find_tools", query="spotify"), AIResponse(content="done")]
        )
        _judge_search(kit, "replace")

        await _text_turn(kit, channel)

        [answer] = [
            str(part.result)
            for message in provider.calls[1].messages
            if message.role == "tool"
            for part in message.content
        ]
        assert answer == '{"matches": []}'
        assert _round_declares(provider, 1) >= FOUND
        await kit.close()

    async def test_a_search_that_finds_nothing_keeps_the_window(self) -> None:
        kit, channel, provider = await _text_kit(
            [
                _calling("find_tools", query="spotify"),
                _calling("find_tools", query="zzzz-nothing-matches"),
                AIResponse(content="done"),
            ]
        )

        await _text_turn(kit, channel)

        assert _round_declares(provider, 2) >= FOUND
        await kit.close()


async def _realtime_kit(
    provider: MockRealtimeProvider,
) -> tuple[RoomKit, RealtimeVoiceChannel, Any]:
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema(n) for n in SPOTIFY],
        tool_handler=_Recorder(),
        tool_search=True,
    )
    kit, session = await _session(channel)
    return kit, channel, session


class TestRealtime:
    async def test_a_blocked_search_reconfigures_nothing(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        seen = _judge_search(kit, "block")
        before = len(provider.calls)

        result = await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        assert json.loads(result) == {"error": "search is not allowed here"}
        assert [c.method for c in provider.calls[before:]] == ["submit_tool_result"]
        assert not channel._tool_search_support._exposed.get(session.id)
        assert [(e.name, e.is_error) for e in seen] == [("find_tools", True)]
        await kit.close()

    async def test_a_served_search_is_declared_after_its_result_went_out(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        before = len(provider.calls)

        await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        # The mock reconfigures by reconnecting, after the result went out.
        methods = [c.method for c in provider.calls[before:]]
        assert methods == ["submit_tool_result", "disconnect", "connect"]
        assert {t.get("name") for t in _declared(provider)} >= FOUND
        await kit.close()

    async def test_a_replaced_search_result_is_what_the_model_reads(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        seen = _judge_search(kit, "replace")

        result = await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        assert result == '{"matches": []}'
        assert [(e.name, e.is_error, e.result) for e in seen] == [
            ("find_tools", False, '{"matches": []}')
        ]
        # The call was served: its matches are revealed whatever the hook
        # replaced its result with, as a served activation opens its gates.
        assert {t.get("name") for t in _declared(provider)} >= FOUND
        await kit.close()

    async def test_a_search_that_finds_nothing_keeps_what_was_revealed(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        await _call(channel, provider, session, "find_tools", {"query": "spotify"})
        before = len(provider.calls)

        await _call(channel, provider, session, "find_tools", {"query": "zzzz-nothing-matches"})

        assert [c.method for c in provider.calls[before:]] == ["submit_tool_result"]
        assert channel._tool_search_support._exposed[session.id] >= FOUND
        await kit.close()

    async def test_a_blocked_list_tools_reads_the_block(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        _judge_search(kit, "block")

        result = await _call(channel, provider, session, "list_tools")

        assert json.loads(result) == {"error": "search is not allowed here"}
        await kit.close()

    async def test_a_fixed_declaration_provider_is_never_reconfigured(self) -> None:
        provider = _FixedProvider()
        kit, channel, session = await _realtime_kit(provider)
        before = len(provider.calls)

        result = await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        assert {m["name"] for m in json.loads(result)["matches"]} >= FOUND
        assert [c.method for c in provider.calls[before:]] == ["submit_tool_result"]
        await kit.close()


class TestRealtimeSearchesOfOneResponse:
    """Each search's result tells the model its matches are declared: two
    searches of one model response reveal both (RMK-606)."""

    async def test_two_searches_of_one_response_reveal_both_their_matches(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        await provider.simulate_response_start(session)

        await _call(channel, provider, session, "find_tools", {"query": "spotify"})
        await _call(channel, provider, session, "find_tools", {"query": "x5"})

        assert {t.get("name") for t in _declared(provider)} >= FOUND | {"x5"}
        await kit.close()

    async def test_a_later_responses_search_still_swaps_the_window(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)
        await provider.simulate_response_start(session)
        await _call(channel, provider, session, "find_tools", {"query": "spotify"})
        await provider.simulate_response_end(session)
        await provider.simulate_response_start(session)

        await _call(channel, provider, session, "find_tools", {"query": "x5"})

        declared = {t.get("name") for t in _declared(provider)}
        assert "x5" in declared and not FOUND & declared
        await kit.close()


class _InBand(MockRealtimeProvider):
    """Reconfigures in band (OpenAI Realtime's ``session.update``): no
    reconnect, the call id in flight stays the session's."""

    async def reconfigure(self, session: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.calls.append(MockCall(method="reconfigure", args=kwargs))


AFTER_HANDOFF = ("spotify_play", *(f"y{i}" for i in range(30)))


class TestRealtimeReview:
    async def test_a_handoff_during_the_judgement_drops_the_reveal(self) -> None:
        """The search matched names in the catalogue the handoff replaced: the
        new catalogue's reveal window stays as the handoff left it."""
        provider = _InBand()
        kit, channel, session = await _realtime_kit(provider)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="handoff")
        async def handoff(event: ToolCallEvent, ctx: Any) -> HookResult:
            if event.name == "find_tools":
                tools = [_schema(n) for n in AFTER_HANDOFF]
                await channel.reconfigure_session(session, tools=tools)
            return HookResult.allow()

        await _call(channel, provider, session, "find_tools", {"query": "spotify"})

        assert not channel._tool_search_support._exposed.get(session.id)
        assert not FOUND & {t.get("name") for t in _declared(provider)}
        await kit.close()

    async def test_a_call_its_hook_s_reconnect_orphaned_reveals_nothing(self) -> None:
        provider = MockRealtimeProvider()
        kit, channel, session = await _realtime_kit(provider)

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="handoff")
        async def handoff(event: ToolCallEvent, ctx: Any) -> HookResult:
            if event.name == "find_tools":
                await channel.reconfigure_session(session, system_prompt="New agent")
                await provider.simulate_tool_call_cancellation(session, [event.tool_call_id])
            return HookResult.allow()

        await provider.simulate_tool_call(session, "c1", "find_tools", {"query": "spotify"})
        for _ in range(50):
            await asyncio.gather(*list(channel._scheduled_tasks), return_exceptions=True)
            await asyncio.sleep(0.01)

        assert provider.tool_results == []
        assert not FOUND & channel._tool_search_support._exposed.get(session.id, set())
        await kit.close()

    async def test_the_hooks_judge_the_whole_result_the_model_reads_bounded(self) -> None:
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "rt",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[_schema(n) for n in SPOTIFY],
            tool_handler=_Recorder(),
            tool_search=True,
            tool_result_max_length=200,
        )
        kit, session = await _session(channel)
        judged: list[int] = []

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="size")
        async def size(event: ToolCallEvent, ctx: Any) -> HookResult:
            judged.append(len(str(event.result)))
            return HookResult.allow()

        read = await _call(channel, provider, session, "list_tools")

        assert judged[0] > 200 and len(read) < judged[0]
        await kit.close()


BIG_TOOL = {"name": "big_tool", "description": "y" * 3000, "parameters": {"type": "object"}}


async def _fixed_list_tools(verdict: str, arguments: dict[str, Any]) -> str:
    provider = _FixedProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[BIG_TOOL, *(_schema(n) for n in SPOTIFY)],
        tool_handler=_Recorder(),
        tool_search=True,
        tool_result_max_length=500,
    )
    kit, session = await _session(channel)
    big = "x" * 5000
    trigger = HookTrigger.BEFORE_TOOL_USE if verdict == "refuse" else HookTrigger.ON_TOOL_CALL

    @kit.hook(trigger, execution=HookExecution.SYNC, name="judge")
    async def judge(event: ToolCallEvent, ctx: Any) -> HookResult:
        if event.name != "list_tools":
            return HookResult.allow()
        if verdict == "replace":
            return HookResult.modify(replace(event, result=json.dumps({"note": big})))
        if verdict in ("block", "refuse"):
            return HookResult.block(big)
        return HookResult.allow()

    read = await _call(channel, provider, session, "list_tools", arguments)
    await kit.close()
    return read


@pytest.mark.parametrize("verdict", ["replace", "block", "refuse"])
async def test_only_the_schema_list_tools_serves_goes_out_uncut(verdict: str) -> None:
    """RFC §21.5: a hook's replacement, a BLOCK or a gate refusal in place of
    ``list_tools(name=...)`` is bounded; the served schema is not."""
    read = await _fixed_list_tools(verdict, {"name": "big_tool"})
    whole = await _fixed_list_tools("allow", {"name": "big_tool"})

    assert len(read) < 1000
    assert json.loads(whole)["tool"]["description"] == "y" * 3000


def _calling_side_by_side(*queries: str) -> AIResponse:
    """One round in which the model ran a search per query, side by side."""
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            AIToolCall(id=f"call-find-{i}", name="find_tools", arguments={"query": query})
            for i, query in enumerate(queries)
        ],
    )


class TestSearchesOfOneRound:
    """Each search's result tells the model its matches are declared next
    round: two searches run side by side in one round reveal both."""

    async def test_two_searches_of_one_round_reveal_both_their_matches(self) -> None:
        kit, channel, provider = await _text_kit(
            [_calling_side_by_side("spotify", "x5"), AIResponse(content="done")]
        )

        await _text_turn(kit, channel)

        assert _round_declares(provider, 1) >= FOUND | {"x5"}
        await kit.close()

    async def test_a_later_rounds_search_still_swaps_the_window(self) -> None:
        kit, channel, provider = await _text_kit(
            [
                _calling("find_tools", query="spotify"),
                _calling("find_tools", query="x5"),
                AIResponse(content="done"),
            ]
        )

        await _text_turn(kit, channel)

        declared = _round_declares(provider, 2)
        assert "x5" in declared and not FOUND & declared
        await kit.close()
