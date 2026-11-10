"""An AI channel's continuation policy goes on an answer that did not act (RFC §6.4).

The loop consults the policy for a round the model ended itself on text,
without a call, a tool declared, whatever the provider's word for the stop;
the policy shares the empty round's bound, the loop keeps its guards, and a
turn whose policy still asks once the bound has run out ends ``unfinished``,
its marker and ON_AI_RESPONSE included. Each case runs on a provider that
streams and on one that only has ``generate()``.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import InboundMessage, RoomKit, TextContent, WebSocketChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.room import Room
from roomkit.models.steering import Cancel
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_LOOKUP = AITool(name="lookup", description="Look it up", parameters={})
_GO_ON = "You announced a check: run it now."


class _Policy:
    """Asks to go on when the round only announced a check, and records what it read."""

    def __init__(self) -> None:
        self.read: list[str] = []

    def __call__(self, text: str) -> str | None:
        self.read.append(text)
        return _GO_ON if text.startswith("I will check") else None


def _said(text: str, finish_reason: str | None = "stop", **usage: int) -> AIResponse:
    return AIResponse(content=text, finish_reason=finish_reason, usage=usage)


async def _turn(
    responses: list[AIResponse], streaming: bool, **kwargs: Any
) -> tuple[LoopRun, MockAIProvider, _Policy, list[AIResponseEvent]]:
    provider = MockAIProvider(ai_responses=responses, streaming=streaming)
    policy = _Policy()
    kwargs.setdefault("tools", [_LOOKUP])
    ch = AIChannel("ai1", provider=provider, continuation=policy, **kwargs)
    reported: list[AIResponseEvent] = []

    async def report(event: AIResponseEvent) -> None:
        reported.append(event)

    ch._after_response_hook = report
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    run = await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )
    return run, provider, policy, reported


@pytest.mark.parametrize("stop", ["stop", "end_turn", "stop_sequence", "STOP"])
async def test_an_announcement_is_continued_once_whatever_the_providers_stop(
    streaming: bool, stop: str
) -> None:
    run, provider, policy, reported = await _turn(
        [_said("I will check the run.", stop), _said("The run finished at noon.")], streaming
    )

    assert len(provider.calls) == 2
    assert policy.read == ["I will check the run.", "The run finished at noon."]
    roles = [(m.role, m.content) for m in provider.calls[1].messages[-2:]]
    assert roles == [("assistant", "I will check the run."), ("user", _GO_ON)]
    assert run.reason == "completed"
    assert [event.loop_end_reason for event in reported] == ["completed"]


async def test_a_second_announcement_ends_the_turn_unfinished(streaming: bool) -> None:
    run, provider, _policy, reported = await _turn(
        [_said("I will check the run."), _said("I will check it now.")], streaming
    )

    assert len(provider.calls) == 2
    assert run.reason == "unfinished"
    assert [event.loop_end_reason for event in reported] == ["unfinished"]


async def test_the_bound_is_the_empty_rounds(streaming: bool) -> None:
    run, provider, _policy, _reported = await _turn(
        [_said("I will check the run.")] * 3, streaming, max_empty_retries=2
    )

    assert len(provider.calls) == 3
    assert run.reason == "unfinished"


async def test_no_continuation_past_the_turns_budget(streaming: bool) -> None:
    """The budget is a limit the loop keeps: an unfinished answer past it ends
    on the budget, and nothing more is generated."""
    run, provider, _policy, _reported = await _turn(
        [_said("I will check the run.", input_tokens=10, output_tokens=5)],
        streaming,
        turn_budget_tokens=1,
    )

    assert len(provider.calls) == 1
    assert run.reason == "budget_exceeded"


@pytest.mark.parametrize("stop", ["length", "MAX_TOKENS", "content_filter", "SAFETY", None])
async def test_a_round_the_model_did_not_end_itself_is_not_the_policys(
    streaming: bool, stop: str | None
) -> None:
    _run, provider, policy, _reported = await _turn(
        [_said("I will check the run.", stop)], streaming
    )

    assert policy.read == []
    assert len(provider.calls) == 1


async def test_without_a_declared_tool_there_is_nothing_to_go_on_with(streaming: bool) -> None:
    run, provider, policy, _reported = await _turn(
        [_said("I will check the run.")], streaming, tools=[]
    )

    assert policy.read == []
    assert len(provider.calls) == 1
    assert run.reason == "completed"


async def test_no_continuation_past_the_turns_deadline(streaming: bool) -> None:
    run, provider, _policy, _reported = await _turn(
        [_said("I will check the run.")], streaming, tool_loop_timeout_seconds=1e-9
    )

    assert len(provider.calls) == 1
    assert run.reason == "timeout"


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return "ok"


async def test_an_earlier_empty_round_spends_the_bound_the_policy_shares(
    streaming: bool,
) -> None:
    """One bound for both tries: once an empty round has used it, an
    announcement ends the turn ``unfinished`` without a continuation."""
    looked_up = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
    )
    run, provider, policy, _reported = await _turn(
        [looked_up, _said(""), _said("I will check the run.")],
        streaming,
        tool_handler=_served,
        max_empty_retries=1,
    )

    assert len(provider.calls) == 3
    assert policy.read == ["I will check the run."]
    assert run.reason == "unfinished"


async def test_a_force_stopped_round_is_not_continued(streaming: bool) -> None:
    """The ripcord's demanded prose ends the turn ``force_stopped``, whatever
    the policy would say of it."""
    repeated = AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id="c0", name="lookup", arguments={"q": "same"})],
    )
    run, _provider, policy, _reported = await _turn(
        [*[repeated] * 6, _said("I will check the run.")], streaming, tool_handler=_served
    )

    assert run.reason == "force_stopped"
    assert policy.read == []


class _CancelledWhileAnswering(MockAIProvider):
    """Answers, and the turn is cancelled while it does."""

    channel: AIChannel

    def _next_response(self) -> AIResponse:
        self.channel.steer(Cancel(reason="the user stopped it"), room_id="r1")
        return super()._next_response()


async def test_a_cancelled_round_is_not_continued(streaming: bool) -> None:
    provider = _CancelledWhileAnswering(
        ai_responses=[_said("I will check the run.")], streaming=streaming
    )
    policy = _Policy()
    ch = AIChannel("ai1", provider=provider, tools=[_LOOKUP], continuation=policy)
    provider.channel = ch
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )

    run = await respond(
        ch, make_event(body="go", channel_id="sms1"), binding, RoomContext(room=Room(id="r1"))
    )

    assert run.reason == "cancelled"
    assert policy.read == []
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    "first",
    [
        _said("I will check the run."),
        # A call the provider could not parse, after words of its own: tried
        # again under the same bound, and its words stay their own segment.
        _said("Let me look.", "malformed_function_call"),
    ],
    ids=["continuation", "malformed-call"],
)
async def test_in_a_room_the_round_tried_again_keeps_its_own_message(
    streaming: bool, first: AIResponse
) -> None:
    """Through the framework: the room stores the round's text and the
    answer as two messages, the segments ON_AI_RESPONSE reports, never one
    run-on sentence (RFC §6.4)."""
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("member"))
    kit.register_channel(
        AIChannel(
            "agent",
            provider=MockAIProvider(
                ai_responses=[first, _said("The run finished at noon.")], streaming=streaming
            ),
            tools=[_LOOKUP],
            continuation=_Policy(),
        )
    )
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "member")
    await kit.attach_channel("room", "agent", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(channel_id="member", sender_id="u1", content=TextContent(body="go"))
    )

    said = [
        event.content.body
        for event in await kit.store.list_events("room")
        if event.type == EventType.MESSAGE and event.source.channel_id == "agent"
    ]
    assert said == [first.content, "The run finished at noon."]
