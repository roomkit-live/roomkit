"""The non-streaming tool loop names its exit too, and counts every round.

The streaming loop got ``LoopEndMarker`` in 0.52.0; the non-streaming loop
kept returning a bare ``AIResponse``, so a force-stopped or round-capped turn
was indistinguishable from a completed one — the exact lie the marker was
introduced to stop, alive on the other path. The reason now rides every
response MESSAGE event's metadata as ``loop_end_reason``.

Same story for usage: the streaming loop sums every round's tokens, while
this one reported only the final generation's — under-counting a multi-round
turn by every round but the last.
"""

from __future__ import annotations

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event

_ECHO_TOOL = {
    "name": "echo",
    "description": "Echo a value.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [_ECHO_TOOL]},
    )


def _tool(i: int = 0, usage: dict[str, int] | None = None) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        usage=usage or {},
        tool_calls=[AIToolCall(id=f"t{i}", name="echo", arguments={"value": str(i)})],
    )


async def _handler(name: str, arguments: dict) -> str:
    return "ok"


async def _final_message(ch: AIChannel) -> RoomEvent:
    output = await ch.on_event(
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )
    messages = [e for e in output.response_events or [] if e.type == EventType.MESSAGE]
    assert messages, "the non-streaming path always emits at least one MESSAGE"
    return messages[-1]


def _channel(responses: list[AIResponse], **kwargs: object) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=responses),
        tool_handler=_handler,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_a_plain_answer_is_marked_completed() -> None:
    message = await _final_message(_channel([AIResponse(content="hello")]))

    assert message.metadata["loop_end_reason"] == "completed"


async def test_an_answer_after_tools_is_still_completed() -> None:
    message = await _final_message(_channel([_tool(), AIResponse(content="done")]))

    assert message.metadata["loop_end_reason"] == "completed"


async def test_the_anti_loop_ripcord_is_named_not_disguised_as_an_answer() -> None:
    """Mirror of the streaming test: six identical calls trip the guard, the
    ripcord strips tools and demands prose — text that is a summary of a cut
    turn, not an answer, and now says so."""
    ch = _channel([*[_tool(0) for _ in range(6)], AIResponse(content="here is what I found")])

    message = await _final_message(ch)

    assert message.metadata["loop_end_reason"] == "force_stopped"
    assert "here is what I found" in message.content.body  # type: ignore[union-attr]


async def test_the_round_cap_is_named() -> None:
    """Exhausting the budget with calls still pending used to end the loop
    with no log and no name — the pending calls just vanished."""
    ch = _channel([_tool(0), _tool(1), _tool(2)], max_tool_rounds=1)

    message = await _final_message(ch)

    assert message.metadata["loop_end_reason"] == "max_rounds"


async def test_an_answer_landing_on_the_last_round_is_a_plain_completion() -> None:
    """Budget exhaustion is only a cut when the model still wanted tools."""
    ch = _channel([_tool(0), AIResponse(content="done", finish_reason="stop")], max_tool_rounds=1)

    message = await _final_message(ch)

    assert message.metadata["loop_end_reason"] == "completed"


async def test_usage_sums_every_round_not_just_the_last() -> None:
    ch = _channel(
        [
            _tool(0, usage={"input_tokens": 10, "output_tokens": 5}),
            _tool(1, usage={"input_tokens": 20, "output_tokens": 6}),
            AIResponse(
                content="done",
                finish_reason="stop",
                usage={"input_tokens": 40, "output_tokens": 7},
            ),
        ]
    )

    message = await _final_message(ch)

    assert message.metadata["ai_usage"] == {"input_tokens": 70, "output_tokens": 18}
