"""An AI channel's earlier tool calls ride its history as tool messages (RFC §6.4).

Each round of an earlier turn reads, at its place in the conversation, as the
assistant's calls then their results, the way the turn itself sent them: a
model answers a follow-up from the result, and never takes a recall for
something the participant asked.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import pytest

from roomkit import HookResult, ToolCallEvent
from roomkit.channels._tool_history import ToolRound, tool_rounds, with_tool_rounds
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.memory import BudgetAwareMemory, SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITool,
    AIToolCall,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel

_TOOLS = [
    AITool(name=name, description=name, parameters={"type": "object", "properties": {}})
    for name in ("lookup", "fail", "big")
]

_BOARDS = " ".join(f"Board {i} holds the cards of its team." for i in range(1, 161))


def _round(*calls: tuple[str, str, dict[str, Any]]) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=i, name=n, arguments=a) for i, n, a in calls],
    )


def _answer(text: str) -> AIResponse:
    return AIResponse(content=text, finish_reason="stop")


async def _handler(name: str, arguments: dict[str, Any]) -> str:
    if name == "fail":
        raise RuntimeError("broken")
    if name == "big":
        return _BOARDS
    return f"{name} found {arguments.get('q')}"


_Hook = Callable[[ToolCallEvent, RoomContext], Awaitable[HookResult]]


async def _turns(
    provider: MockAIProvider,
    *texts: str,
    rooms: tuple[str, ...] = ("r1",),
    hooks: list[tuple[HookTrigger, _Hook]] = (),  # type: ignore[assignment]
    instructed: frozenset[int] = frozenset(),
    **channel: Any,
) -> list[AIContext]:
    """*texts* sent one after the other into a room with an AI channel (the
    nth into the nth of *rooms*, the last one for the rest), the nth as the
    application's instruction to the agent when *instructed* holds n; what
    the provider received, round by round."""
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(
        AIChannel("ai1", provider=provider, tools=_TOOLS, tool_handler=_handler, **channel)
    )
    for n, (trigger, fn) in enumerate(hooks):
        kit.hook(trigger, name=f"hook{n}")(fn)
    for room in rooms:
        await kit.create_room(room_id=room)
        await kit.attach_channel(room, "sms1")
        await kit.attach_channel(room, "ai1", category=ChannelCategory.INTELLIGENCE)
    try:
        for n, text in enumerate(texts):
            instruction = {"event_type": EventType.INSTRUCTION, "addressed_to": ["ai1"]}
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms1",
                    sender_id="u1",
                    content=TextContent(body=text),
                    **(instruction if n in instructed else {}),  # type: ignore[arg-type]
                ),
                room_id=rooms[min(n, len(rooms) - 1)],
            )
    finally:
        await kit.close()
    return provider.calls


def _shape(messages: list[AIMessage]) -> list[tuple[str, Any]]:
    """Each message as (role, what it says): a text, or its calls / results."""
    shaped: list[tuple[str, Any]] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            shaped.append((message.role, content.split("\n\n[Notes kept")[0]))
            continue
        shaped.append(
            (
                message.role,
                [
                    (p.id, p.name, p.arguments)
                    if isinstance(p, AIToolCallPart)
                    else (p.tool_call_id, p.name, p.result, p.is_error)
                    if isinstance(p, AIToolResultPart)
                    else getattr(p, "text", type(p).__name__)
                    for p in content
                ],
            )
        )
    return shaped


async def test_an_earlier_round_reads_as_its_calls_then_their_results(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _round(("c1", "lookup", {"q": "a"}), ("c2", "fail", {})),
            _round(("c3", "lookup", {"q": "b"})),
            _answer("Answer one."),
            _answer("Answer two."),
        ],
        streaming=streaming,
    )

    calls = await _turns(provider, "first question", "second question")

    first_turn_last_round, second_turn = calls[2], calls[3]
    failed = '{"error": "Tool \'fail\' failed (RuntimeError)"}'
    assert _shape(second_turn.messages) == [
        ("user", "first question"),
        ("assistant", [("c1", "lookup", {"q": "a"}), ("c2", "fail", {})]),
        ("tool", [("c1", "lookup", "lookup found a", False), ("c2", "fail", failed, True)]),
        ("assistant", [("c3", "lookup", {"q": "b"})]),
        ("tool", [("c3", "lookup", "lookup found b", False)]),
        ("assistant", "Answer one."),
        ("user", "second question"),
    ]
    # What the turn itself sent of its rounds, word for word.
    assert _shape(second_turn.messages)[:5] == _shape(first_turn_last_round.messages)
    assert "Tools you've already used here" not in str(second_turn.messages[-1].content)


@pytest.mark.parametrize(
    ("name", "replayed"),
    [("multi_tool_use.parallel", False), ("", False), ("no_such_tool", True)],
    ids=["dotted", "empty", "well_formed"],
)
async def test_a_refused_call_is_replayed_only_under_a_name_vendors_take(
    name: str, replayed: bool, streaming: bool
) -> None:
    """A call the model invented under a name no vendor takes is refused like
    any other, but sent back it would make every later request of the room a
    400 (RFC §6.4)."""
    provider = MockAIProvider(
        ai_responses=[_round(("c1", name, {})), _answer("Answer one."), _answer("Answer two.")],
        streaming=streaming,
    )

    calls = await _turns(provider, "first question", "second question")

    sent = [
        part.name
        for message in calls[-1].messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolCallPart)
    ]
    assert sent == ([name] if replayed else [])


def _opening(previous: AIContext, later: AIContext, answer: str) -> tuple[Any, Any]:
    """What *later* opens with, and what *previous* sent then its *answer*:
    equal when *later* extends the request a provider caches a prefix of."""
    sent = [*previous.messages, AIMessage(role="assistant", content=answer)]
    return _shape(later.messages[: len(sent)]), _shape(sent)


async def test_a_later_turn_s_request_extends_the_previous_turn_s(streaming: bool) -> None:
    """Each turn's first request opens with what the previous turn's last
    request sent, word for word, then its answer: every earlier round is
    replayed as it was sent, whatever came after it (RFC §6.4)."""
    turns = 4
    responses = []
    for turn in range(turns):
        responses += [
            _round((f"c{turn}a", "big", {}), (f"c{turn}b", "lookup", {"q": f"{turn}"})),
            _round((f"c{turn}c", "lookup", {"q": f"again {turn}"})),
            _answer(f"Answer {turn}."),
        ]
    provider = MockAIProvider(ai_responses=responses, streaming=streaming)

    calls = await _turns(provider, *(f"question {turn}" for turn in range(turns)))

    for turn in range(1, turns):
        previous_last, first = calls[3 * turn - 1], calls[3 * turn]
        opening, sent = _opening(previous_last, first, f"Answer {turn - 1}.")
        assert opening == sent
        assert first.messages[: len(previous_last.messages)] == previous_last.messages


async def test_a_round_s_text_replays_with_its_calls_in_one_message(streaming: bool) -> None:
    """A step whose model wrote text with its calls reads, in a later turn,
    as the one assistant message the turn sent: its text, then its calls."""
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="Let me look it up.",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={"q": "a"})],
            ),
            _answer("Answer one."),
            _answer("Answer two."),
        ],
        streaming=streaming,
    )

    calls = await _turns(provider, "first question", "second question")

    opening, sent = _opening(calls[1], calls[2], "Answer one.")
    assert opening == sent
    assert calls[2].messages[: len(calls[1].messages)] == calls[1].messages


async def test_a_later_turn_replays_an_instruction_at_its_place(streaming: bool) -> None:
    """The application's instruction a turn took as its input reads, in a
    later turn, where that turn sent it, marked as the application's: the
    later request extends the instructed one (RFC §10.1.1)."""
    provider = MockAIProvider(
        ai_responses=[
            _answer("Answer one."),
            _round(("c1", "lookup", {"q": "a"})),
            _answer("Answer two."),
            _answer("Answer three."),
        ],
        streaming=streaming,
    )

    calls = await _turns(
        provider, "first question", "Look it up now.", "third question", instructed=frozenset({1})
    )

    opening, sent = _opening(calls[2], calls[3], "Answer two.")
    assert opening == sent
    assert calls[3].messages[: len(calls[2].messages)] == calls[2].messages


async def test_an_instruction_that_opened_the_room_replays_with_its_rounds(
    streaming: bool,
) -> None:
    """The round of a turn its instruction opened reads after it, though both
    come before the history's first message, the turn's answer (RFC §10.1.1)."""
    provider = MockAIProvider(
        ai_responses=[_round(("c1", "lookup", {"q": "a"})), _answer("One."), _answer("Two.")],
        streaming=streaming,
    )

    calls = await _turns(provider, "Look it up.", "second question", instructed=frozenset({0}))

    opening, sent = _opening(calls[1], calls[2], "One.")
    assert opening == sent


async def test_a_budget_memory_keeps_whole_turns_with_their_rounds(streaming: bool) -> None:
    """Each round costs the history with the turn it reads in: the memory
    keeps the latest turns whole, and opens on a question (RFC §20). Reserved
    up front, every round cut the messages, and the rounds of the turns cut
    went with them: two questions were left of twelve."""
    turns = 12
    responses = []
    for turn in range(turns):
        responses += [_round((f"c{turn}", "big", {})), _answer(f"Answer {turn}.")]
    provider = MockAIProvider(ai_responses=responses, streaming=streaming)
    memory = BudgetAwareMemory(SlidingWindowMemory(max_events=50), max_context_tokens=16000)

    calls = await _turns(provider, *(f"question {t}" for t in range(turns)), memory=memory)

    roles = [message.role for message in calls[-1].messages]  # the last request
    asked = roles.count("user")
    assert roles[0] == "user"
    assert asked >= 6
    assert roles.count("tool") == asked  # each turn with its round


async def test_the_history_reads_every_row_page_by_page(
    streaming: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every row of the agent's earlier turns, whatever the store's page."""
    monkeypatch.setattr("roomkit.turns._wiring.HISTORY_ROWS_PAGE", 2)
    provider = MockAIProvider(
        ai_responses=[
            _round(("c1", "lookup", {"q": "a"}), ("c2", "lookup", {"q": "b"})),
            _answer("One."),
            _answer("Two."),
        ],
        streaming=streaming,
    )

    calls = await _turns(provider, "first question", "second question")

    opening, sent = _opening(calls[1], calls[2], "One.")
    assert opening == sent


async def test_a_context_narrower_than_the_room_reads_the_instruction_before_it(
    streaming: bool,
) -> None:
    """The rows are read from the room's message before the context's first
    event: the instruction and the round of the turn whose answer opens the
    context read before it (RFC §10.1.1)."""
    provider = MockAIProvider(
        ai_responses=[
            _answer("Zero."),
            _round(("c1", "lookup", {"q": "a"})),
            _answer("One."),
            _answer("Two."),
            _answer("Three."),
        ],
        streaming=streaming,
    )

    calls = await _turns(
        provider, "q0", "Look it up.", "q2", "q3", instructed=frozenset({1}), max_context_events=4
    )

    instructed_turn = _shape(calls[2].messages)[2:]  # its instruction, its round
    assert _shape(calls[-1].messages)[:4] == [*instructed_turn, ("assistant", "One.")]


async def test_a_turn_in_another_room_replays_nothing_of_this_one(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[_round(("c1", "lookup", {"q": "a"})), _answer("One."), _answer("Two.")],
        streaming=streaming,
    )

    calls = await _turns(provider, "first question", "second question", rooms=("r1", "r2"))

    assert [m.role for m in calls[-1].messages] == ["user"]


# -- What a round replays ---------------------------------------------------------


def _row(
    kind: EventType,
    index: int,
    tool_id: str,
    name: str = "lookup",
    *,
    turn: str = "t1",
    arguments: dict[str, Any] | None = None,
    result: Any = None,
    outcome: str | None = "served",
    intent: str | None = None,
) -> RoomEvent:
    end = kind == EventType.TOOL_CALL_END
    return RoomEvent(
        room_id="r1",
        type=kind,
        index=index,
        correlation_id=turn,
        source=EventSource(channel_id="ai1", channel_type=ChannelType.AI),
        content=ToolCallContent(
            tool_name=name,
            tool_id=tool_id,
            arguments=arguments or {},
            result=result if end else None,
            status=("failed" if outcome == "failed" else "completed") if end else "pending",
            outcome=outcome if end else None,
            intent=intent,
        ),
    )


def _call_rows(index: int, tool_id: str, result: Any, **kwargs: Any) -> list[RoomEvent]:
    """A one-call round: its start at *index*, its end right after."""
    return [
        _row(EventType.TOOL_CALL_START, index, tool_id, **kwargs),
        _row(EventType.TOOL_CALL_END, index + 1, tool_id, result=result, **kwargs),
    ]


def _results(rounds: list[ToolRound]) -> list[Any]:
    return [part.result for r in rounds for part in r.results.content]  # type: ignore[union-attr]


def _rounds(
    rows: list[RoomEvent], *, resolves: bool = False, intents: bool = False
) -> list[ToolRound]:
    return tool_rounds(rows, resolves=lambda _: resolves, intents=intents)


@pytest.mark.parametrize(
    ("intents", "replayed"),
    [
        (False, {"to": "<EMAIL_1>"}),
        (True, {"to": "<EMAIL_1>", "_roomkit_intent": "I send the mail."}),
    ],
    ids=["intents off", "intents on"],
)
def test_a_round_replays_the_arguments_the_model_wrote(
    intents: bool, replayed: dict[str, Any]
) -> None:
    """The start row's, never the end's, which a BEFORE_TOOL_USE hook may have
    de-tokenised: the history goes back into the model's prompt. The call's
    intent goes back among them while the agent asks for one (RFC §6.4)."""
    rows = [
        _row(
            EventType.TOOL_CALL_START,
            1,
            "c1",
            arguments={"to": "<EMAIL_1>"},
            intent="I send the mail.",
        ),
        _row(EventType.TOOL_CALL_END, 2, "c1", arguments={"to": "a@b.c"}, result="sent"),
    ]

    [r] = _rounds(rows, intents=intents)

    assert r.calls.content == [AIToolCallPart(id="c1", name="lookup", arguments=replayed)]


def test_every_call_the_turn_ended_is_replayed_but_a_cancelled_one() -> None:
    """Whatever its outcome or tool: the model read each of them in its turn,
    a refusal and a discovery tool of the channel's own included (RMK-1370).
    Not a call cancelled before it answered, whose result no model read."""
    rows = [
        *_call_rows(1, "c1", "ok"),
        *_call_rows(3, "c2", "boom", outcome="failed"),
        *_call_rows(5, "c3", "withheld", outcome="blocked"),
        *_call_rows(7, "c4", "no", outcome="refused"),
        *_call_rows(9, "c5", "", outcome="unserved"),
        *_call_rows(11, "c6", "", outcome="cancelled"),
        *_call_rows(13, "c7", "found", name="find_tools"),
    ]

    rounds = _rounds(rows)

    results = [p for r in rounds for p in r.results.content]
    assert [p.tool_call_id for p in results] == ["c1", "c2", "c3", "c4", "c5", "c7"]  # type: ignore[union-attr]
    assert [p.is_error for p in results] == [False, True, False, False, False, False]  # type: ignore[union-attr]


def test_a_call_with_no_end_row_is_not_replayed() -> None:
    rows = [_row(EventType.TOOL_CALL_START, 1, "c1"), *_call_rows(2, "c2", "ok")]

    assert _results(_rounds(rows)) == ["ok"]


def test_calls_started_together_are_one_round_and_turns_their_own() -> None:
    """Each call under the id the model gave it, as the turn sent it: an id a
    later turn reused too (RMK-1370)."""
    rows = [
        _row(EventType.TOOL_CALL_START, 1, "c1"),
        _row(EventType.TOOL_CALL_START, 2, "c2"),
        _row(EventType.TOOL_CALL_END, 3, "c1", result="a"),
        _row(EventType.TOOL_CALL_END, 4, "c2", result="b"),
        *_call_rows(5, "c3", "c"),
        *_call_rows(7, "c1", "d", turn="t2"),
    ]

    rounds = _rounds(rows)

    assert [[p.id for p in r.calls.content] for r in rounds] == [["c1", "c2"], ["c3"], ["c1"]]  # type: ignore[union-attr]
    assert [r.index for r in rounds] == [1, 5, 7]
    assert _results(rounds) == ["a", "b", "c", "d"]


def test_a_round_s_results_pair_to_its_calls_first_in_first_out() -> None:
    rows = [
        _row(EventType.TOOL_CALL_START, 1, "c1", arguments={"q": "first"}),
        _row(EventType.TOOL_CALL_START, 2, "c1", arguments={"q": "second"}),
        _row(EventType.TOOL_CALL_END, 3, "c1", result="a"),
        _row(EventType.TOOL_CALL_END, 4, "c1", result="b"),
    ]

    rounds = _rounds(rows)

    assert [(p.id, p.arguments) for p in rounds[0].calls.content] == [  # type: ignore[union-attr]
        ("c1", {"q": "first"}),
        ("c1", {"q": "second"}),
    ]
    assert _results(rounds) == ["a", "b"]


def _withheld_end(index: int, tool_id: str) -> RoomEvent:
    """An end row as a deleted one reads: its content withheld (``withheld``)."""
    row = _row(EventType.TOOL_CALL_END, index, tool_id)
    return row.model_copy(update={"content": TextContent(body="")})


def test_a_later_round_s_result_is_not_for_a_call_left_without_its_end() -> None:
    rows = [
        _row(EventType.TOOL_CALL_START, 1, "c1", arguments={"q": "left"}),
        *_call_rows(2, "c2", "other"),
        _row(EventType.TOOL_CALL_START, 4, "c1", arguments={"q": "new"}),
        _row(EventType.TOOL_CALL_END, 5, "c1", result="its own"),
    ]

    rounds = _rounds(rows)

    assert [p.arguments for r in rounds for p in r.calls.content] == [  # type: ignore[union-attr]
        {},
        {"q": "new"},
    ]
    assert _results(rounds) == ["other", "its own"]


def test_a_withheld_end_row_leaves_its_call_without_a_later_call_s_result() -> None:
    rows = [
        _row(EventType.TOOL_CALL_START, 1, "c1", arguments={"q": "left"}),
        _withheld_end(2, "c1"),
        _row(EventType.TOOL_CALL_START, 3, "c1", arguments={"q": "new"}),
        _row(EventType.TOOL_CALL_END, 4, "c1", result="its own"),
    ]

    rounds = _rounds(rows)

    assert [p.arguments for r in rounds for p in r.calls.content] == [{"q": "new"}]  # type: ignore[union-attr]
    assert _results(rounds) == ["its own"]


def test_another_turn_among_a_round_s_starts_leaves_its_results_paired() -> None:
    rows = [
        _row(EventType.TOOL_CALL_START, 1, "c1", arguments={"q": "first"}),
        _row(EventType.TOOL_CALL_START, 2, "c9", turn="t2", arguments={"q": "theirs"}),
        _row(EventType.TOOL_CALL_START, 3, "c1", arguments={"q": "second"}),
        _row(EventType.TOOL_CALL_END, 4, "c9", turn="t2", result="zed"),
        _row(EventType.TOOL_CALL_END, 5, "c1", result="fine"),
        _row(EventType.TOOL_CALL_END, 6, "c1", result="refused", outcome="refused"),
    ]

    rounds = _rounds(rows)

    assert [p.arguments for r in rounds for p in r.calls.content] == [  # type: ignore[union-attr]
        {"q": "first"},
        {"q": "theirs"},
        {"q": "second"},
    ]
    assert _results(rounds) == ["fine", "zed", "refused"]


def test_every_result_is_replayed_whole_however_many_calls_follow() -> None:
    """No window from turn to turn: what a later turn reads of a round does
    not change as calls come after it (RMK-1370)."""
    long = "x" * 7000
    rows = [row for i in range(12) for row in _call_rows(2 * i + 1, f"c{i}", long)]

    assert _results(_rounds(rows)) == [long] * 12


_PLACEHOLDER = (
    "Result too large (2001 tokens). Full output saved as 'evicted_c1'. "
    "Use read_stored_result to read it with pagination.\n\nPreview:\nhead"
)


@pytest.mark.parametrize(
    "result",
    [_PLACEHOLDER, [{"type": "image", "url": "x"}, {"type": "text", "text": _PLACEHOLDER}]],
    ids=["text", "behind-an-image"],
)
def test_a_placeholder_the_room_no_longer_stores_offers_no_stored_id(result: Any) -> None:
    [shown] = _results(_rounds(_call_rows(1, "c1", result), resolves=False))

    assert shown == (
        "Result too large (2001 tokens), the rest is not shown here: call the tool again for it."
    )


def test_a_placeholder_the_room_still_stores_reads_as_the_turn_read_it() -> None:
    assert _results(_rounds(_call_rows(1, "c1", _PLACEHOLDER), resolves=True)) == [_PLACEHOLDER]


def test_a_result_of_parts_reads_as_its_texts() -> None:
    parts = [{"type": "text", "text": "a caption"}, {"type": "image", "url": "x"}]

    assert _results(_rounds(_call_rows(1, "c1", parts))) == ["a caption [non-text part]"]


# -- Where a round goes -------------------------------------------------------------


def _history(*turns: tuple[int, str]) -> tuple[list[AIMessage], list[int | None]]:
    roles = ["user", "assistant"]
    messages = [AIMessage(role=roles[i % 2], content=text) for i, (_, text) in enumerate(turns)]
    return messages, [index for index, _ in turns]


def test_a_round_reads_after_the_turns_before_it() -> None:
    messages, indexes = _history((5, "q1"), (9, "a1"), (10, "q2"), (14, "a2"))
    rounds = _rounds([*_call_rows(6, "c1", "r1"), *_call_rows(11, "c2", "r2")])

    placed, placed_indexes = with_tool_rounds(messages, indexes, rounds, before=15)

    assert [(m.role, str(m.content)[:2]) for m in placed] == [
        ("user", "q1"),
        ("assistant", "[A"),
        ("tool", "[A"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "[A"),
        ("tool", "[A"),
        ("assistant", "a2"),
    ]
    assert placed_indexes == [5, 6, 6, 9, 10, 11, 11, 14]


def test_a_round_outside_the_window_or_of_the_turn_itself_is_left_out() -> None:
    messages, indexes = _history((5, "q1"), (9, "a1"))
    rounds = _rounds(
        [*_call_rows(1, "c0", "old"), *_call_rows(6, "c1", "r1"), *_call_rows(12, "c2", "own")]
    )

    placed, _ = with_tool_rounds(messages, indexes, rounds, before=11)

    assert _results([r for r in rounds if r.calls in placed]) == ["r1"]


# -- Through the room -----------------------------------------------------------------


async def test_a_hooks_rewrite_is_what_is_replayed(streaming: bool) -> None:
    """What the model read in the turn: an ON_TOOL_CALL's rewrite, never the
    data the handler returned."""
    provider = MockAIProvider(
        ai_responses=[_round(("c1", "lookup", {"q": "a"})), _answer("One."), _answer("Two.")],
        streaming=streaming,
    )

    async def rewrite(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        return HookResult.modify(replace(event, result="as the hook rewrote it"))

    calls = await _turns(provider, "first", "second", hooks=[(HookTrigger.ON_TOOL_CALL, rewrite)])

    read_in_turn, replayed = (
        [p.result for m in c.messages if m.role == "tool" for p in m.content]  # type: ignore[union-attr]
        for c in (calls[1], calls[2])
    )
    assert read_in_turn == replayed == ["as the hook rewrote it"]


async def test_an_evicted_result_is_replayed_as_the_turn_read_it(streaming: bool) -> None:
    """The placeholder an oversized result became, with the re-read tool
    declared: the next turn reads it back as the turn itself could."""
    provider = MockAIProvider(
        ai_responses=[_round(("c1", "big", {})), _answer("Twenty boards."), _answer("Done.")],
        streaming=streaming,
    )

    calls = await _turns(provider, "my boards?", "and the fifteenth?", evict_threshold_tokens=1000)

    second = calls[-1]
    [replayed] = [p.result for m in second.messages if m.role == "tool" for p in m.content]  # type: ignore[union-attr]
    assert replayed.startswith("Result too large (")
    assert "read_stored_result" in {t.name for t in second.tools}
