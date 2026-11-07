"""An AI channel's earlier tool calls, replayed in its history (RFC §6.4).

A turn's rounds of tool calls are stored as the channel's TOOL_CALL_START and
TOOL_CALL_END rows, between the turn's input and its answer. Later turns read
them back at that place, the way the turn itself sent them: the assistant's
calls, the text the model wrote with them in the same message, then their
results. A model answers a follow-up from a result it reads there; a recall
riding the participant's message instead reads as something the participant
asked, and models answer it (RMK-1023).

A round replays what the model was given, as it was given, so that a later
turn's request extends the one before it and a provider's prompt cache keeps
it (RMK-1370): the arguments it wrote (the start row's, never a BEFORE_TOOL_USE
rewrite, which may hold the very value a hook kept from it), its intent back
among them while the agent asks for one, and the result it read, whole. Every
call the turn ended is replayed, whatever its tool or its outcome, but for a
call cancelled before it answered, whose result no model read; only an
oversized result's placeholder changes, once the room no longer stores the
result it points to. Nothing else shortens a round from turn to turn: a
compaction does, when a provider refuses the context as too long.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from roomkit.channels._tool_eviction import eviction_placeholder_size, is_eviction_placeholder
from roomkit.models.enums import EventType, ToolCallOutcome
from roomkit.models.event import RoomEvent, TextContent, ToolCallContent
from roomkit.providers.ai.base import AIMessage, AITextPart, AIToolCallPart, AIToolResultPart
from roomkit.providers.ai.tool_intent import with_intent
from roomkit.tools._open_calls import OpenCalls
from roomkit.tools._outcome import OutcomeKind

_NOT_SHOWN = "the rest is not shown here: call the tool again for it"


@dataclass
class _Call:
    index: int
    turn: str | None
    tool_id: str
    name: str
    arguments: dict[str, Any]
    intent: str | None = None
    result: str = ""
    evicted: bool = False
    is_error: bool = False
    outcome: ToolCallOutcome | None = None
    ended: bool = False


@dataclass(frozen=True)
class ToolRound:
    """One round of an earlier turn: the timeline index of its first call, the
    turn it belongs to (its rows' correlation id), and the two messages it
    reads as."""

    index: int
    turn: str | None
    calls: AIMessage
    results: AIMessage


def tool_rounds(
    rows: list[RoomEvent],
    *,
    resolves: Callable[[str], bool],
    intents: bool = False,
) -> list[ToolRound]:
    """*rows*, a channel's stored tool rows in a room oldest first, as the
    rounds its history replays: each call the turn ended, as it was sent, but
    one cancelled before it answered.

    *resolves* says whether the room still stores the result an oversized
    result's placeholder points to, and *intents* whether a call's intent goes
    back among its arguments.
    """
    rounds = [[call for call in calls if _answered(call)] for calls in _rounds(rows)]

    def result(call: _Call) -> str:
        return _shown_result(call, resolves)

    def arguments(call: _Call) -> dict[str, Any]:
        return with_intent(call.arguments, call.intent if intents else None)

    return [_round(calls, result, arguments) for calls in rounds if calls]


def _round(
    calls: list[_Call],
    result: Callable[[_Call], str],
    arguments: Callable[[_Call], dict[str, Any]],
) -> ToolRound:
    """*calls*, one round's, as the assistant's calls then their results,
    each under the id the model gave it, as the turn sent it."""
    return ToolRound(
        index=calls[0].index,
        turn=calls[0].turn,
        calls=AIMessage(
            role="assistant",
            content=[
                AIToolCallPart(id=c.tool_id, name=c.name, arguments=arguments(c)) for c in calls
            ],
        ),
        results=AIMessage(
            role="tool",
            content=[
                AIToolResultPart(
                    tool_call_id=c.tool_id,
                    name=c.name,
                    result=result(c),
                    is_error=c.is_error,
                    outcome=c.outcome,
                )
                for c in calls
            ],
        ),
    )


@dataclass(frozen=True)
class EarlierTurns:
    """What a turn's history replays of the channel's earlier turns beside
    their messages, read from its own rows: their rounds of tool calls and
    the instructions they read."""

    rounds: list[ToolRound] = field(default_factory=list)
    instructions: list[ReplayedInstruction] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayedInstruction:
    """The application's instruction an earlier turn read: the timeline index
    of the row the channel kept it in, the turn it opened (the row's
    correlation id), and the message it reads as, the one that turn sent."""

    index: int
    turn: str | None
    message: AIMessage


def kept_instructions(
    rows: list[RoomEvent], read_as: Callable[[str], str | None]
) -> list[ReplayedInstruction]:
    """The instructions *rows*, a channel's own rows oldest first, keep, each
    as the user message *read_as* makes of its text (RFC §10.1.1)."""
    kept: list[ReplayedInstruction] = []
    for row in rows:
        if row.type != EventType.INSTRUCTION_READ or not isinstance(row.content, TextContent):
            continue
        text = read_as(row.content.body)
        if text is not None:
            message = AIMessage(role="user", content=text)
            kept.append(ReplayedInstruction(row.index, row.correlation_id, message))
    return kept


def with_tool_rounds(
    messages: list[AIMessage],
    indexes: list[int | None],
    rounds: list[ToolRound],
    *,
    before: int | None,
    answered: int | None = None,
    said: Mapping[int, str] | None = None,
    instructions: list[ReplayedInstruction] | None = None,
) -> tuple[list[AIMessage], list[int | None]]:
    """*messages*, the history's turns at the timeline *indexes*, with each of
    *rounds* and *instructions* placed after the turns that came before it,
    and the indexes of the result.

    A round older than the history's first turn is not replayed: its turn's
    input is out of the window, and a history never opens on the assistant's
    calls. Nor is one at or after *before*, the turn's own event. *answered*
    is what a round must come after to read after the event a discussion's turn
    reads at its place, which may open the history or be all of it.

    *said* names the turn each of the channel's own texts in the history
    belongs to, by its index: a round right after a text of its own turn is
    the text the model wrote with its calls, and reads with them as the one
    message the turn sent, the text first.

    An instruction reads where its turn read it, before what the turn wrote:
    one after the history's first turn, or one whose turn the history holds,
    which may then open the history (an instruction that opened the room).
    """
    known = [index for index in indexes if index is not None]
    if not known and answered is None:
        return messages, indexes
    start = min([*known[:1], *([] if answered is None else [answered])])
    placed = [r for r in rounds if r.index > start and (before is None or r.index < before)]
    turns = {turn for turn in (*(said or {}).values(), *(r.turn for r in placed)) if turn}
    read = [
        kept
        for kept in instructions or []
        if (before is None or kept.index < before) and (kept.index > start or kept.turn in turns)
    ]
    if not placed and not read:
        return messages, indexes
    merged: list[tuple[int | None, AIMessage]] = list(zip(indexes, messages, strict=True))
    for kept in read:
        at = _position(merged, kept.index)
        merged[at:at] = [(kept.index, kept.message)]
    for r in placed:
        at = _position(merged, r.index)
        if at and _spoken_with(merged[at - 1], r, said or {}):
            index, text = merged[at - 1]
            merged[at - 1 : at] = [(index, _with_calls(text, r.calls)), (r.index, r.results)]
            continue
        merged[at:at] = [(r.index, r.calls), (r.index, r.results)]
    return [message for _, message in merged], [index for index, _ in merged]


def _spoken_with(
    turn: tuple[int | None, AIMessage], r: ToolRound, said: Mapping[int, str]
) -> bool:
    """Whether *turn* is the text the model wrote with *r*'s calls: the
    channel's own text of the round's turn, right before the round."""
    index, message = turn
    return (
        index is not None
        and r.turn is not None
        and said.get(index) == r.turn
        and message.role == "assistant"
        and isinstance(message.content, str)
    )


def _with_calls(text: AIMessage, calls: AIMessage) -> AIMessage:
    """The one message a round's *text* and its *calls* were sent as."""
    parts = calls.content if isinstance(calls.content, list) else []
    return calls.model_copy(update={"content": [AITextPart(text=str(text.content)), *parts]})


def _position(merged: list[tuple[int | None, AIMessage]], index: int) -> int:
    """Where a round at timeline *index* goes: after the last turn before it."""
    at = 0
    for i, (turn_index, _) in enumerate(merged):
        if turn_index is not None and turn_index < index:
            at = i + 1
    return at


def _rounds(rows: list[RoomEvent]) -> list[list[_Call]]:
    """The calls *rows* hold, round by round: a round is a run of start rows of
    one turn, each call completed by the end row of the same turn and id, the
    first of a round's calls under an id by the first such row (RFC §6.4)."""
    rounds: list[list[_Call]] = []
    waiting: OpenCalls[_Call] = OpenCalls()
    last_start: str | None = None
    opened = False
    for row in rows:
        content = row.content
        turn = row.correlation_id
        if not isinstance(content, ToolCallContent):
            opened = False
            waiting.round_over(turn)
            continue
        if row.type == EventType.TOOL_CALL_START:
            if not opened or turn != last_start:
                rounds.append([])
            call = _started(row.index, turn, content)
            rounds[-1].append(call)
            waiting.hold(turn, content.tool_id, call)
            last_start, opened = turn, True
            continue
        opened = False
        waiting.round_over(turn)
        call = waiting.take(turn, content.tool_id)
        if call is None or row.type != EventType.TOOL_CALL_END:
            continue
        call.ended = True
        call.result = _result_text(content.result)
        call.evicted = _evicted(content.result)
        call.is_error = content.status == "failed"
        call.outcome = content.outcome
    return rounds


def _started(index: int, turn: str | None, start: ToolCallContent) -> _Call:
    """The call a start row of *turn* at timeline *index* opens."""
    return _Call(index, turn, start.tool_id, start.tool_name, dict(start.arguments), start.intent)


def _answered(call: _Call) -> bool:
    """Whether *call* ended with a result the model read: not one cancelled
    before it answered, whose turn was cut or stopped first."""
    return call.ended and call.outcome != OutcomeKind.CANCELLED


def _result_text(result: Any) -> str:
    """A stored result's text: a list of parts read as its texts, a part other
    than text named."""
    if isinstance(result, list):
        texts = (_part_text(part) for part in result)
        return " ".join("[non-text part]" if text is None else text for text in texts)
    return "" if result is None else str(result)


def _evicted(result: Any) -> bool:
    """Whether a stored result is the placeholder an oversized one became,
    wherever it sits in a list of parts (an image may come first)."""
    if isinstance(result, list):
        return any(is_eviction_placeholder(_part_text(part) or "") for part in result)
    return isinstance(result, str) and is_eviction_placeholder(result)


def _part_text(part: Any) -> str | None:
    text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
    return text if isinstance(text, str) else None


def _shown_result(call: _Call, resolves: Callable[[str], bool]) -> str:
    """What a replayed call's result reads as: the result the model read, a
    placeholder whose result the room no longer stores as its size alone,
    never as an id nothing resolves."""
    text = call.result
    if call.evicted and not resolves(text):
        return f"{eviction_placeholder_size(text)}, {_NOT_SHOWN}."
    return text
