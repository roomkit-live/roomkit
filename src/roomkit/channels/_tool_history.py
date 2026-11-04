"""An AI channel's earlier tool calls, replayed in its history (RFC §6.4).

A turn's rounds of tool calls are stored as the channel's TOOL_CALL_START and
TOOL_CALL_END rows, between the turn's input and its answer. Later turns read
them back at that place, the way the turn itself sent them: the assistant's
calls, then their results. A model answers a follow-up from a result it reads
there; a recall riding the participant's message instead reads as something
the participant asked, and models answer it (RMK-1023).

A round replays what the model was given: the arguments it wrote (the start
row's, never a BEFORE_TOOL_USE rewrite, which may hold the very value a hook
kept from it), its intent back among them while the agent asks for one, and
the result it read. An oversized result's placeholder stays
while the room still stores the result it points to, as the turn could read it
back. Which calls are replayed is the tool memory's rule: an answer the tool
gave, served, failed or withheld by a hook, never a refusal, a call nothing
served or one cancelled; never a discovery or housekeeping tool of the
channel's own. Bounded: the last calls, the most recent results whole up to a
size, the older ones as a preview.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from roomkit.channels._tool_eviction import eviction_placeholder_size, is_eviction_placeholder
from roomkit.models.enums import EventType, ToolCallOutcome
from roomkit.models.event import RoomEvent, ToolCallContent
from roomkit.providers.ai.base import AIMessage, AIToolCallPart, AIToolResultPart
from roomkit.providers.ai.tool_intent import with_intent
from roomkit.tools._outcome import kept_in_tool_memory

REPLAYED_CALLS = 8
"""The calls a turn's history replays, the room's most recent."""

WHOLE_RESULTS = 3
"""The most recent replayed calls whose result is replayed whole, up to the
channel's kept size; an older one is replayed as a preview."""

_PREVIEW_CHARS = 120
_NOT_SHOWN = "the rest is not shown here: call the tool again for it"


@dataclass
class _Call:
    round: int
    index: int
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
    """One round of an earlier turn: the timeline index of its first call, and
    the two messages it reads as."""

    index: int
    calls: AIMessage
    results: AIMessage


def tool_rounds(
    rows: list[RoomEvent],
    *,
    replayed: Callable[[str], bool],
    resolves: Callable[[str], bool],
    keep_chars: int,
    intents: bool = False,
) -> list[ToolRound]:
    """*rows*, a channel's stored tool rows in a room oldest first, as the
    rounds its history replays.

    *replayed* says whether a call to a tool is work the agent did, *resolves*
    whether the room still stores the result an oversized result's placeholder
    points to, *keep_chars* how much of a result is replayed whole, and
    *intents* whether a call's intent goes back among its arguments.
    """
    rounds = _rounds(rows)
    kept = [call for calls in rounds for call in calls if _kept(call, replayed)]
    kept = kept[-REPLAYED_CALLS:]
    chosen = {id(call) for call in kept}
    whole = {id(call) for call in kept[-WHOLE_RESULTS:]}
    ids = _unique_ids(kept)

    def result(call: _Call) -> str:
        return _shown_result(call, id(call) in whole, keep_chars, resolves)

    def arguments(call: _Call) -> dict[str, Any]:
        return with_intent(call.arguments, call.intent if intents else None)

    shown = ([call for call in calls if id(call) in chosen] for calls in rounds)
    return [_round(calls, ids, result, arguments) for calls in shown if calls]


def _round(
    calls: list[_Call],
    ids: dict[int, str],
    result: Callable[[_Call], str],
    arguments: Callable[[_Call], dict[str, Any]],
) -> ToolRound:
    """*calls*, one round's, as the assistant's calls then their results."""
    return ToolRound(
        index=calls[0].index,
        calls=AIMessage(
            role="assistant",
            content=[
                AIToolCallPart(id=ids[id(c)], name=c.name, arguments=arguments(c)) for c in calls
            ],
        ),
        results=AIMessage(
            role="tool",
            content=[
                AIToolResultPart(
                    tool_call_id=ids[id(c)],
                    name=c.name,
                    result=result(c),
                    is_error=c.is_error,
                    outcome=c.outcome,
                )
                for c in calls
            ],
        ),
    )


def with_tool_rounds(
    messages: list[AIMessage],
    indexes: list[int | None],
    rounds: list[ToolRound],
    *,
    before: int | None,
    answered: int | None = None,
) -> tuple[list[AIMessage], list[int | None]]:
    """*messages*, the history's turns at the timeline *indexes*, with each of
    *rounds* placed after the turns that came before it, and the indexes of
    the result.

    A round older than the history's first turn is not replayed: its turn's
    input is out of the window, and a history never opens on the assistant's
    calls. Nor is one at or after *before*, the turn's own event. *answered*
    is what a round must come after to read after the event a discussion's turn
    reads at its place, which may open the history or be all of it.
    """
    known = [index for index in indexes if index is not None]
    if not known and answered is None:
        return messages, indexes
    start = min([*known[:1], *([] if answered is None else [answered])])
    placed = [r for r in rounds if r.index > start and (before is None or r.index < before)]
    if not placed:
        return messages, indexes
    merged: list[tuple[int | None, AIMessage]] = list(zip(indexes, messages, strict=True))
    for r in placed:
        at = _position(merged, r.index)
        merged[at:at] = [(r.index, r.calls), (r.index, r.results)]
    return [message for _, message in merged], [index for index, _ in merged]


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
    started: dict[tuple[str | None, str], deque[_Call]] = {}
    last_start: str | None = None
    opened = False
    for row in rows:
        content = row.content
        if not isinstance(content, ToolCallContent):
            opened = False
            continue
        key = (row.correlation_id, content.tool_id)
        if row.type == EventType.TOOL_CALL_START:
            if not opened or row.correlation_id != last_start:
                rounds.append([])
            call = _started(len(rounds), row.index, content)
            rounds[-1].append(call)
            _hold(started.setdefault(key, deque()), call)
            last_start, opened = row.correlation_id, True
            continue
        opened = False
        waiting = started.get(key)
        if not waiting:
            continue
        call = waiting.popleft()
        if row.type != EventType.TOOL_CALL_END:
            continue
        call.ended = True
        call.result = _result_text(content.result)
        call.evicted = _evicted(content.result)
        call.is_error = content.status == "failed"
        call.outcome = content.outcome
    return rounds


def _hold(waiting: deque[_Call], call: _Call) -> None:
    """*call* waits for its end row behind its round's calls under its id. One
    an earlier round left without its end stays so: the row to come is not its."""
    if waiting and waiting[-1].round != call.round:
        waiting.clear()
    waiting.append(call)


def _started(round_number: int, index: int, start: ToolCallContent) -> _Call:
    """The call a start row at timeline *index* opens in round *round_number*."""
    return _Call(
        round_number,
        index,
        start.tool_id,
        start.tool_name,
        dict(start.arguments),
        start.intent,
    )


def _kept(call: _Call, replayed: Callable[[str], bool]) -> bool:
    return call.ended and kept_in_tool_memory(call.outcome) and replayed(call.name)


def _unique_ids(calls: list[_Call]) -> dict[int, str]:
    """Each call's id, one per call: a provider may reuse an id from one turn
    to the next, and a chat API pairs a result with its call by id."""
    ids: dict[int, str] = {}
    seen: set[str] = set()
    for call in calls:
        tool_id = call.tool_id
        if not tool_id or tool_id in seen:
            tool_id = f"{tool_id or 'call'}_{call.index}"
        seen.add(tool_id)
        ids[id(call)] = tool_id
    return ids


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


def _shown_result(
    call: _Call, whole: bool, keep_chars: int, resolves: Callable[[str], bool]
) -> str:
    """What a replayed call's result reads as: whole up to *keep_chars* for
    one of the most recent, a preview for an older one; a placeholder whose
    result the room no longer stores reads as its size alone, never as an id
    nothing resolves."""
    text = call.result
    if call.evicted:
        return text if resolves(text) else f"{eviction_placeholder_size(text)}, {_NOT_SHOWN}."
    if not whole:
        flat = " ".join(text.split())
        if len(flat) <= _PREVIEW_CHARS:
            return flat
        return f"{flat[:_PREVIEW_CHARS]}… [{_NOT_SHOWN}]"
    if len(text) <= keep_chars:
        return text
    return f"{text[:keep_chars]}\n[first {keep_chars} of {len(text)} characters; {_NOT_SHOWN}]"
