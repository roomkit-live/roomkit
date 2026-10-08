"""Execute one delegated turn in a child room and capture its result.

The single code path for running a delegated agent — used by both
``delegate(wait=True)`` (inline) and the background task runner via
:meth:`DelegationMixin`. Persists the worker's full trace (tool calls +
messages) into the child room and returns its output, either as free text or
as the structured payload of the result tool the delegation forces.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.core._failure_log import mark_reported
from roomkit.core._requester import with_requester
from roomkit.core.event_router import responder_turn_entries, stream_record
from roomkit.core.exceptions import TaskCutShortError, TaskTurnFailedError, TurnCutShortError
from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins._response_reader import ResponseReader
from roomkit.core.mixins._result_capture import capture_result
from roomkit.core.mixins._streaming_segments import LaneSink, RowSink, SegmentWriter, TurnScope
from roomkit.core.mixins.lane_execution import DeliverySource
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    Visibility,
)
from roomkit.models.event import (
    EventSource,
    RoomEvent,
    TextContent,
    answer_text,
)
from roomkit.models.response_metadata import TurnEntries, recorded_turn_end
from roomkit.models.store_filter import EventFilter
from roomkit.providers.utils import _aclose_stream

if TYPE_CHECKING:
    from roomkit.channels._tool_registry import ChannelRegistry
    from roomkit.core.event_router import BroadcastResult, StreamingResponse
    from roomkit.core.framework import RoomKit
    from roomkit.models.room import Room
    from roomkit.orchestration.result import ResultTool


_tasks_logger = logging.getLogger("roomkit.tasks")


class _TraceSink:
    """A delegated turn's trace: each row committed as is, with no hook and no lane.

    Nobody is delivered these rows; they are the record of a delegated turn
    (RFC §23.3), so they cross no ``BEFORE_BROADCAST`` hook and ride no
    delivery lane.
    """

    def __init__(self, kit: RoomKit, room_id: str) -> None:
        self._kit = kit
        self._room_id = room_id

    async def commit(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        return await self._kit._commit_indexed(self._room_id, event)


class _ToolRowsOnly:
    """A turn whose text answers no one in its room (a supervisor's first
    pass, whose text is the task it hands on, RFC §19.7.3): its tool calls go
    through *rows* as any turn's, its text is read and kept out."""

    def __init__(self, rows: RowSink) -> None:
        self._rows = rows

    async def commit(self, event: RoomEvent, *, exclude: set[str] | None) -> RoomEvent | None:
        if event.type in (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END):
            return await self._rows.commit(event, exclude=exclude)
        return event


class PersistedTurn(NamedTuple):
    """A turn whose text was kept out of the room: its answer, how it ended,
    and its record (its response metadata with how its loop ended)."""

    answer: str
    end: str | None
    record: dict[str, Any]


async def persist_tool_calls(
    kit: RoomKit, room_id: str, sr: StreamingResponse, context: RoomContext, correlation_id: str
) -> PersistedTurn:
    """Store a turn's tool calls in *room_id* as any streamed turn's, its text
    kept out of the room, under *correlation_id*; the turn's answer, how it
    ended, and its record.

    The rows cross the room's gate and ride its lane (``BEFORE_BROADCAST``,
    the source's right to write, the delivery's visibility), one deeper than
    the event the turn answers and in its thread. The answer is the last
    segment of a turn that completed: a turn its round cap, deadline or
    budget cut short has none, its narration is no answer (RFC §6.4).
    """
    cascade = DeliveryCascade(room_id, reentry_budget=kit._max_chain_depth * 10)
    lane = LaneSink(
        kit,
        room_id=room_id,
        context=context,
        cascade=cascade,
        plan_source=DeliverySource.of(sr.source_channel_id, context),
    )
    scope = TurnScope.answering(sr.trigger_event)
    writer = SegmentWriter(
        kit,
        sr,
        _ToolRowsOnly(lane),
        room_id=room_id,
        chain_depth=scope.chain_depth,
        visibility=scope.visibility,
        response_visibility=scope.response_visibility,
        correlation_id=correlation_id,
        parent_event_id=scope.parent_event_id,
    )
    try:
        answer = await _drain_turn(writer, sr)
    finally:
        await kit._finish_cascade(cascade, room_id, caller_logs=True)
    record = dict(stream_record(sr))
    return _handed_on(answer, _turn_end(record, writer.persisted), record, sr, room_id)


def _handed_on(
    answer: str, reason: str | None, record: dict[str, Any], sr: StreamingResponse, room_id: str
) -> PersistedTurn:
    """The turn as handed on: its answer only when it completed (RFC §6.4)."""
    if reason in (None, "completed"):
        return PersistedTurn(answer, reason, record)
    # A stop someone chose (a steering Cancel) is no failure to log (RFC §19.7.3).
    if reason != "cancelled":
        _tasks_logger.warning(
            "Turn of %s in room %s ended %s: no answer to hand on",
            sr.source_channel_id,
            room_id,
            reason,
        )
    return PersistedTurn("", reason, record)


async def _persist_child_stream(
    kit: RoomKit,
    child_room_id: str,
    sr: Any,
    chain_depth: int,
) -> str:
    """Write a delegated turn's stream into its child room; the worker's answer.

    The rows are the ones a room's streamed turn leaves, by the same writer
    (RFC §23.3): text segments split at tool-call boundaries, each call's
    TOOL_CALL_{START,END} with its structured copy, and the turn's record
    (``loop_end_reason``, ``ai_usage``) on its last message (RFC §6.4). Only
    the commit differs, through :class:`_TraceSink`. A delegation cancelled or
    failed mid-turn closes its open calls and keeps its text (RFC §12.2 step
    13s), then propagates. Returns the last segment's text: the worker's
    answer, as a non-streaming worker's last message is.
    """
    correlation_id = uuid4().hex
    writer = SegmentWriter(
        kit,
        sr,
        _TraceSink(kit, child_room_id),
        room_id=child_room_id,
        chain_depth=chain_depth,
        correlation_id=correlation_id,
    )
    try:
        answer = await _drain_turn(writer, sr)
    except Exception as exc:
        await _report_turn_failure(kit, child_room_id, sr, exc, correlation_id)
        end = _turn_end(stream_record(sr), writer.persisted)
        raise mark_reported(_turn_failure(exc, _last_text(writer.persisted), end))  # noqa: B904
    if (cut := _cut_short(answer, _turn_end(stream_record(sr), writer.persisted))) is not None:
        raise cut
    return answer


async def _report_turn_failure(
    kit: RoomKit, room_id: str, sr: Any, exc: Exception, correlation_id: str
) -> None:
    """Fire ON_ERROR, once, for a delegated turn whose stream failed on the
    trace path, by the hook a room's reader fires for a turn of its own, with
    the turn's scope (RFC §23.3 step 6)."""
    context = await kit._hook_context(room_id, HookTrigger.ON_ERROR)
    if context is None:
        return
    await kit._fire_stream_error_hook(exc, room_id, context, sr, correlation_id)


async def _drain_turn(writer: SegmentWriter, sr: Any) -> str:
    """Read a streamed turn through *writer* to its end; its last segment's text.

    A failed turn still records its end, as a room's does (RFC §6.4), then
    raises.
    """
    failure: Exception | None = None
    try:
        await writer.drain(ResponseReader(sr.stream))
    except Exception as exc:
        failure = exc
    finally:
        # Nothing else reads this response: closing it here ends a cut-short
        # generation (RFC §12.2 step 13s).
        await _aclose_stream(sr.stream)
    await writer.record_on_last_message()
    if failure is not None:
        raise failure
    return _last_text(writer.persisted) or ""


def _last_text(rows: list[RoomEvent]) -> str | None:
    """The text of the last row that answers, if any."""
    return next((text for row in reversed(rows) if (text := answer_text(row)) is not None), None)


async def _persist_response_events(
    kit: RoomKit, child_room_id: str, response_events: list[RoomEvent]
) -> str | None:
    """Persist a non-streaming response's events (tool calls + messages) and
    return the last message text, so the child room keeps the full trace."""
    final_text: str | None = None
    for resp in response_events:
        await kit._commit_indexed(
            child_room_id, resp.model_copy(update={"status": EventStatus.DELIVERED})
        )
        # An interruption marker is not the worker's answer (RFC §6.4)
        if (text := answer_text(resp)) is not None:
            final_text = text
    return final_text


async def _child_context(kit: RoomKit, room: Room, bindings: list[ChannelBinding]) -> RoomContext:
    """The context a delegated turn reads, built after its message is stored.

    Built AFTER storing so the agent's memory provider can see the message in
    recent_events — which is the message just committed, so this read must be
    the room's tail (``newest_first``), not its head. A delegated room that
    outlives 50 events would otherwise hand the agent the opening turns and
    never the new one. Read whole, as ``_build_context`` reads it (RFC §7.5
    rule 8): the per-reader filter drops the refused rows for the channels
    that read it.
    """
    recent = await kit.store.list_events(
        room.id,
        offset=0,
        limit=50,
        newest_first=True,
        event_filter=EventFilter(include_blocked=True),
    )
    return RoomContext(room=room, bindings=bindings, recent_events=recent)


async def _broadcast_and_collect(
    kit: RoomKit, child_room_id: str, message_body: str, *, turns: TurnEntries | None = None
) -> str | None:
    """One delegated turn: store *message_body* as a system message, broadcast it,
    persist the agent's full trace (tool calls + messages), and return its text.
    *turns* holds each responder's entry for this turn, however it ended, in
    place of the turn before (RFC §23.3 step 6: a re-prompted agent's record
    is its last turn's)."""
    if turns is not None:
        turns.clear()
    room = await kit.get_room(child_room_id)
    bindings = await kit.store.list_bindings(child_room_id)

    # The task description, and a result tool's re-prompt, are the delegating
    # side's instruction to the worker: they reach the room's agents and never
    # a transport shared into it (RFC §23.3 step 4).
    msg_event = RoomEvent(
        room_id=child_room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body=message_body),
        status=EventStatus.DELIVERED,
        visibility=Visibility.INTELLIGENCE,
    )
    msg_event = await kit._commit_indexed(child_room_id, msg_event)
    context = await _child_context(kit, room, bindings)

    router = kit._get_router()
    source_binding = ChannelBinding(
        channel_id="system",
        room_id=child_room_id,
        channel_type=ChannelType.SYSTEM,
    )
    result = await router.broadcast(msg_event, source_binding, context)
    # What the broadcast blocked is stored and announced, and its side effects
    # kept, whichever path broadcast the trigger (RFC §8.3).
    await kit._commit_blocked_events(child_room_id, result)
    await kit._persist_side_effects(
        child_room_id, result.tasks, result.observations, msg_event, context
    )
    # A responder that failed answering (raised, or returned its error) fires
    # ON_ERROR once, by the reporter a room's turn uses, whichever path the
    # answer then takes (RFC §23.3 step 6); a stream that fails later fires
    # it where it is read.
    await kit._report_intelligence_errors(msg_event, context, result)
    return await _answer_of_broadcast(
        kit, child_room_id, result, bindings, msg_event.chain_depth + 1, turns
    )


async def _answer_of_broadcast(
    kit: RoomKit,
    child_room_id: str,
    result: BroadcastResult,
    bindings: list[ChannelBinding],
    child_depth: int,
    turns: TurnEntries | None,
) -> str | None:
    """A delegated broadcast's answer, by the room's path when a transport
    is shared, else the trace's; each responder's turn entry goes to *turns*
    however the turn ended (RFC §23.3 steps 5 and 6)."""
    try:
        if _shares_a_transport(bindings):
            return await _deliver_answer(kit, child_room_id, result)
        return await _collect_answer(kit, child_room_id, result, child_depth)
    finally:
        if turns is not None:
            turns.update(responder_turn_entries(result))


def _shares_a_transport(bindings: list[ChannelBinding]) -> bool:
    """Whether the delegated room holds a transport, which only sharing puts there."""
    return any(binding.category == ChannelCategory.TRANSPORT for binding in bindings)


async def _deliver_answer(kit: RoomKit, child_room_id: str, result: BroadcastResult) -> str | None:
    """Commit a delegated turn as a room commits its answers; the answer it kept.

    A transport shared into the child room is told what the agent answers
    (RFC §23.3 step 5), so the answer takes a room's path rather than the
    trace's: each buffered response re-enters, and each stream is read by the
    room's reader, every row crossing ``BEFORE_BROADCAST`` and the agent's
    right to write and riding the child room's delivery lane. The shared
    binding's access and the agent binding's visibility decide what the
    transport receives, as in any room. The answer is the last one the room
    kept (step 6): a hook's rewrite holds for the task result too, and an
    answer the gate refused is none. A response that failed fails the turn
    once every response is read.
    """
    cascade = DeliveryCascade(child_room_id, reentry_budget=kit._max_chain_depth * 10)
    cascade.add_streams(result.streaming_responses)
    await kit._commit_responses(child_room_id, result.reentry_events, None, cascade)
    # The failure is raised to the delegation, which logs it, a stream a
    # shared transport rendered included.
    stream_error, _ = await kit._finish_cascade(
        cascade, child_room_id, caller_logs=True, streamed_too=True
    )
    failure, failed_by = _responder_failure(result, stream_error)
    if failure is not None:
        # The failure carries the end of the responder that failed, never
        # another's (RFC §23.3 step 6).
        text, reason = _answer_of(result, cascade.response_events, failed_by)
        # Reported in the child room, by its reader or the delivery set's.
        raise mark_reported(_turn_failure(failure, text, reason))
    text, reason = _kept_answer(result, cascade.response_events)
    if (cut := _cut_short(text, reason)) is not None:
        raise cut
    return text


def _responder_failure(
    result: BroadcastResult, stream_error: Exception | None
) -> tuple[Exception | None, str | None]:
    """A delegated broadcast's failure, and the responder that failed when it
    can be told: a buffered response's, or the only responder's stream."""
    for cid, out in result.outputs.items():
        if out.error is not None:
            return out.error, cid
    if stream_error is None:
        return None, None
    responders = _responder_records(result)
    return stream_error, (next(iter(responders)) if len(responders) == 1 else None)


def _responder_records(result: BroadcastResult) -> dict[str, Mapping[str, Any]]:
    """Each responder's turn record, by its channel id."""
    records: dict[str, Mapping[str, Any]] = {
        cid: out.response_metadata for cid, out in result.outputs.items() if out.responded
    }
    records.update((sr.source_channel_id, stream_record(sr)) for sr in result.streaming_responses)
    return records


def _answer_of(
    result: BroadcastResult, rows: list[RoomEvent], channel_id: str | None
) -> tuple[str | None, str | None]:
    """*channel_id*'s answer among *rows* and how its turn ended, both read
    off that responder alone; nothing for a responder that cannot be told."""
    if channel_id is None:
        return None, None
    row = _last_answer(rows, channel_id)
    record = _responder_records(result).get(channel_id, {})
    return (answer_text(row) if row is not None else None), _turn_end(
        record, [row] if row is not None else []
    )


def _kept_answer(result: BroadcastResult, rows: list[RoomEvent]) -> tuple[str | None, str | None]:
    """The answer the child room kept, and how the turn that gave it ended.

    Both are read off one responder, never off another's: the first that
    kept an answer among *rows*, its end read from its own record (RFC §23.3
    step 6). With no answer kept, the end is the first responder's that
    names one.
    """
    records = _responder_records(result)
    for cid, record in records.items():
        if (row := _last_answer(rows, cid)) is not None:
            return answer_text(row), _turn_end(record, [row])
    ends = (_turn_end(record, []) for record in records.values())
    return None, next((end for end in ends if end is not None), None)


def _turn_failure(failure: Exception, text: str | None, reason: str | None) -> Exception:
    """A delegated turn that failed after it began: its error, carrying how
    the turn ended and its narration when its record names an end (RFC §23.3
    step 6). An error the turn raised before any end, or a cut, as it is."""
    if reason in (None, "completed") or isinstance(failure, TurnCutShortError):
        return failure
    return TaskTurnFailedError(failure, reason, text or None)


def _cut_short(text: str | None, reason: str | None) -> TaskCutShortError | None:
    """Why a delegated turn has no answer: it ended before it (its round cap,
    deadline or budget), its *text* a narration. ``None`` for a turn that
    completed, or whose stream carried no end (RFC §6.4)."""
    if reason in (None, "completed"):
        return None
    return TaskCutShortError(reason, text or None)


def _turn_end(record: Mapping[str, Any], rows: list[RoomEvent]) -> str | None:
    """How a delegated turn ended: as the turn's record names it, else as its
    last message does, for a channel that records it there only (RFC §6.4).
    ``None`` when neither names an end."""
    return recorded_turn_end(record) or next(
        (
            reason
            for row in reversed(rows)
            if row.type == EventType.MESSAGE and (reason := recorded_turn_end(row.metadata or {}))
        ),
        None,
    )


def _last_answer(rows: list[RoomEvent], channel_id: str) -> RoomEvent | None:
    """*channel_id*'s last answer among *rows*, or ``None`` when it kept none."""
    return next(
        (
            row
            for row in reversed(rows)
            if row.source.channel_id == channel_id and answer_text(row) is not None
        ),
        None,
    )


async def _buffered_outcome(
    kit: RoomKit, child_room_id: str, output: Any
) -> tuple[str | None, Exception | None]:
    """A delegated buffered reply's answer, or why the turn has none.

    Its trace is kept first: ``response_events`` already hold the tool-call
    events, and all of them persist, not just the final text. An error fails
    the turn, whether the reply answered or not, as on the room path
    (:func:`_responder_failure`); a turn its cap, deadline or budget cut has
    no answer (RFC §6.4).
    """
    if not output.responded:
        return None, output.error
    final_text = await _persist_response_events(kit, child_room_id, output.response_events)
    reason = _turn_end(output.response_metadata, output.response_events)
    if output.error is not None:
        # A turn the provider interrupted after a round kept its trace and
        # has no answer.
        return None, _turn_failure(output.error, final_text, reason)
    if (cut := _cut_short(final_text, reason)) is not None:
        return None, cut
    return final_text, None


async def _collect_answer(
    kit: RoomKit, child_room_id: str, result: BroadcastResult, child_depth: int
) -> str | None:
    """Keep the trace of every response a delegated broadcast started; its answer.

    Every response is read to its end (RFC §8.3), so none is left generating
    unread. The first answer is the delegated turn's; a response that failed
    fails the turn once they are all read (RFC §6.4).
    """
    answers: list[str] = []
    failure: Exception | None = None
    for output in result.outputs.values():
        if output.response_stream is not None:
            continue
        answer, failed = await _buffered_outcome(kit, child_room_id, output)
        failure = failure or failed
        if answer is not None:
            answers.append(answer)
    # Streaming: drain the marker stream, persisting tool calls + text segments.
    for sr in result.streaming_responses:
        try:
            text = await _persist_child_stream(kit, child_room_id, sr, child_depth)
        except Exception as exc:
            failure = failure or exc
            continue
        if text:
            answers.append(text)
    if failure is not None:
        # Reported in the child room: a buffered failure by the delivery set,
        # a stream's by its trace writer.
        raise mark_reported(failure)
    return answers[0] if answers else None


#: A cursor larger than any real event index, so ``before_index`` returns the
#: tail (most recent events) — where a worker's final submit_result call lives.
#: Capped at max int32: the postgres store binds ``before_index`` as int4.
_LATEST_TAIL_CURSOR = 2**31 - 1


async def _scan_for_submitted_result(
    kit: RoomKit, child_room_id: str, worker_id: str, result_tool: ResultTool | None = None
) -> dict[str, Any] | None:
    """Find the worker's served call of the result tool (``submit_result`` by
    default) in its persisted trace.

    The one reading of a result, wherever the call was served: through the
    channel's tool loop, or by an MCP server (named ``mcp__<server>__<name>``),
    each call is persisted as a TOOL_CALL event. Only a call that ended
    served counts, made by the worker itself: a refused, failed or blocked
    call is no result, and another channel shared into the room is not the
    worker. Returns the normalized payload, or None.
    """
    from roomkit.orchestration.result import SUBMIT_RESULT

    tool = result_tool or SUBMIT_RESULT
    events = await kit.store.list_events(
        child_room_id, before_index=_LATEST_TAIL_CURSOR, limit=100
    )
    for ev in reversed(events):
        if ev.type != EventType.TOOL_CALL_END or ev.source.channel_id != worker_id:
            continue
        if tool.matches(getattr(ev.content, "tool_name", "") or "") and _served(ev.content):
            return tool.normalize(getattr(ev.content, "arguments", None) or {})
    return None


def _served(content: Any) -> bool:
    """Whether a stored tool end says its call was served; a row written
    before outcomes existed says it by its status."""
    outcome = getattr(content, "outcome", None)
    if outcome is not None:
        return outcome == "served"
    return getattr(content, "status", None) == "completed"


@runtime_checkable
class _InjectableToolChannel(Protocol):
    """A channel a delegation can set a result tool up on, for one room.

    ``_run_with_structured_result`` matches by structure, not by class, so
    duck-typed agent channels (and test doubles) qualify as long as they carry
    a tool registry.
    """

    _registry: ChannelRegistry


async def _run_with_structured_result(
    kit: RoomKit,
    child_room_id: str,
    task_desc: str,
    max_result_retries: int,
    result_tool: ResultTool | None = None,
    *,
    turns: TurnEntries | None = None,
) -> str:
    """Run a delegated agent that must hand its work back via a result tool
    (*result_tool*, ``submit_result`` by default). Injects the tool (for
    function-calling providers), runs the agent, and a deterministic completion
    guard: if the agent ends a turn without calling the tool, it is re-prompted to
    use it (up to *max_result_retries* times); if it still hasn't, the tool's
    ``on_missing`` payload is returned on its behalf.

    The tool is served in *child_room_id* only, by the one
    :func:`~roomkit.core.mixins._result_capture.capture_result` sets up there,
    or by an MCP server; either way the result is read from the room's
    persisted trace, the worker's own call that ended served, once
    ON_TOOL_CALL has judged it. Another room's delegation to the same agent
    never sees this one's result.
    Returns the payload as a JSON string (``on_missing``'s when exhausted)."""
    from roomkit.orchestration.result import SUBMIT_RESULT

    tool = result_tool or SUBMIT_RESULT
    room = await kit.get_room(child_room_id)
    agent_id = (room.metadata or {}).get("task_agent_id")
    channel = kit.channels.get(agent_id) if agent_id else None
    if not isinstance(channel, _InjectableToolChannel):
        # No injectable agent channel — fall back to plain text collection.
        text = await _broadcast_and_collect(kit, child_room_id, task_desc, turns=turns)
        return text or ""
    role = getattr(channel, "role", None) or getattr(channel, "description", None) or str(agent_id)

    with capture_result(channel, child_room_id, tool):
        message = task_desc
        last_text = ""
        for _attempt in range(max_result_retries + 1):
            scanned, text = await _owed_result(
                kit, child_room_id, message, str(agent_id), tool, turns=turns
            )
            if scanned is not None:
                return json.dumps(scanned)
            last_text = text or last_text
            message = tool.reminder
        _tasks_logger.warning(
            "Delegated agent %s never called %s after %d attempts; failing.",
            agent_id,
            tool.name,
            max_result_retries + 1,
        )
        return json.dumps(
            tool.on_missing(role=role, last_output=last_text, attempts=max_result_retries + 1)
        )


async def _owed_result(
    kit: RoomKit,
    child_room_id: str,
    message: str,
    worker_id: str,
    tool: ResultTool,
    *,
    turns: TurnEntries | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """One turn of a worker that owes a result: what it submitted, and its
    text. A turn cut short, or failed after it began, keeps a result it
    submitted before, and fails the task without one (RFC §23.3)."""
    try:
        text = await _broadcast_and_collect(kit, child_room_id, message, turns=turns)
    except (TaskCutShortError, TaskTurnFailedError):
        submitted = await _scan_for_submitted_result(kit, child_room_id, worker_id, tool)
        if submitted is None:
            raise
        return submitted, None
    return await _scan_for_submitted_result(kit, child_room_id, worker_id, tool), text


async def run_agent_in_child_room(
    kit: RoomKit,
    child_room_id: str,
    task_desc: str,
    *,
    require_structured_result: bool = False,
    max_result_retries: int = 3,
    result_tool: ResultTool | None = None,
    turns: TurnEntries | None = None,
) -> str | None:
    """Send a task to a child room and collect the attached agent's response.

    This is the **single code path** for executing a delegated agent, used by
    both ``delegate(wait=True)`` (inline) and the background task runner.

    By default the agent's free-text response is collected and returned. When
    *require_structured_result* is set, the agent must instead hand its work back
    via a result tool, *result_tool* or ``submit_result`` by default (forced
    structure + a guaranteed result); the returned string is then the
    JSON-encoded structured payload (see :func:`_run_with_structured_result`).

    Either way the agent's full trace (tool calls + messages) is persisted in the
    child room, which records its parent via ``metadata.parent_room_id`` (set at
    creation in :meth:`delegate`) so the parent↔child link is rebuildable.

    *turns*, when given, receives each responder's turn entry (its end, its
    usage), the last turn's, however it ended (RFC §23.3 step 6).

    When several people speak in the room the work comes from, the task opens
    with who asked for it (RFC §19.7, :func:`~roomkit.core._requester.requested_line`).
    """
    task_desc = with_requester(task_desc)
    if require_structured_result:
        return await _run_with_structured_result(
            kit, child_room_id, task_desc, max_result_retries, result_tool, turns=turns
        )
    return await _broadcast_and_collect(kit, child_room_id, task_desc, turns=turns)
