"""Lifecycle of the tool calls a streaming provider serves itself.

A call the provider already ran (its result rides it) is reported, and a call
its external tool handler decides is decided, then reported. Every other call
is the channel's own, served by the loop (RFC §9.3, who serves a call).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from roomkit.models.enums import ChannelType
from roomkit.models.streaming import StreamDelta, ToolCallEndMarker, ToolCallStartMarker
from roomkit.models.tool_call import ToolCallEvent, ToolCallObserver
from roomkit.providers.ai.base import StreamToolCall
from roomkit.providers.ai.tool_calls import partial_call_error
from roomkit.realtime.base import EphemeralEventType
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools._turn_calls import AnnouncedCall, reporting
from roomkit.tools.context import _ToolLoopContext
from roomkit.tools.external import ExternalToolHandler, refusal_detail
from roomkit.tools.result import (
    as_tool_result,
    call_id_in_flight_error,
    failure_detail,
    tool_failure,
)

logger = logging.getLogger("roomkit.channels.ai")


class _ToolEventPublisher(Protocol):
    async def __call__(
        self,
        event_type: EphemeralEventType,
        room_id: str | None,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class _Decided:
    """What a still-pending call became, and who reports it: the handler
    (its refusal, its approval) or the channel (a call it refused itself, a
    handler that raised, with what failed)."""

    arguments: dict[str, Any]
    result: str
    kind: OutcomeKind
    by_channel: bool = False
    detail: str | None = None


@dataclass
class _ExternalStreamTools:
    """Turn-scoped routing and lifecycle of the calls the provider serves."""

    channel_id: str
    room_id: str | None
    # The turn's call registry: a call is announced there before its start,
    # and its one report claimed (RFC §9.3).
    loop_ctx: _ToolLoopContext
    publish: _ToolEventPublisher
    # Whether the turn has a tool of the channel's own under a name.
    serves_locally: Callable[[str], bool]
    # The copy of an outcome the model reads, bounded: ``(name, result, call_id)``.
    bound: Callable[[str, str, str], str]
    handler: ExternalToolHandler | None = None
    # ON_TOOL_CALL as a report, for a call the provider already ran (RFC §9.3).
    report: ToolCallObserver | None = None
    # ON_TOOL_CALL's observers only, for a call that never ran and that the
    # channel decided itself (RFC §9.3).
    observe: ToolCallObserver | None = None

    def takes(self, call: StreamToolCall) -> bool:
        """Whether *call* is the provider's: one it already ran (it says so
        on the call, never in its arguments), or one its handler decides
        because no tool of the channel's own carries its name."""
        if call.served is not None:
            return True
        return self.handler is not None and not self.serves_locally(call.name)

    async def stream_call(
        self, call: StreamToolCall, round_idx: int
    ) -> AsyncGenerator[StreamDelta, None]:
        """The lifecycle of a call the provider serves: its start, its outcome
        (the provider's, or its handler's decision) reported once, its end."""
        arguments = dict(call.arguments)
        served = call.served
        result = served.result if served is not None else ""
        failed = served is not None and served.is_error
        kind = OutcomeKind.FAILED if failed else OutcomeKind.SERVED
        started_at = time.monotonic()
        # A call the provider ran means the side effect already happened: its
        # outcome is reported at once, before anything can cut the call. Only a
        # still-pending call can be denied or rewritten before acting.
        pending = served is None and self.handler is not None
        entry = self._announce(call, pending=pending)
        if not pending:
            await self._report(entry, _Decided(arguments, result, kind))
        yield ToolCallStartMarker(tool_name=call.name, tool_id=call.id, arguments=arguments)
        await self._publish_start(call, arguments, round_idx)
        if pending and self.handler is not None:
            decided = await self._settle(entry, self.handler, _Decided(arguments, result, kind))
            arguments, result, kind = decided.arguments, decided.result, decided.kind
            await self._report(entry, decided)
        duration_ms = int((time.monotonic() - started_at) * 1000)
        # The model reads it bounded, as any outcome, and so does its END row;
        # the report above heard it whole (RFC §21.5).
        result = self.bound(call.name, result, call.id)
        yield _end_marker(call, arguments, result, kind, duration_ms)
        await self._publish_end(call, result, kind, round_idx, duration_ms)

    async def _publish_start(
        self, call: StreamToolCall, arguments: dict[str, Any], round_idx: int
    ) -> None:
        if self.room_id:
            await self.publish(
                EphemeralEventType.TOOL_CALL_START,
                self.room_id,
                [call.model_copy(update={"arguments": arguments})],
                round_idx,
            )

    async def _publish_end(
        self,
        call: StreamToolCall,
        result: str,
        kind: OutcomeKind,
        round_idx: int,
        duration_ms: int,
    ) -> None:
        if self.room_id:
            await self.publish(
                EphemeralEventType.TOOL_CALL_END,
                self.room_id,
                [ToolOutcome(kind, result).as_part(call.id, call.name)],
                round_idx,
                duration_ms=duration_ms,
            )

    def _announce(self, call: StreamToolCall, *, pending: bool) -> AnnouncedCall:
        """Put *call* in the turn's registry before anything can cut it: the
        turn's end reports it if nothing did, a *pending* one through the
        handler that was to decide it (RFC §9.3). Under an id another call
        of the round holds, it is a call of its own (RFC §12.4)."""
        copy = call.model_copy(update={"arguments": dict(call.arguments)})
        return self.loop_ctx.calls.announce(copy, external=pending)

    async def _settle(
        self, entry: AnnouncedCall, handler: ExternalToolHandler, pending: _Decided
    ) -> _Decided:
        """What a still-pending call becomes. One under an id another call of
        its round holds is refused by the channel, as on every door, and the
        handler never asked (RFC §12.4)."""
        if entry.duplicate:
            body = call_id_in_flight_error(entry.call.id)
            return _Decided(pending.arguments, body, OutcomeKind.REFUSED, by_channel=True)
        return await self._decide(handler, entry.call, pending)

    async def _decide(
        self, handler: ExternalToolHandler, call: StreamToolCall, pending: _Decided
    ) -> _Decided:
        """What a still-pending call becomes: its arguments, result and outcome.

        A call the response cut before its arguments were complete is refused
        without asking the handler (RFC §6.4); any other is the handler's to
        refuse, rewrite or serve. A handler that raises fails the call, as a
        tool handler that raises does: the model reads the failure's class,
        the log and the observers its message (RFC §9.3).
        """
        arguments = pending.arguments
        if call.partial:
            error = json.dumps(partial_call_error(call.name, garbled=call.garbled))
            return _Decided(arguments, error, OutcomeKind.REFUSED, by_channel=True)
        try:
            decision = await handler.process_tool_call(
                call.name, arguments, tool_call_id=call.id, room_id=self.room_id
            )
        except Exception as exc:
            detail = failure_detail(exc)
            logger.warning("External tool handler failed deciding %s: %s", call.name, detail)
            failure = tool_failure(call.name, exc)
            return _Decided(arguments, failure, OutcomeKind.FAILED, by_channel=True, detail=detail)
        if not decision.approved:
            reason = json.dumps({"error": decision.reason or f"Tool '{call.name}' was denied"})
            return _Decided(arguments, reason, OutcomeKind.REFUSED, detail=decision.detail)
        if decision.modified_input is not None:
            arguments = decision.modified_input
        if decision.result is not None:
            return _Decided(arguments, decision.result, OutcomeKind.SERVED)
        return _Decided(arguments, pending.result, pending.kind)

    async def _report(self, entry: AnnouncedCall, decided: _Decided) -> None:
        """Report the call's outcome once, by whoever decided it: the handler
        its refusal (:meth:`~ExternalToolHandler.on_tool_refused`) or what it
        let through; the channel, to ON_TOOL_CALL, a call it refused itself, a
        handler that raised, or a call the provider ran with no handler. An
        outcome the model already read, so no hook may rewrite it, and a
        refusal reaches the observers only (RFC §9.3).

        The report is claimed where the observers hear it, past the SYNC
        chain: a cut before then leaves it owed, with this outcome, to the
        turn's end. A handler that reports nothing, or raises reporting, has
        made the call's report. The report is the call's own, whichever call
        of its round holds its id.
        """
        call = entry.call
        event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=call.id,
            name=call.name,
            arguments=decided.arguments,
            result=as_tool_result(decided.result),
            room_id=self.room_id,
            is_error=decided.kind is not OutcomeKind.SERVED,
            refused=decided.kind is OutcomeKind.REFUSED,
            error_detail=decided.detail,
        )
        entry.known = event
        with reporting(entry):
            await self._deliver_report(call, decided, event)
            self.loop_ctx.claim_report(call.id)

    async def _deliver_report(
        self, call: StreamToolCall, decided: _Decided, event: ToolCallEvent
    ) -> None:
        """Hand *event* to whoever reports it: the observers alone for a call
        that never ran and that the channel decided (as a local call refused
        or failed), the handler for what it decided, else ON_TOOL_CALL."""
        if decided.by_channel and self.observe is not None:
            await self.observe(event)
            return
        handler = None if decided.by_channel else self.handler
        if handler is None:
            if self.report is not None:
                await self.report(event)
            return
        try:
            await self._hand_to_handler(handler, call, decided, event)
        except Exception:
            # As on the cut path: a handler that raises reporting does not
            # end the turn, and its report is not made twice.
            logger.exception("External tool handler failed reporting the call %s", call.id)

    async def _hand_to_handler(
        self,
        handler: ExternalToolHandler,
        call: StreamToolCall,
        decided: _Decided,
        event: ToolCallEvent,
    ) -> None:
        """The handler reports what it decided: its refusal, with what failed
        when it came from a failure, or the call it let through."""
        if decided.kind is OutcomeKind.REFUSED:
            await handler.on_tool_refused(
                call.name,
                decided.arguments,
                decided.result,
                tool_call_id=call.id,
                room_id=self.room_id,
                **refusal_detail(handler, decided.detail),
            )
            return
        await handler.on_tool_result(
            call.name,
            decided.arguments,
            decided.result,
            is_error=event.is_error,
            tool_call_id=call.id,
            room_id=self.room_id,
        )


async def report_cut(
    handler: ExternalToolHandler, call: StreamToolCall, room_id: str | None
) -> None:
    """Tell *handler* the turn cut *call* before its report: it reports the
    call cancelled, as every channel reports a call it cut (RFC §9.3). A
    handler that raises does not disturb the turn's end."""
    try:
        await handler.on_tool_cancelled(
            call.name, call.arguments, tool_call_id=call.id, room_id=room_id
        )
    except Exception:
        logger.exception("External tool handler failed on the cut call %s", call.id)


def _end_marker(
    call: StreamToolCall,
    arguments: dict[str, Any],
    result: str,
    kind: OutcomeKind,
    duration_ms: int,
) -> ToolCallEndMarker:
    """The stream's END marker of a call the provider serves, with its outcome."""
    failed = kind is not OutcomeKind.SERVED
    return ToolCallEndMarker(
        tool_name=call.name,
        tool_id=call.id,
        arguments=arguments,
        result=result,
        status="failed" if failed else "completed",
        duration_ms=duration_ms,
        error=result if failed else None,
        outcome=kind.value,
    )
