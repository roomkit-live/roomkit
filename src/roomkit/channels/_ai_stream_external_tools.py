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
from roomkit.tools.context import _ToolLoopContext
from roomkit.tools.external import ExternalToolHandler, refusal_detail
from roomkit.tools.result import as_tool_result, failure_detail, tool_failure

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
        self._announce(call, arguments, pending=pending)
        if not pending:
            await self._report(call, _Decided(arguments, result, kind))
        yield ToolCallStartMarker(tool_name=call.name, tool_id=call.id, arguments=arguments)
        await self._publish_start(call, arguments, round_idx)
        if pending and self.handler is not None:
            decided = await self._decide(self.handler, call, _Decided(arguments, result, kind))
            arguments, result, kind = decided.arguments, decided.result, decided.kind
            await self._report(call, decided)
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

    def _announce(self, call: StreamToolCall, arguments: dict[str, Any], *, pending: bool) -> None:
        """Put *call* in the turn's registry before anything can cut it: the
        turn's end reports it if nothing did, a *pending* one through the
        handler that was to decide it (RFC §9.3)."""
        self.loop_ctx.announced_calls[call.id] = call.model_copy(update={"arguments": arguments})
        if pending:
            self.loop_ctx.external_calls.add(call.id)

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

    async def _report(self, call: StreamToolCall, decided: _Decided) -> None:
        """Report the call's outcome once, by whoever decided it: the handler
        its refusal (:meth:`~ExternalToolHandler.on_tool_refused`) or what it
        let through; the channel, to ON_TOOL_CALL, a call it refused itself, a
        handler that raised, or a call the provider ran with no handler. An
        outcome the model already read, so no hook may rewrite it, and a
        refusal reaches the observers only (RFC §9.3).

        The report is claimed where the observers hear it, past the SYNC
        chain: a cut before then leaves it owed, with this outcome, to the
        turn's end. A handler that reports nothing has made the call's report.
        """
        refused = decided.kind is OutcomeKind.REFUSED
        event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=call.id,
            name=call.name,
            arguments=decided.arguments,
            result=as_tool_result(decided.result),
            room_id=self.room_id,
            is_error=decided.kind is not OutcomeKind.SERVED,
            refused=refused,
            error_detail=decided.detail,
        )
        self.loop_ctx.known_outcomes[call.id] = event
        if decided.by_channel and self.observe is not None:
            # It never ran: the observers alone hear of it, as of a local call
            # refused or failed (RFC §9.3).
            await self.observe(event)
            self.loop_ctx.claim_report(call.id)
            return
        handler = None if decided.by_channel else self.handler
        if handler is not None and refused:
            await handler.on_tool_refused(
                call.name,
                decided.arguments,
                decided.result,
                tool_call_id=call.id,
                room_id=self.room_id,
                **refusal_detail(handler, decided.detail),
            )
        elif handler is not None:
            await handler.on_tool_result(
                call.name,
                decided.arguments,
                decided.result,
                is_error=event.is_error,
                tool_call_id=call.id,
                room_id=self.room_id,
            )
        elif self.report is not None:
            await self.report(event)
        self.loop_ctx.claim_report(call.id)


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
