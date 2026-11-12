"""The calls a turn makes to its room owner, on the executor (RFC §10.6.5).

The tools the request lists are the turn's owner tools: the channel declares
them with its own, and a call to one, past its gate, is a ``tool_call`` the
owner answers once. A verb of the turn's ``TurnHost`` is a ``host_call``,
answered the same way. The call keeps the bound the channel gives it; a call cut
(its bound expired, its turn cancelled) is withdrawn with ``call_cancel``.
The handler on the owner reads what a local one would read of the turn,
carried in the call's context, the turn's response-metadata record included:
what it changed of that record is applied to the turn's when its answer comes.
"""

from __future__ import annotations

import asyncio
import contextlib
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from roomkit.channels._tool_registry import TurnOwnerTools, orchestration_tool
from roomkit.tools.context import (
    current_response_metadata,
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_call,
    current_tool_chain_depth,
    current_tool_requester,
    current_tool_room_id,
)
from roomkit.turns._call_record import apply_changes, carried_record
from roomkit.turns._codec import decode_result
from roomkit.turns.wire import CallCancel, CallContext, HostCall, HostResult, ToolCall

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.turns._error_codec import ErrorClasses
    from roomkit.turns.wire import Frame, RecordChanges, ToolResult, TurnRequest


class ExecutorCalls:
    """The owner-served calls of one turn, each awaiting its result."""

    def __init__(
        self,
        request: TurnRequest,
        send: Callable[[Frame], Awaitable[int]],
        errors: ErrorClasses,
    ) -> None:
        self._request = request
        self._send = send
        self._errors = errors
        self._pending: dict[str, asyncio.Future[ToolResult | HostResult]] = {}
        self._settled = False
        self.owner_tools = TurnOwnerTools(
            channel_id=request.channel_id,
            room_id=request.room_id,
            entries={
                owned.tool.name: orchestration_tool(
                    owned.tool, partial(self._call, owned.tool.name), waits=owned.waits_by_design
                )
                for owned in request.owner_tools
            },
        )

    def resolve(self, result: ToolResult | HostResult) -> None:
        """The owner answered a call."""
        future = self._pending.get(result.call_id)
        if future is not None and not future.done():
            future.set_result(result)

    def fail_all(self, error: BaseException) -> None:
        """The owner is gone: no call of the turn will be answered."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    async def settle(self) -> None:
        """The turn ends: wait for the calls still on their way (a verb a
        task of the turn's tools sent), and take no more."""
        self._settled = True
        if self._pending:
            await asyncio.wait(list(self._pending.values()))

    async def host_call(self, verb: str, arguments: dict[str, Any]) -> Any:
        """Run one verb of the turn's host on the owner: its result, as data."""
        call = HostCall(
            call_id=uuid4().hex, verb=verb, arguments=arguments, context=_call_context()
        )
        result = await self._ask(call, f"verb {verb!r}")
        if not isinstance(result, HostResult):
            raise RuntimeError("The room owner answered a verb with a tool's result")
        _take_changes(call.context, result.record_changes)
        if result.error is not None:
            raise self._errors.rebuild(result.error, where="the room owner")
        return result.result

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Serve one call on the owner: its answer, or what the handler raised."""
        call = ToolCall(
            call_id=uuid4().hex, name=name, arguments=arguments, context=_call_context()
        )
        result = await self._ask(call, f"tool {name!r}")
        if isinstance(result, HostResult):
            raise RuntimeError("The room owner answered a tool call with a verb's result")
        _take_changes(call.context, result.record_changes)
        return _answer(result, self._errors)

    async def _ask(self, call: ToolCall | HostCall, what: str) -> ToolResult | HostResult:
        """Send *call* and wait for its one answer; a call cut before it
        came (its bound, its turn) is withdrawn, and the owner cancels it."""
        if self._settled:
            raise RuntimeError(f"The turn has ended: {what} is not sent to the room owner")
        future: asyncio.Future[ToolResult | HostResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[call.call_id] = future
        try:
            try:
                await self._send(call)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"The arguments of {what} cannot cross the turn backend") from exc
            try:
                return await future
            except asyncio.CancelledError:
                if future.cancelled() or not future.done():
                    with contextlib.suppress(Exception):
                        await self._send(CallCancel(call_id=call.call_id))
                raise
        finally:
            del self._pending[call.call_id]


def _call_context() -> CallContext:
    """What the call's handler reads of the turn, as the channel set it."""
    record = current_tool_call()
    allowed = current_tool_allowed_names()
    return CallContext(
        tool_call_id=record.tool_call_id if record is not None else "",
        room_id=current_tool_room_id() or "",
        channel_id=record.channel_id if record is not None else "",
        chain_depth=current_tool_chain_depth(),
        actor_id=current_tool_actor_id(),
        allowed_names=None if allowed is None else sorted(allowed),
        requester=current_tool_requester(),
        response_metadata=carried_record(),
    )


def _take_changes(context: CallContext, changes: RecordChanges | None) -> None:
    """Apply to the turn's record what the owner's handler changed of the
    copy *context* carried, before the call returns (RFC §10.6.5)."""
    record = current_response_metadata()
    if record is not None and context.response_metadata is not None:
        apply_changes(record, context.response_metadata, changes)


def _answer(result: ToolResult, errors: ErrorClasses) -> Any:
    """The handler's answer, its structured copy written on the call as the
    handler would have written it; what it raised, raised here."""
    if result.outcome != "served" or result.result is None:
        if result.error is None:
            raise RuntimeError("The room owner answered a call without its result")
        raise errors.rebuild(result.error, where="the room owner")
    record = current_tool_call()
    if record is not None and result.structured_content is not None:
        record.structured_content = result.structured_content
    return decode_result(result.result)
