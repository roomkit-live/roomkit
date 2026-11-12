"""The calls a turn makes to its room owner, on the owner (RFC §10.6.5).

A ``tool_call`` runs the handler the owner keeps for the tool, and a
``host_call`` a verb of the turn's ``TurnHost``, each as a task of its own,
in a copy of the execution context of the owner task waiting on the turn when
the call arrives (the delivery target awaiting the output, or the reader
awaiting an item), under the turn's id, and under what the turn gave the
call: its room and channel (the request's: a call naming others is refused),
the room's ``Room``, chain depth, actor, allowed names, requester and tool
call id. The call holds the reentrancy that task holds, and a close it calls
spares the turn it serves, as when the call ran in the owner's own code; it
reads no turn's host nor turn's tools, being the owner's code. Its answer
goes back once, as a ``tool_result`` or a ``host_result``. A call the
executor withdraws is cancelled, and so is every call of a turn that ends for
the owner; the client that submitted the turn waits for them when it closes.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable, Coroutine, Generator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import set_turn_owner_tools
from roomkit.core._turn_scope import enter_turn
from roomkit.core.exceptions import ToolRefusedError, UnservedToolCallError
from roomkit.core.task_utils import acting_tasks, close_callers
from roomkit.tools.context import ToolCallContext, owner_served_call
from roomkit.tools.result import as_tool_result
from roomkit.turns._call_record import changes_of
from roomkit.turns._codec import encode_result
from roomkit.turns._error_codec import error_record
from roomkit.turns.base import response_depth
from roomkit.turns.host import READERS, set_turn_host
from roomkit.turns.wire import HostCall, HostResult, ToolCall, ToolResult

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from roomkit.turns.base import OwnerHost, OwnerTools
    from roomkit.turns.wire import CallContext, Control, RecordChanges, TurnRequest

logger = logging.getLogger("roomkit.turns")


def _enter_owner_call(turn_id: str) -> None:
    """The context of a call the owner serves: under its turn's id, outside
    any turn's host or tools (a turn the reading task itself runs under)."""
    enter_turn(turn_id)
    set_turn_host(None)
    set_turn_owner_tools(None)


class OwnerCalls:
    """The owner-served calls of one turn, each a task the turn can cancel.

    *held* is the set of the client's running calls, every turn's, which the
    client waits for when it closes.
    """

    def __init__(
        self,
        request: TurnRequest,
        tools: OwnerTools | None,
        host: OwnerHost | None,
        send: Callable[[Control], Awaitable[None]],
        held: set[asyncio.Task[None]],
    ) -> None:
        self._request = request
        self._tools = tools
        self._host = host
        self._send = send
        self._held = held
        self._running: dict[str, asyncio.Task[None]] = {}

    def start(self, call: ToolCall | HostCall, waiting: contextvars.Context | None) -> None:
        """Run *call* in a copy of *waiting*, the context of the task waiting
        on the turn when it arrived (a fresh one when none has waited yet)."""
        context = waiting.copy() if waiting is not None else contextvars.Context()
        context.run(_enter_owner_call, self._request.turn_id)
        # The work is begun by the task, never before: a task cancelled before
        # its first step (the turn ending as the call arrives) leaves no
        # coroutine unawaited.
        if isinstance(call, HostCall):
            work, failed = partial(self._verb_result, call), partial(_verb_failed, call.call_id)
        else:
            work, failed = partial(self._tool_result, call), partial(_tool_failed, call.call_id)
        task = asyncio.get_running_loop().create_task(
            self._answer(work, failed), context=context, name=f"roomkit-owner-call-{call.call_id}"
        )
        self._running[call.call_id] = task
        self._held.add(task)
        task.add_done_callback(lambda _task: self._forget(call.call_id, task))

    def withdraw(self, call_id: str) -> None:
        """The executor withdrew the call: it is cancelled."""
        task = self._running.get(call_id)
        if task is not None and not task.cancelling():
            task.cancel("withdrawn")

    def cancel_all(self) -> None:
        """The turn is over for the owner: every call still running is
        cancelled, but the one running this code, and one waiting for a
        kit's close it called (the close cut the turn it runs in)."""
        acting = acting_tasks() | close_callers()
        for task in list(self._running.values()):
            if task not in acting and not task.cancelling():
                task.cancel("turn_ended")

    def _forget(self, call_id: str, task: asyncio.Task[None]) -> None:
        self._running.pop(call_id, None)
        self._held.discard(task)

    async def _answer(
        self,
        work: Callable[[], Coroutine[Any, Any, Control]],
        failed: Callable[[BaseException], Control],
    ) -> None:
        """Send what *work* makes of the call, once."""
        try:
            result = await work()
        except asyncio.CancelledError as cancelled:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            # The call ended on a cancellation of something it awaited, not
            # of the call: the call failed, and the turn goes on.
            result = failed(cancelled)
        try:
            await self._send(result)
        except (TypeError, ValueError) as exc:
            refusal = TypeError(f"The result of a call cannot cross the turn backend: {exc}")
            await self._send(failed(refusal))

    async def _tool_result(self, call: ToolCall) -> ToolResult:
        """What the tool's handler made of *call*, as it crosses back."""
        served = _Served(call.context.response_metadata)
        try:
            with self._serving(call.context, served) as record:
                if self._tools is None:
                    raise UnservedToolCallError(
                        f"No tool {call.name!r} is served by the room owner"
                    )
                answer = await self._tools.serve(self._request, call)
        except UnservedToolCallError as exc:
            return ToolResult(
                call_id=call.call_id,
                outcome="unserved",
                error=error_record(exc),
                record_changes=served.changes(),
            )
        except ToolRefusedError as exc:
            return ToolResult(
                call_id=call.call_id,
                outcome="refused",
                error=error_record(exc),
                record_changes=served.changes(),
            )
        except Exception as exc:
            logger.debug("Owner-served call %s failed", call.name, exc_info=True)
            return _tool_failed(call.call_id, exc, served.changes())
        return ToolResult(
            call_id=call.call_id,
            outcome="served",
            result=encode_result(as_tool_result(answer)),
            structured_content=record.structured_content,
            record_changes=served.changes(),
        )

    async def _verb_result(self, call: HostCall) -> HostResult:
        """What the verb of *call* made, as it crosses back."""
        served = _Served(call.context.response_metadata)
        try:
            if self._host is None:
                raise ValueError(f"No verb {call.verb!r} is served by the room owner")
            if call.verb in READERS:
                # A reader of the turn's room, made while the turn builds its
                # input: no tool call's context is its.
                value = await self._host.call(self._request, call.verb, call.arguments)
            else:
                with self._serving(call.context, served):
                    value = await self._host.call(self._request, call.verb, call.arguments)
        except Exception as exc:
            logger.debug("Host call %s failed", call.verb, exc_info=True)
            return _verb_failed(call.call_id, exc, served.changes())
        return HostResult(call_id=call.call_id, result=value, record_changes=served.changes())

    @contextmanager
    def _serving(self, context: CallContext, served: _Served) -> Generator[ToolCallContext]:
        """Run the enclosed call under what the turn gave it, the room and
        the channel being the request's, the turn's record a copy *served*
        keeps; a call naming others is refused."""
        request = self._request
        if context.room_id not in ("", request.room_id) or context.channel_id not in (
            "",
            request.channel_id,
        ):
            raise UnservedToolCallError(
                f"A call of the turn in room {request.room_id!r} named room "
                f"{context.room_id!r} and channel {context.channel_id!r}"
            )
        record = ToolCallContext(
            room_id=request.room_id,
            tool_call_id=context.tool_call_id,
            channel_id=request.channel_id,
        )
        room = request.context.room
        with owner_served_call(
            record,
            room=room if room.id == request.room_id else None,
            actor_id=context.actor_id,
            requester=context.requester,
            # A call made outside the tool loop (a hook fired before it, an
            # ACP agent's turn) carries depth 0: the turn's work is at least
            # as deep as its response.
            chain_depth=max(context.chain_depth, response_depth(request)),
            allowed_names=context.allowed_names,
            response_metadata=context.response_metadata,
        ) as worked:
            served.worked = worked
            yield record


class _Served:
    """The turn's record of one served call: the copy it carried, and the
    one its handler worked on."""

    def __init__(self, carried: dict[str, Any] | None) -> None:
        self.carried = carried
        self.worked: Mapping[str, Any] | None = None

    def changes(self) -> RecordChanges | None:
        """What the handler changed of what the call carried."""
        if self.carried is None or self.worked is None:
            return None
        return changes_of(self.carried, self.worked)


def _tool_failed(
    call_id: str, exc: BaseException, changes: RecordChanges | None = None
) -> ToolResult:
    return ToolResult(
        call_id=call_id, outcome="failed", error=error_record(exc), record_changes=changes
    )


def _verb_failed(
    call_id: str, exc: BaseException, changes: RecordChanges | None = None
) -> HostResult:
    return HostResult(call_id=call_id, error=error_record(exc), record_changes=changes)
