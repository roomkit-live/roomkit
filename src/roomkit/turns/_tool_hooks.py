"""ON_TOOL_CALL and BEFORE_TOOL_USE for the channels an executor wires (RFC
§9.3, §10.6.6): the gate before a call, the chain on its outcome and its one
report, for every door that serves a call.

The executor's own hooks run them, with a room context built when they fire;
the framework events they emit go where the turn's go (RFC §10.6.2).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import SyncPipelineResult
from roomkit.core.task_utils import shielded
from roomkit.models.enums import HookTrigger
from roomkit.models.tool_call import (
    ToolCallVerdict,
    observed_call_event,
    tool_call_chain_fold,
    withheld_call_event,
)
from roomkit.tools.context import turn_report_claim
from roomkit.tools.external import BeforeToolDecision, note_handler_report
from roomkit.tools.result import before_tool_use_detail, hook_errors_detail, tool_call_verdict

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.models.context import RoomContext
    from roomkit.models.tool_call import ToolCallCallback, ToolCallEvent, ToolCallObserver
    from roomkit.tools.external import BeforeToolCallback

logger = logging.getLogger("roomkit.turns")

HookContext = Callable[..., Awaitable["RoomContext | None"]]
"""The room's context for a trigger's hooks: ``(room_id, trigger, *,
carrying=None)``, ``None`` when it will not build."""

Emit = Callable[..., Awaitable[None]]
"""Emits a framework event: ``(event_type, *, room_id, channel_id, data)``."""


def before_tool_decision(event: ToolCallEvent, hook_result: Any) -> BeforeToolDecision:
    """What the BEFORE_TOOL_USE chain decided about *event*.

    The call runs with the arguments the chain ended with: a MODIFY's
    payload, or the event edited in place, each read by the next hook. A
    rewrite under ``metadata["arguments"]``, which no later hook reads, has
    the last word (RFC §9.3). Rewritten arguments that are not an object
    refuse the call. A hook that failed closed refused it too: its error
    rides the decision for the observers, never the model.
    """
    rewritten = hook_result.metadata.get("arguments")
    if "arguments" in hook_result.metadata and not isinstance(rewritten, dict):
        logger.error(
            "BEFORE_TOOL_USE hook returned non-object arguments for %s — denying tool call",
            event.name,
        )
        return BeforeToolDecision(allowed=False)
    blocked = not hook_result.allowed and not hook_result.failed_closed
    return BeforeToolDecision(
        allowed=hook_result.allowed,
        arguments=rewritten if isinstance(rewritten, dict) else _modified(event, hook_result),
        detail=before_tool_use_detail(hook_result),
        reason=hook_result.reason if blocked else None,
    )


def _modified(event: ToolCallEvent, hook_result: Any) -> dict[str, Any] | None:
    """The arguments a MODIFY left the call with, ``None`` when the chain
    ended on *event* itself, edited in place or not."""
    ended = hook_result.event
    if ended is event or not isinstance(ended, type(event)) or ended.arguments == event.arguments:
        return None
    return ended.arguments


class ToolHooks:
    """The tool-call triggers of one executor's channels.

    Args:
        hooks: The executor's hook engine.
        context: Builds the room's context for a trigger's hooks.
        emit: Emits a framework event.
    """

    def __init__(self, hooks: HookEngine, context: HookContext, emit: Emit) -> None:
        self._hooks = hooks
        self._context = context
        self._emit = emit

    # -- What a channel is handed -----------------------------------------------

    def build_tool_call_hook(self, channel_id: str) -> ToolCallCallback:
        """The ON_TOOL_CALL callback for a call a channel served.

        The returned callback runs ON_TOOL_CALL's SYNC hooks as a chain on the
        call's outcome (RFC §9.3, ``tool_call_chain_fold``) and returns their
        :class:`ToolCallVerdict` (see :func:`tool_call_verdict`). The ASYNC
        observers see the final outcome as the model reads it, or, after a
        BLOCK, the failure. Emits a ``tool_call`` framework event.

        A call nothing served arrives with no result: the chain is the hooks'
        chance to serve it, not a report on it. The observers then see the
        result a hook supplied, or the block; when none serves it, the channel
        reports the failure itself, once (RFC §9.3).
        """

        async def _callback(
            event: ToolCallEvent, *, claim: Callable[[], bool] | None = None
        ) -> ToolCallVerdict | None:
            return await self.judge_tool_call(event, channel_id, claim=claim)

        return _callback

    def build_tool_report_hook(self, channel_id: str) -> ToolCallObserver:
        """The ON_TOOL_CALL callback for a call an external handler ran.

        A report, by construction (RFC §9.3): the agent already read the
        provider's result, so nothing a hook returns reaches it, no rewrite is
        folded in, and a BLOCK withholds nothing. The observers see the
        provider's outcome and its own ``is_error``, a block included. The
        channel's turn that announced the call claims its one report once the
        observers hear it.
        """

        async def _callback(event: ToolCallEvent) -> None:
            # The turn that announced the call claims its one report where
            # the observers hear it (RFC §9.3).
            turn_claim = turn_report_claim(event.tool_call_id, channel_id)

            def claim() -> bool:
                # Where the observers hear it: a handler that raises past
                # this point has made the call's report, and its door does
                # not make it again.
                note_handler_report(event.tool_call_id)
                return turn_claim() if turn_claim is not None else True

            await self.report_tool_call(event, channel_id, claim=claim)

        return _callback

    def build_tool_observer_hook(self, channel_id: str) -> ToolCallObserver:
        """The ON_TOOL_CALL callback for a call that failed or was refused.

        The counterpart of :meth:`build_tool_call_hook`: it runs the ASYNC
        observers of ON_TOOL_CALL and returns nothing. A refused call has no
        result for a hook to provide or correct, and must not reach a hook
        that would serve it.
        """

        async def _callback(event: ToolCallEvent) -> None:
            claim = turn_report_claim(event.tool_call_id, channel_id)
            await self.observe_failed_tool_call(event, channel_id, claim=claim)

        return _callback

    def build_before_tool_call_hook(self, channel_id: str) -> BeforeToolCallback:
        """The BEFORE_TOOL_USE callback for a channel.

        The returned callback runs BEFORE_TOOL_USE sync hooks. If any hook
        blocks, the tool call is denied. A hook may also rewrite the call's
        arguments: by a MODIFY, or the event edited in place, which the next
        hook reads, or by returning them under ``metadata["arguments"]``, the
        mirror of what ON_TOOL_CALL does with ``metadata["result"]`` on the
        way out, which has the last word (:func:`before_tool_decision`).
        """

        async def _callback(event: ToolCallEvent) -> BeforeToolDecision:
            decision, _ = await self.decide_before_tool_use(event, channel_id)
            return decision

        return _callback

    # -- ON_TOOL_CALL -------------------------------------------------------------

    async def judge_tool_call(
        self,
        event: ToolCallEvent,
        channel_id: str,
        *,
        carrying: RoomContext | None = None,
        claim: Callable[[], bool] | None = None,
    ) -> ToolCallVerdict | None:
        """ON_TOOL_CALL's verdict on a call a channel served, its observers told.

        What :meth:`build_tool_call_hook`'s callback runs, for every channel;
        *carrying* is a context the caller already built for this call, which
        spares the room history a second read. *claim* claims the call's one
        report between the chain and the observers: when it answers ``False``
        the outcome was already reported (a cancellation that landed while
        the chain ran) and nothing more is (RFC §12.4).
        """
        if not event.room_id:
            return None
        chain = await self.run_tool_call_chain(event, event.room_id, carrying=carrying)
        if chain is None:
            unreachable = self._unreachable_tool_call_verdict(event.room_id, event)
            # The framework event says what the model reads: the failure of
            # a hook that fails closed (RFC §9.3).
            reported = (
                withheld_call_event(event, str(unreachable.result)) if unreachable else event
            )
            await self._report_unreachable_tool_call(reported, channel_id, claim)
            return unreachable
        hook_result, context = chain
        verdict = tool_call_verdict(hook_result, event)
        read = verdict.result if verdict.result is not None else event.result
        if read is None:
            # Served by nothing: the channel reports the failure, once,
            # with its own framework event, and what any hook that
            # failed said for the observers.
            return replace(verdict, error_detail=hook_errors_detail(hook_result))
        if claim is not None and not claim():
            return verdict
        # Observers and the framework event see what the model reads: the
        # result, or the failure of a call a hook withheld (RFC §9.3). The
        # report is claimed now: it is made to its end, a cut of the call
        # meanwhile included, or the call would have none.
        observed = observed_call_event(hook_result, event, read)
        await shielded(self.tell_tool_call(observed, context, channel_id))
        return verdict

    async def tell_tool_call(
        self, observed: ToolCallEvent, context: RoomContext | None, channel_id: str
    ) -> None:
        """Tell ON_TOOL_CALL's observers and the framework event of a call
        whose one report was claimed."""
        if context is not None:
            await self._observe_tool_call(observed, context)
        await self._emit_tool_call_event(observed, channel_id)

    async def run_tool_call_chain(
        self, event: ToolCallEvent, room_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[SyncPipelineResult, RoomContext | None] | None:
        """ON_TOOL_CALL's SYNC chain on a served call, and the context it ran with.

        The one runner of the chain, for every channel (RFC §9.3). With no
        ON_TOOL_CALL hook registered the call stands as served, and no
        context is built. ``None`` when the context would not build.
        *carrying* is a context the caller already built for this call, which
        spares the room history a second read.
        """
        if not self._hooks.has_hooks_for(room_id, HookTrigger.ON_TOOL_CALL, event):
            return SyncPipelineResult(event=event), None
        context = await self._context(room_id, HookTrigger.ON_TOOL_CALL, carrying=carrying)
        if context is None:
            return None
        hook_result = await self._hooks.run_sync_hooks(
            room_id,
            HookTrigger.ON_TOOL_CALL,
            event,
            context,
            fold=tool_call_chain_fold(event),
            fire_observers=False,
        )
        return hook_result, context

    async def report_tool_call(
        self,
        event: ToolCallEvent,
        channel_id: str,
        *,
        claim: Callable[[], bool] | None = None,
    ) -> None:
        """Fire ON_TOOL_CALL as a report on a call whose outcome the model already read.

        Every hook runs and nothing it returns is applied (RFC §9.3): the
        observers see *event* as it stands, a BLOCK included. For a call an
        external handler or a provider ran. A call the turn cut or a
        gate or handler refused never ran: its ASYNC observers alone hear of
        it, as of a local call cut or refused.
        *claim* claims the call's one report between the chain and the
        observers, as :meth:`judge_tool_call` does: a report cut while the
        chain ran is still owed, and one the observers heard is made.
        """
        if not event.room_id:
            return
        if event.cancelled or event.refused:
            # Neither ran: the observers alone hear of it, on every door.
            await self.observe_failed_tool_call(event, channel_id, claim=claim)
            return
        context: RoomContext | None = None
        if self._hooks.has_hooks(HookTrigger.ON_TOOL_CALL):
            # The one runner of the chain, so each hook sees the call as the
            # previous one left it, whichever form its rewrite took; nothing it
            # returns reaches the agent.
            chain = await self.run_tool_call_chain(event, event.room_id)
            if chain is None:
                await self._report_unreachable_tool_call(event, channel_id, claim)
                return
            _, context = chain
        if claim is not None and not claim():
            return
        if context is not None:
            await self._observe_tool_call(event, context)
        await self._emit_tool_call_event(event, channel_id)

    async def observe_failed_tool_call(
        self,
        event: ToolCallEvent,
        channel_id: str,
        *,
        claim: Callable[[], bool] | None = None,
    ) -> None:
        """Tell ON_TOOL_CALL's ASYNC observers a call failed, was refused or was
        cancelled, or one whose report a cut left unmade, and emit its
        ``tool_call`` framework event, both as *event* stands (RFC §9.3).

        The one report of such a call, for every channel: nothing that could
        serve the call reads it. *claim* claims that report once the context
        the observers read is built, as every report runner does: a report cut
        before then is still owed.
        """
        if not event.room_id:
            return
        context: RoomContext | None = None
        if self._hooks.has_hooks_for(event.room_id, HookTrigger.ON_TOOL_CALL, event):
            context = await self._context(event.room_id, HookTrigger.ON_TOOL_CALL)
            if context is None:
                # The observers need the context that failed; the framework
                # event still reports the call once, as for a served call.
                await self._report_unreachable_tool_call(event, channel_id, claim)
                return
        if claim is not None and not claim():
            return
        if context is not None:
            await self._observe_tool_call(event, context)
        await self._emit_tool_call_event(event, channel_id)

    async def _observe_tool_call(self, event: ToolCallEvent, context: RoomContext) -> None:
        """Fire ON_TOOL_CALL's ASYNC observers alone on *event*."""
        await self._hooks.run_observers(
            str(event.room_id),
            HookTrigger.ON_TOOL_CALL,
            event,
            context,
        )

    async def _emit_tool_call_event(self, event: ToolCallEvent, channel_id: str) -> None:
        """The ``tool_call`` framework event of one reported call, its failure
        and cancellation markers included, whichever path reported it."""
        data: dict[str, Any] = {
            "tool_name": event.name,
            "tool_call_id": event.tool_call_id,
            "channel_type": str(event.channel_type),
        }
        if event.is_error:
            data["is_error"] = True
        if event.cancelled:
            data["cancelled"] = True
        if event.refused:
            data["refused"] = True
        if event.refused_but_ran:
            data["refused_but_ran"] = True
        await self._emit("tool_call", room_id=event.room_id, channel_id=channel_id, data=data)

    async def _report_unreachable_tool_call(
        self, event: ToolCallEvent, channel_id: str, claim: Callable[[], bool] | None = None
    ) -> None:
        """Report a call whose ON_TOOL_CALL hooks could not run, as *event* stands.

        The observers need the context that failed: the ``tool_call``
        framework event is then the call's one report, on every channel,
        saying what the model reads (a fail-closed hook's failure included,
        which the judge hands it) (RFC §9.3). A call with no result yet is the
        channel's to report, as the failure it is.
        """
        if event.result is not None and (claim is None or claim()):
            await self._emit_tool_call_event(event, channel_id)

    def _unreachable_tool_call_verdict(
        self, room_id: str, event: ToolCallEvent
    ) -> ToolCallVerdict | None:
        """The verdict on a call whose ON_TOOL_CALL hooks could not run.

        None, keeping the result, unless a hook there fails closed: then the
        result is withheld as that hook's own failure would withhold it, since
        the redaction it exists for must not be skipped.
        """
        closed = self._hooks.fail_closed_hook(room_id, HookTrigger.ON_TOOL_CALL, event)
        if closed is None:
            return None
        reason = json.dumps({"error": f"hook_error:{closed}"})
        return ToolCallVerdict(result=reason, blocked=True)

    # -- BEFORE_TOOL_USE ------------------------------------------------------------

    async def decide_before_tool_use(
        self, event: ToolCallEvent, channel_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[BeforeToolDecision, RoomContext | None]:
        """What BEFORE_TOOL_USE decides about *event*, for every channel, and
        the context its hooks ran with, ``None`` when none was built.

        *carrying* is a context the caller already built for this call. A
        context that will not build denies the call: an authorization failure
        MUST NOT silently permit it (RFC §9.3).
        """
        if not event.room_id:
            return BeforeToolDecision(allowed=True), None  # Allow if no room context
        ran = await self._run_before_tool_use(event, event.room_id, carrying=carrying)
        if ran is None:
            logger.warning("BEFORE_TOOL_USE could not run: tool %s denied", event.name)
            return BeforeToolDecision(allowed=False), None
        hook_result, context = ran
        await self._emit(
            "before_tool_use",
            room_id=event.room_id,
            channel_id=channel_id,
            data={
                "tool_name": event.name,
                "tool_call_id": event.tool_call_id,
                "allowed": hook_result.allowed,
                "reason": hook_result.reason,
            },
        )
        return before_tool_decision(event, hook_result), context

    async def _run_before_tool_use(
        self, event: ToolCallEvent, room_id: str, *, carrying: RoomContext | None = None
    ) -> tuple[SyncPipelineResult, RoomContext | None] | None:
        """What the BEFORE_TOOL_USE hooks decide about *event*, and the context
        they ran with; ``None`` when the room's context could not be built.

        With no hook registered there is nothing to run and no context is built.
        """
        if not self._hooks.has_hooks_for(room_id, HookTrigger.BEFORE_TOOL_USE, event):
            return SyncPipelineResult(event=event), None
        context = await self._context(room_id, HookTrigger.BEFORE_TOOL_USE, carrying=carrying)
        if context is None:
            return None
        hook_result = await self._hooks.run_sync_hooks(
            room_id,
            HookTrigger.BEFORE_TOOL_USE,
            event,
            context,
        )
        return hook_result, context
