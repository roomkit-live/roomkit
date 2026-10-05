"""Mapping between ACP session updates and RoomKit stream/realtime events."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from roomkit.channels._acp_client import (
    _SDK,
    _config_labels,
    _config_values,
    _model_dump,
    _option_kind,
    _result_text,
    _ToolState,
    _TurnState,
)
from roomkit.channels._acp_usage import (
    _apply_transport_usage,
    _observe_usage,
    _report_context,
    _transport_usage,
)
from roomkit.channels._tool_event_result import tool_event_result
from roomkit.core.task_utils import shielded
from roomkit.models.streaming import (
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.models.tool_call import ToolCallEvent
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
from roomkit.tools.external import detail_keyword, handler_reported, takes_keyword, warn_once
from roomkit.tools.result import cancelled_tool_error, failure_detail, tool_failure

if TYPE_CHECKING:
    from roomkit.channels.acp_transport import ACPTransport
    from roomkit.models.enums import ChannelType, ToolCallOutcome
    from roomkit.models.tool_call import ToolCallObserver
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.tools.external import ExternalToolHandler

logger = logging.getLogger("roomkit.channels.acp")

_TURN_ENDED_ERROR = "The ACP turn ended before the tool reported a result"


@dataclass(frozen=True, slots=True)
class _ToolEnd:
    """How an ACP tool call ended, as each report of its end states it."""

    result: Any
    status: Literal["completed", "failed"]
    error: str | None
    duration_ms: int
    display: Any
    """The agent's display payload for the call (ACP tool content), dumped."""
    outcome: ToolCallOutcome
    """How the call ended, as every channel names it (RFC §6.4)."""


def _end_outcome(tool: _ToolState, status: str, *, interrupted: bool) -> ToolCallOutcome:
    """The outcome of an ACP call: RoomKit's decision on its permission when
    the call did not run (refused, or failed when a handler raised deciding
    it), whether the agent closed it failed or the turn ended under it; else
    cancelled when the turn ended under it, or what the agent said."""
    if interrupted or status == "failed":
        if tool.refused:
            return "refused"
        if tool.failure is not None:
            return "failed"
    if interrupted:
        return "cancelled"
    return "failed" if status == "failed" else "served"


def _decision_detail(tool: _ToolState) -> str | None:
    """What failed in RoomKit's decision on the call's permission, for the
    observers only: its handler's failure, or what its refusal came from
    (RFC §9.3)."""
    return tool.failure or tool.refusal_detail


def _decided_error(tool: _ToolState) -> str | None:
    """The error of a call RoomKit decided, as the AI door words it: its
    refusal, or its handler's failure (RFC §9.3); ``None`` for any other
    call."""
    if tool.refused:
        return json.dumps({"error": tool.refusal or f"Tool '{tool.name}' was denied"})
    return tool.failure_error


_UNAPPLIABLE = "ACP cannot apply an approval that rewrites the call's input or result"
"""Why the channel refuses a permission its handler approved with an override."""


@dataclass(frozen=True, slots=True)
class _PermissionDecision:
    """RoomKit's decision on an ACP call's permission."""

    approved: bool = False
    failure: str | None = None
    """What failed, when the handler raised deciding it."""
    failure_error: str | None = None
    """That failure as a model reads it."""
    channel_refused: bool = False
    refusal: str | None = None
    refusal_detail: str | None = None
    """What failed, when the handler's refusal came from a failure."""


def _record_decision(tool: _ToolState, decision: _PermissionDecision) -> None:
    """Keep *decision* on its call, the last one standing: a refused call's
    end reads as refused, not as a tool that failed; one approved later can
    fail on its own. A handler that raised refused nothing: the call failed,
    and the channel reports it with what failed (RFC §9.3); the agent was
    still answered with a rejection, which a call it runs anyway went past."""
    tool.decided = True
    tool.failure = decision.failure
    tool.failure_error = decision.failure_error
    tool.rejected = not decision.approved
    tool.refused = tool.rejected and decision.failure is None
    tool.refusal = decision.refusal if tool.refused else None
    tool.refusal_detail = decision.refusal_detail if tool.refused else None
    tool.channel_refused = decision.channel_refused


def _permission_response(sdk: Any, options: list[Any], *, approved: bool) -> Any:
    """The ACP answer to a permission request: the agent's allow option when
    approved, its reject option otherwise, a denial when it offered neither."""
    preferred = ("allow_once", "allow_always") if approved else ("reject_once", "reject_always")
    for kind in preferred:
        option = next((item for item in options if _option_kind(item) == kind), None)
        if option is not None:
            return sdk.schema.RequestPermissionResponse(
                outcome=sdk.schema.AllowedOutcome(outcome="selected", option_id=option.option_id)
            )
    return sdk.schema.RequestPermissionResponse(
        outcome=sdk.schema.DeniedOutcome(outcome="cancelled")
    )


def _tool_end(
    tool: _ToolState, status: str, error: str | None, *, interrupted: bool = False
) -> _ToolEnd:
    """Read how a tool call ended off what the agent reported of it."""
    display = _model_dump(tool.content) if tool.content is not None else None
    result = tool.raw_output
    if result is None and display is not None:
        result = display
    end_status: Literal["completed", "failed"] = "failed" if status == "failed" else "completed"
    # A tool cut short has no result to explain itself with, so the caller
    # says why instead: "never returned" and "returned an error" read the
    # same in the timeline otherwise.
    if error is None and end_status == "failed" and result is not None:
        # Bounded first: the text of a raw output carrying a screenshot
        # would put its base64 in the event's error field whole.
        error = _result_text(tool_event_result(result))
    duration_ms = max(0, int((time.monotonic() - tool.started_at) * 1000))
    outcome = _end_outcome(tool, status, interrupted=interrupted)
    return _ToolEnd(result, end_status, error, duration_ms, display, outcome)


def _end_marker(tool: _ToolState, end: _ToolEnd) -> ToolCallEndMarker:
    """The stream's END marker for a tool call, which the segment writer stores."""
    # ACP's tool content is the display-intended payload (diffs, formatted
    # text); carry it beside the raw result so UI surfaces can render it —
    # structured_content is the field they read.
    structured = {"acp_content": end.display} if end.display is not None else None
    return ToolCallEndMarker(
        tool_name=tool.name,
        tool_id=tool.tool_id,
        arguments=tool.arguments,
        result=_model_dump(end.result),
        status=end.status,
        duration_ms=end.duration_ms,
        error=end.error,
        structured_content=structured,
        outcome=end.outcome,
        refused_but_ran=_ran_despite_refusal(tool, end),
    )


def _ran_despite_refusal(tool: _ToolState, end: _ToolEnd) -> bool:
    """Whether the agent ran a call whose permission RoomKit rejected, refused
    or failed deciding, and closed it completed: reported served, as it ran,
    and marked (RFC §9.3)."""
    return tool.rejected and end.outcome == "served"


def _handler_reports(handler: ExternalToolHandler, tool: _ToolState, end: _ToolEnd) -> bool:
    """Whether *handler* reports the call's end. A call the agent ran past
    RoomKit's refusal is reported with that marker: an override whose
    ``on_tool_result`` cannot take it leaves the report to the channel, never
    a report without the marker, nor none (RFC §9.3)."""
    if not _ran_despite_refusal(tool, end) or takes_keyword(
        handler.on_tool_result, "refused_but_ran"
    ):
        return True
    warn_once(
        handler,
        "refused_but_ran",
        "%s.on_tool_result takes no 'refused_but_ran': the channel reports the calls the "
        "agent runs past its refusal itself; accept **kwargs to hear them",
    )
    return False


def _handler_report(
    handler: ExternalToolHandler, room_id: str | None, tool: _ToolState, end: _ToolEnd
) -> Awaitable[None]:
    """The handler's report of a call's end: a call the turn cut has no
    result, and the handler reports it cancelled; one it refused, refused;
    any other with its result, marked when the agent ran it past a refusal
    (RFC §9.3)."""
    if end.outcome == "cancelled":
        return handler.on_tool_cancelled(
            tool.name, tool.arguments, tool_call_id=tool.tool_id, room_id=room_id
        )
    if end.outcome == "refused":
        return handler.on_tool_refused(
            tool.name,
            tool.arguments,
            _reported_body(tool, end),
            tool_call_id=tool.tool_id,
            room_id=room_id,
            **detail_keyword(handler, "on_tool_refused", _decision_detail(tool)),
        )
    return handler.on_tool_result(
        tool.name,
        tool.arguments,
        _reported_body(tool, end),
        is_error=end.status == "failed",
        tool_call_id=tool.tool_id,
        room_id=room_id,
        **_ran_past_refusal_keywords(handler, tool, end),
    )


def _ran_past_refusal_keywords(
    handler: ExternalToolHandler, tool: _ToolState, end: _ToolEnd
) -> dict[str, Any]:
    """The keywords *handler*'s ``on_tool_result`` takes for a call the
    agent ran past RoomKit's rejection: the marker, and what failed in the
    rejection as the channel's own report carries it (RFC §9.3)."""
    if not _ran_despite_refusal(tool, end):
        return {}
    detail = detail_keyword(handler, "on_tool_result", _decision_detail(tool))
    return {"refused_but_ran": True, **detail}


def _channel_decided(tool: _ToolState, end: _ToolEnd) -> bool:
    """Whether the channel, not the handler, decided a call that never ran:
    it refused a permission the handler approved with what ACP cannot apply,
    or the handler raised deciding it, and the agent reported the call
    refused or failed."""
    if end.outcome == "refused" and tool.channel_refused:
        return True
    return tool.failure is not None and end.status == "failed"


def _reported_body(tool: _ToolState, end: _ToolEnd) -> str:
    """What a call's report carries, with an external handler or without:
    a cancelled call's cancellation envelope, a failed or refused call's
    bounded error, a served call's result text (RFC §9.3)."""
    if end.outcome == "cancelled":
        return cancelled_tool_error(tool.name, "The turn ended before its result.")
    if end.status == "failed":
        return end.error or ""
    return _result_text(end.result)


class ACPEventsMixin:
    """Consume agent updates and enforce the ACP permission boundary."""

    channel_id: str
    channel_type: ChannelType
    _turns: dict[str, _TurnState]
    _session_rooms: dict[str, str]
    _session_options: dict[str, list[Any]]
    # Implemented by the other mixins; annotations, so they shadow nothing.
    _is_room_session: Callable[[str], bool]
    _sdk: Callable[[], _SDK]
    _transport: ACPTransport
    _external_tool_handler: ExternalToolHandler | None
    _tool_report_hook: ToolCallObserver | None
    _tool_observer_hook: ToolCallObserver | None
    _realtime: RealtimeBackend | None

    async def _receive_update(self, session_id: str, update: Any) -> None:
        if (turn := self._turns.get(session_id)) is not None:
            turn.activity_seen = True
        update_type = str(getattr(update, "session_update", ""))
        handler = self._UPDATE_HANDLERS.get(update_type)
        if handler is not None:
            await handler(self, session_id, update)

    async def _on_message_chunk(self, session_id: str, update: Any) -> None:
        turn = self._turns.get(session_id)
        text = getattr(getattr(update, "content", None), "text", None)
        if turn is not None and isinstance(text, str) and text:
            if not turn.segments:
                turn.segments.append([])
            turn.segments[-1].append(text)
            turn.queue.put_nowait(text)

    async def _on_thought_chunk(self, session_id: str, update: Any) -> None:
        thinking = getattr(getattr(update, "content", None), "text", None)
        if not isinstance(thinking, str) or not thinking:
            return
        turn = self._turns.get(session_id)
        if turn is not None:
            if not turn.thinking_open:
                turn.thinking_open = True
                await self._publish(
                    turn.room_id,
                    EphemeralEventType.THINKING_START,
                    {"thinking": "", "round": 0},
                )
            turn.thinking.append(thinking)
            turn.queue.put_nowait(ThinkingDeltaMarker(thinking=thinking))
        room_id = self._session_rooms.get(session_id)
        if room_id is not None:
            await self._publish(
                room_id,
                EphemeralEventType.THINKING_DELTA,
                {
                    "thinking": thinking[:1000],
                    "thinking_length": len(thinking),
                    "round": 0,
                },
            )

    async def _on_plan_update(self, session_id: str, update: Any) -> None:
        room_id = self._session_rooms.get(session_id)
        if room_id is None:
            return
        await self._publish(
            room_id,
            EphemeralEventType.CUSTOM,
            {
                "type": "acp_plan_update",
                "session_id": session_id,
                "update": _model_dump(update),
            },
        )

    async def _on_usage_update(self, session_id: str, update: Any) -> None:
        envelope = _transport_usage(update) if self._transport.provides_usage_metadata else None
        report = (
            envelope.get("usage_report")
            if envelope is not None
            else _observe_usage(
                update, _config_values(self._session_options.get(session_id)).get("model")
            )
        )
        turn = self._turns.get(session_id)
        if (
            turn is not None
            and not turn.usage_finalized
            and (envelope is None or envelope.get("session_id") == session_id)
        ):
            # Kept, not only announced: the end-of-turn report is the only
            # place this reaches a host that is not watching the ephemeral
            # stream.
            turn.context = _report_context(report)
            if envelope is not None:
                _apply_transport_usage(turn.usage_metadata, envelope, terminal=False)
            else:
                turn.usage_metadata["usage_report"] = deepcopy(report)
        room_id = self._session_rooms.get(session_id)
        # The turn keeps its own accounting above; the room's gauge follows
        # the room's session only, never a standalone turn's empty one.
        if room_id is None or not self._is_room_session(session_id):
            return
        await self._publish(
            room_id,
            EphemeralEventType.CUSTOM,
            {
                "type": "acp_usage",
                "session_id": session_id,
                "usage": _model_dump(update),
                "usage_metadata": envelope
                if envelope is not None
                else {
                    "session_id": session_id,
                    "usage_report": report,
                },
            },
        )

    async def _on_config_option_update(self, session_id: str, update: Any) -> None:
        """Track a config change the agent reports — the model, most visibly.

        Agents send this whenever a tunable moves, including changes the user
        made from inside the agent (``/model`` is handled locally by the
        agent, never reaching RoomKit as a prompt) and ones it made itself (a
        refusal fallback switching model). Recording it keeps
        :meth:`ACPChannel.session_config` truthful; the ephemeral event lets
        UI surfaces follow along live.
        """
        # Only a room's session describes the room: a standalone turn's, open
        # or already closed, would read as the room's tunables changing.
        if not self._is_room_session(session_id):
            return
        options = _model_dump(getattr(update, "config_options", None))
        if isinstance(options, list):
            self._session_options[session_id] = options
        await self._publish_config_options(session_id, options, _config_values(options))

    async def _publish_config_options(
        self,
        session_id: str,
        options: Any,
        values: dict[str, str | bool],
    ) -> None:
        """Announce a session's tunables — the state, not the delta.

        Emitted for a new session too, not only on change: a surface that
        shows the running model has nothing to show otherwise until the
        first change happens to occur.
        """
        room_id = self._session_rooms.get(session_id)
        if room_id is None:
            return
        await self._publish(
            room_id,
            EphemeralEventType.CUSTOM,
            {
                "type": "acp_config_options",
                "session_id": session_id,
                "values": values,
                "labels": _config_labels(options),
                "config_options": _model_dump(options),
            },
        )

    async def _tool_start(self, session_id: str, update: Any) -> None:
        turn = self._turns.get(session_id)
        room_id = self._session_rooms.get(session_id)
        tool = self._merge_tool(turn, update)
        if tool is None:
            return
        await self._emit_tool_start(turn, room_id, tool)
        status = str(getattr(update, "status", "") or "")
        if status in {"completed", "failed"}:
            await self._emit_tool_end(turn, room_id, tool, status)

    async def _tool_progress(self, session_id: str, update: Any) -> None:
        turn = self._turns.get(session_id)
        room_id = self._session_rooms.get(session_id)
        tool = self._merge_tool(turn, update)
        if tool is None:
            return
        await self._emit_tool_start(turn, room_id, tool)
        status = str(getattr(update, "status", "") or "")
        if status in {"completed", "failed"}:
            await self._emit_tool_end(turn, room_id, tool, status)
        elif room_id is not None:
            await self._publish(
                room_id,
                EphemeralEventType.CUSTOM,
                {
                    "type": "acp_tool_progress",
                    "session_id": session_id,
                    "tool_call": _model_dump(update),
                },
            )

    _UPDATE_HANDLERS: ClassVar[dict[str, Callable[..., Awaitable[None]]]] = {
        "agent_message_chunk": _on_message_chunk,
        "agent_thought_chunk": _on_thought_chunk,
        "tool_call": _tool_start,
        "tool_call_update": _tool_progress,
        "plan": _on_plan_update,
        "plan_update": _on_plan_update,
        "plan_removed": _on_plan_update,
        "usage_update": _on_usage_update,
        "config_option_update": _on_config_option_update,
    }

    @staticmethod
    def _merge_tool(turn: _TurnState | None, update: Any) -> _ToolState | None:
        if turn is None:
            return None
        tool_id = str(getattr(update, "tool_call_id", "") or "")
        if not tool_id:
            return None
        tool = turn.tools.setdefault(tool_id, _ToolState(tool_id=tool_id))
        title = getattr(update, "title", None)
        kind = getattr(update, "kind", None)
        if title:
            tool.name = str(title)
        elif kind and tool.name == "tool":
            tool.name = str(kind)
        raw_input = getattr(update, "raw_input", None)
        if isinstance(raw_input, Mapping):
            tool.arguments = {str(key): value for key, value in raw_input.items()}
        elif raw_input is not None:
            tool.arguments = {"value": raw_input}
        raw_output = getattr(update, "raw_output", None)
        if raw_output is not None:
            tool.raw_output = raw_output
        content = getattr(update, "content", None)
        if content is not None:
            tool.content = content
        return tool

    async def _emit_tool_start(
        self,
        turn: _TurnState | None,
        room_id: str | None,
        tool: _ToolState,
    ) -> None:
        if tool.started:
            return
        tool.started = True
        if turn is not None:
            # The call cuts the agent's text: whatever follows is the next
            # segment of the end-of-turn report.
            if turn.segments and turn.segments[-1]:
                turn.segments.append([])
            turn.queue.put_nowait(
                ToolCallStartMarker(
                    tool_name=tool.name,
                    tool_id=tool.tool_id,
                    arguments=tool.arguments,
                )
            )
        if room_id is not None:
            await self._publish(
                room_id,
                EphemeralEventType.TOOL_CALL_START,
                {
                    "tool_calls": [
                        {
                            "id": tool.tool_id,
                            "name": tool.name,
                            "arguments": tool.arguments,
                        }
                    ],
                    "round": 0,
                },
            )

    async def _close_open_tools(
        self,
        turn: _TurnState,
        room_id: str | None,
        *,
        stream: bool,
    ) -> bool:
        """Close every tool the turn started and never closed, and every call
        RoomKit decided that the agent never announced.

        A turn can die mid-tool — the agent process restarts, the node goes
        away, the user presses Stop — and the agent then never sends the
        terminal ``tool_call_update``. Without this the ``TOOL_CALL_START``
        stands alone and its card spins forever, on every reload, because the
        stored row says the call is still pending.

        ``stream`` says whether the closing markers can still reach the
        stream: they must, for the *stored* ``TOOL_CALL_END`` to exist at
        all — it is persisted from the marker. Once the consumer is gone
        (``stream=False``) only the ephemeral event can go out.

        Returns whether anything was closed, so a caller can tell a turn that
        left something open from one that ended clean. Idempotent: a tool
        already finished is skipped by :meth:`_emit_tool_end`.
        """
        # A call RoomKit decided is closed even if the agent never announced
        # it: its refusal or failure is reported, an approval cut by the
        # turn's end is reported cancelled.
        open_tools = [
            tool
            for tool in turn.tools.values()
            if not tool.finished and (tool.started or tool.decided)
        ]
        if not open_tools:
            return False
        # Said out loud: an agent that stops mid-tool leaves no other trace,
        # and whoever reads the timeline afterwards sees a failed tool without
        # knowing the turn died under it.
        logger.info(
            "ACP turn ended with %d tool call(s) unfinished in room %s: %s",
            len(open_tools),
            room_id,
            ", ".join(tool.name for tool in open_tools),
        )
        for tool in open_tools:
            await self._emit_tool_start(turn if stream else None, room_id, tool)
            await self._emit_tool_end(
                turn if stream else None,
                room_id,
                tool,
                "failed",
                error=_decided_error(tool) or _TURN_ENDED_ERROR,
                interrupted=True,
            )
        return True

    async def _emit_tool_end(
        self,
        turn: _TurnState | None,
        room_id: str | None,
        tool: _ToolState,
        status: str,
        *,
        error: str | None = None,
        interrupted: bool = False,
    ) -> None:
        """Close a tool call once, in every report its end reaches."""
        if tool.finished:
            return
        tool.finished = True
        end = _tool_end(tool, status, error, interrupted=interrupted)
        if turn is not None:
            turn.queue.put_nowait(_end_marker(tool, end))
        # The call is closed from here: its end reaches its reports even if
        # this task is cut meanwhile (RFC §9.3, one report per call).
        await shielded(self._announce_tool_end(room_id, tool, end))

    async def _announce_tool_end(
        self, room_id: str | None, tool: _ToolState, end: _ToolEnd
    ) -> None:
        """Publish a closed call's end and report it, once."""
        if room_id is not None:
            await self._publish_tool_end(room_id, tool, end)
        if _channel_decided(tool, end):
            # A call that never ran because the channel refused it or its
            # handler raised: the channel reports it, to the observers only.
            await self._report_agent_call(room_id, tool, end, observe=True)
        elif self._external_tool_handler is not None and _handler_reports(
            self._external_tool_handler, tool, end
        ):
            reported = await self._report_tool_end(self._external_tool_handler, room_id, tool, end)
            if not reported:
                # Never lost: a call whose report the handler raised before
                # making is the channel's to report, as it ended, its
                # refused_but_ran marker included (RFC §9.3).
                await self._report_agent_call(room_id, tool, end)
        elif self._tool_report_hook is not None:
            # No handler to report it: ON_TOOL_CALL still hears of every call,
            # as of a call an AI provider ran itself (RFC §9.3).
            await self._report_agent_call(room_id, tool, end)

    async def _publish_tool_end(self, room_id: str, tool: _ToolState, end: _ToolEnd) -> None:
        await self._publish(
            room_id,
            EphemeralEventType.TOOL_CALL_END,
            {
                "tool_calls": [
                    {
                        "id": tool.tool_id,
                        "name": tool.name,
                        "result": (end.error or _result_text(end.result))[:500],
                        "status": end.status,
                    }
                ],
                "round": 0,
                "duration_ms": end.duration_ms,
            },
        )

    async def _report_agent_call(
        self, room_id: str | None, tool: _ToolState, end: _ToolEnd, *, observe: bool = False
    ) -> None:
        """Report a call the agent ran to ON_TOOL_CALL, through the kit: with
        the body a handler would report (:func:`_reported_body`), refused or
        cancelled to the observers only (RFC §9.3), and to them alone when
        *observe* (a call the channel decided that never ran)."""
        report = (self._tool_observer_hook if observe else None) or self._tool_report_hook
        if report is None:
            return
        event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=self.channel_type,
            tool_call_id=tool.tool_id,
            name=tool.name,
            arguments=tool.arguments,
            result=_reported_body(tool, end),
            room_id=room_id,
            is_error=end.status == "failed",
            cancelled=end.outcome == "cancelled",
            refused=end.outcome == "refused",
            error_detail=_decision_detail(tool),
            refused_but_ran=_ran_despite_refusal(tool, end),
        )
        try:
            await report(event)
        except Exception:
            logger.exception("ACP tool-call report failed")

    @staticmethod
    async def _report_tool_end(
        handler: ExternalToolHandler, room_id: str | None, tool: _ToolState, end: _ToolEnd
    ) -> bool:
        """Hand a call's end to the handler (:func:`_handler_report`).
        ``False`` when the handler raised before its report reached
        ON_TOOL_CALL: the call's report is then the channel's (RFC §9.3)."""
        report = partial(_handler_report, handler, room_id, tool, end)
        return await handler_reported(
            report, f"reporting the ACP call {tool.tool_id}", tool.tool_id
        )

    async def _request_permission(
        self,
        session_id: str,
        tool_call: Any,
        options: list[Any],
    ) -> Any:
        sdk = self._sdk()
        room_id = self._session_rooms.get(session_id)
        turn = self._turns.get(session_id)
        if turn is not None:
            turn.activity_seen = True
        tool = self._merge_tool(turn, tool_call)
        tool_id = str(getattr(tool_call, "tool_call_id", "") or "")
        tool_name = str(getattr(tool_call, "title", "") or "") or (
            tool.name if tool is not None else "tool"
        )
        raw_input = getattr(tool_call, "raw_input", None)
        arguments = (
            {str(key): value for key, value in raw_input.items()}
            if isinstance(raw_input, Mapping)
            else (tool.arguments if tool is not None else {})
        )

        decision = await self._decide_permission(
            tool_name, arguments, tool_id=tool_id, session_id=session_id, room_id=room_id
        )
        if tool is not None:
            _record_decision(tool, decision)
        return _permission_response(sdk, options, approved=decision.approved)

    async def _decide_permission(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        tool_id: str,
        session_id: str,
        room_id: str | None,
    ) -> _PermissionDecision:
        """The external tool handler's decision on a call's permission: no
        handler refuses it; one that approves with an input or a result ACP
        cannot apply is refused by the channel; one that raises fails it."""
        handler = self._external_tool_handler
        if handler is None:
            return _PermissionDecision()
        try:
            decision = await handler.process_tool_call(
                tool_name, arguments, tool_call_id=tool_id, session_id=session_id, room_id=room_id
            )
        except Exception as exc:
            logger.exception("ACP external permission handler failed")
            return _PermissionDecision(
                failure=failure_detail(exc), failure_error=tool_failure(tool_name, exc)
            )
        if decision.modified_input is not None or decision.result is not None:
            logger.warning(
                "ACP cannot apply ExternalToolHandler input/result overrides; "
                "rejecting tool call %s",
                tool_id,
            )
            return _PermissionDecision(channel_refused=True, refusal=_UNAPPLIABLE)
        return _PermissionDecision(
            approved=decision.approved, refusal=decision.reason, refusal_detail=decision.detail
        )

    async def _publish(
        self,
        room_id: str,
        event_type: EphemeralEventType,
        data: dict[str, Any],
    ) -> None:
        if self._realtime is None:
            return
        try:
            await self._realtime.publish_to_room(
                room_id,
                EphemeralEvent(
                    room_id=room_id,
                    type=event_type,
                    user_id=self.channel_id,
                    channel_id=self.channel_id,
                    data={**data, "channel_id": self.channel_id},
                ),
            )
        except Exception:
            logger.debug("Failed to publish ACP realtime event", exc_info=True)
