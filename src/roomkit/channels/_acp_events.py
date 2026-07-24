"""Mapping between ACP session updates and RoomKit stream/realtime events."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from roomkit.channels._acp_client import (
    _SDK,
    _model_dump,
    _option_kind,
    _result_text,
    _ToolState,
    _TurnState,
)
from roomkit.models.streaming import (
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType

if TYPE_CHECKING:
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.tools.external import ExternalToolHandler

logger = logging.getLogger("roomkit.channels.acp")


class ACPEventsMixin:
    """Consume agent updates and enforce the ACP permission boundary."""

    channel_id: str
    _turns: dict[str, _TurnState]
    _session_rooms: dict[str, str]
    _external_tool_handler: ExternalToolHandler | None
    _realtime: RealtimeBackend | None

    def _sdk(self) -> _SDK:
        raise NotImplementedError

    async def _receive_update(self, session_id: str, update: Any) -> None:
        update_type = str(getattr(update, "session_update", ""))
        handler = self._UPDATE_HANDLERS.get(update_type)
        if handler is not None:
            await handler(self, session_id, update)

    async def _on_message_chunk(self, session_id: str, update: Any) -> None:
        turn = self._turns.get(session_id)
        text = getattr(getattr(update, "content", None), "text", None)
        if turn is not None and isinstance(text, str) and text:
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
        room_id = self._session_rooms.get(session_id)
        if room_id is None:
            return
        await self._publish(
            room_id,
            EphemeralEventType.CUSTOM,
            {
                "type": "acp_usage",
                "session_id": session_id,
                "usage": _model_dump(update),
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

    async def _emit_tool_end(
        self,
        turn: _TurnState | None,
        room_id: str | None,
        tool: _ToolState,
        status: str,
    ) -> None:
        if tool.finished:
            return
        tool.finished = True
        duration_ms = max(0, int((time.monotonic() - tool.started_at) * 1000))
        result = tool.raw_output
        if result is None and tool.content is not None:
            result = _model_dump(tool.content)
        marker_status = "failed" if status == "failed" else "completed"
        error = _result_text(result) if marker_status == "failed" and result is not None else None
        if turn is not None:
            turn.queue.put_nowait(
                ToolCallEndMarker(
                    tool_name=tool.name,
                    tool_id=tool.tool_id,
                    arguments=tool.arguments,
                    result=_model_dump(result),
                    status=marker_status,
                    duration_ms=duration_ms,
                    error=error,
                )
            )
        if room_id is not None:
            await self._publish(
                room_id,
                EphemeralEventType.TOOL_CALL_END,
                {
                    "tool_calls": [
                        {
                            "id": tool.tool_id,
                            "name": tool.name,
                            "result": _result_text(result)[:500],
                            "status": marker_status,
                        }
                    ],
                    "round": 0,
                    "duration_ms": duration_ms,
                },
            )
        if self._external_tool_handler is not None:
            try:
                await self._external_tool_handler.on_tool_result(
                    tool.name,
                    tool.arguments,
                    _result_text(result),
                    is_error=marker_status == "failed",
                    tool_call_id=tool.tool_id,
                    room_id=room_id,
                )
            except Exception:
                logger.exception("ACP external tool-result handler failed")

    async def _request_permission(
        self,
        session_id: str,
        tool_call: Any,
        options: list[Any],
    ) -> Any:
        sdk = self._sdk()
        room_id = self._session_rooms.get(session_id)
        turn = self._turns.get(session_id)
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

        approved = False
        if self._external_tool_handler is not None:
            try:
                decision = await self._external_tool_handler.process_tool_call(
                    tool_name,
                    arguments,
                    tool_call_id=tool_id,
                    session_id=session_id,
                    room_id=room_id,
                )
                approved = decision.approved
                if decision.modified_input is not None or decision.result is not None:
                    logger.warning(
                        "ACP cannot apply ExternalToolHandler input/result overrides; "
                        "rejecting tool call %s",
                        tool_id,
                    )
                    approved = False
            except Exception:
                logger.exception("ACP external permission handler failed")

        preferred = (
            ("allow_once", "allow_always") if approved else ("reject_once", "reject_always")
        )
        for kind in preferred:
            option = next((item for item in options if _option_kind(item) == kind), None)
            if option is not None:
                return sdk.schema.RequestPermissionResponse(
                    outcome=sdk.schema.AllowedOutcome(
                        outcome="selected",
                        option_id=option.option_id,
                    )
                )
        return sdk.schema.RequestPermissionResponse(
            outcome=sdk.schema.DeniedOutcome(outcome="cancelled")
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
