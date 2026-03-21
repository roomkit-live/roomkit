"""AIChannel mixin for publishing ephemeral tool-call and thinking events."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from roomkit.providers.ai.base import AIToolResultPart
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType

if TYPE_CHECKING:
    from roomkit.realtime.base import RealtimeBackend

logger = logging.getLogger("roomkit.channels.ai")


class AIEventsMixin:
    """Publishes tool-call and thinking ephemeral events over the realtime backend."""

    _realtime: RealtimeBackend | None
    channel_id: str

    async def _publish_tool_event(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = None,
    ) -> None:
        """Publish a tool call ephemeral event. Best-effort, never breaks the loop."""
        if self._realtime is None or not room_id:
            return
        result_preview = 500
        if event_type == EphemeralEventType.TOOL_CALL_START:
            tc_data = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ]
        else:  # TOOL_CALL_END — tool_calls are AIToolResultPart
            tc_data = [
                {
                    "id": tc.tool_call_id,
                    "name": tc.name,
                    "result": tc.result[:result_preview]
                    if len(tc.result) > result_preview
                    else tc.result,
                }
                for tc in tool_calls
                if isinstance(tc, AIToolResultPart)
            ]
        data: dict[str, Any] = {
            "tool_calls": tc_data,
            "round": round_idx,
            "channel_id": self.channel_id,
        }
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        try:
            await self._realtime.publish_to_room(
                room_id,
                EphemeralEvent(
                    room_id=room_id,
                    type=event_type,
                    user_id=self.channel_id,
                    channel_id=self.channel_id,
                    data=data,
                ),
            )
        except Exception:
            logger.debug("Failed to publish tool call event", exc_info=True)

    async def _publish_thinking_event(
        self,
        event_type: EphemeralEventType,
        room_id: str,
        thinking: str,
        round_idx: int,
    ) -> None:
        """Publish a thinking ephemeral event. Best-effort, never breaks the loop."""
        if self._realtime is None or not room_id:
            return
        preview_limit = 1000
        data: dict[str, Any] = {
            "thinking": thinking[:preview_limit] if len(thinking) > preview_limit else thinking,
            "thinking_length": len(thinking),
            "round": round_idx,
            "channel_id": self.channel_id,
        }
        try:
            await self._realtime.publish_to_room(
                room_id,
                EphemeralEvent(
                    room_id=room_id,
                    type=event_type,
                    user_id=self.channel_id,
                    channel_id=self.channel_id,
                    data=data,
                ),
            )
        except Exception:
            logger.debug("Failed to publish thinking event", exc_info=True)
