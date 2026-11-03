"""AIChannel mixin for publishing ephemeral tool-call and thinking events."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from roomkit.providers.ai.base import AIToolResultPart
from roomkit.providers.ai.tool_intent import with_intent_field
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import ThinkingHook
    from roomkit.realtime.base import RealtimeBackend

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")

# Per-event cap on the ``thinking`` payload published to the realtime bus.
# Producers that must deliver longer text without loss (e.g. the streaming
# coalescer) split it into multiple events of at most this size.
THINKING_PREVIEW_LIMIT = 1000

# Characters of a tool result a TOOL_CALL_END event previews.
_RESULT_PREVIEW = 500


class AIEventsMixin(_AIChannelContract):
    """Publishes tool-call and thinking ephemeral events over the realtime backend."""

    _realtime: RealtimeBackend | None
    _thinking_hook: ThinkingHook | None
    _publish_thinking_text: bool
    channel_id: str

    async def _publish_tool_event(
        self,
        event_type: EphemeralEventType,
        room_id: str | None,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = None,
        for_calls: Sequence[Any] = (),
    ) -> None:
        """Publish a tool call ephemeral event. Best-effort, never breaks the loop.

        ``tool_calls`` carries one of three shapes, picked by ``event_type``:
        ``StreamToolCall`` for START, ``AIToolResultPart`` for END, and
        already-built dicts for DELTA — the composition frames are assembled by
        the streaming coalescer, which is the only place that knows a call's
        running size. An END's *for_calls* are the calls its parts answer, in the
        same order, for their intents.
        """
        if self._realtime is None or not room_id:
            return
        # Payload build is inside the try to honour the best-effort contract in
        # this method's docstring.
        try:
            if event_type == EphemeralEventType.TOOL_CALL_START:
                tc_data = _started_calls(tool_calls)
            elif event_type == EphemeralEventType.TOOL_CALL_DELTA:
                tc_data = _composing_calls(tool_calls)
            else:  # TOOL_CALL_END — tool_calls are AIToolResultPart
                tc_data = _ended_calls(tool_calls, for_calls)
            data: dict[str, Any] = {
                "tool_calls": tc_data,
                "round": round_idx,
                "channel_id": self.channel_id,
            }
            if duration_ms is not None:
                data["duration_ms"] = duration_ms
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
        room_id: str | None,
        thinking: str,
        round_idx: int,
    ) -> None:
        """Publish a thinking ephemeral event and fire ON_AI_THINKING.

        Best-effort, never breaks the loop. The hook fires whether or not a
        realtime backend is configured (RFC §9.2) — reasoning is observable
        without a live UI attached.
        """
        if not room_id:
            return
        if event_type == EphemeralEventType.THINKING_START and self._thinking_hook is not None:
            try:
                await self._thinking_hook(room_id, thinking, round_idx)
            except Exception:
                logger.debug("ON_AI_THINKING hook failed", exc_info=True)
        if self._realtime is None:
            return
        data: dict[str, Any] = {
            "thinking_length": len(thinking),
            "round": round_idx,
            "channel_id": self.channel_id,
        }
        if self._publish_thinking_text:
            data["thinking"] = thinking[:THINKING_PREVIEW_LIMIT]
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

    async def _publish_text_event(self, room_id: str | None, text: str, round_idx: int) -> None:
        """Publish a ``TEXT_DELTA`` of the answer being written. Best-effort."""
        if not room_id or self._realtime is None:
            return
        data: dict[str, Any] = {"text": text, "round": round_idx, "channel_id": self.channel_id}
        try:
            await self._realtime.publish_to_room(
                room_id,
                EphemeralEvent(
                    room_id=room_id,
                    type=EphemeralEventType.TEXT_DELTA,
                    user_id=self.channel_id,
                    channel_id=self.channel_id,
                    data=data,
                ),
            )
        except Exception:
            logger.debug("Failed to publish text event", exc_info=True)


def _started_calls(calls: list[Any]) -> list[dict[str, Any]]:
    """A TOOL_CALL_START's calls: each one's id, name and arguments, and its
    intent when it has one (RFC §6.4)."""
    return [
        with_intent_field(
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments},
            getattr(tc, "intent", None),
        )
        for tc in calls
    ]


def _composing_calls(calls: list[Any]) -> list[dict[str, Any]]:
    """A TOOL_CALL_DELTA's calls: each one's name and running argument size.

    Projected field by field rather than copied wholesale: the argument text
    can be megabytes or personal data, and a guarantee the publish boundary
    does not enforce is a guarantee one careless caller away from being
    false. START delivers the arguments, once, when the call is complete. An
    empty list is the terminal frame: nothing is being composed any more.
    """
    return [
        {"id": tc["id"], "name": tc["name"], "arguments_chars": tc["arguments_chars"]}
        for tc in calls
    ]


def _ended_calls(parts: list[Any], calls: Sequence[Any]) -> list[dict[str, Any]]:
    """A TOOL_CALL_END's calls: each one's result preview and status, and the
    intent of the call in *calls* at its place, when it has one.

    ``status`` as the stored row and the ACP channel's END carry it, so a
    live surface tells a failed call from a served one without reading the
    preview. ``as_text()`` flattens a multimodal result, so the preview never
    carries raw base64 image data. Matched by place, not by id: two calls of
    a round can share one (RFC §12.4).
    """
    intents = [getattr(call, "intent", None) for call in calls]
    intents += [None] * (len(parts) - len(intents))
    return [
        with_intent_field(
            {
                "id": part.tool_call_id,
                "name": part.name,
                "result": part.as_text()[:_RESULT_PREVIEW],
                "status": "failed" if part.is_error else "completed",
            },
            intent,
        )
        for part, intent in zip(parts, intents, strict=False)
        if isinstance(part, AIToolResultPart)
    ]
