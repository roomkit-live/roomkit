"""Tool call handling for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.telemetry.base import Attr, SpanKind

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")


@runtime_checkable
class RealtimeToolsHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeToolsMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _session_spans: Active telemetry session span per session.
        _turn_spans: Active telemetry turn span per session.
        _session_tools: Per-session tool definitions.
        _tool_handler: User-provided tool handler callback.
        _tools: Default tool definitions.
        _mute_on_tool_call: Whether to mute mic during tool execution.
        _tool_result_max_length: Max characters for tool result.
        _skill_support: Skill infrastructure support.
        _provider: The realtime voice provider.
        _transport: The voice backend transport.
        _framework: The RoomKit framework instance (or None).
        channel_id: Channel identifier.
        _telemetry_provider: Telemetry provider for spans.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _tool_handler: Any
    _tools: Any
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...


class RealtimeToolsMixin:
    """Tool call execution for RealtimeVoiceChannel.

    Host contract: :class:`RealtimeToolsHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _tool_handler: Any
    _tools: Any
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    _track_task: Any  # see RealtimeToolsHost — cross-mixin

    def _on_provider_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Handle tool call from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_tool_call(session, call_id, name, arguments),
            name=f"rt_tool_call:{session.id}:{call_id}",
        )

    async def _handle_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Execute a tool call and submit the result to the provider.

        If a ``tool_handler`` was provided, it is called directly.
        The ``ON_TOOL_CALL`` hook is then fired (handler result, if any,
        is passed as ``event.result`` so the hook can observe or override).
        """
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
            _rt_parent = self._session_spans.get(session.id)
            parent = self._turn_spans.get(session.id) or _rt_parent

        from roomkit.telemetry.context import reset_span, set_current_span

        _rt_tok = set_current_span(_rt_parent) if _rt_parent else None

        telemetry = self._telemetry_provider
        tool_span_id = telemetry.start_span(
            SpanKind.REALTIME_TOOL_CALL,
            f"realtime_tool:{name}",
            parent_id=parent,
            attributes={Attr.REALTIME_TOOL_NAME: name},
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
        )

        if self._mute_on_tool_call and self._transport is not None:
            self._transport.set_input_muted(session, True)

        try:
            result_str: str

            # Step 0: Skill infrastructure tools — handle internally
            if self._skill_support and self._skill_support.is_skill_tool(name):
                result_str = await self._skill_support.handle_tool_call(
                    name, arguments, session.id
                )
                if name == TOOL_ACTIVATE_SKILL:
                    base = self._session_tools.get(session.id, self._tools or [])
                    all_tools = self._skill_support.skill_tool_dicts() + base
                    updated = self._skill_support.newly_visible_after_activation(
                        all_tools, session.id, arguments.get("name", "")
                    )
                    if updated is not None:
                        await self._provider.reconfigure(session, tools=updated)
                await self._provider.submit_tool_result(session, call_id, result_str)
                telemetry.end_span(tool_span_id)
                logger.info(
                    "Skill tool %s(%s) handled for session %s",
                    name,
                    call_id,
                    session.id,
                )
                return

            # Step 1: Run tool_handler (if exists)
            handler_result: str | None = None
            if self._tool_handler is not None:
                logger.info(
                    "Executing tool %s(%s) via handler for session %s",
                    name,
                    call_id,
                    session.id,
                )
                from roomkit.channels._realtime_context import _current_voice_session

                token = _current_voice_session.set(session)
                try:
                    raw = await self._tool_handler(name, arguments)
                finally:
                    _current_voice_session.reset(token)
                handler_result = raw if isinstance(raw, str) else json.dumps(raw)

            # Step 2: Run ON_TOOL_CALL hook (if framework + room)
            from roomkit.models.tool_call import ToolCallEvent

            tool_event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.REALTIME_VOICE,
                tool_call_id=call_id,
                name=name,
                arguments=arguments,
                result=handler_result,
                room_id=room_id,
                session=session,
            )

            if self._framework and room_id:
                result_str = await self._fire_tool_hook(
                    tool_event, room_id, handler_result, name, call_id, session
                )
            elif handler_result is not None:
                result_str = handler_result
            else:
                result_str = json.dumps({"error": f"No handler for tool {name}"})

            if len(result_str) > self._tool_result_max_length:
                result_str = self._truncate_tool_result(result_str, name, call_id, session.id)

            await self._provider.submit_tool_result(session, call_id, result_str)

            telemetry.end_span(tool_span_id)
            logger.info(
                "Tool call %s(%s) handled for session %s",
                name,
                call_id,
                session.id,
            )

        except Exception:
            telemetry.end_span(tool_span_id, status="error", error_message=f"tool {name} failed")
            logger.exception("Error handling tool call %s for session %s", call_id, session.id)
            try:
                await self._provider.submit_tool_result(
                    session,
                    call_id,
                    json.dumps({"error": "Internal error handling tool call"}),
                )
            except Exception:
                logger.exception("Error submitting fallback tool result")
        finally:
            if self._mute_on_tool_call and self._transport is not None:
                self._transport.set_input_muted(session, False)
            if _rt_tok is not None:
                reset_span(_rt_tok)

    async def _fire_tool_hook(
        self,
        tool_event: Any,
        room_id: str,
        handler_result: str | None,
        name: str,
        call_id: str,
        session: VoiceSession,
    ) -> str:
        """Fire ON_TOOL_CALL hook and determine final result."""
        assert self._framework is not None  # guarded by caller  # noqa: S101
        context = await self._framework._build_context(room_id)
        hook_result = await self._framework.hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.ON_TOOL_CALL,
            tool_event,
            context,
            skip_event_filter=True,
        )

        if not hook_result.allowed:
            result_str = json.dumps({"error": hook_result.reason or "Tool call blocked by hook"})
        elif "result" in hook_result.metadata:
            hook_val = hook_result.metadata["result"]
            result_str = hook_val if isinstance(hook_val, str) else json.dumps(hook_val)
        elif handler_result is not None:
            result_str = handler_result
        elif hook_result.hook_errors:
            errors = "; ".join(f"{e['hook']}: {e['error']}" for e in hook_result.hook_errors)
            result_str = json.dumps(
                {
                    "error": f"Tool call failed: {errors}. Take a fresh screenshot and retry.",
                }
            )
        else:
            result_str = json.dumps({"status": "ok"})

        await self._framework._emit_framework_event(
            "tool_call",
            room_id=room_id,
            channel_id=self.channel_id,
            data={
                "tool_name": name,
                "tool_call_id": call_id,
                "channel_type": str(ChannelType.REALTIME_VOICE),
            },
        )
        return result_str

    def _truncate_tool_result(
        self,
        result_str: str,
        name: str,
        call_id: str,
        session_id: str,
    ) -> str:
        """Truncate an oversized tool result with a notice."""
        original_len = len(result_str)
        logger.warning(
            "Tool result for %s(%s) truncated from %d to %d chars (session %s)",
            name,
            call_id,
            original_len,
            self._tool_result_max_length,
            session_id,
        )
        notice = (
            f"\n... [truncated — original result was {original_len} chars. "
            "The full content has been delivered to the client.]"
        )
        return result_str[: self._tool_result_max_length - len(notice)] + notice
