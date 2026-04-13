"""Tool-call-in-text recovery for RealtimeVoiceChannel.

Some voice models (notably Gemini Live) occasionally emit tool calls as
spoken text instead of using the function calling API.  This mixin detects
the ``call:{name}{key:value,...}`` pattern in assistant transcriptions,
parses the arguments, and dispatches the tool call through the normal
handler pipeline.

Because the model did not issue a real function call, we do NOT call
``submit_tool_result`` on the provider.  Instead, the tool result is
injected back as silent text context so the model can reference it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.telemetry.base import Attr, SpanKind

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")

# Matches ``call:tool_name{`` with optional leading text.
_TEXT_TOOL_CALL_RE = re.compile(r"call:(\w+)\s*\{(.+)", re.DOTALL)


@runtime_checkable
class RealtimeToolRecoveryHost(Protocol):
    """Contract: capabilities a host class must provide for this mixin.

    Attributes come from ``RealtimeToolsMixin`` and the channel ``__init__``.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _tools: list[dict[str, Any]] | None
    _session_tools: dict[str, list[dict[str, Any]]]
    _tool_handler: Any
    _tool_recovery_enabled: bool
    _provider: RealtimeVoiceProvider
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...


class RealtimeToolRecoveryMixin:
    """Detect and recover tool calls that a voice model emitted as text.

    Host contract: :class:`RealtimeToolRecoveryHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _tools: list[dict[str, Any]] | None
    _session_tools: dict[str, list[dict[str, Any]]]
    _tool_handler: Any
    _tool_recovery_enabled: bool
    _provider: RealtimeVoiceProvider
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    _track_task: Any  # cross-mixin

    # ------------------------------------------------------------------
    # Public entry point (called from _realtime_transcription.py)
    # ------------------------------------------------------------------

    def _try_recover_tool_call_from_text(
        self,
        session: VoiceSession,
        text: str,
    ) -> tuple[bool, str | None]:
        """Detect a tool call in *text* and dispatch it if found.

        Returns ``(recovered, remaining_text)``:

        - ``(False, None)`` — no tool call detected, nothing changed.
        - ``(True, None)``  — entire text was a tool call, suppress it.
        - ``(True, "...")``  — tool call found; remaining speech to emit.
        """
        if not self._tool_recovery_enabled:
            return False, None

        match = _TEXT_TOOL_CALL_RE.search(text)
        if not match:
            return False, None

        tool_name = match.group(1)
        known = self._known_tool_names(session.id)
        if tool_name not in known:
            return False, None

        raw_args = match.group(2)
        param_names = self._tool_param_names(tool_name, session.id)
        arguments = _parse_args(raw_args, param_names)
        # Coerce string values to schema types (boolean, integer, number)
        param_types = self._tool_param_types(tool_name, session.id)
        arguments = _coerce_types(arguments, param_types)

        # Extract any leading speech before "call:"
        prefix = text[: match.start()].strip()
        remaining = prefix if prefix else None

        logger.warning(
            "Recovered tool call from assistant text: tool=%s, args=%s, "
            "session=%s, raw=%.300s",
            tool_name,
            list(arguments.keys()),
            session.id,
            text,
        )

        # Dispatch asynchronously — mirrors _on_provider_tool_call pattern.
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False, None

        self._track_task(
            loop,
            self._dispatch_recovered_tool_call(session, tool_name, arguments, text),
            name=f"rt_tool_recovery:{session.id}:{tool_name}",
        )
        return True, remaining

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _known_tool_names(self, session_id: str) -> set[str]:
        with self._state_lock:
            session_tools = self._session_tools.get(session_id)
        tools = session_tools or self._tools or []
        return {
            t["name"] for t in tools if isinstance(t, dict) and "name" in t
        }

    def _tool_param_names(self, tool_name: str, session_id: str) -> list[str]:
        with self._state_lock:
            session_tools = self._session_tools.get(session_id)
        tools = session_tools or self._tools or []
        for t in tools:
            if isinstance(t, dict) and t.get("name") == tool_name:
                return list(
                    t.get("parameters", {}).get("properties", {}).keys()
                )
        return []

    def _tool_param_types(self, tool_name: str, session_id: str) -> dict[str, str]:
        """Return ``{param_name: json_type}`` for the given tool."""
        with self._state_lock:
            session_tools = self._session_tools.get(session_id)
        tools = session_tools or self._tools or []
        for t in tools:
            if isinstance(t, dict) and t.get("name") == tool_name:
                props = t.get("parameters", {}).get("properties", {})
                return {k: v.get("type", "string") for k, v in props.items()}
        return {}

    async def _dispatch_recovered_tool_call(
        self,
        session: VoiceSession,
        tool_name: str,
        arguments: dict[str, Any],
        raw_text: str,
    ) -> None:
        """Execute a recovered tool call and inject the result as context."""
        call_id = f"recovered-{uuid4().hex[:12]}"

        with self._state_lock:
            room_id = self._session_rooms.get(session.id)

        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            SpanKind.REALTIME_TOOL_RECOVERY,
            f"recovered_tool:{tool_name}",
            attributes={
                Attr.REALTIME_TOOL_NAME: tool_name,
            },
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
        )

        try:
            # Step 1: Run tool_handler
            handler_result: str | None = None
            if self._tool_handler is not None:
                from roomkit.channels._realtime_context import _current_voice_session

                token = _current_voice_session.set(session)
                try:
                    raw = await self._tool_handler(tool_name, arguments)
                finally:
                    _current_voice_session.reset(token)
                handler_result = raw if isinstance(raw, str) else json.dumps(raw)

            # Step 2: Fire ON_TOOL_CALL hook (for observability / overrides)
            from roomkit.models.tool_call import ToolCallEvent

            tool_event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.REALTIME_VOICE,
                tool_call_id=call_id,
                name=tool_name,
                arguments=arguments,
                result=handler_result,
                room_id=room_id,
                session=session,
            )

            result_str = handler_result or json.dumps({"status": "ok"})
            if self._framework and room_id:
                context = await self._framework._build_context(room_id)
                hook_result = await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.ON_TOOL_CALL,
                    tool_event,
                    context,
                    skip_event_filter=True,
                )
                if not hook_result.allowed:
                    result_str = json.dumps(
                        {"error": hook_result.reason or "Tool call blocked by hook"}
                    )
                elif "result" in hook_result.metadata:
                    hook_val = hook_result.metadata["result"]
                    result_str = hook_val if isinstance(hook_val, str) else json.dumps(hook_val)

            # Step 3: Inject result as silent context (NOT submit_tool_result).
            # Gemini didn't issue the call, so it doesn't expect a FunctionResponse.
            summary = result_str[:8000] if len(result_str) > 8000 else result_str
            await self._provider.inject_text(
                session,
                f"[Tool {tool_name} completed: {summary}]",
                role="user",
                silent=True,
            )

            telemetry.end_span(span_id)
            logger.info(
                "Recovered tool %s(%s) executed for session %s (result_len=%d)",
                tool_name,
                call_id,
                session.id,
                len(result_str),
            )

        except Exception:
            telemetry.end_span(span_id, status="error", error_message=f"recovery:{tool_name}")
            logger.exception(
                "Error executing recovered tool call %s for session %s",
                tool_name,
                session.id,
            )


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def _parse_args(raw: str, param_names: list[str]) -> dict[str, Any]:
    """Parse ``key:value,...`` text using known parameter names as delimiters.

    Finds the *first* occurrence of each ``param_name:`` in *raw*, sorts
    by position, and slices values between consecutive boundaries.
    This avoids false splits when a value contains a substring like
    ``task:`` (only the first, true boundary is used per param).
    """
    if not param_names or not raw:
        return {}

    # Find the first occurrence of each param followed by ':'
    positions: list[tuple[int, int, str]] = []
    for name in param_names:
        pattern = re.compile(re.escape(name) + r"\s*:")
        match = pattern.search(raw)
        if match:
            positions.append((match.start(), match.end(), name))

    if not positions:
        return {}

    positions.sort(key=lambda x: x[0])

    args: dict[str, Any] = {}
    for i, (_start, colon_end, name) in enumerate(positions):
        value_end = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
        value = raw[colon_end:value_end].strip()
        # Strip trailing delimiters that separate params
        value = value.rstrip(",").rstrip("}").rstrip(",").strip()
        if value:
            args[name] = value

    return args


def _coerce_types(args: dict[str, Any], param_types: dict[str, str]) -> dict[str, Any]:
    """Coerce string values to their schema types (best-effort)."""
    for key, value in list(args.items()):
        if not isinstance(value, str):
            continue
        expected = param_types.get(key, "string")
        if expected == "boolean":
            args[key] = value.lower() in ("true", "1", "yes")
        elif expected == "integer":
            with contextlib.suppress(ValueError):
                args[key] = int(value)
        elif expected == "number":
            with contextlib.suppress(ValueError):
                args[key] = float(value)
    return args
