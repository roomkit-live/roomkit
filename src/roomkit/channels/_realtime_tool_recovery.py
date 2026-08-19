"""Tool-call-in-text recovery for RealtimeVoiceChannel.

Some voice models (notably Gemini Live) occasionally emit tool calls as
spoken text instead of using the function calling API.  This mixin detects
the ``call:{name}{key:value,...}`` pattern in assistant transcriptions,
parses the arguments, and dispatches the tool call through the normal
handler pipeline — behind the same pre-execution gate as a call that came
through the function calling API, because arguments rebuilt from free text
are the least trustworthy the channel handles.

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

from roomkit.channels._realtime_tools import result_text
from roomkit.models.enums import ChannelType
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
    _tool_result_max_length: int
    _provider: RealtimeVoiceProvider
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    async def _authorize_realtime_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
    ) -> tuple[dict[str, Any], str | None]: ...

    async def _fire_tool_hook(
        self,
        tool_event: Any,
        room_id: str,
        handler_result: str | None,
        name: str,
        call_id: str,
        session: VoiceSession,
    ) -> str: ...

    def _truncate_tool_result(
        self, result_str: str, name: str, call_id: str, session_id: str
    ) -> str: ...


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
    _tool_result_max_length: int
    _provider: RealtimeVoiceProvider
    _framework: RoomKit | None
    channel_id: str
    _telemetry_provider: Any

    _track_task: Any  # cross-mixin
    _authorize_realtime_tool: Any  # cross-mixin (RealtimeToolsMixin)
    _fire_tool_hook: Any  # cross-mixin (RealtimeToolsMixin)
    _truncate_tool_result: Any  # cross-mixin (RealtimeToolsMixin)

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
            "Recovered tool call from assistant text: tool=%s, args=%s, session=%s, raw=%.300s",
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
        return {t["name"] for t in tools if isinstance(t, dict) and "name" in t}

    def _tool_param_names(self, tool_name: str, session_id: str) -> list[str]:
        with self._state_lock:
            session_tools = self._session_tools.get(session_id)
        tools = session_tools or self._tools or []
        for t in tools:
            if isinstance(t, dict) and t.get("name") == tool_name:
                return list(t.get("parameters", {}).get("properties", {}).keys())
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

    async def _inject_recovered_result(
        self,
        session: VoiceSession,
        tool_name: str,
        call_id: str,
        result_str: str,
        *,
        denied: bool = False,
    ) -> None:
        """Hand an outcome back to the model as silent context.

        Never ``submit_tool_result``: the model spoke the call instead of
        issuing it, so it has no pending ``FunctionResponse`` to answer. A
        denial travels the same way as a result — the model reads why it was
        refused and can correct itself on its next turn.

        Oversized results go through the channel's own truncation, so this path
        honours the host's ``tool_result_max_length`` and the model is told the
        result was cut instead of reading a sentence that stops mid-word.
        """
        summary = result_str
        if len(summary) > self._tool_result_max_length:
            summary = self._truncate_tool_result(summary, tool_name, call_id, session.id)
        verb = "denied" if denied else "completed"
        await self._provider.inject_text(
            session,
            f"[Tool {tool_name} {verb}: {summary}]",
            role="user",
            silent=True,
        )

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
            # Same pre-execution gate as a call that arrived through the
            # function calling API (_realtime_tools._handle_tool_call): the
            # arguments here were reconstructed from free text, so they are
            # less trustworthy than a real function call's, not more.
            arguments, denial = await self._authorize_realtime_tool(
                tool_name, arguments, call_id, room_id, session
            )
            if denial is not None:
                await self._inject_recovered_result(
                    session, tool_name, call_id, denial, denied=True
                )
                # The recovery span is the only signal this path emits, so a
                # refusal has to be legible in a trace, not just in the logs.
                telemetry.end_span(span_id, attributes={Attr.REALTIME_TOOL_DENIED: True})
                logger.info(
                    "Recovered tool %s(%s) denied before execution for session %s",
                    tool_name,
                    call_id,
                    session.id,
                )
                return

            # Run tool_handler.
            handler_result: str | None = None
            if self._tool_handler is not None:
                from roomkit.channels._realtime_context import _current_voice_session

                token = _current_voice_session.set(session)
                try:
                    raw = await self._tool_handler(tool_name, arguments)
                finally:
                    _current_voice_session.reset(token)
                handler_result = result_text(raw)

            # Fire ON_TOOL_CALL hook (for observability / overrides).
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

            # Same ON_TOOL_CALL dispatch as the API path, so a hook that
            # raises is reported to the model rather than swallowed, and a
            # recovered call shows up in the framework event feed like any
            # other tool call.
            if self._framework and room_id:
                result_str = await self._fire_tool_hook(
                    tool_event, room_id, handler_result, tool_name, call_id, session
                )
            elif handler_result is not None:
                result_str = handler_result
            else:
                result_str = json.dumps({"status": "ok"})

            await self._inject_recovered_result(session, tool_name, call_id, result_str)

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
