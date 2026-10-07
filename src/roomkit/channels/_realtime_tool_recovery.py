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

import asyncio
import contextlib
import logging
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable
from uuid import uuid4

from roomkit.channels._realtime_tool_calls import RealtimeToolCall
from roomkit.channels._realtime_tool_executor import ToolCallHost, run_tool_call
from roomkit.telemetry.base import SpanKind
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools.result import result_text
from roomkit.voice.base import VoiceSessionState

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")

# ``call:tool_name{...}`` said as a sentence of its own that ends the utterance:
# at the start of the text or of a line, or after a sentence's end (a closing
# quote allowed, no space needed after a CJK full stop), and nothing after its
# closing brace but a final stop. Speech before it is kept; a sentence that
# mentions the form mid-way ("type call:x{...} to search") calls nothing. The
# arguments admit one level of nested braces, so a later brace in the speech
# never reads as the call's own.
_TEXT_TOOL_CALL_RE = re.compile(
    r"(?:^[ \t]*|(?<=[.!?…])[ \t\u00a0]+|(?<=[.!?…][\"'»”’)])[ \t\u00a0]+|(?<=[。！？]))"
    r"call:(\w+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}[.!?。]?\s*\Z",
    re.MULTILINE,
)

# A ``key:`` token where a key can legitimately start — at the beginning of the
# argument text or just after a comma. Anywhere else (``at 3:30``) is a value,
# and a token opening with a digit (``, 3:30 pm``) is a time, not a key.
_UNDECLARED_KEY_RE = re.compile(r"(?:^|,)\s*([A-Za-z_]\w*)\s*:")


@runtime_checkable
class RealtimeToolRecoveryHost(Protocol):
    """Contract: capabilities a host class must provide for this mixin.

    Attributes come from ``RealtimeToolsMixin`` and the channel ``__init__``.
    """

    _tool_recovery_enabled: bool
    _provider: RealtimeVoiceProvider

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _open_tool_call(self, call: RealtimeToolCall) -> bool: ...

    def _close_tool_call(self, call: RealtimeToolCall) -> None: ...

    def _tool_call_span(self, call: RealtimeToolCall, kind: Any, prefix: str) -> Any: ...

    def _session_catalogue(self, session_id: str) -> list[dict[str, Any]]: ...

    async def inject_text(
        self, session: VoiceSession, text: str, *, role: str = "user", silent: bool = False
    ) -> Any: ...


class RealtimeToolRecoveryMixin:
    """Detect and recover tool calls that a voice model emitted as text.

    Host contract: :class:`RealtimeToolRecoveryHost`.
    """

    _tool_recovery_enabled: bool
    _provider: RealtimeVoiceProvider

    _track_task: Any  # cross-mixin
    _open_tool_call: Any  # cross-mixin (RealtimeToolsMixin)
    _close_tool_call: Any  # cross-mixin (RealtimeToolsMixin)
    _tool_call_span: Any  # cross-mixin (RealtimeToolsMixin)
    _session_catalogue: Any  # cross-mixin (RealtimeToolsMixin)
    inject_text: Any  # cross-mixin (RealtimeVoiceChannel)

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
        if not self._tool_recovery_enabled or session.state == VoiceSessionState.ENDED:
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False, None

        # Booked on arrival, as the provider door books its calls: a session
        # that ends before the task runs still reports it (RFC §12.4).
        call = self._book_recovered_call(session, tool_name, arguments)
        call.task = self._track_task(
            loop,
            self._serve_recovered_call(call),
            name=f"rt_tool_recovery:{session.id}:{tool_name}",
        )
        return True, remaining

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _known_tool_names(self, session_id: str) -> set[str]:
        tools = self._session_catalogue(session_id)
        return {t["name"] for t in tools if isinstance(t, dict) and "name" in t}

    def _tool_param_names(self, tool_name: str, session_id: str) -> list[str]:
        for t in self._session_catalogue(session_id):
            if isinstance(t, dict) and t.get("name") == tool_name:
                return list(t.get("parameters", {}).get("properties", {}).keys())
        return []

    def _tool_param_types(self, tool_name: str, session_id: str) -> dict[str, str]:
        """Return ``{param_name: json_type}`` for the given tool."""
        for t in self._session_catalogue(session_id):
            if isinstance(t, dict) and t.get("name") == tool_name:
                props = t.get("parameters", {}).get("properties", {})
                return {k: v.get("type", "string") for k, v in props.items()}
        return {}

    async def _inject_recovered_result(
        self,
        session: VoiceSession,
        tool_name: str,
        result_str: str,
        *,
        verb: Literal["completed", "denied", "failed"] = "completed",
    ) -> None:
        """Hand an outcome back to the model as silent context.

        Never ``submit_tool_result``: the model spoke the call instead of
        issuing it, so it has no pending ``FunctionResponse`` to answer. A
        denial travels the same way as a result — the model reads why it was
        refused and can correct itself on its next turn. The result arrives
        bounded by the channel's ``tool_result_max_length`` (RFC §21.5).
        Injected through the channel, which announces it to
        ON_REALTIME_TEXT_INJECTED as every injection (RFC §12.4); an ended
        session takes none.
        """
        await self.inject_text(
            session,
            f"[Tool {tool_name} {verb}: {result_str}]",
            role="user",
            silent=True,
        )

    def _book_recovered_call(
        self, session: VoiceSession, tool_name: str, arguments: dict[str, Any]
    ) -> RealtimeToolCall:
        """A call recovered from speech, on the books under an id of its own."""
        call = RealtimeToolCall(session, f"recovered-{uuid4().hex[:12]}", tool_name, arguments)
        self._open_tool_call(call)
        return call

    async def _serve_recovered_call(self, call: RealtimeToolCall) -> None:
        """Serve a booked recovered call behind the gate, as any realtime call
        (RFC §12.4), and inject its outcome as context; a call on a session
        that ended still gets its one report, cancelled."""
        try:
            with self._tool_call_span(
                call, SpanKind.REALTIME_TOOL_RECOVERY, "recovered_tool"
            ) as span:
                # The channel's RealtimeToolsMixin is the host of every door.
                host = cast("ToolCallHost", self)
                outcome = await run_tool_call(host, call, _RecoveredDoor(self))
                span.close(outcome)
        finally:
            self._close_tool_call(call)
        logger.info(
            "Recovered tool %s(%s) %s for session %s",
            call.name,
            call.call_id,
            outcome.kind,
            call.session.id,
        )


class _RecoveredDoor:
    """A call recovered from speech: the model issued no call, so its outcome
    goes back as injected context, never as a tool result (RFC §12.4)."""

    channel_serves = False
    can_activate = True

    def __init__(self, channel: RealtimeToolRecoveryMixin) -> None:
        self._channel = channel

    async def deliver(self, call: RealtimeToolCall, outcome: ToolOutcome) -> bool:
        verb = _VERBS.get(outcome.kind, "failed")
        await self._channel._inject_recovered_result(
            call.session, call.name, result_text(outcome.result), verb=verb
        )
        return True


_VERBS: dict[OutcomeKind, Literal["completed", "denied", "failed"]] = {
    OutcomeKind.SERVED: "completed",
    OutcomeKind.REFUSED: "denied",
}


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def _parse_args(raw: str, param_names: list[str]) -> dict[str, Any]:
    """Parse ``key:value,...`` text using known parameter names as delimiters.

    Finds the *first* occurrence of each ``param_name:`` in *raw*, sorts
    by position, and slices values between consecutive boundaries.
    This avoids false splits when a value contains a substring like
    ``task:`` (only the first, true boundary is used per param).

    A key the tool does not declare also ends the value before it, provided it
    sits where a key belongs — at the start or just after a comma. It is
    returned under its own name so the schema gate can refuse it by name. The
    alternative is worse than a refusal: the undeclared text would otherwise be
    swallowed into the preceding value, and ``{"city": "Paris,country:FR"}``
    passes a ``{"city": {"type": "string"}}`` schema, handing the tool a
    corrupted argument nothing downstream can catch.

    The cost is a free-text value containing ``, word:`` — refused rather than
    truncated silently. On a path that reconstructs a call the model failed to
    issue properly, saying so beats guessing.
    """
    if not param_names or not raw:
        return {}

    # Find the first occurrence of each param followed by ':', where a key can
    # start: at the beginning, after a comma, or after whitespace. Without that
    # left boundary, an undeclared key *ending* with a declared name — the
    # ``name`` inside ``username:`` — opens the value in its place.
    positions: list[tuple[int, int, str]] = []
    for name in param_names:
        pattern = re.compile(r"(?:^|[,\s])\s*(" + re.escape(name) + r")\s*:")
        match = pattern.search(raw)
        if match:
            positions.append((match.start(1), match.end(), name))

    # Undeclared keys, only where a key can legitimately start.
    declared = set(param_names)
    taken = {start for start, _end, _name in positions}
    for match in _UNDECLARED_KEY_RE.finditer(raw):
        name = match.group(1)
        start = match.start(1)
        if name not in declared and start not in taken:
            positions.append((start, match.end(), name))
            taken.add(start)

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
