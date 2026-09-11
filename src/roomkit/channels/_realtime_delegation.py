"""Reasoning delegation for RealtimeVoiceChannel (RFC §12.4.1).

A full-duplex provider's model holds the conversation and hands reasoning and
tool use to a backend; the provider announces each hand-over through
``on_delegation``. This mixin turns every announcement into
``ON_REALTIME_DELEGATION`` and serves the integrator-side ones through the
channel's :class:`~roomkit.voice.realtime.reasoning.ReasoningBackend`: it keeps
the transcript ledger the backend reads, returns every output through
``submit_delegation_output``, answers a backend that fails or says nothing
with a spoken fallback, and runs the backend's tool calls through the same
pre-execution gate as any realtime tool call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

from roomkit.channels._realtime_context import _current_voice_session
from roomkit.channels._realtime_tools import result_text
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.models.tool_call import ToolCallEvent
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.context import reset_span
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.events import RealtimeDelegationEvent
from roomkit.voice.realtime.reasoning import ReasoningBackend, ReasoningRequest, TranscriptLine

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")

#: Spoken to the model when its delegation cannot be served (RFC §12.4.1).
FALLBACK_NO_BACKEND = "No backend is available to handle delegated work in this session."
FALLBACK_NO_OUTPUT = "The delegated work finished without an answer."
FALLBACK_FAILED = "The delegated work could not be completed."
FALLBACK_TIMEOUT = "The delegated work took too long and was abandoned."


class RealtimeDelegationHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeDelegationMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _session_spans: Telemetry session span per session.
        _session_tools: Resolved declared tool catalogue per session.
        _framework: The RoomKit framework instance (or None).
        _provider: The realtime voice provider.
        _reasoning_backend: The integrator-side backend, or None.
        _reasoning_timeout_s: Bound on one delegation's run.
        _transcript_ledger: Per-session ``[role, text]`` fragments since the
            previous delegation.
        _delegated_before: Sessions that already served a delegation.
        _pending_delegations: Delegation ids still running, per session.
        _tool_handler: The channel's tool handler, or None.
        _tool_result_max_length: Cap on a tool result's length.
        channel_id: The channel identifier.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task, _rt_span_ctx, _update_idle_event, _telemetry_provider,
        _authorize_realtime_tool, _fire_tool_hook, _truncate_tool_result.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, str]
    _session_tools: dict[str, list[dict[str, Any]]]
    _framework: RoomKit | None
    _provider: RealtimeVoiceProvider
    _reasoning_backend: ReasoningBackend | None
    _reasoning_timeout_s: float
    _transcript_ledger: dict[str, list[list[str]]]
    _delegated_before: set[str]
    _pending_delegations: dict[str, set[str]]
    _tool_handler: Any
    _tool_result_max_length: int
    channel_id: str

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _rt_span_ctx(self, session_id: str) -> tuple[Any, Any]: ...

    def _update_idle_event(self, session_id: str) -> None: ...


class RealtimeDelegationMixin:
    """Reasoning-delegation handling for :class:`RealtimeVoiceChannel`.

    Host contract: :class:`RealtimeDelegationHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, str]
    _session_tools: dict[str, list[dict[str, Any]]]
    _framework: RoomKit | None
    _provider: RealtimeVoiceProvider
    _reasoning_backend: ReasoningBackend | None
    _reasoning_timeout_s: float
    _transcript_ledger: dict[str, list[list[str]]]
    _delegated_before: set[str]
    _pending_delegations: dict[str, set[str]]
    _tool_handler: Any
    _tool_result_max_length: int
    channel_id: str

    _track_task: Any  # see RealtimeDelegationHost — cross-mixin
    _rt_span_ctx: Any  # see RealtimeDelegationHost — cross-mixin
    _update_idle_event: Any  # see RealtimeDelegationHost — cross-mixin
    _telemetry_provider: Any  # see RealtimeDelegationHost — cross-mixin
    _authorize_realtime_tool: Any  # see RealtimeToolsMixin
    _fire_tool_hook: Any  # see RealtimeToolsMixin
    _truncate_tool_result: Any  # see RealtimeToolsMixin

    # -----------------------------------------------------------------
    # Transcript ledger
    # -----------------------------------------------------------------

    def _on_transcript_fragment(
        self, session: VoiceSession, text: str, role: str, is_final: bool
    ) -> None:
        """Second transcription callback: keep the ledger a backend will read.

        Partials of a full-duplex provider carry deltas for both roles, and
        a delegation lands before the turn that caused it is declared final
        — so the ledger is fed from partials, merging consecutive fragments
        of one speaker, and finals are left out (they repeat the deltas).
        """
        if is_final or not text or self._reasoning_backend is None:
            return
        if not self._provider.full_duplex or role not in ("user", "assistant"):
            return
        with self._state_lock:
            ledger = self._transcript_ledger.setdefault(session.id, [])
            if ledger and ledger[-1][0] == role:
                ledger[-1][1] += text
            elif text.strip():
                ledger.append([role, text.lstrip()])

    def _take_transcript(self, session_id: str) -> list[TranscriptLine]:
        """Hand over what was recorded since the previous delegation."""
        with self._state_lock:
            taken = self._transcript_ledger.pop(session_id, [])
        lines: list[TranscriptLine] = []
        for role, text in taken:
            kind: Literal["user", "assistant"] = "user" if role == "user" else "assistant"
            if text.strip():
                lines.append(TranscriptLine(role=kind, text=text.strip()))
        return lines

    # -----------------------------------------------------------------
    # Delegation callback
    # -----------------------------------------------------------------

    def _on_provider_delegation(
        self, session: VoiceSession, delegation_id: str, target: str
    ) -> Any:
        """Provider callback: the model handed work to a backend."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if session.state == VoiceSessionState.ENDED:
            return
        logger.info(
            "Reasoning delegation %s to the %s backend (session %s)",
            delegation_id,
            target,
            session.id,
        )
        self._track_task(
            loop,
            self._fire_delegation_hook(session, delegation_id, target),
            name=f"rt_delegation_hook:{session.id}:{delegation_id}",
        )
        if target != "integrator":
            return
        if self._reasoning_backend is None:
            self._track_task(
                loop,
                self._decline_delegation(session, delegation_id),
                name=f"rt_delegation:{session.id}:{delegation_id}",
            )
            return
        self._begin_delegation(session.id, delegation_id)
        task = self._track_task(
            loop,
            self._serve_delegation(session, delegation_id),
            name=f"rt_delegation:{session.id}:{delegation_id}",
        )
        task.add_done_callback(lambda _: self._finish_delegation(session.id, delegation_id))

    def _begin_delegation(self, session_id: str, delegation_id: str) -> None:
        with self._state_lock:
            self._pending_delegations.setdefault(session_id, set()).add(delegation_id)
        self._update_idle_event(session_id)

    def _finish_delegation(self, session_id: str, delegation_id: str) -> None:
        with self._state_lock:
            pending = self._pending_delegations.get(session_id)
            if pending is not None:
                pending.discard(delegation_id)
        self._update_idle_event(session_id)

    async def _fire_delegation_hook(
        self, session: VoiceSession, delegation_id: str, target: str
    ) -> None:
        """Announce the delegation to ON_REALTIME_DELEGATION (RFC §12.4.1)."""
        if self._framework is None:
            return
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
        if not room_id:
            return
        if not self._framework.hook_engine.has_hooks(HookTrigger.ON_REALTIME_DELEGATION):
            return

        kind: Literal["hosted", "integrator"] = "hosted" if target == "hosted" else "integrator"
        event = RealtimeDelegationEvent(session=session, delegation_id=delegation_id, target=kind)
        _, _tok = self._rt_span_ctx(session.id)
        try:
            context = await self._framework._build_context(room_id)  # noqa: SLF001
            await self._framework.hook_engine.run_async_hooks(
                room_id,
                HookTrigger.ON_REALTIME_DELEGATION,
                event,
                context,
                skip_event_filter=True,
            )
        except Exception:
            logger.warning(
                "ON_REALTIME_DELEGATION failed for session %s", session.id, exc_info=True
            )
        finally:
            if _tok is not None:
                reset_span(_tok)

    # -----------------------------------------------------------------
    # Serving the integrator backend
    # -----------------------------------------------------------------

    async def _decline_delegation(self, session: VoiceSession, delegation_id: str) -> None:
        """No backend configured: say so, or the model waits for nothing."""
        logger.warning(
            "Delegation %s declined: no reasoning_backend on channel %s (session %s)",
            delegation_id,
            self.channel_id,
            session.id,
        )
        await self._fallback(session, delegation_id, FALLBACK_NO_BACKEND)

    async def _serve_delegation(self, session: VoiceSession, delegation_id: str) -> None:
        backend = self._reasoning_backend
        if backend is None:
            return
        with self._state_lock:
            first = session.id not in self._delegated_before
            self._delegated_before.add(session.id)
            tools = [dict(t) for t in self._session_tools.get(session.id, [])]
        request = ReasoningRequest(
            session=session,
            delegation_id=delegation_id,
            transcript=self._take_transcript(session.id),
            first=first,
            tools=tools,
            execute_tool=lambda name, arguments: self._execute_backend_tool(
                session, delegation_id, name, arguments
            ),
        )
        answered = False

        async def _relay() -> None:
            nonlocal answered
            async for output in backend.run(request):
                if session.state == VoiceSessionState.ENDED:
                    return
                text = output.text.strip()
                if not text:
                    continue
                answered = True
                await self._provider.submit_delegation_output(
                    session, delegation_id, text, spoken=output.spoken
                )

        try:
            await asyncio.wait_for(_relay(), timeout=self._reasoning_timeout_s)
        except TimeoutError:
            logger.warning(
                "Delegation %s exceeded %.0fs (session %s)",
                delegation_id,
                self._reasoning_timeout_s,
                session.id,
            )
            await self._fallback(session, delegation_id, FALLBACK_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Delegation %s failed (session %s)", delegation_id, session.id)
            await self._fallback(session, delegation_id, FALLBACK_FAILED)
        else:
            if not answered:
                logger.warning(
                    "Delegation %s produced no output (session %s)", delegation_id, session.id
                )
                await self._fallback(session, delegation_id, FALLBACK_NO_OUTPUT)
            else:
                logger.info("Delegation %s served (session %s)", delegation_id, session.id)

    async def _fallback(self, session: VoiceSession, delegation_id: str, text: str) -> None:
        """One spoken output, so the model does not wait for an answer that never comes."""
        if session.state == VoiceSessionState.ENDED:
            return
        try:
            await self._provider.submit_delegation_output(
                session, delegation_id, text, spoken=True
            )
        except Exception:
            logger.exception(
                "Could not return the fallback for delegation %s (session %s)",
                delegation_id,
                session.id,
            )

    # -----------------------------------------------------------------
    # Backend tool calls go through the channel gate
    # -----------------------------------------------------------------

    async def _execute_backend_tool(
        self,
        session: VoiceSession,
        delegation_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Run a backend's tool call as the framework runs any realtime tool call.

        Same gate (declared catalogue, argument schema, skill gating,
        ``BEFORE_TOOL_USE``), same handler, same ``ON_TOOL_CALL`` observation
        and truncation; the one difference is where the result goes — back to
        the backend model, not to the provider (RFC §12.4.1).
        """
        if session.state == VoiceSessionState.ENDED:
            return json.dumps({"error": "The session has ended."})
        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
            parent = self._session_spans.get(session.id)
        call_id = f"{delegation_id}:{uuid4().hex[:8]}"
        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            SpanKind.REALTIME_TOOL_CALL,
            f"realtime_tool:{name}",
            parent_id=parent,
            attributes={
                Attr.REALTIME_TOOL_NAME: name,
                "tool_call_id": call_id,
                "delegation_id": delegation_id,
            },
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
        )
        try:
            arguments, denial, gate_context = await self._authorize_realtime_tool(
                name, arguments, call_id, room_id, session
            )
            if denial is not None:
                logger.info(
                    "Backend tool %s denied before execution (delegation %s, session %s)",
                    name,
                    delegation_id,
                    session.id,
                )
                telemetry.end_span(span_id)
                return denial

            handler_result: str | None = None
            if self._tool_handler is not None:
                token = _current_voice_session.set(session)
                try:
                    raw = await self._tool_handler(name, arguments)
                finally:
                    _current_voice_session.reset(token)
                handler_result = result_text(raw)

            if self._framework is not None and room_id:
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
                result_str: str = await self._fire_tool_hook(
                    tool_event, room_id, handler_result, name, call_id, session, gate_context
                )
            elif handler_result is not None:
                result_str = handler_result
            else:
                result_str = json.dumps({"error": f"No handler for tool {name}"})

            if len(result_str) > self._tool_result_max_length:
                result_str = self._truncate_tool_result(result_str, name, call_id, session.id)
            telemetry.end_span(span_id)
            logger.info(
                "Backend tool %s handled (delegation %s, session %s)",
                name,
                delegation_id,
                session.id,
            )
            return result_str
        except asyncio.CancelledError:
            telemetry.end_span(span_id, status="cancelled")
            raise
        except Exception:
            telemetry.end_span(span_id, status="error", error_message=f"tool {name} failed")
            logger.exception(
                "Error handling backend tool %s (delegation %s, session %s)",
                name,
                delegation_id,
                session.id,
            )
            return json.dumps({"error": "Internal error handling tool call", "tool": name})
