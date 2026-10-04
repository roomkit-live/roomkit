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
import logging
import threading
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from uuid import uuid4

from roomkit.channels._realtime_tool_calls import RealtimeToolCall
from roomkit.channels._realtime_tool_executor import (
    ToolCallHost,
    report_cancelled_call,
    report_failed_call,
    report_served_elsewhere,
    run_tool_call,
)
from roomkit.core._failure_log import log_failure
from roomkit.core._fallback import FALLBACK_FAILED
from roomkit.core.exceptions import TurnCutShortError
from roomkit.core.task_utils import shielded
from roomkit.models.enums import HookTrigger
from roomkit.telemetry.base import SpanKind
from roomkit.telemetry.context import reset_span
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools.result import result_text
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.events import RealtimeDelegationEvent
from roomkit.voice.realtime.reasoning import (
    ReasoningBackend,
    ReasoningRequest,
    ToolCallResult,
    TranscriptLine,
    model_call_id,
)

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")

#: Spoken to the model when its delegation cannot be served (RFC §12.4.1).
FALLBACK_NO_BACKEND = "No backend is available to handle delegated work in this session."
FALLBACK_NO_OUTPUT = "The delegated work finished without an answer."
FALLBACK_TIMEOUT = "The delegated work took too long and was abandoned."


class RealtimeDelegationHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeDelegationMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _session_tools: Resolved declared tool catalogue per session.
        _framework: The RoomKit framework instance (or None).
        _provider: The realtime voice provider.
        _reasoning_backend: The integrator-side backend, or None.
        _reasoning_timeout_s: Bound on one delegation's run.
        _transcript_ledger: Per-session ``[role, text]`` fragments since the
            previous delegation.
        _delegated_before: Sessions that already served a delegation.
        _pending_delegations: Delegation ids still running, per session.
        channel_id: The channel identifier.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task, _rt_span_ctx, _update_idle_event,
        _access_cause, _door_exempt,
        _refresh_session_policies, _open_tool_call, _close_tool_call,
        _tool_call_span,
        and the executor's host steps.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_tools: dict[str, list[dict[str, Any]]]
    _framework: RoomKit | None
    _provider: RealtimeVoiceProvider
    _reasoning_backend: ReasoningBackend | None
    _reasoning_timeout_s: float
    _transcript_ledger: dict[str, list[list[str]]]
    _delegated_before: set[str]
    _pending_delegations: dict[str, set[str]]
    channel_id: str

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _rt_span_ctx(self, session_id: str) -> tuple[Any, Any]: ...

    def _update_idle_event(self, session_id: str) -> None: ...

    def _expect_provider_output(self, session_id: str) -> None: ...


class RealtimeDelegationMixin:
    """Reasoning-delegation handling for :class:`RealtimeVoiceChannel`.

    Host contract: :class:`RealtimeDelegationHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_tools: dict[str, list[dict[str, Any]]]
    _session_catalogue: Any  # cross-mixin (RealtimeToolsMixin)
    _framework: RoomKit | None
    _provider: RealtimeVoiceProvider
    _reasoning_backend: ReasoningBackend | None
    _reasoning_timeout_s: float
    _transcript_ledger: dict[str, list[list[str]]]
    _delegated_before: set[str]
    _pending_delegations: dict[str, set[str]]
    channel_id: str

    _track_task: Any  # see RealtimeDelegationHost — cross-mixin
    _rt_span_ctx: Any  # see RealtimeDelegationHost — cross-mixin
    _fire_session_error_hook: Any  # see RealtimeResponseMixin
    _expect_provider_output: Any
    _update_idle_event: Any  # see RealtimeDelegationHost — cross-mixin
    _access_cause: Any  # see RealtimeToolsMixin
    _door_exempt: Any  # see RealtimeToolGateMixin
    _refresh_session_policies: Any  # see RealtimeToolGateMixin
    _open_tool_call: Any  # see RealtimeToolsMixin
    _close_tool_call: Any  # see RealtimeToolsMixin
    _tool_calls: Any  # see RealtimeToolsMixin
    _tool_call_span: Any  # see RealtimeToolsMixin

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
            self._serve_delegation_under_session(session, delegation_id),
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

    async def _serve_delegation_under_session(
        self, session: VoiceSession, delegation_id: str
    ) -> None:
        """Serve a delegation under its session's span: the backend's own turn
        is traced as part of the session."""
        _, token = self._rt_span_ctx(session.id)
        try:
            await self._serve_delegation(session, delegation_id)
        finally:
            if token is not None:
                reset_span(token)

    async def _serve_delegation(self, session: VoiceSession, delegation_id: str) -> None:
        backend = self._reasoning_backend
        if backend is None:
            return
        handover = self._take_handover(session)
        answered = False

        async def _relay() -> None:
            nonlocal answered
            request = await self._reasoning_request(session, delegation_id, handover)
            async for output in backend.run(request):
                if session.state == VoiceSessionState.ENDED:
                    return
                text = output.text.strip()
                if not text:
                    continue
                answered = True
                if output.spoken:
                    self._expect_provider_output(session.id)
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
        except Exception as exc:
            log_failure(logger, exc, f"Delegation {delegation_id} (session {session.id})")
            # A failed turn is an error a host renders, as a room turn's is
            # (RFC §12.4.1): ON_ERROR once, beside the spoken fallback. A turn
            # its cap, deadline or budget cut is an expected end, not one.
            if not isinstance(exc, TurnCutShortError):
                await self._fire_session_error_hook(
                    session, str(exc), type(exc).__name__, "reasoning", type(backend).__name__
                )
            await self._fallback(session, delegation_id, FALLBACK_FAILED)
        else:
            if not answered:
                logger.warning(
                    "Delegation %s produced no output (session %s)", delegation_id, session.id
                )
                await self._fallback(session, delegation_id, FALLBACK_NO_OUTPUT)
            else:
                logger.info("Delegation %s served (session %s)", delegation_id, session.id)

    def _take_handover(self, session: VoiceSession) -> _Handover:
        """What a delegation takes as it is handed over, in the order the
        delegations were announced: whether it is the session's first, and
        the transcript recorded since the previous one (RFC §12.4.1)."""
        with self._state_lock:
            first = session.id not in self._delegated_before
            self._delegated_before.add(session.id)
            room_id = self._session_rooms.get(session.id) or session.room_id
        return _Handover(first, self._take_transcript(session.id), room_id)

    async def _reasoning_request(
        self, session: VoiceSession, delegation_id: str, handover: _Handover
    ) -> ReasoningRequest:
        """What the backend receives for one delegation: what it took when
        handed over, the session's catalogue, and the door to its tools.

        The catalogue follows the participant's role and the room's agent as
        they stand when the backend is handed the delegation, as the gate
        reads them at each call; their read runs inside the delegation's
        bound, so one that fails is answered by its spoken fallback
        (RFC §12.4.1).
        """
        await self._refresh_session_policies(session, handover.room_id)
        return ReasoningRequest(
            session=session,
            delegation_id=delegation_id,
            transcript=handover.transcript,
            first=handover.first,
            tools=self._backend_catalogue(session.id),
            unavailable=self._backend_unavailable(session.id),
            execute_tool=lambda name, arguments: self._execute_backend_tool(
                session, delegation_id, name, arguments
            ),
            execute_tool_call=lambda name, arguments: self._execute_backend_tool_call(
                session, delegation_id, name, arguments
            ),
            report_refusal=partial(self._report_backend_refusal, session, delegation_id),
            report_call=partial(self._report_backend_call, session, delegation_id),
        )

    def _backend_catalogue(self, session_id: str) -> list[dict[str, Any]]:
        """The session's tools a backend may call: none its policy denies or a
        skill gates, read as its door's gate reads them (nothing of the
        channel's own escapes), so a tool the backend is offered is one it
        may call."""
        with self._state_lock:
            declared = session_id in self._session_tools
        tools = self._session_catalogue(session_id) if declared else []
        exempt = self._door_exempt(False)
        return [
            dict(t)
            for t in tools
            if self._access_cause(str(t.get("name", "")), session_id, exempt) is None
        ]

    def _backend_unavailable(self, session_id: str) -> dict[str, str]:
        """The session's tools a backend is not offered, each with the refusal
        its call reads: the policy's, or a skill's worded for a model that
        cannot activate one (RFC §21.1)."""
        with self._state_lock:
            declared = session_id in self._session_tools
        names = (str(t.get("name", "")) for t in self._session_catalogue(session_id))
        exempt = self._door_exempt(False)
        causes = {
            name: self._access_cause(name, session_id, exempt, can_activate=False)
            for name in names
            if declared
        }
        return {name: cause for name, cause in causes.items() if name and cause is not None}

    async def _fallback(self, session: VoiceSession, delegation_id: str, text: str) -> None:
        """One spoken output, so the model does not wait for an answer that never comes."""
        if session.state == VoiceSessionState.ENDED:
            return
        try:
            self._expect_provider_output(session.id)
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
        """A backend's tool call, as the text its model reads (``execute_tool``)."""
        done = await self._execute_backend_tool_call(session, delegation_id, name, arguments)
        return done.text

    async def _execute_backend_tool_call(
        self,
        session: VoiceSession,
        delegation_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        """Run a backend's tool call as the framework runs any realtime tool call.

        Same gate (declared catalogue, tool policy, skill gating, argument
        schema, ``BEFORE_TOOL_USE``), the same serving, ON_TOOL_CALL and
        bound; the one difference is where the outcome goes — back to the
        backend model, with whether it failed, not to the provider (RFC §12.4.1).
        """
        # Under the id the backend's model gave it, as its other reports are
        # (RFC §9.3).
        call_id = _backend_call_id(delegation_id, model_call_id())
        call = RealtimeToolCall(session, call_id, name, arguments)
        # The delegation's task serves the call: a call that ends its own
        # session is then not taken for one the session's end interrupted.
        call.task = asyncio.current_task()
        self._open_tool_call(call)
        try:
            with self._tool_call_span(
                call, SpanKind.REALTIME_TOOL_CALL, "realtime_tool", delegation_id=delegation_id
            ) as span:
                # The channel's RealtimeToolsMixin is the host of every door.
                host = cast("ToolCallHost", self)
                try:
                    outcome = await run_tool_call(host, call, _BackendDoor())
                except asyncio.CancelledError:
                    await self._report_cut_backend_call(call)
                    raise
                span.close(outcome)
        finally:
            self._close_tool_call(call)
        logger.info(
            "Backend tool %s %s (delegation %s, session %s)",
            name,
            outcome.kind,
            delegation_id,
            session.id,
        )
        return ToolCallResult(
            result_text(outcome.result),
            is_error=outcome.failed,
            refused=outcome.kind is OutcomeKind.REFUSED,
        )

    async def _report_cut_backend_call(self, call: RealtimeToolCall) -> None:
        """Report a backend call the end of its delegation cut, once, as
        cancelled (RFC §9.3). A call the session's end took off the books is
        reported there, with its own reason."""
        if not self._tool_calls.holds(call):
            return
        await shielded(
            report_cancelled_call(cast("ToolCallHost", self), call, "The delegation ended")
        )

    def _backend_call(
        self,
        session: VoiceSession,
        delegation_id: str,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> RealtimeToolCall:
        """A call a backend reports outside the gate, under an id of its
        delegation, in the session's room."""
        call = RealtimeToolCall(
            session, _backend_call_id(delegation_id, tool_call_id), name, arguments
        )
        call.room_id = self._session_rooms.get(session.id) or session.room_id
        return call

    async def _report_backend_refusal(
        self,
        session: VoiceSession,
        delegation_id: str,
        name: str,
        arguments: dict[str, Any],
        body: str,
        *,
        cancelled: bool = False,
        refused: bool = True,
        detail: str | None = None,
    ) -> None:
        """Report a call the backend's own loop ended before the gate (its
        arguments did not read, a stop cut it), as the gate reports one, with
        its outcome: cancelled, refused, or failed, and what failed; under
        the id its model gave it."""
        call = self._backend_call(session, delegation_id, name, arguments, model_call_id())
        if cancelled:
            kind = OutcomeKind.CANCELLED
        else:
            kind = OutcomeKind.REFUSED if refused else OutcomeKind.FAILED
        outcome = ToolOutcome(kind, body, detail=detail)
        await report_failed_call(cast("ToolCallHost", self), call, outcome)

    async def _report_backend_call(
        self,
        session: VoiceSession,
        delegation_id: str,
        name: str,
        arguments: dict[str, Any],
        result: str,
        *,
        is_error: bool = False,
        detail: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Report a call the backend's own provider served, outside the gate,
        as an AIChannel reports one: served or failed, once (RFC §9.3)."""
        call = self._backend_call(session, delegation_id, name, arguments, tool_call_id)
        await report_served_elsewhere(
            cast("ToolCallHost", self), call, result, is_error=is_error, detail=detail
        )


def _backend_call_id(delegation_id: str, tool_call_id: str | None) -> str:
    """The id a backend's call is reported under: its model's within the
    delegation's, or one minted for a backend that gives none."""
    return f"{delegation_id}:{tool_call_id or uuid4().hex[:8]}"


@dataclass(frozen=True)
class _Handover:
    """What a delegation took when it was handed over."""

    first: bool
    transcript: list[TranscriptLine]
    room_id: str | None


class _BackendDoor:
    """A reasoning backend's call: its outcome goes back to the backend's
    model, which awaits it, never to the provider (RFC §12.4.1)."""

    channel_serves = False
    can_activate = False

    async def deliver(self, call: RealtimeToolCall, outcome: ToolOutcome) -> bool:
        return True
