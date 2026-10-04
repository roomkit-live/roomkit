"""Tool call handling for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._realtime_context import (
    _current_voice_session,
    serving_call,
    spare_own_orphaned_call,
)
from roomkit.channels._realtime_skills import RequiredToolsCheck
from roomkit.channels._realtime_tool_calls import RealtimeToolCall, ToolCallBook
from roomkit.channels._realtime_tool_executor import (
    ABANDONED_BY_PROVIDER,
    SESSION_ENDED,
    ToolCallDoor,
    deliver_once,
    ended_outcome,
    judge_tool_call,
    report_cancelled_call,
    report_interrupted_calls,
    run_tool_call,
    serve_tool_call,
    serve_unbooked,
    serving_tool_call,
    submit_tool_outcome,
    tool_loop_context,
)
from roomkit.channels._served_tools import dict_tool_name
from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
from roomkit.channels._tool_registry import ChannelRegistry, schema_tool
from roomkit.channels._tool_search_constants import TOOL_CALL_TOOL, TOOL_LIST_TOOLS
from roomkit.core.exceptions import ToolRefusedError, UnservedToolCallError
from roomkit.models.tool_call import (
    ToolCallEvent,
)
from roomkit.skills.models import missing_tools_error, serves_exactly
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.context import reset_span, set_current_span
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools.result import (
    GateRefusal,
    bounded_result,
    declined_answer,
    result_text,
)
from roomkit.tools.timeout import ToolTimeouts, answer_within
from roomkit.voice.base import VoiceSessionState

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.models.enums import ChannelType
    from roomkit.tools._human_input_channel import ChannelHumanInput
    from roomkit.tools.context import _ToolLoopContext
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")

_LOOP_SEGMENT_BUDGET_S = 0.050
"""Sync work on the event loop past this delays realtime pacing.

The SIP pacer's jitter headroom is 60ms — one fused stretch beyond it is
an audible drop-out on a concurrent call.  Tool-call segments are timed
individually so the culprit is named in the logs without an asyncio
set_debug hunt."""


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
        _tool_reports: The reports of abandoned calls under way, which
            ``close()`` lets finish before it cancels the channel's tasks.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _session_config_locks: dict[str, asyncio.Lock]
    _tool_handler: Any
    _tools: Any
    _system_prompt: str | None
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _registry: ChannelRegistry
    _tool_timeouts: ToolTimeouts
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    _tool_calls: ToolCallBook
    _tool_reports: set[asyncio.Task[Any]]
    channel_id: str
    channel_type: ChannelType
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _update_idle_event(self, session_id: str) -> None: ...

    def _expect_provider_output(self, session_id: str) -> None: ...


def _timed_result_text(name: str, raw: Any) -> str:
    """*raw* as the text a voice provider reads, warning when flattening it
    held the event loop past the realtime segment budget."""
    started = time.perf_counter()
    text = result_text(raw)
    held = time.perf_counter() - started
    if held > _LOOP_SEGMENT_BUDGET_S:
        # Pure sync CPU (wall == loop hold), and it runs on the FULL result
        # before truncation caps it.
        logger.warning(
            "Tool %s result serialization held the event loop for "
            "%.0fms (%d chars, budget ~%.0fms) — concurrent "
            "realtime audio may underrun; return a string or a "
            "compact reference instead of a large object",
            name,
            held * 1000,
            len(text),
            _LOOP_SEGMENT_BUDGET_S * 1000,
        )
    return text


class _ProviderDoor:
    """The provider's function call: its outcome goes back as the call's result."""

    channel_serves = True
    can_activate = True

    def __init__(self, channel: RealtimeToolsMixin) -> None:
        self._channel = channel

    async def deliver(self, call: RealtimeToolCall, outcome: ToolOutcome) -> bool:
        text = result_text(outcome.result)
        return await self._channel._submit_realtime_tool_result(call, text, failed=outcome.failed)


class _ToolCallSpan:
    """The telemetry span of one realtime tool call, closed by its outcome."""

    def __init__(self, telemetry: Any, span_id: Any, name: str) -> None:
        self._telemetry = telemetry
        self._span_id = span_id
        self._name = name
        self._open = True

    def close(self, outcome: ToolOutcome) -> None:
        """End the span as *outcome* ended the call."""
        if outcome.kind is OutcomeKind.REFUSED:
            self.end(attributes={Attr.REALTIME_TOOL_DENIED: True})
        elif outcome.kind is OutcomeKind.CANCELLED:
            self.end(status="cancelled")
        elif outcome.kind is OutcomeKind.FAILED:
            self.end(status="error", error_message=f"tool {self._name} failed")
        else:
            self.end()

    def end(self, **kwargs: Any) -> None:
        if self._open:
            self._open = False
            self._telemetry.end_span(self._span_id, **kwargs)


class RealtimeToolsMixin:
    """Tool call execution for RealtimeVoiceChannel.

    Host contract: :class:`RealtimeToolsHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _session_config_locks: dict[str, asyncio.Lock]
    _tool_handler: Any
    _human_input: ChannelHumanInput
    _tools: Any
    _system_prompt: str | None
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _registry: ChannelRegistry
    _tool_timeouts: ToolTimeouts
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    _tool_calls: ToolCallBook
    _tool_reports: set[asyncio.Task[Any]]
    channel_id: str
    channel_type: ChannelType
    _telemetry_provider: Any

    _track_task: Any  # see RealtimeToolsHost — cross-mixin
    _session_answer_depth: Any  # RealtimeTranscriptionMixin — cross-mixin
    _expect_provider_output: Any
    _update_idle_event: Any
    _compose_session_prompt: Any
    _compose_session_tools: Any
    _authorize_realtime_tool: Any  # RealtimeToolGateMixin — cross-mixin
    _session_base_tools: Any  # RealtimeToolGateMixin — cross-mixin
    _session_catalogue: Any  # RealtimeToolGateMixin — cross-mixin
    _admitted_catalogue: Any  # RealtimeToolGateMixin — cross-mixin
    _session_declared_tools: Any  # RealtimeToolGateMixin — cross-mixin
    _session_policy_check: Any  # RealtimeToolGateMixin — cross-mixin
    _tool_reachable: Any  # RealtimeToolGateMixin — cross-mixin
    _channel_tool_names: Any  # RealtimeToolGateMixin — cross-mixin

    def _on_provider_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any] | str,
    ) -> Any:
        """Handle tool call from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        # A call on a session that ended still gets its one report, cancelled
        # (RFC §9.3): the executor serves nothing for it.
        call = self._provider_call(session, call_id, name, arguments)
        if not self._open_tool_call(call):
            # No result can name it (no id, or an id that still names a call
            # in flight): it takes the path of any call, the session's end
            # and the transcription barrier included, is refused there and
            # sends nothing (RFC §12.4).
            # Off the books, so the session's end does not cancel it.
            kind = "duplicate" if call_id else "unidentified"
            self._track_task(
                loop,
                self._serve_unbooked_call(call),
                name=f"rt_tool_{kind}:{session.id}:{call_id or call.name}",
            )
            return
        call.task = self._track_task(
            loop,
            self._handle_tool_call(call),
            name=f"rt_tool_call:{session.id}:{call_id}",
        )
        call.task.add_done_callback(lambda _: self._close_tool_call(call))

    def _provider_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any] | str,
    ) -> RealtimeToolCall:
        """The call a provider issued, unwrapped first, so the books, the span
        and every report, a refusal at entry included, name the tool a
        fixed-declaration call_tool carries, not the transport."""
        call = RealtimeToolCall.from_provider(
            session, call_id, name, arguments, mutes=self._mute_on_tool_call
        )
        if call.unreadable is None:
            call.unreadable = self._unwrap_call_tool(call)
        return call

    async def _report_start_calls(
        self, session: VoiceSession, issued: list[tuple[Any, ...]]
    ) -> None:
        """Report each call the provider *issued* while a start that failed
        was pending, once, cancelled: none was served, nothing was sent
        (RFC §12.4)."""
        calls = [self._provider_call(session, *args) for args in issued]
        for call in calls:
            call.room_id = self._session_room(session) or session.room_id or None
        await report_interrupted_calls(self, calls, SESSION_ENDED)

    def _on_provider_tool_call_cancelled(self, session: VoiceSession, call_ids: list[str]) -> Any:
        """Provider callback: the model will not read these calls' results (RFC §12.4).

        The provider freed each call's id: the channel frees it too, so
        nothing is sent for the call and a call issued under the id is a new
        one (RFC §12.4). The handler still running for one of them is working
        for a result nobody will read: its task is cancelled and the call is
        reported to ON_TOOL_CALL's observers as cancelled. A call no longer in
        the books (its result left before the cancellation arrived) has no
        event to build: the provider dropped the stale result and logged it. A
        call whose outcome the observers already received is left to finish,
        sending nothing: a second event would put two outcomes on one
        ``tool_call_id``. A reconnect the call's own handler caused (a handoff
        reconfiguring its session) orphans that call too, and it is not
        abandoned: it runs on, its id freed and its result kept off the wire
        (RFC §9.3).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if session.state == VoiceSessionState.ENDED:
            return
        for call_id in call_ids:
            call = self._tool_calls.get(session.id, call_id)
            if call is None:
                logger.debug(
                    "Cancelled tool call %s is not in flight for session %s", call_id, session.id
                )
                continue
            if self._spared_by_own_reconnect(call):
                continue
            if self._tool_calls.release(session.id, call_id) is None or not call.interruptible:
                logger.debug(
                    "Cancelled tool call %s already gave its outcome for session %s",
                    call_id,
                    session.id,
                )
                continue
            assert call.task is not None  # interruptible  # noqa: S101
            call.task.cancel()
            logger.info(
                "Tool call %s(%s) abandoned by the provider for session %s",
                call.name,
                call_id,
                session.id,
            )
            report = self._track_task(
                loop,
                report_cancelled_call(self, call, ABANDONED_BY_PROVIDER),
                name=f"rt_tool_cancelled:{session.id}:{call_id}",
            )
            self._tool_reports.add(report)
            report.add_done_callback(self._tool_reports.discard)

    async def _settle_tool_reports(self) -> None:
        """Wait for the reports of abandoned calls under way: the channel's
        close would otherwise cut them before their claim (RFC §9.3). A wait
        cancelled leaves them running."""
        if self._tool_reports:
            await asyncio.wait(list(self._tool_reports), timeout=5.0)

    @staticmethod
    def _spared_by_own_reconnect(call: RealtimeToolCall) -> bool:
        """Whether the call's own handler caused the reconnect that orphaned it.

        Such a call is not abandoned: its handler runs on, its result stays
        off the wire, and its outcome is reported as usual (RFC §9.3). A call
        issued under its id after its result went out is another call, which
        the reconnect abandons as any other.
        """
        if not spare_own_orphaned_call(call):
            return False
        logger.info(
            "Tool call %s(%s) lost its id to the reconnect its own handler caused; "
            "the handler runs on and its result stays off the wire (session %s)",
            call.name,
            call.call_id,
            call.session.id,
        )
        return True

    def _session_room(self, session: VoiceSession) -> str | None:
        with self._state_lock:
            return self._session_rooms.get(session.id)

    def _open_tool_call(self, call: RealtimeToolCall) -> bool:
        """Put *call* on the books; False when its id names a call in flight.

        A call that holds the input muted mutes it, and the input stays muted
        until the last such call in flight ends (RFC §12.4). The call holds
        idle while it runs, and a result it sends holds it until the
        continuation (``_expect_provider_output``).
        """
        session_id = call.session.id
        if call.room_id is None:
            # A session already ended no longer maps its room: its calls are
            # reported there all the same (RFC §9.3).
            call.room_id = self._session_room(call.session) or call.session.room_id or None
        muted_before = self._tool_calls.muting(session_id)
        if not self._tool_calls.open(call):
            return False
        if call.mutes and not muted_before and self._transport is not None:
            self._transport.set_input_muted(call.session, True)
        self._update_idle_event(session_id)
        return True

    def _close_tool_call(self, call: RealtimeToolCall) -> None:
        """Take *call* off the books, releasing the input once no call holds it."""
        session_id = call.session.id
        self._tool_calls.close(call)
        released = call.mutes and not self._tool_calls.muting(session_id)
        if released and self._transport is not None:
            self._transport.set_input_muted(call.session, False)
        self._update_idle_event(session_id)

    async def _handle_tool_call(self, call: RealtimeToolCall) -> None:
        with serving_call(call):
            await self._execute_tool_call(call)

    async def _serve_unbooked_call(self, call: RealtimeToolCall) -> None:
        """Take a call no result can name down the path of any call, off the
        books (RFC §12.4).

        It sends nothing, so it serves no call's context: a reconnect its
        observers cause orphans the call its id names, not this one. A cut
        before its outcome (the channel closing while it waits behind the
        transcription barrier) still reports it once, cancelled (RFC §9.3).
        """
        await serve_unbooked(self, call, lambda: self._execute_tool_call(call), SESSION_ENDED)

    async def _execute_tool_call(self, call: RealtimeToolCall) -> None:
        """Serve a provider's function call and submit its outcome (RFC §12.4)."""
        session = call.session
        await self._after_earlier_transcriptions(session)
        call.room_id = self._session_room(session) or call.room_id
        with self._tool_call_span(call, SpanKind.REALTIME_TOOL_CALL, "realtime_tool") as span:
            outcome = await run_tool_call(self, call, _ProviderDoor(self))
            span.close(outcome)
        logger.info(
            "Tool call %s(%s) %s for session %s", call.name, call.call_id, outcome.kind, session.id
        )

    async def _after_earlier_transcriptions(self, session: VoiceSession) -> None:
        """Wait for the transcriptions the provider emitted before this call.

        A tool call must not overtake them: the user final that closes the
        current utterance travels the serialised transcription queue, while
        tool calls run in their own task — unbarriered, the tool reaches the
        application first and the late final reads as new user speech. The
        call passes through the same FIFO lock, then releases it: tool
        execution itself must not hold transcriptions back. No lock, no
        transcription ahead of it: the call creates none, so one arriving
        once its session ended leaves nothing behind.
        """
        with self._state_lock:
            order_lock = self._transcription_order_locks.get(session.id)
        if order_lock is None:
            return
        async with order_lock:
            pass

    @contextlib.contextmanager
    def _tool_call_span(
        self, call: RealtimeToolCall, kind: SpanKind, prefix: str, **attributes: Any
    ) -> Iterator[_ToolCallSpan]:
        """The call's telemetry span, under the session's, the session's span
        current while the call runs; a call cut short ends it cancelled."""
        session_id = call.session.id
        with self._state_lock:
            session_span = self._session_spans.get(session_id)
            parent = self._turn_spans.get(session_id) or session_span
        token = set_current_span(session_span) if session_span else None
        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            kind,
            f"{prefix}:{call.name}",
            parent_id=parent,
            attributes={
                Attr.REALTIME_TOOL_NAME: call.name,
                "tool_call_id": call.call_id,
                **attributes,
            },
            room_id=call.room_id,
            session_id=session_id,
            channel_id=self.channel_id,
        )
        span = _ToolCallSpan(telemetry, span_id, call.name)
        try:
            yield span
        except asyncio.CancelledError:
            span.end(status="cancelled")
            raise
        finally:
            span.end()
            if token is not None:
                reset_span(token)

    def _unwrap_call_tool(self, call: RealtimeToolCall) -> str | None:
        """Unwrap the tool a fixed-declaration ``call_tool`` carries into *call*,
        so its books and its reports name that tool; what the model reads when
        the transport is unreadable, if it is."""
        support = self._tool_search_support
        session = call.session
        if not (
            call.name == TOOL_CALL_TOOL
            and support
            and support.uses_call_tool
            and support.active(session.id)
        ):
            return None
        call.name, call.arguments, transport_error = support.unwrap_call(
            call.arguments, session.id
        )
        return None if transport_error is None else json.dumps(transport_error)

    # -- ToolCallHost: the steps the executor serves a call with -------------

    def _tool_framework(self, call: RealtimeToolCall) -> RoomKit | None:
        return self._framework if call.room_id else None

    def _tool_event(self, call: RealtimeToolCall, result: str | None) -> ToolCallEvent:
        """The ON_TOOL_CALL event of one call on this session."""
        return ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=self.channel_type,
            tool_call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            room_id=call.room_id,
            session=call.session,
        )

    async def _authorize_call(
        self, call: RealtimeToolCall, door: ToolCallDoor
    ) -> tuple[GateRefusal | None, RoomContext | None]:
        """The pre-execution gate (RFC §12.4)."""
        call.arguments, denial, context = await self._authorize_realtime_tool(
            call.name,
            call.arguments,
            call.call_id,
            call.room_id,
            call.session,
            channel_serves=door.channel_serves,
            can_activate=door.can_activate,
        )
        return denial, context

    def _call_ended(self, call: RealtimeToolCall) -> bool:
        return call.session.state == VoiceSessionState.ENDED

    async def _serve_channel_tool(
        self, call: RealtimeToolCall, door: ToolCallDoor, carrying: RoomContext | None
    ) -> ToolOutcome | None:
        """Tool Search and skill activation, which reconfigure the session
        around their delivery; ``None`` for any other call."""
        if self._tool_search_support and self._tool_search_support.is_search_tool(call.name):
            return await self._serve_tool_search(call, door, carrying)
        if not (self._skill_support and self._skill_support.is_skill_tool(call.name)):
            return None
        if call.name == TOOL_ACTIVATE_SKILL:
            return await self._serve_skill_activation(call, door, carrying)
        return await serve_tool_call(
            self, call, carrying, answer_with=lambda: self._skill_answer(call, carrying)
        )

    async def _skill_answer(self, call: RealtimeToolCall, carrying: RoomContext | None) -> str:
        """A skill tool's answer (a reference, a script's output), bounded in
        time, inside the tool call context as a handler's (RFC §12.4)."""
        async with self._tool_call_scope(call, carrying) as loop_ctx:
            answer = self._skill_support.handle_tool_call(
                call.name, call.arguments, call.session.id
            )
            timeout = self._call_timeout(call.name, loop_ctx.room_id)
            return await answer_within(timeout, call.name, answer)

    async def _answer_call(self, call: RealtimeToolCall, carrying: RoomContext | None) -> str:
        """The handler's answer, as the text the model reads.

        Raises :class:`~roomkit.core.exceptions.UnservedToolCallError` when
        nothing serves the call, and lets the handler's
        :class:`~roomkit.core.exceptions.ToolRefusedError` and
        :class:`~roomkit.core.exceptions.ToolFailedError` through.
        """
        name, session = call.name, call.session
        if not self._serves_tool(name, call.room_id or session.room_id):
            raise UnservedToolCallError(name)
        logger.info(
            "Executing tool %s(%s) via handler for session %s", name, call.call_id, session.id
        )
        t_seg = time.perf_counter()
        raw = await self._call_tool_handler(call, carrying)
        logger.debug(
            "tool %s handler segment: %.0fms wall", name, (time.perf_counter() - t_seg) * 1000
        )
        text = _timed_result_text(name, raw)
        # Yield so realtime pacing gets a slot between the handler segment and
        # hook dispatch — sync hooks run inline next and would otherwise fuse
        # with this segment into one loop step.
        await asyncio.sleep(0)
        return text

    def _bound_call_result(self, call: RealtimeToolCall, text: str, *, served: bool = True) -> str:
        """*text* within ``tool_result_max_length`` (RFC §21.5). The complete
        schema ``list_tools(name=...)`` serves is exempt: the model needs it
        whole to call the tool. A refusal or a hook's replacement in its
        place (*served* false) is bounded as any result. (An activated
        skill's instructions, exempt too, go out with the activation itself.)"""
        if served and call.name == TOOL_LIST_TOOLS and call.arguments.get("name"):
            return text
        return bounded_result(text, self._tool_result_max_length, call.name)

    async def _call_tool_handler(
        self, call: RealtimeToolCall, gate_context: RoomContext | None
    ) -> Any:
        """The answer to one call, run inside the tool call context (RFC §21.4),
        whichever door brought the call: the tool orchestration set up for the
        room, else the host's handler."""
        async with self._tool_call_scope(call, gate_context) as loop_ctx:
            timeout = self._call_timeout(call.name, loop_ctx.room_id)
            answer = self._answer(call.name, call.arguments, loop_ctx.room_id)
            return await answer_within(timeout, call.name, answer)

    @contextlib.asynccontextmanager
    async def _tool_call_scope(
        self, call: RealtimeToolCall, gate_context: RoomContext | None
    ) -> AsyncIterator[_ToolLoopContext]:
        """The tool call context one call is served in (RFC §21.4): its room,
        actor and chain depth, the session, and ``current_tool_call()``."""
        session = call.session
        loop_ctx = await tool_loop_context(
            self._framework,
            call.room_id or session.room_id,
            actor_id=session.participant_id,
            # The call belongs to the model's answer, as deep as its transcript.
            chain_depth=self._session_answer_depth(session.id).answer,
            room=gate_context.room if gate_context is not None else None,
        )
        # What the session declares is the call's resolved toolset, which
        # ``current_tool_allowed_names()`` answers, as a turn's (RFC §21.4); a
        # session that declares no catalogue names no list, its gate still
        # judging each call.
        if self._session_catalogue(session.id):
            loop_ctx.all_context_tools = [
                schema_tool(tool)
                for tool in self._session_declared_tools(session.id)
                if dict_tool_name(tool)  # a provider's native tool has no name
            ]
            loop_ctx.admits = self._session_policy_check(session.id)
        token = _current_voice_session.set(session)
        try:
            with serving_tool_call(call, self.channel_id, loop_ctx):
                yield loop_ctx
        finally:
            _current_voice_session.reset(token)

    def _call_timeout(self, name: str, room_id: str | None) -> float | None:
        """The bound of one call to *name* (RFC §21.6): the channel's, unless
        the tool keeps a bound of its own (a person's answer, under its
        handler's timeout)."""
        own = self._human_input.serves(name)
        return self._registry.bound(name, room_id, self._tool_timeouts, own=own)

    async def _answer(self, name: str, arguments: dict[str, Any], room_id: str | None) -> Any:
        """The answer of what orchestration set up for *room_id*, else of the
        person's tools, else of the host's handler; a handler that declines
        the call raises :class:`~roomkit.core.exceptions.UnservedToolCallError`
        (RFC §21.4).

        Only the host's answer may be the "not mine" envelope: orchestration's
        tools answer what they ran, and a person what they said.
        """
        entry = self._registry.lookup(name, room_id)
        if entry is not None and entry.serve is not None:
            result = entry.serve(arguments)
            return await result if inspect.isawaitable(result) else result
        if self._human_input.serves(name):
            return await self._human_input.serve(name, arguments)
        return declined_answer(await self._tool_handler(name, arguments), name)

    def _serves_tool(self, name: str, room_id: str | None) -> bool:
        """Whether something serves a call to *name* in *room_id*: what
        orchestration set up there, the person's tools, or the host's handler."""
        entry = self._registry.lookup(name, room_id)
        if entry is not None and entry.serve is not None:
            return True
        return self._human_input.serves(name) or self._tool_handler is not None

    async def _serve_skill_activation(
        self, call: RealtimeToolCall, door: ToolCallDoor, carrying: RoomContext | None
    ) -> ToolOutcome:
        """Serve ``activate_skill``: ON_TOOL_CALL decides, then delivery, then gates.

        The SYNC hooks run on the result before it goes out, as for any other
        tool, so a hook that blocks ``activate_skill`` blocks the activation
        too: the model reads the refusal and no gate opens. They run outside
        the session's configuration lock, which a hook may itself need to
        reconfigure the session.
        """
        support, session = self._skill_support, call.session
        lock = self._session_config_locks.get(session.id)
        if lock is None:
            return ended_outcome(call)
        # A skill requires tools the session declares, whoever set them up
        # (its catalogue, orchestration), once its tool policy is applied: a
        # tool the policy denies is no tool the skill can use, and its schema
        # never goes out with the activation (RFC §24.3).
        tools = self._admitted_catalogue(session.id)
        try:
            result, skill = await support.prepare_activation(call.arguments, session.id, tools)
        except ToolRefusedError as refusal:
            return ToolOutcome(OutcomeKind.REFUSED, refusal.message)
        catalogue = self._session_base_tools(session.id)
        result, hinted = self._unknown_skill_hint(call, result, skill, tools)
        if skill is None and not hinted:
            # Nothing to reveal: the call named no skill, refused (RFC §9.3).
            return ToolOutcome(OutcomeKind.REFUSED, result)
        outcome, skill = await self._judge_activation(call, carrying, result, skill)
        # Provider updates (discovery, handoff, activation) are serialised on
        # this lock.
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return ended_outcome(call)
            return await self._deliver_activation(
                call, door, outcome, result, skill, hinted, catalogue=catalogue
            )

    def _unknown_skill_hint(
        self, call: RealtimeToolCall, result: str, skill: Any, tools: list[dict[str, Any]]
    ) -> tuple[str, list[str]]:
        """An activation that found no skill, with the session's tools its
        name matches hinted, and those tools, as on the text path."""
        name = call.arguments.get("name")
        if skill is not None or not isinstance(name, str):
            return result, []
        session_id = call.session.id
        # Only the turn's own catalogue, never the channel's own tools, as on
        # the text path (RFC §24.4).
        own = self._channel_tool_names()
        reachable = [
            tool_name
            for tool in tools
            if (tool_name := dict_tool_name(tool))
            and tool_name not in own
            and self._tool_reachable(tool_name, session_id)
        ]
        search = self._tool_search_support
        call_tool = search is not None and search.uses_call_tool and search.active(session_id)
        return self._skill_support.unknown_skill_hint(result, name, reachable, call_tool=call_tool)

    async def _judge_activation(
        self, call: RealtimeToolCall, carrying: RoomContext | None, result: str, skill: Any
    ) -> tuple[ToolOutcome, Any]:
        """The activation as ON_TOOL_CALL judged it, and the skill it opens.

        A tool the skill requires may leave the catalogue while the hooks run
        (a handoff): that is checked once, after the hooks and before anyone is
        told, so the observers read what the model reads (RFC §9.3). A handoff
        landing after that does not withdraw the activation; the skill's calls
        to a tool it removed are then refused as undeclared.
        """
        support = self._skill_support
        closed = support.closed_for(skill, call.session.id) if support and skill else ()
        match = support.requires_match if support else serves_exactly
        check = RequiredToolsCheck(
            skill, lambda: self._admitted_catalogue(call.session.id), closed, match
        )
        served = ToolOutcome(OutcomeKind.SERVED, result)
        outcome = await judge_tool_call(self, call, served, carrying, admit=check.held)
        if check.missing is None:  # no framework judged the call
            check.held()
        if check.missing and outcome.kind is OutcomeKind.SERVED:
            return ToolOutcome(OutcomeKind.REFUSED, missing_tools_error(check.missing)), None
        return outcome, skill

    async def _deliver_activation(
        self,
        call: RealtimeToolCall,
        door: ToolCallDoor,
        outcome: ToolOutcome,
        result: str,
        skill: Any,
        hinted: list[str],
        *,
        catalogue: list[dict[str, Any]],
    ) -> ToolOutcome:
        """Deliver the judged activation, then open its gates when it was
        served, or reveal the tools a name that is no skill matched
        (*hinted*, matched in the session's *catalogue*)."""
        # An activated skill's instructions go out whole (RFC §21.5); a
        # refusal, a block or a hook's replacement is bounded.
        if not (outcome.kind is OutcomeKind.SERVED and outcome.result == result):
            text = result_text(outcome.result)
            outcome = replace(outcome, result=self._bound_call_result(call, text, served=False))
        # The call ID belongs to the current connection. Deliver before
        # native reconfiguration can replace that connection.
        delivered = await deliver_once(call, door, outcome)
        if not (delivered and outcome.kind is OutcomeKind.SERVED):
            return outcome
        if skill is not None:
            await self._open_skill_gates(call.session, skill)
        elif hinted:
            await self._reveal_names(call.session, hinted, catalogue)
        return outcome

    async def _reveal_names(
        self, session: VoiceSession, names: list[str], catalogue: list[dict[str, Any]]
    ) -> None:
        """Reveal the tools a served ``find_tools`` call matched, or an
        activation's hint named, in the session's *catalogue*.

        A reconfiguration that gave the session another catalogue while the
        call was judged (a handoff) reset the reveal window, and the names
        were matched in the old one: nothing is revealed. Their observers
        judged the call before it went out, so a failed reconfiguration is
        logged by :meth:`_reveal_tools`, never reported to them.
        """
        if self._session_base_tools(session.id) is not catalogue:
            logger.debug("Reveal for session %s dropped: its catalogue changed", session.id)
            return
        search = self._tool_search_support
        if (
            search is not None
            and search.expose(session.id, names)
            and self._provider.supports_mid_session_reconfigure
        ):
            await self._reveal_tools(session)

    async def _open_skill_gates(self, session: VoiceSession, skill: Any) -> None:
        """Give the session a delivered activation's rules, then commit it."""
        support = self._skill_support
        if self._provider.supports_mid_session_reconfigure:
            base_tools = self._session_base_tools(session.id)
            visible = self._compose_session_tools(session, base_tools, pending_skill=skill)
            addendum = support.activated_skills_prompt(session.id, skill)
            if addendum or skill.metadata.gated_tool_names:
                prompt = self._compose_session_prompt(
                    session,
                    session.metadata.get("system_prompt", self._system_prompt),
                    pending_skill=skill,
                )
                await self._provider.reconfigure(session, tools=visible, system_prompt=prompt)
        if session.state != VoiceSessionState.ENDED:
            support.commit_activation(session.id, skill)

    async def _submit_realtime_tool_result(
        self, call: RealtimeToolCall, result: str, *, failed: bool = False
    ) -> bool:
        """Send *call*'s result, marked as an error when the call *failed* and
        the provider's protocol can say so (RFC §12.4); whether it reached a
        live session.

        False when the session ended.
        """
        session = call.session
        if session.state == VoiceSessionState.ENDED:
            return False
        self._expect_provider_output(session.id)
        await submit_tool_outcome(self._provider, session, call.call_id, result, failed=failed)
        return session.state != VoiceSessionState.ENDED

    async def _serve_tool_search(
        self, call: RealtimeToolCall, door: ToolCallDoor, carrying: RoomContext | None
    ) -> ToolOutcome:
        """Serve a Tool Search call as ``activate_skill`` is served:
        ON_TOOL_CALL decides, then delivery, then the reveal (RFC §6.4).

        The answer is computed under the session's configuration lock, so an
        activation or a handoff in progress settles first. The SYNC hooks then
        run on it outside the lock (a hook may request a handoff), so a hook
        that blocks ``find_tools`` blocks the reveal too: the model reads the
        refusal and the session declares nothing new. Under the lock again the
        result goes out before the session is reconfigured to declare its
        matches: the call id belongs to the current connection, and a provider
        update can replace it.
        """
        session = call.session
        lock = self._session_config_locks.get(session.id)
        if lock is None:
            return ended_outcome(call)
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return ended_outcome(call)
            catalogue = self._session_base_tools(session.id)
            result, names = await self._tool_search_support.handle_tool_call(
                call.name, call.arguments, session.id
            )
        served = ToolOutcome(OutcomeKind.SERVED, result)
        judged = await judge_tool_call(self, call, served, carrying)
        # The schema list_tools(name=...) serves goes out whole; a refusal or
        # a hook's replacement in its place is bounded (RFC §21.5).
        whole = judged.kind is OutcomeKind.SERVED and judged.result == result
        text = self._bound_call_result(call, result_text(judged.result), served=whole)
        outcome = replace(judged, result=text)
        async with lock:
            if session.state == VoiceSessionState.ENDED:
                return ended_outcome(call)
            delivered = await deliver_once(call, door, outcome)
            if delivered and outcome.kind is OutcomeKind.SERVED and names:
                await self._reveal_names(session, names, catalogue)
        logger.info(
            "Tool-search %s(%s) %s for session %s (%d matches)",
            call.name,
            call.call_id,
            outcome.kind,
            session.id,
            len(names),
        )
        return outcome

    async def _reveal_tools(self, session: VoiceSession) -> None:
        """Declare to the session the tools Tool Search revealed, logging a
        reconfiguration that failed: the model already read the call's result
        and its observers heard of it."""
        with self._state_lock:
            base_tools = self._session_tools.get(session.id, self._tools or [])
        try:
            await self._provider.reconfigure(
                session,
                tools=self._compose_session_tools(session, base_tools),
                system_prompt=self._compose_session_prompt(
                    session, session.metadata.get("system_prompt", self._system_prompt)
                ),
            )
        except Exception:
            logger.exception("Revealing tools to session %s failed", session.id)
