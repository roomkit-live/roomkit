"""External tool handler ABC for provider-executed tools.

When an AI provider executes tools internally (e.g., Claude Code sandbox),
the :class:`ExternalToolHandler` ABC provides control (approve/deny/modify)
and observability (post-execution hooks) without RoomKit trying to execute
the tools locally.

The ABC is transport-agnostic — subclasses decide HOW tool events arrive
(HTTP, WebSocket, in-process queue, etc.). The framework injects hook
callbacks so the handler can fire :attr:`~roomkit.models.enums.HookTrigger.BEFORE_TOOL_USE`
and :attr:`~roomkit.models.enums.HookTrigger.ON_TOOL_CALL` hooks.

Usage::

    from roomkit.tools.external import ExternalToolHandler, ToolDecision

    class MyToolHandler(ExternalToolHandler):
        async def process_tool_call(self, tool_name, tool_input, **kwargs):
            # Apply policy, ask user, etc.
            if tool_name == "Bash":
                return ToolDecision(approved=False, reason="Bash not allowed")
            return ToolDecision(approved=True)

        async def on_tool_result(self, tool_name, tool_input, result, **kwargs):
            print(f"Tool {tool_name} returned: {result[:100]}")

    agent = Agent(
        "my-agent",
        provider=provider,
        external_tool_handler=MyToolHandler(),
    )
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent
from roomkit.tools.policy import ToolPolicy, call_admitted, policy_refusal
from roomkit.tools.result import cancelled_tool_error, pre_execution_denial

logger = logging.getLogger("roomkit.tools.external")


@dataclass
class ToolDecision:
    """Decision for an external tool call.

    Returned by :meth:`ExternalToolHandler.process_tool_call`.
    """

    approved: bool
    """Whether the tool call is allowed to proceed."""

    modified_input: dict[str, Any] | None = None
    """If set, overrides the original tool input."""

    result: str | None = None
    """If set, overrides the tool result (e.g. human-provided answer)."""

    reason: str = ""
    """Human-readable reason for the decision (used in deny messages)."""

    detail: str | None = None
    """What failed when a refusal comes from a failure (a ``BEFORE_TOOL_USE``
    hook that failed closed): for the observers (``ToolCallEvent.error_detail``)
    and the log, never for the agent (RFC §9.3)."""


@dataclass(frozen=True)
class BeforeToolDecision:
    """What the ``BEFORE_TOOL_USE`` hooks decided about one tool call.

    Mirrors what ``ON_TOOL_CALL`` already offers on the way out: that hook
    can replace a tool's *result*, this one can replace its *arguments*.
    A redaction hook needs both halves — it hands the model tokenised text
    and must put the real values back before the tool acts on them.

    Truthiness is :attr:`allowed`, so a handler can gate on the decision
    directly (``if not decision:``) and only reach for :attr:`arguments`
    when it cares about a rewrite.
    """

    allowed: bool
    """Whether the tool call may proceed."""

    arguments: dict[str, Any] | None = None
    """Rewritten arguments, or ``None`` to keep the model's own."""

    detail: str | None = None
    """The error of a hook that failed closed and so refused the call, for the
    observers only (``ToolCallEvent.error_detail``), never for the model."""

    reason: str | None = None
    """A BLOCK's reason, the hook's own words for the model (RFC §9.3);
    ``None`` when the hook gave none or failed closed."""

    def __bool__(self) -> bool:
        return self.allowed


def takes_keyword(method: Callable[..., Any], name: str) -> bool:
    """Whether *method* accepts the keyword *name* (named, or ``**kwargs``):
    a handler override that does not is never handed it (RFC §9.3)."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def detail_keyword(
    handler: ExternalToolHandler,
    report: Literal["on_tool_refused", "on_tool_result"],
    detail: str | None,
) -> dict[str, Any]:
    """The ``detail`` keyword a door hands *handler*'s *report* method (a
    refusal's :meth:`~ExternalToolHandler.on_tool_refused`, or the
    :meth:`~ExternalToolHandler.on_tool_result` of a call run past a
    refusal): only when there is one and the override takes it. One that
    does not still makes its report, without what failed, and the log says
    so."""
    if detail is None:
        return {}
    if takes_keyword(getattr(handler, report), "detail"):
        return {"detail": detail}
    warn_once(
        handler,
        f"{report}.detail",
        f"%s.{report} takes no 'detail': what failed is left out of its reports; "
        "accept **kwargs and hand it on to the ON_TOOL_CALL report",
    )
    return {}


@dataclass
class _ReportWatch:
    """A handler's report of one call being awaited, and whether it reached
    ON_TOOL_CALL."""

    call_id: str
    reached: bool = False


_report_watch: ContextVar[_ReportWatch | None] = ContextVar("_report_watch", default=None)


def note_handler_report(call_id: str) -> None:
    """Mark that a handler's report of call *call_id* reached ON_TOOL_CALL's
    observers, for the :func:`handler_reported` running it, if any."""
    watch = _report_watch.get()
    if watch is not None and watch.call_id == call_id:
        watch.reached = True


async def handler_reported(report: Callable[[], Awaitable[Any]], what: str, call_id: str) -> bool:
    """Run an external handler's report of call *call_id*: ``False`` when it
    raised, at the call (an override that cannot take the arguments) or
    while running, before that report reached ON_TOOL_CALL's observers;
    logged with *what* it was reporting. The door then reports the call
    itself, so a call is reported once on every door (RFC §9.3)."""
    watch = _ReportWatch(call_id)
    token = _report_watch.set(watch)
    try:
        await report()
    except Exception:
        logger.exception("External tool handler failed %s", what)
        return watch.reached
    finally:
        _report_watch.reset(token)
    return True


_WARNED: set[tuple[type, str]] = set()


def warn_once(handler: ExternalToolHandler, keyword: str, message: str) -> None:
    """Log *message* (``%s``, the handler's class) once per handler class and
    keyword: an override that cannot take *keyword* says so once, not at
    every call."""
    key = (type(handler), keyword)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, type(handler).__name__)


# Callback type injected by the framework.
BeforeToolCallback = Callable[[ToolCallEvent], Awaitable["BeforeToolDecision"]]
# Its return is discarded: an external call already ran outside the channel,
# and its firing is a report, whose override a hook cannot apply (RFC §9.3).
OnToolCallback = Callable[[ToolCallEvent], Awaitable[Any]]


class ExternalToolHandler(ABC):
    """Controls and observes tools executed by an external provider.

    Subclasses decide HOW tool events arrive (HTTP callback, WebSocket,
    in-process, queue, etc.). The ABC defines the contract for processing
    them and bridging to RoomKit's hook system.

    Lifecycle:
        1. ``register_channel()`` injects ``_before_tool_hook`` and ``_on_tool_hook``
        2. ``start()`` is called (subclass sets up transport)
        3. External provider sends tool events via transport
        4. Subclass calls ``process_tool_call()`` / ``on_tool_result()``
        5. These fire RoomKit hooks via injected callbacks
        6. ``stop()`` is called on shutdown

    Attributes set by the framework (do not override):
        _before_tool_hook: Fires BEFORE_TOOL_USE sync hooks. Returns a truthy
            :class:`BeforeToolDecision` when allowed, optionally with arguments.
        _on_tool_hook: Fires ON_TOOL_CALL sync hooks, as a report: its return
            is discarded.
        _channel_id: Channel ID this handler is attached to.
    """

    _before_tool_hook: BeforeToolCallback | None = None
    _on_tool_hook: OnToolCallback | None = None
    _channel_id: str = ""

    @abstractmethod
    async def process_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        room_id: str | None = None,
    ) -> ToolDecision:
        """Called BEFORE the external provider executes a tool.

        The implementation should apply its own logic (policy checks,
        user approval, etc.) and return a :class:`ToolDecision`.

        The default implementation fires ``BEFORE_TOOL_USE`` hooks via
        the injected callback. Subclasses that override this should call
        ``await self._fire_before_hook(...)`` to preserve hook integration.

        This method MAY block (e.g., waiting for user approval via UI).
        The external provider's hook (e.g., Claude Code's ``PreToolUse``)
        blocks until this returns.

        Args:
            tool_name: Name of the tool (e.g., "Write", "Bash", "Read").
            tool_input: Tool arguments as a dict.
            tool_call_id: Provider-assigned ID for this tool call.
            job_id: Job identifier for tracking.
            session_id: Session identifier.
            tenant_id: Tenant identifier for multi-tenant isolation.
            room_id: RoomKit room ID.

        Returns:
            ToolDecision with approved/denied status and optional modified input.
        """

    @abstractmethod
    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        *,
        is_error: bool = False,
        tool_call_id: str = "",
        job_id: str | None = None,
        room_id: str | None = None,
        refused_but_ran: bool = False,
        detail: str | None = None,
    ) -> None:
        """Called AFTER the external provider executed a tool.

        Fires ``ON_TOOL_CALL`` hooks via the injected callback for
        observability. Subclasses that override this should call
        ``await self._fire_on_tool_hook(...)`` to preserve hook integration,
        within this call, and MUST pass ``is_error`` and ``tool_call_id`` on
        to it: ``is_error`` is the outcome of the call, which nothing
        downstream can recover from the result body, and ``tool_call_id`` is
        how the turn knows this call's one report was made (RFC §9.3).

        Args:
            tool_name: Name of the tool.
            tool_input: Tool arguments.
            result: Tool execution result (stdout, content, etc.).
            is_error: Whether the tool execution failed.
            tool_call_id: Provider-assigned ID for this tool call.
            job_id: Job identifier.
            room_id: RoomKit room ID.
            refused_but_ran: The call ran although RoomKit refused it (an
                ACP agent past a rejected permission); passed only then, so
                an override should take ``**kwargs`` and hand it on to
                ``_fire_on_tool_hook``.
            detail: What failed in that refusal (a hook that failed closed,
                or this handler raising while it decided); passed only with
                ``refused_but_ran`` and when there is one, for
                ``_fire_on_tool_hook``'s ``error_detail``.
        """

    async def on_tool_cancelled(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        room_id: str | None = None,
    ) -> None:
        """Called when the turn cut a call this handler was to decide, before
        its outcome was reported.

        The turn's cancellation, or a transport that stopped reading before
        the call was decided, ended it: the handler may never have been asked
        about it, or may still be deciding it. On an AI channel the turn's
        cancellation also cancels a pending :meth:`process_tool_call`; an ACP
        agent's permission request runs in the connection's own task and is
        not cancelled, so an override resolves it itself (it denies the
        approval still pending). The call has no result: it is reported to
        ``ON_TOOL_CALL``'s observers as cancelled, as every channel reports a
        call it cut (RFC §9.3). Override it to withdraw what the call left
        pending (an approval prompt, say), and call
        ``await super().on_tool_cancelled(...)`` to keep the report. It runs in
        the turn's teardown: keep it short.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool arguments.
            tool_call_id: Provider-assigned ID for this tool call.
            job_id: Job identifier.
            room_id: RoomKit room ID.
        """
        await self._fire_on_tool_hook(
            tool_name,
            tool_input,
            cancelled_tool_error(tool_name, "The turn ended before its result."),
            is_error=True,
            cancelled=True,
            tool_call_id=tool_call_id,
            room_id=room_id,
        )

    async def on_tool_refused(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        reason: str,
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        room_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Called when this handler refused a call: :meth:`process_tool_call`
        denied it, or an ACP permission it decided was rejected.

        The call never ran. It is reported to ``ON_TOOL_CALL``'s observers as
        refused, *reason* being what the model reads, and reaches no SYNC hook,
        as every channel reports a call it refused (RFC §9.3). Override it to
        record the refusal, and call ``await super().on_tool_refused(...)`` to
        keep the report. :meth:`on_tool_result` no longer hears of a refusal.

        Args:
            tool_name: Name of the tool.
            tool_input: Tool arguments.
            reason: What the model reads of the refusal.
            tool_call_id: Provider-assigned ID for this tool call.
            job_id: Job identifier.
            room_id: RoomKit room ID.
            detail: What failed, when the refusal came from a failure
                (:attr:`ToolDecision.detail`); passed only then, so an
                override should take ``**kwargs``.
        """
        await self._fire_on_tool_hook(
            tool_name,
            tool_input,
            reason,
            is_error=True,
            refused=True,
            error_detail=detail,
            tool_call_id=tool_call_id,
            room_id=room_id,
        )

    @property
    def channel_id(self) -> str:
        """The channel this handler serves, or ``""`` before registration.

        A handler is wired to one channel (``register_channel`` injects that
        channel's hook callbacks), so this is the answer to "who is asking?"
        — the question a permission prompt must put to a human when several
        agents share one terminal. Empty until the channel is registered:
        handlers are usually constructed before that, so read it when a tool
        call arrives, not in ``__init__``.
        """
        return self._channel_id

    async def start(self) -> None:  # noqa: B027
        """Start receiving tool events.

        Override for transports that need setup (HTTP server, WebSocket
        connection, queue subscription, etc.). Default: no-op.
        """

    async def stop(self) -> None:  # noqa: B027
        """Stop and release resources. Default: no-op."""

    # ── Hook bridge helpers ───────────────────────────────────────────

    async def _fire_before_hook(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        room_id: str | None = None,
    ) -> BeforeToolDecision:
        """Fire BEFORE_TOOL_USE hooks.

        The returned decision is truthy when the call is allowed, so
        ``if not await self._fire_before_hook(...)`` still reads correctly;
        read :attr:`BeforeToolDecision.arguments` to honour a hook that
        rewrote the input (pass it back as ``ToolDecision.modified_input``).
        """
        if self._before_tool_hook is None:
            return BeforeToolDecision(allowed=True)
        event = ToolCallEvent(
            channel_id=self._channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=tool_call_id,
            name=tool_name,
            arguments=tool_input,
            result=None,
            room_id=room_id,
        )
        return await self._before_tool_hook(event)

    async def _fire_on_tool_hook(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        *,
        is_error: bool = False,
        cancelled: bool = False,
        refused: bool = False,
        error_detail: str | None = None,
        tool_call_id: str = "",
        room_id: str | None = None,
        refused_but_ran: bool = False,
    ) -> None:
        """Fire ON_TOOL_CALL hooks for observation.

        ``is_error`` is the external provider's own verdict on the call, and it
        MUST be forwarded: this boundary is the only place that holds it. The
        body is the provider's — a terminal's stderr, an SDK's message — and
        recognising a failure in it is guesswork, so an observer handed the
        body alone reads a failed tool as a completed one. ``cancelled`` marks
        a call the turn cut before its outcome (:meth:`on_tool_cancelled`),
        ``refused`` one this handler refused (:meth:`on_tool_refused`): either
        reaches the observers only. ``error_detail`` is what failed, for the
        observers and the log, never the model (RFC §9.3).
        """
        if self._on_tool_hook is None:
            return
        event = ToolCallEvent(
            channel_id=self._channel_id,
            channel_type=ChannelType.AI,
            tool_call_id=tool_call_id,
            name=tool_name,
            arguments=tool_input,
            result=result,
            room_id=room_id,
            is_error=is_error,
            cancelled=cancelled,
            refused=refused,
            error_detail=error_detail,
            refused_but_ran=refused_but_ran,
        )
        await self._on_tool_hook(event)


class PolicyExternalToolHandler(ExternalToolHandler):
    """Auto-approve tools based on a ToolPolicy. No UI, no blocking.

    Useful for standalone/testing scenarios where no human approval is needed.
    Fires all RoomKit hooks for observability.

    Usage::

        handler = PolicyExternalToolHandler(
            policy=ToolPolicy(deny=["Bash", "Write"]),
        )
    """

    def __init__(self, policy: Any = None) -> None:
        self._policy: ToolPolicy | None = policy

    async def process_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str = "",
        job_id: str | None = None,
        session_id: str | None = None,
        tenant_id: str | None = None,
        room_id: str | None = None,
    ) -> ToolDecision:
        # The policy before BEFORE_TOOL_USE, as every gate orders them (RFC
        # §21.1): an approval hook is never asked about a tool it may not run.
        # Under an MCP alias too: the agent names MCP tools that way (RFC §21.1).
        if self._policy and not call_admitted(self._policy, tool_name):
            return ToolDecision(approved=False, reason=policy_refusal(tool_name))

        decision = await self._fire_before_hook(
            tool_name, tool_input, tool_call_id=tool_call_id, room_id=room_id
        )
        if not decision:
            if decision.detail is not None:
                # A hook that failed closed: its error for the log and the
                # observers, never the agent.
                logger.warning("BEFORE_TOOL_USE refused %s: %s", tool_name, decision.detail)
            return ToolDecision(
                approved=False,
                reason=pre_execution_denial(tool_name, decision.reason),
                detail=decision.detail,
            )

        return ToolDecision(approved=True, modified_input=decision.arguments)

    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        *,
        is_error: bool = False,
        tool_call_id: str = "",
        job_id: str | None = None,
        room_id: str | None = None,
        refused_but_ran: bool = False,
        detail: str | None = None,
    ) -> None:
        await self._fire_on_tool_hook(
            tool_name,
            tool_input,
            result,
            is_error=is_error,
            error_detail=detail,
            tool_call_id=tool_call_id,
            room_id=room_id,
            refused_but_ran=refused_but_ran,
        )
