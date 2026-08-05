"""Human-in-the-loop tool primitive for pausing tool execution until human input.

Provides two layers:

* :class:`HumanInputHandler` — core async primitive that manages pending
  requests (create / wait / resolve / reject).
* :class:`HumanInputToolHandler` — :data:`ToolHandler` wrapper that composes
  with :func:`compose_tool_handlers` for the native AIChannel path.

For the external-provider path (Claude Code sandbox), applications use
:class:`HumanInputHandler` directly inside their
:class:`~roomkit.tools.external.ExternalToolHandler` implementation.

Usage (native)::

    from roomkit.tools.human_input import HumanInputToolHandler
    from roomkit.tools.compose import compose_tool_handlers

    human = HumanInputToolHandler(
        tool_names={"AskUserQuestion"},
        timeout=300,
    )
    ai = AIChannel(
        "agent",
        provider=provider,
        tool_handler=other_handler,
        human_input_handler=human,
    )

    # When user answers (from REST endpoint, WebSocket, etc.):
    human.handler.resolve(pending_id, answer_json)

Usage (external / Claude Code)::

    from roomkit.tools.human_input import HumanInputHandler

    handler = HumanInputHandler()

    # Inside ExternalToolHandler.process_tool_call():
    pending = await handler.create("AskUserQuestion", arguments, room_id=room_id, ...)
    result = await handler.wait(pending.pending_id, timeout=300)

When the runtime owns its own tool loop and the answer travels back some
other way — nobody calls ``wait()`` — say so, and free the request when
done::

    pending = await handler.create_detached("AskUserQuestion", arguments, ...)
    ...
    handler.release(pending.pending_id)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from roomkit.core.task_utils import log_task_exception
from roomkit.models.enums import ChannelType
from roomkit.models.pending_input import PendingInput, PendingInputEvent, PendingInputStatus
from roomkit.providers.ai.base import AITool

logger = logging.getLogger("roomkit.tools.human_input")

# Callback type: fires when input is needed.  Set by the framework
# (via register_channel hook builder) or by the application directly.
# Returns True to proceed, False to deny the pending request.
OnInputRequiredCallback = Callable[[PendingInputEvent], Awaitable[bool]]


# ── Tool-call context propagation ────────────────────────────────────
#
# The ToolHandler protocol is (name, arguments) → str — it does not
# receive room_id, tool_call_id, or channel_id.  This contextvar
# bridges the gap: _ai_tools._run_one() sets it before calling the
# handler, and HumanInputToolHandler reads it.
#
# Safe with asyncio.gather because gather creates Tasks with copied
# contexts.


@dataclass
class ToolCallContext:
    """Contextvar payload carrying tool-call metadata.

    ``structured_content`` is the reverse channel: the ToolHandler contract
    returns only a string, but MCP tools can produce a structured result
    (``CallToolResult.structuredContent``) that UI surfaces need verbatim —
    the LLM-facing string may be truncated/evicted when large. A handler
    that has one sets it here; ``_run_one()`` reads it back after the call
    and carries it on the tool-call events untouched by eviction.
    """

    room_id: str = ""
    tool_call_id: str = ""
    channel_id: str = ""
    structured_content: dict[str, Any] | None = None


_current_tool_call: contextvars.ContextVar[ToolCallContext | None] = contextvars.ContextVar(
    "_current_tool_call", default=None
)


class HumanInputHandler:
    """Manages pending human input requests.

    Core lifecycle::

        pending = await handler.create("AskUser", args, room_id="r1", ...)
        # → request is answerable from here on; the
        #   ON_USER_INPUT_REQUIRED notification runs alongside
        result  = await handler.wait(pending.pending_id, timeout=300)
        # → blocks until resolve() / reject() / timeout

    Two invariants the caller can rely on:

    * **The notification never gates the answer.** ``create()`` arms the
      request and returns; the ``_on_input_required`` callback runs in a
      background task.  A human who answers while that callback is still
      running — a slow WebSocket broadcast, a hook burning its 30 s budget —
      is answering a request that is already listening.  A denial coming back
      from the callback rejects the request, and ``wait()`` reports it.
    * **A consumed outcome stays readable.** ``wait()`` retires the request
      into a bounded retention (*retention* entries, newest kept) and replays
      it for a second ``wait()``.  Only a genuinely unknown id raises
      ``ValueError``, so a caller that drops its own bookkeeping cannot turn
      an answer that arrived into a hard failure.

    The ``_on_input_required`` callback is injected by the framework
    (via ``register_channel`` hook builder) or set by the application
    directly.
    """

    def __init__(self, *, retention: int = 128) -> None:
        self._pending: dict[str, PendingInput] = {}
        self._recent: OrderedDict[str, PendingInput] = OrderedDict()
        self._retention = max(0, retention)
        self._notify_tasks: set[asyncio.Task[None]] = set()
        self._on_input_required: OnInputRequiredCallback | None = None

    @property
    def pending(self) -> dict[str, PendingInput]:
        """Active pending requests (read-only snapshot)."""
        return dict(self._pending)

    async def create(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        room_id: str = "",
        tool_call_id: str = "",
        channel_id: str = "",
        channel_type: ChannelType = ChannelType.AI,
    ) -> PendingInput:
        """Register a new pending input request and schedule the callback.

        Returns as soon as the request is answerable — the
        ``_on_input_required`` callback runs in a background task and a
        denial from it rejects the request wherever ``wait()`` has got to.

        ``wait()`` owns this request's cleanup; for a request no one will
        wait on, use :meth:`create_detached`.
        """
        return self._arm(
            tool_name,
            arguments,
            room_id=room_id,
            tool_call_id=tool_call_id,
            channel_id=channel_id,
            channel_type=channel_type,
            detached=False,
        )

    async def create_detached(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        room_id: str = "",
        tool_call_id: str = "",
        channel_id: str = "",
        channel_type: ChannelType = ChannelType.AI,
    ) -> PendingInput:
        """Register a pending request that no one will :meth:`wait` on.

        For runtimes that own their own tool loop — a Claude Code sandbox,
        say — where ``create()`` exists to raise the request and the answer
        travels back another way.  Nothing here retires the request, so its
        creator MUST call :meth:`release` when done with it; otherwise the
        entry lives as long as the handler.
        """
        return self._arm(
            tool_name,
            arguments,
            room_id=room_id,
            tool_call_id=tool_call_id,
            channel_id=channel_id,
            channel_type=channel_type,
            detached=True,
        )

    def _arm(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        room_id: str,
        tool_call_id: str,
        channel_id: str,
        channel_type: ChannelType,
        detached: bool,
    ) -> PendingInput:
        """Register the request, then schedule its notification."""
        pending_id = uuid4().hex
        pending = PendingInput(
            pending_id=pending_id,
            tool_name=tool_name,
            arguments=arguments,
            room_id=room_id,
            tool_call_id=tool_call_id,
            channel_id=channel_id,
            detached=detached,
        )
        self._pending[pending_id] = pending

        if self._on_input_required is not None:
            event = PendingInputEvent(
                pending_id=pending_id,
                tool_name=tool_name,
                arguments=arguments,
                room_id=room_id,
                tool_call_id=tool_call_id,
                channel_id=channel_id,
                channel_type=channel_type,
            )
            task = asyncio.get_running_loop().create_task(
                self._notify(event), name=f"human-input-notify-{pending_id}"
            )
            self._notify_tasks.add(task)
            task.add_done_callback(self._notify_tasks.discard)
            task.add_done_callback(log_task_exception)

        return pending

    async def _notify(self, event: PendingInputEvent) -> None:
        """Run the ON_USER_INPUT_REQUIRED callback off the waiting path."""
        callback = self._on_input_required
        if callback is None:
            return
        try:
            allowed = await callback(event)
        except Exception:
            # Fail-open, as RFC §9.3 requires of this trigger: a broken
            # notification loses the notification, not the request.
            logger.exception("_on_input_required callback failed for pending %s", event.pending_id)
            return
        if not allowed:
            self.reject(event.pending_id, "Denied by ON_USER_INPUT_REQUIRED hook")

    async def wait(self, pending_id: str, *, timeout: float = 300) -> str:
        """Block until the request is resolved, rejected, or times out.

        An outcome already reached and consumed is replayed from the
        retention, so waiting twice — or waiting after someone else dropped
        their own record of the request — reports what happened rather than
        an error.

        Returns:
            The result string on resolution.

        Raises:
            asyncio.TimeoutError: If the timeout expires, or if a retained
                request had timed out.
            RuntimeError: If the request was rejected.
            ValueError: If *pending_id* is unknown — never seen, or retired
                long enough ago to have been evicted from the retention.
        """
        pending = self._pending.get(pending_id)
        if pending is None:
            retained = self._recent.get(pending_id)
            if retained is None:
                msg = f"No pending request with id {pending_id}"
                raise ValueError(msg)
            return self._outcome(retained)

        try:
            await asyncio.wait_for(pending._event.wait(), timeout=timeout)
        except TimeoutError:
            pending.status = PendingInputStatus.TIMED_OUT
            self._retire(pending_id)
            raise

        self._retire(pending_id)
        return self._outcome(pending)

    def release(self, pending_id: str) -> bool:
        """Drop a request whose cleanup the caller owns.

        The counterpart of :meth:`create_detached`.  A request still
        unanswered is rejected on the way out, so a stray waiter unblocks
        instead of hanging; the outcome goes to the retention either way and
        stays readable by :meth:`wait`.

        Returns ``True`` if an active request was dropped.
        """
        pending = self._pending.get(pending_id)
        if pending is None:
            return False
        if pending.status == PendingInputStatus.PENDING:
            pending.reject_reason = "Released before an answer arrived"
            pending.status = PendingInputStatus.REJECTED
            pending._event.set()
        self._retire(pending_id)
        return True

    def _retire(self, pending_id: str) -> None:
        """Move a finished request out of the active set, into retention."""
        pending = self._pending.pop(pending_id, None)
        if pending is None or self._retention == 0:
            return
        self._recent[pending_id] = pending
        while len(self._recent) > self._retention:
            self._recent.popitem(last=False)

    @staticmethod
    def _outcome(pending: PendingInput) -> str:
        """Report a terminal request as its result or its failure."""
        if pending.status == PendingInputStatus.REJECTED:
            raise RuntimeError(pending.reject_reason or "Request rejected")

        if pending.status == PendingInputStatus.RESOLVED:
            return pending.result or ""

        if pending.status == PendingInputStatus.TIMED_OUT:
            raise TimeoutError

        msg = f"Unexpected pending status: {pending.status}"
        raise RuntimeError(msg)

    def resolve(self, pending_id: str, result: str) -> bool:
        """Resolve a pending request with a result.

        Returns ``True`` if the request was found and resolved.
        """
        pending = self._pending.get(pending_id)
        if pending is None or pending.status != PendingInputStatus.PENDING:
            return False
        pending.result = result
        pending.status = PendingInputStatus.RESOLVED
        pending._event.set()
        return True

    def reject(self, pending_id: str, reason: str = "") -> bool:
        """Reject a pending request.

        Returns ``True`` if the request was found and rejected.
        """
        pending = self._pending.get(pending_id)
        if pending is None or pending.status != PendingInputStatus.PENDING:
            return False
        pending.reject_reason = reason
        pending.status = PendingInputStatus.REJECTED
        pending._event.set()
        return True


class HumanInputToolHandler:
    """ToolHandler wrapper that blocks on human input for specified tools.

    Composes with other handlers via
    :func:`~roomkit.tools.compose.compose_tool_handlers`.  Falls through
    (returns ``"Unknown tool"`` error) for non-matching tool names so
    the compose chain continues to the next handler.

    Pass this to :class:`~roomkit.channels.ai.AIChannel` via the
    ``human_input_handler`` parameter — the channel auto-composes it
    and the framework injects the ``ON_USER_INPUT_REQUIRED`` hook
    callback at registration time.
    """

    def __init__(
        self,
        tool_names: set[str],
        timeout: float = 300,
        handler: HumanInputHandler | None = None,
        tool_definitions: list[AITool] | None = None,
    ) -> None:
        if not tool_names:
            msg = "tool_names must not be empty"
            raise ValueError(msg)
        self.tool_names = set(tool_names)
        self.timeout = timeout
        self._handler = handler or HumanInputHandler()
        self._tool_definitions: list[AITool] = []
        if tool_definitions:
            for td in tool_definitions:
                if td.name not in self.tool_names:
                    msg = f"Tool definition '{td.name}' not in tool_names {self.tool_names}"
                    raise ValueError(msg)
            self._tool_definitions = list(tool_definitions)

    @property
    def handler(self) -> HumanInputHandler:
        """The underlying :class:`HumanInputHandler` for resolve/reject access."""
        return self._handler

    @property
    def tools(self) -> list[AITool]:
        """Tool definitions to inject into the AI context."""
        return list(self._tool_definitions)

    async def __call__(self, name: str, arguments: dict[str, Any]) -> str:
        """ToolHandler protocol — blocks on matching tools, falls through otherwise."""
        if name not in self.tool_names:
            return json.dumps({"error": f"Unknown tool: {name}"})

        ctx = _current_tool_call.get()
        room_id = ctx.room_id if ctx else ""
        tool_call_id = ctx.tool_call_id if ctx else ""
        channel_id = ctx.channel_id if ctx else ""

        pending = await self._handler.create(
            tool_name=name,
            arguments=arguments,
            room_id=room_id,
            tool_call_id=tool_call_id,
            channel_id=channel_id,
        )

        try:
            return await self._handler.wait(pending.pending_id, timeout=self.timeout)
        except TimeoutError:
            return json.dumps(
                {"error": f"Human input timed out after {self.timeout}s for tool '{name}'"}
            )
        except RuntimeError as exc:
            return json.dumps({"error": f"Human input rejected: {exc}"})
