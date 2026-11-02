"""Human-in-the-loop tool primitive for pausing tool execution until human input.

Provides two layers:

* :class:`HumanInputHandler` — core async primitive that manages pending
  requests (create / wait / resolve / reject).
* :class:`HumanInputToolHandler` — the tools that ask a person, which an
  AIChannel, a RealtimeVoiceChannel or a conference's realtime configuration
  serves itself when given it as ``human_input_handler=``.

For the external-provider path (Claude Code sandbox), applications use
:class:`HumanInputHandler` directly inside their
:class:`~roomkit.tools.external.ExternalToolHandler` implementation.

Usage (native)::

    from roomkit.tools import HumanInputToolHandler, compose_tool_handlers

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

    from roomkit.tools import HumanInputHandler

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
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from roomkit.core.exceptions import (
    HumanInputRejectedError,
    ToolFailedError,
    ToolRefusedError,
    UnservedToolCallError,
)
from roomkit.core.task_utils import log_task_exception
from roomkit.models.enums import ChannelType
from roomkit.models.pending_input import PendingInput, PendingInputEvent, PendingInputStatus
from roomkit.providers.ai.base import AITool
from roomkit.tools._turn_inputs import TurnInputs, current_turn_inputs
from roomkit.tools.context import _current_tool_call, current_tool_actor_id

logger = logging.getLogger("roomkit.tools.human_input")

# Callback type: fires when input is needed.  Set by the framework
# (via register_channel hook builder) or by the application directly.
# Returns True to proceed, False to deny the pending request.
OnInputRequiredCallback = Callable[[PendingInputEvent], Awaitable[bool]]


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
    * **A recorded outcome stays readable.** A request settling — answered,
      rejected, timed out — is kept in a bounded retention (*retention*
      entries, newest kept), and ``wait()`` replays it once the request has
      left the active set.  Only a genuinely unknown id raises
      ``ValueError``, so neither a second read nor a host that keeps its own
      bookkeeping can turn an answer that arrived into a hard failure.
    * **A channel scope belongs to a channel object, not to its id.** A host
      that rebuilds the channel serving an id — the same agent re-attached to
      a second room — hands the same shared handler a succession of owners.
      Registering re-opens the scope, and the departing owner's ``close()``
      is a no-op once a newer one has taken over, so a predecessor being
      torn down cannot silence its live successor.

    The ``_on_input_required`` callback is injected by the framework
    (via ``register_channel`` hook builder) or set by the application
    directly.
    """

    def __init__(self, *, retention: int = 128) -> None:
        self._pending: dict[str, PendingInput] = {}
        self._recent: OrderedDict[str, PendingInput] = OrderedDict()
        self._retention = max(0, retention)
        self._notify_tasks: set[asyncio.Task[None]] = set()
        self._notify_channels: dict[asyncio.Task[None], str] = {}
        self._on_input_required: OnInputRequiredCallback | None = None
        self._on_input_required_by_channel: dict[str, OnInputRequiredCallback] = {}
        self._registrations: dict[str, int] = {}
        self._closed_channels: set[str] = set()
        self._closed = False
        # The requests opened in a turn the turn backend runs: their owner
        # keeps the record that answers them (RFC §10.6.5).
        self._turn_of: dict[str, TurnInputs] = {}

    @property
    def pending(self) -> dict[str, PendingInput]:
        """Active pending requests (read-only snapshot)."""
        return dict(self._pending)

    async def close(
        self, *, channel_id: str | None = None, registration: int | None = None
    ) -> None:
        """Stop notifications and settle requests owned by one channel.

        When ``channel_id`` is omitted, all work owned by this handler is
        stopped. Channel-scoped closing lets a handler be shared safely by
        multiple :class:`~roomkit.channels.ai.AIChannel` instances.

        ``registration`` is the token :meth:`_set_on_input_required` handed
        the closing channel. Passing it makes the close belong to that channel
        object rather than to the id it used: a channel displaced from the
        registry and torn down later closes nothing, because the id is already
        serving its replacement. Omitting it closes the scope unconditionally,
        which is what a lone owner and a manual host call both want.
        """
        # Mark the scope closed before inspecting current work. ``create()``
        # and this prefix contain no suspension point, so an asyncio caller is
        # either registered in time to be rejected below or observes the
        # closed marker and cannot arm a request after the snapshot.
        if channel_id is None:
            self._closed = True
            self._on_input_required_by_channel.clear()
            self._registrations.clear()
        else:
            if registration is not None and self._registrations.get(channel_id) != registration:
                return
            self._closed_channels.add(channel_id)
            self._on_input_required_by_channel.pop(channel_id, None)
            self._registrations.pop(channel_id, None)

        for pending_id, pending in list(self._pending.items()):
            if channel_id is not None and pending.channel_id != channel_id:
                continue
            if pending.status == PendingInputStatus.PENDING:
                pending.reject_reason = "Human input handler closed"
                pending.status = PendingInputStatus.REJECTED
                pending._withdrawn = True
                pending._event.set()
            self._retire(pending_id)

        tasks = [
            task
            for task in self._notify_tasks
            if channel_id is None or self._notify_channels.get(task) == channel_id
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _set_on_input_required(self, channel_id: str, callback: OnInputRequiredCallback) -> int:
        """Register the framework callback belonging to one channel.

        The application-level ``_on_input_required`` fallback remains for
        direct handler use. Framework callbacks need their own routing table:
        one shared handler must not let the last registered AI channel replace
        every earlier channel's hook and observability context.

        A registration re-opens the id's scope: a channel object registering
        under an id whose previous owner closed is a new owner, and refusing
        it would strand the id for the handler's whole life. Returns the token
        identifying this owner, to be handed back to :meth:`close`.
        """
        self._ensure_open(channel_id, reopening=True)
        self._closed_channels.discard(channel_id)
        self._on_input_required_by_channel[channel_id] = callback
        registration = self._registrations.get(channel_id, 0) + 1
        self._registrations[channel_id] = registration
        return registration

    async def create(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        room_id: str = "",
        tool_call_id: str = "",
        channel_id: str = "",
        channel_type: ChannelType = ChannelType.AI,
        actor_id: str | None = None,
    ) -> PendingInput:
        """Register a new pending input request and schedule the callback.

        Returns as soon as the request is answerable — the
        ``_on_input_required`` callback runs in a background task and a
        denial from it rejects the request wherever ``wait()`` has got to.

        ``wait()`` owns this request's cleanup; for a request no one will
        wait on, use :meth:`create_detached`.

        ``actor_id`` names whose turn raised the request, so a notification
        layer can ask that person rather than the whole room. The native
        :class:`HumanInputToolHandler` fills it from the tool loop; a caller
        driving its own loop passes what it knows.
        """
        return self._arm(
            tool_name,
            arguments,
            room_id=room_id,
            tool_call_id=tool_call_id,
            channel_id=channel_id,
            channel_type=channel_type,
            actor_id=actor_id,
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
        actor_id: str | None = None,
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
            actor_id=actor_id,
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
        actor_id: str | None,
        detached: bool,
    ) -> PendingInput:
        """Register the request, then schedule its notification."""
        self._ensure_open(channel_id)
        pending_id = uuid4().hex
        pending = PendingInput(
            pending_id=pending_id,
            tool_name=tool_name,
            arguments=arguments,
            room_id=room_id,
            tool_call_id=tool_call_id,
            channel_id=channel_id,
            actor_id=actor_id,
            detached=detached,
        )
        self._pending[pending_id] = pending
        inputs = current_turn_inputs()
        if inputs is not None and not inputs.open:
            # Asked from a task that outlived its turn: settled here alone.
            inputs = None
        if inputs is not None:
            self._turn_of[pending_id] = inputs

        if self._callback_for(channel_id) is not None or inputs is not None:
            event = PendingInputEvent(
                pending_id=pending_id,
                tool_name=tool_name,
                arguments=arguments,
                room_id=room_id,
                tool_call_id=tool_call_id,
                channel_id=channel_id,
                channel_type=channel_type,
                actor_id=actor_id,
            )
            task = asyncio.get_running_loop().create_task(
                self._notify(event), name=f"human-input-notify-{pending_id}"
            )
            self._notify_tasks.add(task)
            self._notify_channels[task] = channel_id
            task.add_done_callback(self._forget_notify_task)
            task.add_done_callback(log_task_exception)

        return pending

    def _ensure_open(self, channel_id: str, *, reopening: bool = False) -> None:
        """Reject work created after the owning lifecycle has closed.

        ``reopening`` is for the one caller that legitimately outlives a
        closed channel scope — a fresh registration under the same id, which
        is a new owner rather than late work from the old one. The handler's
        own close still refuses everything: that lifecycle has no successor.
        """
        if self._closed:
            raise RuntimeError("Human input handler is closed")
        if not reopening and channel_id in self._closed_channels:
            raise RuntimeError(f"Human input handler is closed for channel {channel_id}")

    def _callback_for(self, channel_id: str) -> OnInputRequiredCallback | None:
        """Resolve a channel callback without misrouting an ambiguous request."""
        callback = self._on_input_required_by_channel.get(channel_id)
        if callback is not None:
            return callback
        if not channel_id and len(self._on_input_required_by_channel) == 1:
            # Preserve the convenient single-channel/manual-create behavior;
            # an empty id with several owners is ambiguous and must not be
            # attributed to whichever callback happened to register last.
            return next(iter(self._on_input_required_by_channel.values()))
        return self._on_input_required

    def _forget_notify_task(self, task: asyncio.Task[None]) -> None:
        """Drop bookkeeping for a completed notification task."""
        self._notify_tasks.discard(task)
        self._notify_channels.pop(task, None)

    async def _notify(self, event: PendingInputEvent) -> None:
        """Run the ON_USER_INPUT_REQUIRED callback off the waiting path, the
        turn's owner told first of a request it answers (RFC §10.6.5)."""
        if self._closed or event.channel_id in self._closed_channels:
            return
        await self._open_on_owner(event.pending_id)
        callback = self._callback_for(event.channel_id)
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

    async def _open_on_owner(self, pending_id: str) -> None:
        """Tell the owner of the request's turn that it is open, when it is."""
        inputs = self._turn_of.get(pending_id)
        pending = self._pending.get(pending_id)
        if inputs is None or pending is None or pending.status != PendingInputStatus.PENDING:
            return
        try:
            await inputs.opened(pending, self)
        except Exception:
            logger.exception("Pending %s could not be opened on its room owner", pending_id)

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
            HumanInputRejectedError: If the request was rejected, by a human
                or an ``ON_USER_INPUT_REQUIRED`` hook (a :class:`RuntimeError`).
            RuntimeError: If the handler gave the request up before an
                answer (its close, a release), or is closed.
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
            # Settled as the deadline passed: an answer accepted then is the
            # outcome, never retracted by the timeout (RFC §9.3).
            if pending.status == PendingInputStatus.PENDING:
                pending.status = PendingInputStatus.TIMED_OUT
        except asyncio.CancelledError:
            # Nobody waits for the answer any more (the call was cut): the
            # request is withdrawn, so a late answer is not taken for it.
            self.release(pending_id)
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
            pending._withdrawn = True
            pending._event.set()
        self._retire(pending_id)
        return True

    def _retire(self, pending_id: str) -> None:
        """Move a finished request out of the active set."""
        pending = self._pending.pop(pending_id, None)
        if pending is not None:
            self._retain(pending)

    def _retain(self, pending: PendingInput) -> None:
        """Record a settled outcome, evicting the oldest past the cap.

        Called the moment a request settles, not when someone reads it: a
        host that keeps its own bookkeeping and drops the request on
        ``resolve()`` must not be able to turn the answer it just recorded
        into a failure for whoever asked the question.
        """
        inputs = self._turn_of.pop(pending.pending_id, None)
        if inputs is not None:
            # Settled here, where the tool waits, first come first served:
            # the owner closes its record (RFC §10.6.5).
            inputs.closed(pending.pending_id, pending.status.value)
        if self._retention == 0:
            return
        self._recent[pending.pending_id] = pending
        while len(self._recent) > self._retention:
            self._recent.popitem(last=False)

    @staticmethod
    def _outcome(pending: PendingInput) -> str:
        """Report a terminal request as its result or its failure."""
        if pending.status == PendingInputStatus.REJECTED:
            reason = pending.reject_reason or "Request rejected"
            if pending._withdrawn:
                # The handler gave the request up: nobody's answer (RFC §9.3).
                raise RuntimeError(reason)
            raise HumanInputRejectedError(reason)

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
        self._retain(pending)
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
        self._retain(pending)
        return True

    def settle(self, pending_id: str, *, value: str | None, reason: str | None) -> bool:
        """Settle a request with the outcome its turn's owner recorded: an
        answer (*value*) or a rejection (*reason*), when it is still open."""
        self._turn_of.pop(pending_id, None)
        if value is not None:
            return self.resolve(pending_id, value)
        return self.reject(pending_id, reason or "")

    def withdraw(self, pending_id: str) -> None:
        """Withdraw a request whose turn ended while it was open (RFC §10.6.5):
        its waiter reads it given up, as after a release. A detached request
        is its creator's to release (:meth:`create_detached`): the turn only
        lets go of it."""
        self._turn_of.pop(pending_id, None)
        pending = self._pending.get(pending_id)
        if pending is None or not pending.detached:
            self.release(pending_id)


class HumanInputToolHandler:
    """ToolHandler wrapper that blocks on human input for specified tools.

    Composes with other handlers via
    :func:`~roomkit.tools.compose.compose_tool_handlers`.  Declines a
    non-matching tool name by raising
    :class:`~roomkit.core.exceptions.UnservedToolCallError`, so the compose
    chain continues to the next handler (RFC §21.4).

    Pass this as ``human_input_handler=`` to an
    :class:`~roomkit.channels.ai.AIChannel`, a
    :class:`~roomkit.channels.realtime_voice.RealtimeVoiceChannel` or a
    :class:`~roomkit.conference.models.ConferenceRealtimeConfig`: the channel
    declares :attr:`tools` and serves :attr:`tool_names` before the host's
    handler, on every door, under :attr:`timeout` rather than its default
    call bound, and the framework injects the ``ON_USER_INPUT_REQUIRED``
    hook callback at registration time (RFC §9.3, §21.6). Given as a plain
    ``tool_handler`` it is served as any other handler, without those rules.
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
        """ToolHandler protocol: :meth:`ask` on an AI channel's door."""
        return await self.ask(name, arguments)

    async def ask(
        self, name: str, arguments: dict[str, Any], *, channel_type: ChannelType = ChannelType.AI
    ) -> str:
        """Block on a person's answer to the call *name*; decline any other.

        *channel_type* is the door the call came through, which the request
        and its ``ON_USER_INPUT_REQUIRED`` event name. Raises
        :class:`~roomkit.core.exceptions.UnservedToolCallError` for a name
        outside :attr:`tool_names`, so a compose chain goes on to the next
        handler (RFC §21.4).

        Raises :class:`~roomkit.core.exceptions.ToolRefusedError` when the
        request is rejected (a human, an ``ON_USER_INPUT_REQUIRED`` hook), the
        model reading why, and
        :class:`~roomkit.core.exceptions.ToolFailedError` when nobody answers
        within :attr:`timeout`: the tool ran and got no answer. Any other
        error, the handler giving the request up included, takes the generic
        failure path, its message withheld from the model (RFC §9.3).
        """
        if name not in self.tool_names:
            raise UnservedToolCallError(f"tool {name!r} is not served here")

        ctx = _current_tool_call.get()
        room_id = ctx.room_id if ctx else ""
        tool_call_id = ctx.tool_call_id if ctx else ""
        channel_id = ctx.channel_id if ctx else ""

        try:
            pending = await self._handler.create(
                tool_name=name,
                arguments=arguments,
                room_id=room_id,
                tool_call_id=tool_call_id,
                channel_id=channel_id,
                channel_type=channel_type,
                # Asking a human is where "whose turn is it" matters most: a
                # request that names nobody has to be broadcast to the room,
                # and whoever answers first answers for someone else.
                actor_id=current_tool_actor_id(),
            )
            return await self._handler.wait(pending.pending_id, timeout=self.timeout)
        except TimeoutError:
            # The tool ran and nobody answered: a failure, in words the model
            # reads (RFC §9.3).
            raise ToolFailedError(
                json.dumps(
                    {"error": f"Human input timed out after {self.timeout}s for tool '{name}'"}
                )
            ) from None
        except HumanInputRejectedError as exc:
            # The request was rejected (a human, a hook): a refusal, and why.
            raise ToolRefusedError(json.dumps({"error": f"Human input rejected: {exc}"})) from exc
