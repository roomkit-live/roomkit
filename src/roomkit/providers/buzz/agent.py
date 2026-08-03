"""BuzzAgent — run a RoomKit app as a first-class Buzz agent.

A live Buzz agent makes promises the platform's remote-agents contract states
for *every* launcher, however the process was started: its presence reflects
reality, its owner's ``!shutdown`` stops it gracefully, an opt-in inactivity
bound reaps it when nobody needs it, and an intentional stop is final — the
process exits its graceful path once, with everything closed behind it.

:class:`BuzzAgent` packages those promises around an already-configured
:class:`~roomkit.core.framework.RoomKit` and its Buzz sources, so the app's
job ends at wiring rooms and channels::

    agent = BuzzAgent(kit, sources=[source], exit_after_inactivity=7200)
    cause = await agent.run()      # blocks: owner !shutdown / SIGTERM / idle
    sys.exit(0)                    # intentional exit is clean — code 0

``run()`` attaches the sources, installs SIGTERM/SIGINT handlers, arms the
optional inactivity reaper, and — whatever the stop cause — exits through one
graceful path: ``kit.close()`` (two-phase drain, presence ``offline``, relay
sockets closed). The kit is consumed: after ``run()`` returns, it is closed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookExecution, HookTrigger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from roomkit.core.framework import RoomKit
    from roomkit.models.event import RoomEvent
    from roomkit.models.hook import HookResult, RoomContext
    from roomkit.sources.buzz import BuzzOwnerCommandCallback, BuzzRelaySource

logger = logging.getLogger("roomkit.providers.buzz.agent")

# Upper bound on the inactivity re-check interval. The effective reap moment
# is t ∈ [T, T + interval) — immaterial at real-world bounds (minutes+).
_INACTIVITY_CHECK_CAP = 60.0


@unique
class BuzzAgentStopCause(StrEnum):
    """Why :meth:`BuzzAgent.run` returned. Every cause exits gracefully."""

    OWNER_SHUTDOWN = "owner_shutdown"
    SIGNAL = "signal"
    INACTIVITY = "inactivity"


class BuzzAgent:
    """Lifecycle runner turning a RoomKit app into a conforming Buzz agent.

    The agent owns *waiting and dying*, not wiring: rooms, channels and hooks
    stay the app's job, and the sources are handed over **unattached** —
    ``run()`` attaches them (with the callbacks already in place) so no
    owner command can slip through before the takeover.

    Args:
        kit: The configured RoomKit. ``run()`` closes it on exit.
        sources: Buzz relay sources to attach and supervise. Their
            ``on_owner_command`` is taken over by the agent (a warning is
            logged if one was already set).
        exit_after_inactivity: Optional idle bound in seconds — the agent
            stops itself after that long with no inbound dispatched and no
            broadcast in any room (the platform's opt-in self-stop; default
            off, and deliberately *not* named like the per-turn timeouts).
        on_owner_command: Optional passthrough for ``"cancel"``/``"rotate"``
            (``"shutdown"`` is the agent's, always).
    """

    def __init__(
        self,
        kit: RoomKit,
        sources: Sequence[BuzzRelaySource],
        *,
        exit_after_inactivity: float | None = None,
        on_owner_command: BuzzOwnerCommandCallback | None = None,
    ) -> None:
        if exit_after_inactivity is not None and exit_after_inactivity <= 0:
            raise ValueError("exit_after_inactivity must be positive (or None to disable)")
        if not sources:
            raise ValueError("BuzzAgent needs at least one source")
        self._kit = kit
        self._sources = list(sources)
        self._exit_after_inactivity = exit_after_inactivity
        self._on_owner_command = on_owner_command
        self._cause: BuzzAgentStopCause | None = None
        self._stopped = asyncio.Event()
        self._last_activity = datetime.now(UTC)
        self._ran = False

    async def run(self) -> BuzzAgentStopCause:
        """Serve until the owner, a signal, or the inactivity bound stops us.

        Single-shot: the kit is closed on the way out, whatever the cause,
        so every exit is the same graceful path — sources stopped (presence
        ``offline`` published while the socket is up), channels drained and
        closed (RFC 12.10.4). Raises whatever ``kit.close()`` raises, after
        the rest of the shutdown ran to completion.

        A failure during startup takes the same exit: whatever had already
        been started is stopped and the kit is closed before the exception
        reaches the caller.
        """
        if self._ran:
            raise RuntimeError("BuzzAgent.run() is single-shot; build a new agent to restart")
        self._ran = True
        self._last_activity = datetime.now(UTC)

        for source in self._sources:
            if source.on_owner_command is not None:
                logger.warning(
                    "BuzzAgent takes over on_owner_command of %s; pass the callback "
                    "to BuzzAgent(on_owner_command=...) instead",
                    source.name,
                )
            source.on_owner_command = self._handle_owner_command

        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        reaper: asyncio.Task | None = None
        serving = False
        # Everything acquired below is released by the finally, startup
        # included: a source that fails to attach (an unreachable relay, two
        # sources sharing a channel_id) must not leave the reaper running,
        # the signal handlers installed, the earlier sources connected and
        # the kit open.
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, self._request_stop, BuzzAgentStopCause.SIGNAL)
                    installed.append(sig)
                except (NotImplementedError, RuntimeError):
                    # Non-Unix event loop (or nested loop): the substrate's stop
                    # signal cannot be wired here — document/hand-wire instead.
                    logger.warning("Cannot install %s handler on this event loop", sig.name)

            if self._exit_after_inactivity is not None:
                # Any broadcast in any room — inbound routed or AI answering —
                # counts as activity, so the hook alone nearly covers "events
                # dispatched"; the sources' own last-message timestamps below
                # also count inbound that hooks later blocked.
                self._kit.hook(
                    HookTrigger.AFTER_BROADCAST,
                    execution=HookExecution.ASYNC,
                    name="buzz_agent_activity",
                )(self._record_activity)
                # The reaper runs on its own timer, never gated on other state:
                # the idle agent it exists to stop is exactly the one nothing
                # else would wake.
                reaper = asyncio.create_task(self._inactivity_loop())

            for source in self._sources:
                await self._kit.attach_source(source.channel_id, source, auto_restart=True)
            logger.info(
                "Buzz agent running: %d source(s)%s",
                len(self._sources),
                f", inactivity bound {self._exit_after_inactivity:.0f}s"
                if self._exit_after_inactivity
                else "",
            )

            serving = True
            await self._stopped.wait()
        finally:
            if reaper is not None:
                reaper.cancel()
            for sig in installed:
                loop.remove_signal_handler(sig)
            cause = self._cause or BuzzAgentStopCause.SIGNAL
            logger.info("Buzz agent stopping (%s)", cause if serving else "startup failed")
            await self._kit.close()

        return cause

    # ------------------------------------------------------------- internals

    def _request_stop(self, cause: BuzzAgentStopCause) -> None:
        """First cause wins; idempotent thereafter."""
        if self._cause is None:
            self._cause = cause
        self._stopped.set()

    async def _handle_owner_command(self, command: str, event: dict[str, Any]) -> None:
        if command == "shutdown":
            self._request_stop(BuzzAgentStopCause.OWNER_SHUTDOWN)
            return
        if self._on_owner_command is not None:
            result = self._on_owner_command(command, event)
            if inspect.isawaitable(result):
                await result

    async def _record_activity(self, event: RoomEvent, ctx: RoomContext) -> HookResult | None:
        self._last_activity = datetime.now(UTC)
        return None

    async def _idle_seconds(self) -> float:
        last = self._last_activity
        for source in self._sources:
            health = await source.healthcheck()
            if health.last_message_at is not None and health.last_message_at > last:
                last = health.last_message_at
        return (datetime.now(UTC) - last).total_seconds()

    async def _inactivity_loop(self) -> None:
        bound = self._exit_after_inactivity
        if bound is None:  # only spawned when a bound is set
            return
        while True:
            remaining = bound - await self._idle_seconds()
            if remaining <= 0:
                logger.info("Inactivity bound reached (%.0fs) — stopping", bound)
                self._request_stop(BuzzAgentStopCause.INACTIVITY)
                return
            await asyncio.sleep(min(remaining, _INACTIVITY_CHECK_CAP))
