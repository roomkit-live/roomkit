"""``VoiceTrace`` — the timeline of a voice conversation, for tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.voice.base import VoiceSession

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext

VOICE_TRIGGERS: tuple[HookTrigger, ...] = (
    HookTrigger.ON_SESSION_STARTED,
    HookTrigger.ON_SPEECH_START,
    HookTrigger.ON_SPEECH_END,
    HookTrigger.ON_PARTIAL_TRANSCRIPTION,
    HookTrigger.ON_TRANSCRIPTION,
    HookTrigger.BEFORE_TTS,
    HookTrigger.AFTER_TTS,
    HookTrigger.ON_BARGE_IN,
    HookTrigger.ON_TTS_CANCELLED,
    HookTrigger.ON_DTMF,
    HookTrigger.ON_TURN_COMPLETE,
    HookTrigger.ON_TURN_INCOMPLETE,
)
"""The hooks a voice turn fires, in the order a turn usually fires them."""


@dataclass(frozen=True)
class TraceEntry:
    """One hook firing, as the trace saw it."""

    t: float
    """``time.monotonic()`` when the hook ran: the clock the channel's own
    timings (barge-in confirmation, echo windows) read."""
    trigger: HookTrigger
    payload: Any
    """What the hook received: a :class:`VoiceSession` (speech start and end),
    a dataclass of :mod:`roomkit.voice.events`, the ``SessionStartedEvent``, or
    the text (``BEFORE_TTS`` / ``AFTER_TTS``)."""
    room_id: str | None
    session_id: str | None


def _session_id(payload: Any) -> str | None:
    if isinstance(payload, VoiceSession):
        return payload.id
    session = getattr(payload, "session", None)
    return session.id if isinstance(session, VoiceSession) else None


class VoiceTrace:
    """Subscribes to the voice hooks of a kit and records when each fired.

    The record replaces the ``asyncio.sleep`` a voice test otherwise waits
    with: ``await trace.wait_for(HookTrigger.ON_TRANSCRIPTION)`` returns the
    moment the transcription hook ran, or raises ``TimeoutError`` naming what
    did fire. Every entry keeps the hook's payload and its monotonic time, so
    an order (:meth:`sequence`) or a latency (:meth:`elapsed_ms`) is read off
    the timeline rather than off the channel's private state.

    Observers are ``ASYNC`` hooks, which also see the triggers the channel
    runs synchronously (``ON_TRANSCRIPTION``, ``BEFORE_TTS``): the engine
    fires async observers after the sync chain. They are fire-and-forget, so
    a ``wait_for`` returns when the entry is recorded, not when the channel is
    done with the turn. Hooks are global to the kit and cannot be removed;
    build one trace per kit, in a test that owns the kit.
    """

    def __init__(
        self,
        kit: RoomKit,
        *,
        triggers: Iterable[HookTrigger] = VOICE_TRIGGERS,
    ) -> None:
        self._entries: list[TraceEntry] = []
        self._arrived = asyncio.Event()
        self._triggers = tuple(triggers)
        for trigger in self._triggers:
            kit.hook(trigger, HookExecution.ASYNC, name=f"voice_trace:{trigger.value}")(
                self._observer(trigger)
            )

    @property
    def triggers(self) -> tuple[HookTrigger, ...]:
        """The triggers this trace observes."""
        return self._triggers

    def _observer(self, trigger: HookTrigger) -> Any:
        async def observe(event: Any, context: RoomContext) -> None:
            self._record(trigger, event, context)

        return observe

    def _record(self, trigger: HookTrigger, payload: Any, context: RoomContext | None) -> None:
        room = getattr(context, "room", None)
        self._entries.append(
            TraceEntry(
                t=time.monotonic(),
                trigger=trigger,
                payload=payload,
                room_id=getattr(room, "id", None),
                session_id=_session_id(payload),
            )
        )
        # Wake every waiter on the current event, then arm a fresh one for
        # the next wait: a waiter always holds the event of the entry it
        # missed, never one set by an earlier firing.
        arrived, self._arrived = self._arrived, asyncio.Event()
        arrived.set()

    # -------------------------------------------------------------------------
    # Reading the timeline
    # -------------------------------------------------------------------------

    def entries(
        self,
        trigger: HookTrigger | None = None,
        *,
        session_id: str | None = None,
        after: TraceEntry | float | None = None,
    ) -> list[TraceEntry]:
        """The entries so far, oldest first, filtered by trigger, session and time.

        *after* is a :class:`TraceEntry` or a monotonic time; only entries
        strictly later are returned.
        """
        since = after.t if isinstance(after, TraceEntry) else after
        return [
            e
            for e in self._entries
            if (trigger is None or e.trigger is trigger)
            and (session_id is None or e.session_id == session_id)
            and (since is None or e.t > since)
        ]

    def sequence(self) -> list[HookTrigger]:
        """The triggers in the order they fired."""
        return [e.trigger for e in self._entries]

    def first(self, trigger: HookTrigger) -> TraceEntry | None:
        matches = self.entries(trigger)
        return matches[0] if matches else None

    def last(self, trigger: HookTrigger) -> TraceEntry | None:
        matches = self.entries(trigger)
        return matches[-1] if matches else None

    @staticmethod
    def elapsed_ms(start: TraceEntry, end: TraceEntry) -> float:
        """Milliseconds from *start* to *end*."""
        return (end.t - start.t) * 1000.0

    def clear(self) -> None:
        """Forget every entry recorded so far."""
        self._entries.clear()

    # -------------------------------------------------------------------------
    # Waiting on the timeline
    # -------------------------------------------------------------------------

    async def wait_for(
        self,
        trigger: HookTrigger,
        *,
        timeout: float = 2.0,
        after: TraceEntry | float | None = None,
        session_id: str | None = None,
    ) -> TraceEntry:
        """Return the first entry for *trigger* (later than *after*, for
        *session_id*), waiting up to *timeout* seconds for it to arrive.

        Raises ``TimeoutError`` naming the triggers that did fire, which is
        what a failing voice test needs to read first.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            matches = self.entries(trigger, session_id=session_id, after=after)
            if matches:
                return matches[0]
            remaining = deadline - loop.time()
            if remaining <= 0:
                seen = ", ".join(t.value for t in self.sequence()) or "nothing"
                raise TimeoutError(
                    f"{trigger.value} not seen within {timeout:.3g}s (seen so far: {seen})"
                )
            arrived = self._arrived
            try:
                await asyncio.wait_for(arrived.wait(), remaining)
            except TimeoutError:
                continue
