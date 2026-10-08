"""What the agent thinks while it listens, on the AI channel (RFC §6.4).

A channel that has a speak policy and a thinker keeps, per room, the agent's
thought. On an event the policy leaves silent, the thinker rewrites it from the
event's context, one call at a time per room; back in time with something to
say, the policy is asked again. When the agent speaks, the turn's notes carry
its thought and what it wanted to say is emptied.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING

from roomkit.speaking.thought import Thought, ThoughtEvent, thought_note

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import ThoughtHook
    from roomkit.channels._ai_contract import _AIChannelContract
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.providers.ai.base import AIContext
    from roomkit.speaking.base import SpeakDecision, SpeakPolicy
    from roomkit.speaking.thinker import Thinker
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


class _RoomMind:
    """One room's thought, and the thinker call that rewrites it: one at a time;
    an event listened to meanwhile is thought about in the next call, from the
    latest context, so a fast exchange never queues stale work."""

    def __init__(self, room_id: str, thinker: Thinker, channel: AIThinkingMixin) -> None:
        self.room_id = room_id
        self.thought = Thought()
        self._thinker = thinker
        self._channel = channel
        self._task: asyncio.Task[None] | None = None
        self._changed = asyncio.Condition()
        self._pending: tuple[AIContext, int] | None = None
        """The latest context to think about, and how often the agent had spoken then."""
        self._asked = 0
        self._done = 0
        self._spoke = 0

    def think(self, context: AIContext) -> int:
        """Have the thinker think about *context*; the ticket :meth:`settled` waits on."""
        self._asked += 1
        self._pending = (context, self._spoke)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"thinker:{self.room_id}")
        return self._asked

    async def settled(self, ticket: int, timeout: float) -> bool:
        """Whether the thought covering *ticket* came back within *timeout* seconds."""
        async with self._changed:
            try:
                await asyncio.wait_for(
                    self._changed.wait_for(lambda: self._done >= ticket), timeout
                )
            except TimeoutError:
                return False
        return True

    def spoke(self) -> Thought:
        """The agent speaks with its thought: what it wanted to say is on the table.
        Returns the thought it spoke with."""
        thought, self.thought = self.thought, self.thought.said()
        self._spoke += 1
        return thought

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while self._pending is not None:
            (context, spoke), ticket = self._pending, self._asked
            self._pending = None
            started = time.monotonic()
            thought = await self._next(self.thought, context)
            duration_ms = round((time.monotonic() - started) * 1000)
            if self._spoke != spoke:
                thought = thought.said()  # it spoke since the context: that is said now
            async with self._changed:
                # What the thought replaces is read now: the agent may have
                # spoken during the call, emptying the one the call started from.
                replaced, self.thought = self.thought, thought
                self._done = ticket
                self._changed.notify_all()
            if thought != replaced:
                await self._channel._report_thought(self.room_id, thought, replaced, duration_ms)

    async def _next(self, previous: Thought, context: AIContext) -> Thought:
        try:
            return await self._thinker.think(previous, context)
        except Exception:
            logger.warning("Thinker failed in room %s; thought kept", self.room_id, exc_info=True)
            return previous


class AIThinkingMixin(_AIChannelContract):
    """Keeps the agent's thought per room and runs its thinker while it listens."""

    channel_id: str
    _thinker: Thinker | None
    _think_wait: float
    _thought_hook: ThoughtHook | None
    _minds: dict[str, _RoomMind]

    def _store_thinker(
        self, thinker: Thinker | None, wait: float, policy: SpeakPolicy | None
    ) -> None:
        if thinker is not None and policy is None:
            raise ValueError(
                "a thinker needs a speak_policy: it thinks on the events the agent listens to"
            )
        if wait < 0:
            raise ValueError("think_wait must not be negative")
        self._thinker = thinker
        self._think_wait = wait
        self._thought_hook = None
        self._minds = {}

    def _thought_of(self, room_id: str) -> Thought | None:
        """What the agent has in mind in *room_id*; ``None`` without a thinker."""
        if self._thinker is None:
            return None
        mind = self._minds.get(room_id)
        return mind.thought if mind is not None else Thought()

    async def _think_while_listening(
        self,
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
        decision: SpeakDecision,
    ) -> SpeakDecision:
        """The agent stays silent on *event*: it thinks about it, and when the thought
        comes back within the wait with something to say, the policy decides again."""
        thinker = self._thinker
        if thinker is None:
            return decision
        room_id = context.room.id if context.room else event.room_id
        mind = self._minds.get(room_id) or self._minds.setdefault(
            room_id, _RoomMind(room_id, thinker, self)
        )
        ai_context = await self._thinking_context(event, binding, context)
        if ai_context is None:
            return decision  # BEFORE_AI_GENERATION kept it from the thinker
        ticket = mind.think(ai_context)
        if not self._think_wait or not await mind.settled(ticket, self._think_wait):
            return decision
        if not mind.thought.want_to_say:
            return decision
        again = await self._speak_decision(event, context, mind.thought, asked_again=True)
        return again if again is not None else decision

    async def _thought_notes(self, room_id: str) -> tuple[str, ...]:
        """The agent speaks on a decided event: its thought joins the turn's notes,
        and what it wanted to say is emptied, a new thought ``ON_THOUGHT`` reports
        (RFC §6.4)."""
        mind = self._minds.get(room_id)
        if mind is None:
            return ()
        spoken = mind.spoke()
        if mind.thought != spoken:
            await self._report_thought(room_id, mind.thought, spoken, None)
        note = thought_note(spoken)
        return (note,) if note else ()

    async def _report_thought(
        self, room_id: str, thought: Thought, previous: Thought, duration_ms: int | None
    ) -> None:
        """Fire ``ON_THOUGHT`` with *thought*, and how long the thinker call that
        brought it took (``None`` without a call)."""
        hook = self._thought_hook
        if hook is None:
            return
        try:
            await hook(ThoughtEvent(room_id, self.channel_id, thought, previous, duration_ms))
        except Exception:
            logger.warning("ON_THOUGHT failed in room %s", room_id, exc_info=True)

    async def _forget_thought(self, room_id: str) -> None:
        """Drop *room_id*'s thought and stop its thinker call."""
        mind = self._minds.pop(room_id, None)
        if mind is not None:
            await mind.close()

    async def _close_minds(self) -> None:
        for mind in self._minds.values():
            await mind.close()
        self._minds.clear()
