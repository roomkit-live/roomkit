"""The hooks a voice event announces, with the room's context read once and
only for a trigger a hook is registered on."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.core.hooks import SyncPipelineResult

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.models.enums import HookTrigger


class RoomHooks:
    """A room's hooks for one voice event: an utterance, a speech boundary, a partial.

    A context is store reads (the room, its bindings, its participants), and
    an utterance announces up to four triggers while most rooms register a
    hook on none of them: the context is read for the first trigger a hook is
    registered on, shared by the event's later triggers, and never read when
    none is. A hook that must see the room after other work, as AFTER_TTS
    does after the synthesis, gets a ``RoomHooks`` of its own. A read that
    fails raises to the caller, which knows what the failure costs its event.
    """

    def __init__(self, framework: RoomKit, room_id: str) -> None:
        self._framework = framework
        self._room_id = room_id
        self._read: RoomContext | None = None

    async def run_async(self, trigger: HookTrigger, event: Any) -> None:
        """Run *trigger*'s hooks on *event*, when one is registered, and
        return once they have ended: a trigger of a session's stream (a
        speech boundary, a partial, a barge-in, what the agent said) is
        observed in order with the stream's next triggers, its SYNC ones
        included (RFC §9.3)."""
        engine = self._framework.hook_engine
        if not engine.has_hooks(trigger):
            return
        context = await self._context()
        await engine.run_async_hooks(
            self._room_id, trigger, event, context, skip_event_filter=True
        )

    async def fire(
        self,
        trigger: HookTrigger,
        event: Any,
        *,
        then: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Fire *trigger*'s hooks on *event* as an ASYNC trigger's, outside
        a session's stream (a recording's start or end): started and not
        waited for, but for an awaited trigger's (``HookEngine._fire_async_hooks``,
        RFC §9.3). *then*, the framework event that reports it, comes once
        they have ended."""
        engine = self._framework.hook_engine
        if not engine.has_hooks(trigger):
            if then is not None:
                await then()
            return
        context = await self._context()
        await engine._fire_async_hooks(
            self._room_id, trigger, event, context, skip_event_filter=True, then=then
        )

    async def run_sync(self, trigger: HookTrigger, event: Any) -> SyncPipelineResult:
        """Run *trigger*'s hook chain on *event*; with no hook, *event* passes as is."""
        engine = self._framework.hook_engine
        if not engine.has_hooks(trigger):
            return SyncPipelineResult(event=event)
        context = await self._context()
        return await engine.run_sync_hooks(
            self._room_id, trigger, event, context, skip_event_filter=True
        )

    async def _context(self) -> RoomContext:
        """The room's context, read on the first call."""
        if self._read is None:
            self._read = await self._framework._build_context(self._room_id)
        return self._read
