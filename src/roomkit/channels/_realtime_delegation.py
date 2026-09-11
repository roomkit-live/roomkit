"""Reasoning delegation for RealtimeVoiceChannel (RFC §12.4.1).

A full-duplex provider's model holds the conversation and hands reasoning and
tool use to a backend; the provider announces each hand-over through
``on_delegation``. This mixin turns every announcement into
``ON_REALTIME_DELEGATION``, whichever backend the delegation went to.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol

from roomkit.models.enums import HookTrigger
from roomkit.telemetry.context import reset_span
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.events import RealtimeDelegationEvent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeDelegationHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeDelegationMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _framework: The RoomKit framework instance (or None).
        _provider: The realtime voice provider.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
        _rt_span_ctx: Get the telemetry span context for a session.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _framework: RoomKit | None
    _provider: RealtimeVoiceProvider

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...

    def _rt_span_ctx(self, session_id: str) -> tuple[Any, Any]: ...


class RealtimeDelegationMixin:
    """Reasoning-delegation handling for :class:`RealtimeVoiceChannel`.

    Host contract: :class:`RealtimeDelegationHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _framework: RoomKit | None
    _provider: RealtimeVoiceProvider

    _track_task: Any  # see RealtimeDelegationHost — cross-mixin
    _rt_span_ctx: Any  # see RealtimeDelegationHost — cross-mixin

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
