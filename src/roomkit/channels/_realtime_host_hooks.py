"""The hooks a channel hosting a realtime model announces, the same way on
every host: a realtime voice channel and a conference with a realtime model
plugged in (RFC §12.5). ON_REALTIME_TEXT_INJECTED for text entering the
model's context, ON_ERROR for a failure of its session."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.channels.realtime_host_hooks")


async def fire_text_injected(
    framework: RoomKit | None,
    source: EventSource,
    session: VoiceSession,
    text: str,
    *,
    role: str,
    injected_from: RoomEvent | None = None,
) -> None:
    """Announce *text* injected into *session* with the intent *role*, the
    event of the injection itself on every host: its source the host, its
    ``injected_role`` and ``session_id``, and for a broadcast the event it
    came from (``injected_from``: its channel and its id).

    Fired wherever text enters the model's conversation context, not only
    where an inbound event drove it: the hook is how an integrator audits
    what reached the model. A hook failure is logged, never raised into the
    injection that already happened.
    """
    if framework is None or not session.room_id:
        return
    metadata: dict[str, object] = {"injected_role": role, "session_id": session.id}
    if injected_from is not None:
        metadata["injected_from"] = {
            "channel_id": injected_from.source.channel_id,
            "event_id": injected_from.id,
        }
    event = RoomEvent(
        room_id=session.room_id,
        source=source,
        content=TextContent(body=text),
        metadata=metadata,
    )
    try:
        context = await framework._build_context(session.room_id)  # noqa: SLF001
        await framework.hook_engine.run_async_hooks(
            session.room_id,
            HookTrigger.ON_REALTIME_TEXT_INJECTED,
            event,
            context,
            skip_event_filter=True,
        )
    except Exception:
        logger.warning(
            "ON_REALTIME_TEXT_INJECTED failed for session %s", session.id, exc_info=True
        )


async def fire_session_error(
    framework: RoomKit | None,
    source: EventSource,
    session: VoiceSession,
    *,
    error: str,
    error_type: str,
    category: str,
) -> None:
    """Fire ON_ERROR for a failure of *session*'s, as its host *source*: the
    provider's, or a reasoning backend's turn (RFC §12.4.1, §12.5). A hook
    failure is logged, never raised into the provider's receive loop."""
    if framework is None or not session.room_id:
        return
    try:
        context = await framework._build_context(session.room_id)  # noqa: SLF001
        await framework._fire_error_hook(  # noqa: SLF001
            session.room_id,
            context,
            source,
            error=error,
            error_type=error_type,
            error_category=category,
        )
    except Exception:
        logger.warning("ON_ERROR could not be fired for session %s", session.id, exc_info=True)
