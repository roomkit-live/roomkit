"""ON_REALTIME_TEXT_INJECTED, announced the same way by every channel whose
realtime model takes injected text: a realtime voice channel and a conference
with a realtime model plugged in (RFC §12.5)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roomkit.models.enums import HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.channels.realtime_text_injected")


async def fire_text_injected(
    framework: RoomKit | None,
    source: EventSource,
    session: VoiceSession,
    text: str,
    *,
    role: str,
) -> None:
    """Announce *text* injected into *session* with the intent *role*.

    Fired wherever text enters the model's conversation context, not only
    where an inbound event drove it: the hook is how an integrator audits
    what reached the model. A hook failure is logged, never raised into the
    injection that already happened.
    """
    if framework is None or not session.room_id:
        return
    event = RoomEvent(
        room_id=session.room_id,
        source=source,
        content=TextContent(body=text),
        metadata={"injected_role": role, "session_id": session.id},
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
