"""The hooks a channel hosting a realtime model announces, the same way on
every host: a realtime voice channel and a conference with a realtime model
plugged in (RFC §12.5). ON_REALTIME_TEXT_INJECTED for text entering the
model's context, ON_ERROR for a failure of its session,
ON_REALTIME_DELEGATION for a delegation, and what both hosts share around
them: the spoken fallback a delegation is answered with when nothing else
answers it, the one test of whether a session still takes text, and what a
text another channel broadcast becomes in a session."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from roomkit._text import quoted
from roomkit.channels._mark_copies import without_mark_copies
from roomkit.channels._runtime_record import written_by_runtime
from roomkit.channels._speaker import channel_label, turn_labels
from roomkit.core.visibility import visible_events
from roomkit.models.enums import HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.voice.base import VoiceSessionState
from roomkit.voice.realtime.events import RealtimeDelegationEvent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_host_hooks")

FALLBACK_NO_BACKEND = "No backend is available to handle delegated work in this session."
FALLBACK_NO_OUTPUT = "The delegated work finished without an answer."
FALLBACK_TIMEOUT = "The delegated work took too long and was abandoned."

BROADCAST_INTENT = "user"
"""The intent a text another channel broadcast enters a realtime session with:
content someone else wrote, never the application's instruction, whatever the
event carries (RFC §12.4)."""

BROADCAST_TEXT_LIMIT = 4000
"""The characters of a broadcast text a session takes."""


def broadcast_text(
    event: RoomEvent, text: str, context: RoomContext, channel_id: str
) -> str | None:
    """*text*, which *event* broadcast, as a realtime host injects it: quoted
    after the label the conversation gives its author (RFC §6.4, §12.4), read
    among the recent turns *channel_id*, the session's channel, may see.
    ``Marie: “…”``: on one line, it cannot end its quote nor add a line of
    its own, a copy of a runtime mark in it replaced unless the runtime wrote
    *event*. ``None`` for a blank text, which no session takes. The caller
    compiles the marks' patterns first
    (:func:`~roomkit.channels._mark_copies.compile_mark_patterns`)."""
    if not text.strip():
        return None
    if not written_by_runtime(event.metadata):
        text = without_mark_copies(text)
    seen = visible_events(context, channel_id)
    label = turn_labels([*seen, event], context).get(event.id)
    author = label or channel_label(event.source.channel_id)
    return f"{author}: {quoted(text, BROADCAST_TEXT_LIMIT)}"


def serves(held: VoiceSession | None, session: VoiceSession) -> bool:
    """Whether *session* still takes text, on every host: it is the session
    the host holds (*held*) and the provider has not ended it, even before
    the host has let it go (RFC §12.4)."""
    return held is session and session.state != VoiceSessionState.ENDED


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


async def fire_delegation(
    framework: RoomKit | None,
    room_id: str | None,
    session: VoiceSession,
    delegation_id: str,
    target: str,
) -> None:
    """Announce a provider's delegation to ON_REALTIME_DELEGATION (RFC
    §12.4.1), the same way on every host. A hook failure is logged, never
    raised into the provider's receive loop."""
    if framework is None or not room_id:
        return
    if not framework.hook_engine.has_hooks(HookTrigger.ON_REALTIME_DELEGATION):
        return
    kind: Literal["hosted", "integrator"] = "hosted" if target == "hosted" else "integrator"
    event = RealtimeDelegationEvent(session=session, delegation_id=delegation_id, target=kind)
    try:
        context = await framework._build_context(room_id)  # noqa: SLF001
        await framework.hook_engine.run_async_hooks(
            room_id,
            HookTrigger.ON_REALTIME_DELEGATION,
            event,
            context,
            skip_event_filter=True,
        )
    except Exception:
        logger.warning("ON_REALTIME_DELEGATION failed for session %s", session.id, exc_info=True)


async def speak_fallback(
    provider: RealtimeVoiceProvider, session: VoiceSession, delegation_id: str, text: str
) -> bool:
    """Answer a delegation with one spoken output, so the model does not wait
    for an answer that never comes (RFC §12.4.1), on every host. Whether it
    was sent: an output not sent is not waited for."""
    if session.state == VoiceSessionState.ENDED:
        return False
    try:
        await provider.submit_delegation_output(session, delegation_id, text, spoken=True)
    except Exception:
        logger.exception(
            "Could not return the fallback for delegation %s (session %s)",
            delegation_id,
            session.id,
        )
        return False
    return True
