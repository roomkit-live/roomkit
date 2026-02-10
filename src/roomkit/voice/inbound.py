"""Voice inbound helpers — convert voice sessions to InboundMessage."""

from __future__ import annotations

from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.models.event import SystemContent

from .base import VoiceSession


def parse_voice_session(
    session: VoiceSession,
    channel_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> InboundMessage:
    """Convert a VoiceSession into an InboundMessage for ``process_inbound``.

    Equivalent to ``parse_telegram_webhook`` / ``parse_twilio_webhook``
    but for voice sessions.  Creates an ``InboundMessage`` with the
    session attached so that ``process_inbound`` handles room routing,
    hooks, and session binding automatically.

    Args:
        session: The voice session from a backend (e.g. SIP ``on_call``).
        channel_id: The voice channel ID to route to.
        metadata: Extra metadata merged into the message.  Session
            metadata is included automatically.

    Returns:
        An ``InboundMessage`` ready for ``kit.process_inbound()``.

    Example::

        @sip_backend.on_call
        async def handle(session):
            await kit.process_inbound(
                parse_voice_session(session, channel_id="voice")
            )
    """
    merged_metadata = {**session.metadata}
    if metadata:
        merged_metadata.update(metadata)

    return InboundMessage(
        channel_id=channel_id,
        sender_id=session.participant_id,
        content=SystemContent(
            body="Voice session started",
            code="session_started",
            data={
                "session_id": session.id,
                "channel_id": channel_id,
                "caller": session.metadata.get("caller"),
            },
        ),
        event_type=EventType.SYSTEM,
        session=session,
        metadata=merged_metadata,
    )
