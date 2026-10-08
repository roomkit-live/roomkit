"""Wire video vision results into a model's context.

What a vision provider saw rides an AIChannel's turn notes on its own, as a
``<vision>`` block (RFC §12.8.7): every AIChannel attached to a room with an
analysed video channel reads it, and its system prompt and binding never
change with the camera. :func:`setup_realtime_vision` gives the same block to
a RealtimeVoiceChannel's sessions.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from roomkit.channels._video_hooks import vision_note
from roomkit.video.vision.base import VisionResult

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.framework_event import FrameworkEvent

logger = logging.getLogger("roomkit.video.ai_integration")


def setup_video_vision(
    kit: RoomKit,
    room_id: str,
    ai_channel_id: str,
    *,
    context_prefix: str = "You can see a live camera feed. Current view:",
) -> None:
    """Deprecated: warns, and wires nothing.

    .. deprecated::
        What a vision provider saw rides every AIChannel's turn notes in the
        room on its own (RFC §12.8.7), never a system prompt, where the text a
        camera read would pass for the application's instruction. This call
        writes no prompt and no binding; *context_prefix* and *ai_channel_id*
        are ignored.
    """
    logger.warning(
        "setup_video_vision() does nothing: what the camera sees rides every AI "
        "channel's turn notes in room %s (RFC §12.8.7); its context_prefix is ignored",
        room_id,
    )
    warnings.warn(
        "setup_video_vision() is deprecated and does nothing: what the camera "
        "sees rides every AI channel's turn notes in the room (RFC §12.8.7).",
        DeprecationWarning,
        stacklevel=2,
    )


def setup_realtime_vision(
    kit: RoomKit,
    room_id: str,
    voice_channel_id: str,
    *,
    context_prefix: str = "You can see the screen. Current view:",
) -> None:
    """Wire video vision results into a RealtimeVoiceChannel via inject_text.

    Registers a framework event handler that listens for
    ``video_vision_result`` events and injects what the camera saw into
    active voice sessions using ``inject_text(silent=True)``, as a
    ``<vision>`` block under *context_prefix* (RFC §12.8.7).

    Includes dedup: unchanged descriptions are not re-injected.

    Args:
        kit: The RoomKit instance.
        room_id: The room where video and voice channels are attached.
        voice_channel_id: The RealtimeVoiceChannel to receive vision context.
        context_prefix: Text prepended to the vision description.
    """
    _last_description: list[str] = [""]

    async def _on_vision(event: FrameworkEvent) -> None:
        if event.room_id != room_id:
            return
        description = event.data.get("description", "")
        if not description:
            return

        # Dedup: skip if description unchanged
        if description == _last_description[0]:
            return
        _last_description[0] = description

        seen = VisionResult(
            description=description,
            labels=event.data.get("labels") or [],
            text=event.data.get("text"),
        )
        # Set apart as data: the text the camera read is no instruction (RFC §6.4).
        vision_context = vision_note(seen, lead=context_prefix)

        try:
            from roomkit.channels.realtime_voice import RealtimeVoiceChannel

            channel = kit.get_channel(voice_channel_id)
            if not isinstance(channel, RealtimeVoiceChannel):
                return

            sessions = channel.get_room_sessions(room_id)
            for session in sessions:
                await channel.inject_text(session, vision_context, silent=True)
                logger.debug(
                    "Vision injected into realtime session %s (len=%d)",
                    session.id,
                    len(vision_context),
                )
        except Exception:
            logger.exception("Failed to inject vision into realtime voice channel")

    kit.on("video_vision_result")(_on_vision)
