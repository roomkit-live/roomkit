"""Context variables for RealtimeVoiceChannel tool calls."""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roomkit.voice.base import VoiceSession

_current_voice_session: contextvars.ContextVar[VoiceSession | None] = contextvars.ContextVar(
    "_current_voice_session",
    default=None,
)


def get_current_voice_session() -> VoiceSession | None:
    """Get the voice session for the current tool call.

    Available inside tool handlers called by RealtimeVoiceChannel.
    Returns None outside of a tool call context.
    """
    return _current_voice_session.get()
