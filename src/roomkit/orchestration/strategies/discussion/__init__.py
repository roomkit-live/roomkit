"""Discussion orchestration strategy (RFC §19.7.5).

Several agents and one or more people hold one conversation in a room: any
agent may address any other by ``@name``, and one agent speaks at a time.
"""

from __future__ import annotations

from .models import SpeakQueue, SpeakQueueChange, SpeakQueueEvent

__all__ = ["SpeakQueue", "SpeakQueueChange", "SpeakQueueEvent"]
