"""Realtime backend for ephemeral events."""

from roomkit.realtime.base import (
    EphemeralCallback,
    EphemeralEvent,
    EphemeralEventType,
    RealtimeBackend,
)
from roomkit.realtime.memory import InMemoryRealtime

__all__ = [
    "EphemeralCallback",
    "EphemeralEvent",
    "EphemeralEventType",
    "InMemoryRealtime",
    "RealtimeBackend",
]

# RedisRealtimeBackend requires redis>=5.0.1 (optional dep).
# Import fails cleanly at construction time if redis is absent.
try:
    from roomkit.realtime.redis import RedisRealtimeBackend

    __all__ += ["RedisRealtimeBackend"]
except ImportError:
    pass
