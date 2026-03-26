"""Persistent delivery backend for RoomKit."""

from __future__ import annotations

from roomkit.delivery.base import DeliveryBackend, DeliveryItem, DeliveryItemStatus
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.serialization import deserialize_strategy, serialize_strategy
from roomkit.delivery.worker import execute_delivery, run_worker_loop

__all__ = [
    "DeliveryBackend",
    "DeliveryItem",
    "DeliveryItemStatus",
    "InMemoryDeliveryBackend",
    "deserialize_strategy",
    "execute_delivery",
    "run_worker_loop",
    "serialize_strategy",
]

# RedisDeliveryBackend requires redis>=5.0 (optional dep).
# Import fails cleanly at construction time if redis is absent.
try:
    from roomkit.delivery.redis import RedisDeliveryBackend

    __all__ += ["RedisDeliveryBackend"]
except ImportError:
    pass
