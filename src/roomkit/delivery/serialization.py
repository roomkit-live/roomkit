"""Strategy serialization for persistent delivery queues.

Converts ``DeliveryStrategy`` instances to JSON-serializable dicts
and back so that they can be stored in Redis Streams, NATS JetStream,
or any other persistent queue.
"""

from __future__ import annotations

import logging
from typing import Any

from roomkit.core.delivery import (
    _STRATEGY_MAP,
    DeliveryStrategy,
    Immediate,
    Queued,
    WaitForIdle,
)

logger = logging.getLogger("roomkit.delivery.serialization")


def serialize_strategy(strategy: DeliveryStrategy | None) -> dict[str, Any]:
    """Convert a strategy instance to a JSON-serializable dict."""
    if strategy is None or isinstance(strategy, Immediate):
        return {"type": "immediate", "params": {}}
    if isinstance(strategy, WaitForIdle):
        return {
            "type": "wait_for_idle",
            "params": {
                "buffer": strategy.buffer,
                "playback_timeout": strategy.playback_timeout,
            },
        }
    if isinstance(strategy, Queued):
        return {
            "type": "queued",
            "params": {
                "buffer": strategy.buffer,
                "playback_timeout": strategy.playback_timeout,
                "separator": strategy.separator,
            },
        }
    logger.warning(
        "Cannot serialize strategy %s, falling back to immediate",
        type(strategy).__name__,
    )
    return {"type": "immediate", "params": {}}


def deserialize_strategy(data: dict[str, Any]) -> DeliveryStrategy:
    """Reconstruct a strategy from its serialized form."""
    stype = data.get("type", "immediate")
    params = data.get("params", {})
    cls = _STRATEGY_MAP.get(stype)
    if cls is None:
        logger.warning("Unknown strategy type %r, falling back to Immediate", stype)
        return Immediate()
    return cls(**params)
