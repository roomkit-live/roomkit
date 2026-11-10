"""Turn execution behind a turn backend (RFC §10.6).

Every turn of an intelligence channel runs through a :class:`TurnBackend`
between the room owner and a :class:`TurnExecutor`, as serialized messages.
The default :class:`InMemoryTurnBackend` keeps it in the owner's process;
:class:`RedisTurnBackend` (``pip install roomkit[redis]``) carries it to
executors in other processes.
"""

from __future__ import annotations

from roomkit.core._turn_scope import current_turn_id
from roomkit.turns.base import (
    ExecutorLink,
    OwnerEvents,
    OwnerHost,
    OwnerTools,
    RoomOwner,
    TurnBackend,
    TurnLink,
    TurnTimeouts,
)
from roomkit.turns.executor import TurnExecutor
from roomkit.turns.host import DeclaredVerb, TurnHost, current_turn_host
from roomkit.turns.memory import InMemoryTurnBackend
from roomkit.turns.owner import TurnClient
from roomkit.turns.presence import AgentPresence, ExecutorInfo
from roomkit.turns.redis import RedisTurnBackend
from roomkit.turns.wire import (
    ChannelCall,
    ChannelControl,
    ChannelResult,
    CommittedEvent,
    OwnerTool,
    PreviousTurn,
    RoomAttached,
    RoomCall,
    RoomDetached,
    RoomPublication,
    TurnRequest,
)

__all__ = [
    "AgentPresence",
    "ChannelCall",
    "CommittedEvent",
    "ChannelControl",
    "ChannelResult",
    "DeclaredVerb",
    "ExecutorInfo",
    "ExecutorLink",
    "InMemoryTurnBackend",
    "OwnerEvents",
    "OwnerHost",
    "OwnerTool",
    "OwnerTools",
    "PreviousTurn",
    "RedisTurnBackend",
    "RoomAttached",
    "RoomCall",
    "RoomOwner",
    "RoomPublication",
    "RoomDetached",
    "TurnBackend",
    "TurnClient",
    "TurnExecutor",
    "TurnHost",
    "TurnLink",
    "TurnRequest",
    "TurnTimeouts",
    "current_turn_host",
    "current_turn_id",
]
