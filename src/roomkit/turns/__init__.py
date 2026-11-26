"""Turn execution behind a turn backend (RFC §10.6).

Every turn of an intelligence channel runs through a :class:`TurnBackend`
between the room owner and a :class:`TurnExecutor`, as serialized messages.
The default :class:`InMemoryTurnBackend` keeps it in the owner's process;
:class:`DistributedTurnBackend` carries it to executors in other processes,
over a :class:`TurnBroker` and a :class:`TurnRegistry`. On Redis,
:class:`RedisTurnBackend` (``pip install roomkit[redis]``) is that backend
assembled; :class:`InMemoryTurnBroker` and :class:`InMemoryTurnRegistry`
keep both services in one process.
"""

from __future__ import annotations

from roomkit.core._turn_scope import current_turn_id
from roomkit.turns._memory_ports import InMemoryTurnBroker, InMemoryTurnRegistry
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
from roomkit.turns.broker import (
    Beat,
    Beats,
    BeatWriter,
    BrokerMessage,
    InboxEntry,
    RegistryView,
    TurnBroker,
    TurnInbox,
    TurnRegistry,
)
from roomkit.turns.distributed import DistributedTurnBackend
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
    "Beat",
    "BeatWriter",
    "Beats",
    "BrokerMessage",
    "ChannelCall",
    "CommittedEvent",
    "ChannelControl",
    "ChannelResult",
    "DeclaredVerb",
    "DistributedTurnBackend",
    "ExecutorInfo",
    "ExecutorLink",
    "InMemoryTurnBackend",
    "InMemoryTurnBroker",
    "InMemoryTurnRegistry",
    "InboxEntry",
    "OwnerEvents",
    "OwnerHost",
    "OwnerTool",
    "OwnerTools",
    "PreviousTurn",
    "RedisTurnBackend",
    "RegistryView",
    "RoomAttached",
    "RoomCall",
    "RoomOwner",
    "RoomPublication",
    "RoomDetached",
    "TurnBackend",
    "TurnBroker",
    "TurnClient",
    "TurnExecutor",
    "TurnHost",
    "TurnInbox",
    "TurnLink",
    "TurnRegistry",
    "TurnRequest",
    "TurnTimeouts",
    "current_turn_host",
    "current_turn_id",
]
