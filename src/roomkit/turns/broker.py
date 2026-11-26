"""What a turn backend across processes asks of the services that carry its
messages between processes and keep what they share (RFC §10.6.7, §10.6.8):
a broker for the messages, a registry for the rest. One service may give
both. The backend reaches them through these classes alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BrokerMessage:
    """A message of a node's inbox. The backend writes and reads its fields;
    a broker carries them as they are."""

    kind: str
    """What the message is: ``request``, ``frame``, ``control``,
    ``released``, ``channel``, ``owner`` (a room call), ``work`` (work after
    a turn nothing answers) or ``answer``."""
    sender: str
    """The node that posted it, where an answer goes."""
    body: str
    """Its content, serialized."""
    target: str = ""
    """What it is for on the node that reads it: a turn, an executor."""
    ref: str = ""
    """The reference an answer names, empty when it expects none."""


@dataclass(frozen=True, slots=True)
class InboxEntry:
    """An entry read from an inbox."""

    id: str
    """The entry's id in its inbox: the cursor of a later read, and what an
    acknowledgement names."""
    message: BrokerMessage | None
    """Its message; none for an entry the broker could not read as one."""


@dataclass(frozen=True, slots=True)
class Beat:
    """A node's beat: what it declares, what it withdrew and the turns it
    runs, written to the registry as one change."""

    node: str
    """The node that beats."""
    generation: int
    """The count of the node's changes: a beat older than the node's last
    one is ignored."""
    declared: dict[str, str] = field(default_factory=dict)
    """Each executor the node declares, by its member name
    (``<node>:<executor id>``), with its declaration, serialized."""
    undeclared: list[str] = field(default_factory=list)
    """The executors that still beat and declare nothing."""
    gone: list[str] = field(default_factory=list)
    """The executors that are gone: forgotten."""
    running: list[str] = field(default_factory=list)
    """The turns the node runs."""
    owner: bool = False
    """Whether the node is a room owner."""


@dataclass(frozen=True, slots=True)
class RegistryView:
    """The executors and the room owners beating, as the registry keeps them."""

    now: int
    """The registry's time, in milliseconds since the Unix epoch: the clock
    of every beat, which an executor's last beat reads."""
    executors: dict[str, tuple[int, str | None]]
    """Each executor's last beat, and its declaration when it has one."""
    owners: dict[str, int]
    """Each room owner's last beat, by its node."""


@dataclass(frozen=True, slots=True)
class Beats:
    """What :meth:`TurnRegistry.beats` reads."""

    last: list[int | None]
    """The last beat of each executor asked, in the order asked; none for
    one the registry does not know."""
    running: list[list[bool]]
    """For each node asked, whether it runs each of the turns named."""


class TurnInbox(ABC):
    """A reader of one node's inbox, on a connection of its own."""

    @abstractmethod
    async def read(self, last: str) -> list[InboxEntry]:
        """The entries after the one *last* names, in the order posted,
        waiting for the first no longer than the reader was told; from the
        inbox's first entry for an empty *last*."""

    async def close(self) -> None:  # noqa: B027 - a reader that holds nothing closes nothing
        """Let go of what the reader holds: it reads no more."""


class BeatWriter(ABC):
    """A writer of beats on a connection of its own: the heartbeat's, which
    runs apart from the turns, in a thread of its own."""

    @abstractmethod
    async def beat(self, beat: Beat) -> None:
        """As :meth:`TurnRegistry.beat`."""

    async def close(self) -> None:  # noqa: B027 - a writer that holds nothing closes nothing
        """Let go of what the writer holds: it beats no more."""


class TurnBroker(ABC):
    """The messages between the processes of a turn backend across
    processes: an inbox a node, read in the order posted."""

    @abstractmethod
    async def post(self, node: str, messages: Sequence[BrokerMessage]) -> None:
        """Add *messages* to *node*'s inbox, in their order, after those it
        holds."""

    @abstractmethod
    async def inbox(self, node: str, block: float) -> TurnInbox:
        """A reader of *node*'s inbox, each read waiting *block* seconds at
        most."""

    @abstractmethod
    async def acknowledge(self, node: str, ids: Sequence[str]) -> None:
        """Let go of entries read from *node*'s inbox: *ids* in the order
        read, each entry before the last of them read too."""


class TurnRegistry(ABC):
    """What the processes of a turn backend across processes share: who
    beats, who holds a room, the presence last published."""

    @abstractmethod
    async def connect(self) -> None:
        """Reach the registry, a first time only."""

    @abstractmethod
    def connected(self) -> bool:
        """Whether the registry was reached: a node that never reached it
        has nothing to withdraw there."""

    @abstractmethod
    async def beat(self, beat: Beat) -> None:
        """Write a node's beat at the registry's time and as one change. A
        beat older than the node's last one is ignored."""

    @abstractmethod
    async def beater(self) -> BeatWriter:
        """A writer of beats on a connection of its own."""

    @abstractmethod
    async def view(self, forget_after_ms: int) -> RegistryView:
        """The executors and the owners beating; what last beat more than
        *forget_after_ms* ago is forgotten."""

    @abstractmethod
    async def forget_owner(self, node: str) -> None:
        """*node* is no room owner any more."""

    @abstractmethod
    async def beats(
        self, members: Sequence[str], running: Sequence[tuple[str, Sequence[str]]]
    ) -> Beats:
        """The last beat of each of *members*, and for each node of
        *running* whether it runs each of the turns named; either may name
        none. What is read of one node is read as one of its beats left it."""

    @abstractmethod
    async def holder(self, channel: str, room: str) -> str | None:
        """The executor holding the room's state of *channel*."""

    @abstractmethod
    async def hold(self, channel: str, room: str, member: str) -> None:
        """Record *member* as the executor holding the room's state of
        *channel* from now."""

    @abstractmethod
    async def forget_holder(self, channel: str, room: str) -> None:
        """Forget the room's holder of *channel*."""

    @abstractmethod
    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        """Record the presence last published of *channel_id* in *room_id*:
        whether it is a change, which one caller alone is told, the one that
        then publishes it; ``None`` forgets it."""
