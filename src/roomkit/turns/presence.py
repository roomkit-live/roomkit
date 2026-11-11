"""Which executors are online, and which agents they cover (RFC §10.6.7,
§10.6.8).

A turn backend knows the executors it carries turns to: its own process's,
and, for a distributed one, the executors of the other processes from their
beats. The kit reads them (``kit.executors()``), and tells a room's agents
online or offline from them (``kit.agent_presence(room_id)``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, PrivateAttr

from roomkit.turns._hosting import Hosted
from roomkit.turns._organizations import serves_organization

if TYPE_CHECKING:
    from roomkit.turns.executor import TurnExecutor


class ExecutorInfo(BaseModel):
    """An executor a turn backend carries turns to, as the backend sees it.

    Attributes:
        executor_id: The id it answers with.
        node: The process it runs in, as the backend names it; ``None`` for
            a backend that knows only its own process.
        hosts: The channel ids and prefixes it declares (``"acme-*"``);
            ``None`` for a predicate or every id, which keeps to its own
            process.
        organizations: The organizations whose rooms it serves (``"*"``
            every one); ``None`` for the rooms without an organization
            alone (RFC §10.6.7).
        room_hooks: The names of the room hooks it defines (RFC §10.6.6).
        last_beat: When it last beat; ``None`` when the backend reads no
            beat for it (an executor of this process, a backend without
            beats).
        online: Whether it takes turns now: beating within the heartbeat
            timeout, not closed.
    """

    model_config = ConfigDict(frozen=True)

    executor_id: str
    node: str | None = None
    hosts: frozenset[str] | None = None
    organizations: frozenset[str] | None = None
    room_hooks: frozenset[str] = frozenset()
    last_beat: datetime | None = None
    online: bool = True
    # How an executor of this process tells a channel it takes, a predicate
    # included; ``None`` reads ``hosts``.
    _declares: Callable[[str], bool] | None = PrivateAttr(default=None)

    def covers(
        self,
        channel_id: str,
        room_hooks: Iterable[str] = (),
        organization_id: str | None = None,
    ) -> bool:
        """Whether, online, it takes the turns of *channel_id* in a room with
        the hooks named *room_hooks*, of *organization_id* (``None``: a room
        without an organization) (RFC §10.6.7)."""
        if not self.online or not set(room_hooks) <= self.room_hooks:
            return False
        if not serves_organization(self.organizations, organization_id):
            return False
        if self._declares is not None:
            return self._declares(channel_id)
        return self.hosts is not None and Hosted(self.hosts).covers(channel_id)


class AgentPresence(BaseModel):
    """Whether one of a room's agents can answer there now.

    Attributes:
        channel_id: The agent's channel id.
        online: Whether an executor online covers it in the room: declares
            it and defines the room's enabled hooks; an agent the kit holds
            live is online while the kit runs.
        executors: The ids of the executors that cover it.
    """

    model_config = ConfigDict(frozen=True)

    channel_id: str
    online: bool
    executors: list[str]


def local_info(executor: TurnExecutor, node: str | None = None) -> ExecutorInfo:
    """The info of *executor*, served in this process: it tells the channels
    it covers as the executor does, a predicate included."""
    info = ExecutorInfo(
        executor_id=executor.executor_id,
        node=node,
        hosts=executor.hosted,
        organizations=executor.organizations,
        room_hooks=frozenset(executor.hooks.defined_room_hooks()),
        online=not executor.closed,
    )
    info._declares = executor.hosts_channel
    return info
