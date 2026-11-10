"""The presence of a kit's agents, published in their rooms (RFC §10.6.7,
§8.4).

An agent is online in a room while an executor online covers it there, and
an agent the kit holds live while the kit's own executor takes its turns.
The kit publishes an agent's presence in a room as the agent is attached to
it, and again when it changes; a detach, and the room's close, forget it
there. It reads the executors its turn backend carries turns to at each
heartbeat interval, and at once when another executor of its process is
served or closes. When the executors covering an agent the kit holds by its
declaration change, it reads the agent's rooms in the store and publishes
``presence_online`` or ``presence_offline`` in each room whose state changed;
the turn backend's claim leaves one room owner process to publish each
change, and that owner publishes again at a later look, while it runs, a
publication or a claim that failed (closing, it lets go of the claims it
could not publish). With a turn backend that does not list its executors
(``TurnBackend.executors`` left as it is), nothing is published for the
agents held by their declaration: the kit cannot tell whether they are
served.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any

from roomkit.channels.declared import DeclaredChannel
from roomkit.models.enums import ChannelCategory, RoomStatus
from roomkit.models.store_filter import RoomFilter
from roomkit.turns.base import TurnTimeouts, heartbeat_tick, lists_executors

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.framework_event import FrameworkEvent
    from roomkit.turns.presence import ExecutorInfo

logger = logging.getLogger("roomkit.turns")

# The most rooms of one agent a change is published in.
ROOMS_READ = 10_000
_LIVE_ROOMS = RoomFilter(statuses=[RoomStatus.ACTIVE, RoomStatus.PAUSED])


class AgentPresenceWatch:
    """Publishes the presence of a kit's agents in their rooms."""

    def __init__(self, kit: RoomKit) -> None:
        self._kit = kit
        # For each agent held by its declaration, the executors covering it
        # at the last look, each with its room hooks.
        self._covering: dict[str, frozenset[tuple[str, frozenset[str]]]] = {}
        # What this owner could not publish yet, by agent and room: the state
        # it claimed (it holds the claim, so no other owner publishes it), or
        # None when the claim itself failed.
        self._unpublished: dict[tuple[str, str], bool | None] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        # The forgets of rooms that closed, by room: an attach there waits.
        self._forgetting: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def start(self) -> None:
        """Watch the executors, once the kit holds an agent by its declaration
        (on a backend that lists its executors) or has a presence to publish
        again, and from a running loop; nothing otherwise."""
        if self._task is not None or self._closed:
            return
        if not (self._declared() and self._lists_executors()) and not self._unpublished:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._kit._turns.backend._watch_executors(self.check)
        self._task = loop.create_task(self._watch(), name="roomkit-agent-presence")

    async def close(self) -> None:
        """Stop watching, and let go of the claims this owner could not
        publish: the room's next change or attach, or an owner that starts,
        publishes them."""
        self._closed = True
        for task in [self._task, *self._forgetting.values()]:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._task = None
        backend = self._kit._turns.backend
        for (agent, room_id), claimed in list(self._unpublished.items()):
            if claimed is not None:
                with contextlib.suppress(Exception):
                    await backend.claim_presence(agent, room_id, None)
        self._unpublished.clear()

    async def attached(self, room_id: str, channel_id: str) -> None:
        """*channel_id* was attached to *room_id*: its presence there now,
        unless a look at the executors published it as the binding landed.
        A failure is logged, never raised into the attach."""
        self.start()
        await self._quietly("an attach", self._publish_in(room_id, [channel_id]))

    async def detached(self, room_id: str, channel_id: str) -> None:
        """*channel_id* left *room_id*: what was published there is
        forgotten, so a later attach publishes again."""
        await self._quietly("a detach", self._forget(room_id, [channel_id]))

    async def room_changed(self, room_id: str) -> None:
        """The room's enabled hooks changed: its agents' presence there."""
        await self._quietly("a room hook change", self._publish_in(room_id, None))

    async def room_closed(self, event: FrameworkEvent) -> None:
        """A room closed, by a call or by its timer: no change is published
        there any more, and what was published there is forgotten, apart
        from the close, which runs under the room's lock."""
        room_id = event.room_id
        if room_id is None or self._closed:
            return
        task = asyncio.get_running_loop().create_task(
            self._forget_closed(room_id), name="roomkit-presence-forget"
        )
        self._forgetting[room_id] = task
        task.add_done_callback(lambda done: self._forgot(room_id, done))

    async def _forget_closed(self, room_id: str) -> None:
        # The forget's coroutine is made here, in the task: a task cut before
        # its first step leaves no coroutine unawaited.
        await self._quietly("a room close", self._forget(room_id, None))

    def _forgot(self, room_id: str, task: asyncio.Task[None]) -> None:
        if self._forgetting.get(room_id) is task:
            del self._forgetting[room_id]

    async def check(self) -> None:
        """Publish again what this owner could not publish, then the presence
        of each agent held by its declaration whose covering executors
        changed since the last look, in its rooms."""
        if self._closed:
            return
        async with self._lock:
            infos = await self._kit.executors()
            await self._publish_unpublished(infos)
            for agent in self._declared():
                covering = frozenset(
                    (info.executor_id, info.room_hooks) for info in infos if info.covers(agent)
                )
                if self._covering.get(agent) == covering:
                    continue
                try:
                    await self._publish_in_rooms(agent, infos)
                except Exception:
                    # Its rooms unread: the next look tries the change again.
                    logger.warning("Reading the rooms of agent %s failed", agent, exc_info=True)
                    continue
                self._covering[agent] = covering

    async def _publish_in(self, room_id: str, channel_ids: list[str] | None) -> None:
        """Publish in *room_id* the presence of *channel_ids*, or of all its
        intelligence channels, once a forget of the room's close is done."""
        forgetting = self._forgetting.get(room_id)
        if forgetting is not None:
            await asyncio.wait({forgetting})
        async with self._lock:
            infos = await self._kit.executors()
            for channel_id in channel_ids or await self._agents_of(room_id):
                await self._publish_or_queue(channel_id, room_id, infos)

    async def _forget(self, room_id: str, channel_ids: list[str] | None) -> None:
        async with self._lock:
            backend = self._kit._turns.backend
            for channel_id in channel_ids or await self._agents_of(room_id):
                self._unpublished.pop((channel_id, room_id), None)
                await backend.claim_presence(channel_id, room_id, None)

    async def _agents_of(self, room_id: str) -> list[str]:
        return [
            binding.channel_id
            for binding in await self._kit.store.list_bindings(room_id)
            if binding.category == ChannelCategory.INTELLIGENCE
        ]

    async def _quietly(self, what: str, work: Coroutine[Any, Any, None]) -> None:
        """Run *work*, logging what it raises: a presence that could not be
        published never fails the call that changed it."""
        try:
            await work
        except Exception:
            logger.warning("Publishing the agents' presence on %s failed", what, exc_info=True)

    async def _watch(self) -> None:
        backend = self._kit._turns.backend
        interval = (backend.timeouts or TurnTimeouts()).heartbeat_interval
        while True:
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Reading the agents' presence failed", exc_info=True)
            await heartbeat_tick(interval)

    def _lists_executors(self) -> bool:
        """Whether the turn backend lists its executors: one that does not
        reads every agent held by its declaration as offline, untruly."""
        return lists_executors(self._kit._turns.backend)

    def _declared(self) -> list[str]:
        return [cid for cid, ch in self._kit.channels.items() if isinstance(ch, DeclaredChannel)]

    async def _publish_in_rooms(self, agent: str, infos: list[ExecutorInfo]) -> None:
        """Publish *agent*'s presence in each of its rooms whose state
        changed; a room whose publication fails waits for the next look."""
        store = self._kit.store
        room_ids = await store.find_room_ids_by_channel(agent, _LIVE_ROOMS, limit=ROOMS_READ)
        if len(room_ids) >= ROOMS_READ:
            logger.warning(
                "Agent %s is attached to %d rooms or more: its presence is published "
                "in the %d oldest only",
                agent,
                ROOMS_READ,
                ROOMS_READ,
            )
        for room_id in room_ids:
            await self._publish_or_queue(agent, room_id, infos)

    async def _publish_unpublished(self, infos: list[ExecutorInfo]) -> None:
        """Publish what this owner could not publish, in the rooms the agent
        is still attached to: as it claimed it when it still holds, else the
        presence now, claimed anew."""
        kit = self._kit
        for (agent, room_id), claimed in list(self._unpublished.items()):
            try:
                if await kit.store.get_binding(room_id, agent) is None:
                    del self._unpublished[(agent, room_id)]
                    continue
                presence = kit._presence_of(agent, kit._turns.room_hook_names(room_id), infos)
                if claimed is not None and presence.online == claimed:
                    await self._send(agent, room_id, claimed)
                    continue
            except Exception:
                logger.debug("Publishing the presence of %s in %s failed again", agent, room_id)
                continue
            del self._unpublished[(agent, room_id)]
            await self._publish_or_queue(agent, room_id, infos, retry=True)

    async def _publish_or_queue(
        self, agent: str, room_id: str, infos: list[ExecutorInfo], *, retry: bool = False
    ) -> None:
        """:meth:`_publish`, a failure kept for a later look: logged as it
        first fails, at debug level as a *retry* fails again."""
        try:
            await self._publish(agent, room_id, infos)
        except Exception:
            if retry:
                logger.debug("Publishing the presence of %s in %s failed again", agent, room_id)
                return
            logger.warning(
                "Publishing the presence of %s in room %s failed: it is tried again",
                agent,
                room_id,
                exc_info=True,
            )
            self.start()

    async def _publish(self, agent: str, room_id: str, infos: list[ExecutorInfo]) -> None:
        """Publish *agent*'s presence in *room_id* when the claim says it
        changed: one room owner process publishes each change. A claim that
        fails is kept for a later look."""
        kit = self._kit
        if isinstance(kit.channels.get(agent), DeclaredChannel) and not self._lists_executors():
            return
        presence = kit._presence_of(agent, kit._turns.room_hook_names(room_id), infos)
        try:
            claimed = await kit._turns.backend.claim_presence(agent, room_id, presence.online)
        except Exception:
            self._unpublished[(agent, room_id)] = None
            raise
        if claimed:
            await self._send(agent, room_id, presence.online)

    async def _send(self, agent: str, room_id: str, online: bool) -> None:
        """Publish the presence this owner claimed; one that fails is kept,
        with its claim, for this owner's later look."""
        try:
            await self._kit.publish_presence(room_id, agent, "online" if online else "offline")
        except Exception:
            self._unpublished[(agent, room_id)] = online
            raise
        self._unpublished.pop((agent, room_id), None)
