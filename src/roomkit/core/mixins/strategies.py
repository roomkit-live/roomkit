"""StrategyMixin — one orchestration strategy per room, installed while the room lives (RFC §19.7).

``install_strategy`` claims the room for one strategy, registers and attaches
the strategy's agents as ``create_room`` does, runs its install, and records
what the install added for the room. ``uninstall_strategy`` runs the
strategy's own uninstall step, then takes back exactly what was recorded: the
room hooks, the tools and turn runners set up on the room's agents, the
agents the install attached and the room metadata it wrote. A room can then
take another strategy; its timeline stays. Recording what changed, rather
than asking each strategy what it set up, holds for the host's own
strategies too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import ChannelCategory

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.orchestration.base import Orchestration
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")


@dataclass
class _Snapshot:
    """What a room holds that a strategy's install may add to."""

    hooks: set[str]
    tools: dict[str, frozenset[str]]
    runners: set[str]
    bindings: set[str]
    keys: set[str]


@dataclass
class _Installed:
    """A room's strategy, and what its install added for the room."""

    strategy: Orchestration
    organization_id: str | None = None
    hooks: list[str] = field(default_factory=list)
    tools: dict[str, frozenset[str]] = field(default_factory=dict)
    runners: list[str] = field(default_factory=list)
    attached: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)


class StrategyMixin(HelpersMixin):
    """Adds ``install_strategy()``, ``uninstall_strategy()`` and ``room_strategy()``."""

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _hook_engine: HookEngine
    _channels: dict[str, Channel]
    _discussions: dict[str, Any]
    _room_strategies: dict[str, _Installed]

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # RoomLifecycleMixin
    attach_channel: Any  # ChannelOpsMixin
    detach_channel: Any  # ChannelOpsMixin
    register_channel: Any  # ChannelOpsMixin

    def room_strategy(
        self, room_id: str, *, organization_id: str | None = None
    ) -> Orchestration | None:
        """The strategy installed in *room_id* through this kit, if any.
        *organization_id* scopes the read to one tenant (RFC §17.2): a room of
        another organization reads as holding none."""
        installed = self._room_strategies.get(room_id)
        if installed is None:
            return None
        if organization_id is not None and installed.organization_id != organization_id:
            return None
        return installed.strategy

    async def install_strategy(
        self, room_id: str, strategy: Orchestration, *, organization_id: str | None = None
    ) -> None:
        """Install *strategy* in a room, new or already in use (RFC §19.7).

        The strategy's agents are registered and attached as intelligence
        channels, then its install runs; the room's timeline stays, and the
        agents' turns read it. *organization_id* scopes the call to one
        tenant (RFC §17.2).

        Raises:
            ValueError: the room holds a strategy already (uninstall it first),
                or the strategy refuses the room.
        """
        room = await self.get_room(room_id, organization_id=organization_id)
        await self._install_strategy(room_id, strategy, organization_id=room.organization_id)

    async def uninstall_strategy(
        self, room_id: str, *, organization_id: str | None = None
    ) -> bool:
        """Take the room's strategy back out (RFC §19.7): its own uninstall
        step, then what its install added for the room. The room answers by
        its ``agent_response_policy`` again and may take another strategy.
        Whether a strategy was installed."""
        await self.get_room(room_id, organization_id=organization_id)
        installed = self._room_strategies.get(room_id)
        if installed is None:
            return False
        await installed.strategy.uninstall(self, room_id)  # ty: ignore[invalid-argument-type]
        # Released only once all of it is taken back: a room left with part of
        # a strategy is still that strategy's, never open to a second one.
        await self._take_back(room_id, installed)
        del self._room_strategies[room_id]
        return True

    def claim_room_strategy(self, room_id: str, strategy: Orchestration) -> None:
        """Claim *room_id* for *strategy*, or refuse: a room holds one strategy
        (RFC §19.7). :meth:`install_strategy` claims for the strategy it
        installs; a strategy installed by calling its ``install`` directly
        calls this first, as the built-in ones do.

        Raises:
            ValueError: the room holds another strategy or a discussion.
        """
        installed = self._room_strategies.get(room_id)
        if installed is not None and installed.strategy is not strategy:
            raise ValueError(
                f"Room {room_id} holds a strategy already: uninstall_strategy() first"
            )
        if installed is None and room_id in self._discussions:
            raise ValueError(f"Room {room_id} holds a discussion: no other strategy can join it")

    async def _install_strategy(
        self, room_id: str, strategy: Orchestration, *, organization_id: str | None
    ) -> None:
        self.claim_room_strategy(room_id, strategy)
        if room_id in self._room_strategies:
            raise ValueError(f"Room {room_id} holds this strategy already")
        # Claimed before the first await: two installs at once are one too many.
        self._room_strategies[room_id] = _Installed(strategy, organization_id)
        # Under the room lock, what changes in the room meanwhile is the install's.
        async with self._lock_manager.locked(room_id):
            before = await self._snapshot(room_id)
            try:
                await self._attach_agents(room_id, strategy)
                await strategy.install(self, room_id)  # ty: ignore[invalid-argument-type]
            except BaseException:
                await self._undo_install(room_id, strategy, organization_id, before)
                raise
            self._room_strategies[room_id] = _added(
                strategy, organization_id, before, await self._snapshot(room_id)
            )

    async def _undo_install(
        self,
        room_id: str,
        strategy: Orchestration,
        organization_id: str | None,
        before: _Snapshot,
    ) -> None:
        """Take back what a failed install added; the room stays claimed for
        what could not be, so ``uninstall_strategy`` can finish it."""
        added = _added(strategy, organization_id, before, await self._snapshot(room_id))
        try:
            await self._take_back(room_id, added)
        except Exception:
            self._room_strategies[room_id] = added
            logger.exception("A failed install in room %s was not taken back whole", room_id)
            return
        del self._room_strategies[room_id]

    async def _attach_agents(self, room_id: str, strategy: Orchestration) -> None:
        bound = {b.channel_id for b in await self._store.list_bindings(room_id)}
        for agent in strategy.agents():
            if agent.channel_id not in self._channels:
                self.register_channel(agent)
            if agent.channel_id not in bound:
                await self.attach_channel(
                    room_id, agent.channel_id, category=ChannelCategory.INTELLIGENCE
                )

    async def _snapshot(self, room_id: str) -> _Snapshot:
        room = await self._store.get_room(room_id)
        registries = self._registries()
        return _Snapshot(
            hooks=set(self._hook_engine.room_hook_names(room_id)),
            tools={cid: reg.room_tool_names(room_id) for cid, reg in registries.items()},
            runners={cid for cid, reg in registries.items() if reg.turn_runner(room_id)},
            bindings={b.channel_id for b in await self._store.list_bindings(room_id)},
            keys=set(room.metadata) if room is not None else set(),
        )

    def _registries(self) -> dict[str, ChannelRegistry]:
        return {
            cid: registry
            for cid, channel in self._channels.items()
            if isinstance(registry := getattr(channel, "_registry", None), ChannelRegistry)
        }

    async def _take_back(self, room_id: str, installed: _Installed) -> None:
        """Remove what a strategy's install added for *room_id*: everything it
        can, then a failure if anything stays (what was taken back is dropped
        from *installed*, so a retry finishes the rest)."""
        for name in installed.hooks:
            self._hook_engine.remove_room_hook(room_id, name)
        installed.hooks = []
        registries = self._registries()
        for cid in {*installed.tools, *installed.runners}:
            registry = registries.get(cid)
            if registry is not None:
                registry.release(
                    room_id, installed.tools.get(cid, ()), turn_runner=cid in installed.runners
                )
        installed.tools, installed.runners = {}, []
        stayed: list[str] = []
        for cid in list(installed.attached):
            try:
                await self.detach_channel(room_id, cid)
                installed.attached.remove(cid)
            except Exception:
                logger.exception("Agent %s was not detached from room %s", cid, room_id)
                stayed.append(cid)
        if installed.keys:
            await self._store.patch_room_metadata(room_id, dict.fromkeys(installed.keys))
            installed.keys = []
        if stayed:
            raise RuntimeError(f"Room {room_id} still binds {stayed} its strategy attached")


def _added(
    strategy: Orchestration, organization_id: str | None, before: _Snapshot, after: _Snapshot
) -> _Installed:
    """What a strategy's install added, from the room before and after it."""
    return _Installed(
        strategy,
        organization_id,
        hooks=sorted(after.hooks - before.hooks),
        tools={
            cid: names - before.tools.get(cid, frozenset())
            for cid, names in after.tools.items()
            if names - before.tools.get(cid, frozenset())
        },
        runners=sorted(after.runners - before.runners),
        attached=sorted(after.bindings - before.bindings),
        keys=sorted(after.keys - before.keys),
    )
