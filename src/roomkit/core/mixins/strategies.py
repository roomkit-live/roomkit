"""StrategyMixin — one orchestration strategy per room, installed while the room lives (RFC §19.7).

``install_strategy`` claims the room for one strategy, registers and attaches
the strategy's agents as ``create_room`` does, runs its install, and records
what the install added for the room. ``uninstall_strategy`` runs the
strategy's own uninstall step, then takes back exactly what was recorded: the
room hooks, the tools set up on the room's agents, the turn runners the kit
keeps for them, the agents the install attached and the room metadata it
wrote. A room can then
take another strategy; its timeline stays. Recording what changed, rather
than asking each strategy what it set up, holds for the host's own
strategies too.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import ChannelRegistry
from roomkit.channels.declared import id_of, register_strategy_agents
from roomkit.core.locks import _has_room_lock
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import ChannelCategory

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core._turn_runners import TurnRunners
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.orchestration.base import Orchestration
    from roomkit.store.base import ConversationStore
    from roomkit.turns._registry_tools import OwnerToolStore

logger = logging.getLogger("roomkit.framework")


@dataclass
class _Snapshot:
    """What a room holds that a strategy's install may add to."""

    hooks: set[tuple[bool, str]]  # (on the turn executor, name)
    tools: dict[str, frozenset[str]]
    runners: set[str]
    bindings: set[str]
    keys: set[str]


@dataclass
class _Installed:
    """A room's strategy, and what its install added for the room."""

    strategy: Orchestration
    organization_id: str | None = None
    hooks: list[tuple[bool, str]] = field(default_factory=list)  # (on the executor, name)
    tools: dict[str, frozenset[str]] = field(default_factory=dict)
    runners: list[str] = field(default_factory=list)
    attached: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    installing: bool = False
    """The record claims the room while the install runs."""
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    """Set once the install that claimed the room with this record is over,
    whatever its end: what an uninstall called meanwhile waits for."""


class StrategyMixin(HelpersMixin):
    """Adds ``install_strategy()``, ``uninstall_strategy()``, ``room_strategy()``
    and ``room_strategy_is_current()``."""

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _hook_engine: HookEngine
    _turn_hooks: HookEngine  # the kit's turn executor's (RFC §10.6.6)
    _turns: Any  # RoomKit's TurnClient
    _turn_runners: TurnRunners  # the strategies' runners, kept by the owner (RFC §10.6.1)
    _owner_tools: OwnerToolStore  # the tools installs set up, kept by the owner (RFC §10.6.2)
    _channels: dict[str, Channel]
    _discussions: dict[str, Any]
    _room_strategies: dict[str, _Installed]

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # RoomLifecycleMixin
    attach_channel: Any  # ChannelOpsMixin
    detach_channel: Any  # ChannelOpsMixin
    register_channel: Any  # ChannelOpsMixin
    _follow_discussion: Any  # DiscussionMixin
    _replaces_discussion: Any  # DiscussionMixin
    _hold_attached: Any  # DiscussionMixin
    _runs_discussion: Any  # DiscussionMixin
    _stored_attached: Any  # DiscussionMixin
    _stored_with_room: Any  # DiscussionMixin
    _discussion_is_current: Any  # DiscussionMixin
    _uninstall_stored_discussion: Any  # DiscussionMixin

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

    async def room_strategy_is_current(
        self,
        room_id: str,
        kind: type[Orchestration],
        *,
        organization_id: str | None = None,
    ) -> bool:
        """Whether the strategy of *kind* this process installed in *room_id*
        is the one the room stores now (RFC §19.7).

        A strategy stored with the room, a discussion (RFC §19.7.5 rule 16),
        may be replaced by another process's install with another
        configuration, after which this process gives no turn. False then,
        when this process installed none, and once it was uninstalled; the
        room's stored state is read when called. An uninstall another process
        made is taken here at once: :meth:`room_strategy` answers None from
        then on, where a replaced install still names this process's own.
        A host that finds it false installs again only while its own state
        still wants the discussion, else it would bring back one another
        process took out::

            if wanted and not await kit.room_strategy_is_current(room_id, Discussion):
                await kit.install_strategy(room_id, Discussion(agents))

        *organization_id* scopes the call to one tenant (RFC §17.2).

        Raises:
            TypeError: *kind* is not a strategy stored with the room.
            RoomNotFoundError: the room does not exist in the scope.
        """
        if not self._stored_with_room(kind):
            raise TypeError(
                f"room_strategy_is_current() takes a strategy class stored with the room "
                f"(Discussion), not {kind!r}: it is not a strategy stored with the room"
            )
        return await self._discussion_is_current(room_id, kind, organization_id=organization_id)

    async def install_strategy(
        self, room_id: str, strategy: Orchestration, *, organization_id: str | None = None
    ) -> None:
        """Install *strategy* in a room, new or already in use (RFC §19.7).

        The strategy's agents are registered and attached as intelligence
        channels, then its install runs; the room's timeline stays, and the
        agents' turns read it. *organization_id* scopes the call to one
        tenant (RFC §17.2).

        A Discussion installed in a room that holds a discussion, installed
        through this kit or stored by another process, takes its place and
        keeps its queue (RFC §19.7.5 rule 1): how a process that restarted
        takes the discussion back, and how the host changes its agents.

        Raises:
            ValueError: the room holds a strategy already (uninstall it first),
                or the strategy refuses the room.
        """
        room = await self.get_room(room_id, organization_id=organization_id)
        # A discussion another process stored is this room's here too (rule 16).
        self._follow_discussion(room)
        await self._install_strategy(room_id, strategy, organization_id=room.organization_id)

    async def uninstall_strategy(
        self, room_id: str, *, organization_id: str | None = None
    ) -> bool:
        """Take the room's strategy back out (RFC §19.7): its own uninstall
        step, then what its install added for the room. The room answers by
        its ``agent_response_policy`` again and may take another strategy.
        A discussion is uninstalled from any process serving the room, one
        where no host installed it included (RFC §19.7.5 rule 16). Whether a
        strategy was installed."""
        room = await self.get_room(room_id, organization_id=organization_id)
        installed = await self._installed_once_over(room_id)
        if installed is not None and installed.installing and not installed.ended.is_set():
            # An install the caller holds the room's lock against, or runs
            # within: nothing is installed yet (RFC §19.7).
            return False
        if installed is None:
            return await self._uninstall_stored_discussion(room)
        await installed.strategy.uninstall(self, room_id)  # ty: ignore[invalid-argument-type]
        # A discussion's agents another process's install attached too.
        installed.attached = sorted({*installed.attached, *self._stored_attached(room)})
        # Released only once all of it is taken back: a room left with part of
        # a strategy is still that strategy's, never open to a second one.
        await self._take_back(room_id, installed)
        del self._room_strategies[room_id]
        return True

    async def _installed_once_over(self, room_id: str) -> _Installed | None:
        """The record of *room_id*'s strategy once no install of it runs: an
        install under way is waited for, so an uninstall called meanwhile
        takes back what it installed rather than let it record itself again
        (RFC §19.7). An install is not waited for by code that holds the
        room's lock, which that install waits for or runs under: the
        record is returned as it stands. An install over that left its
        claim (one that failed before it could take it back) ends the
        wait too."""
        installed = self._room_strategies.get(room_id)
        while installed is not None and installed.installing and not installed.ended.is_set():
            if _has_room_lock(room_id, self._lock_manager):
                break
            await installed.ended.wait()
            installed = self._room_strategies.get(room_id)
        return installed

    async def claim_room_strategy(
        self, room_id: str, strategy: Orchestration, *, organization_id: str | None = None
    ) -> None:
        """Claim *room_id* for *strategy*, or refuse: a room holds one strategy
        (RFC §19.7). :meth:`install_strategy` claims for the strategy it
        installs; a strategy installed by calling its ``install`` directly
        calls this first, as the built-in ones do.

        A Discussion may claim a room holding a discussion no install of this
        kit recorded, to take its place (RFC §19.7.5 rule 1). The room is
        read first, as an install reads it: a discussion another process
        stored is this room's here too (rule 16).

        *organization_id* is the organization the caller acts for (RFC
        §17.2): the room is read with it, and another organization's room is
        not found, whatever strategy it holds.

        Raises:
            RoomNotFoundError: *organization_id* is set and the room is
                missing or another organization's.
            ValueError: the room holds another strategy or a discussion.
        """
        if organization_id is not None:
            room = await self.get_room(room_id, organization_id=organization_id)
        else:
            room = await self._store.get_room(room_id)
        if room is not None:
            self._follow_discussion(room)
        self._check_claim(room_id, strategy)

    def _check_claim(self, room_id: str, strategy: Orchestration) -> None:
        """Refuse *strategy* a room holding another strategy or a discussion
        it does not replace, with no await: an install claims and records in
        one step (RFC §19.7)."""
        installed = self._room_strategies.get(room_id)
        if installed is not None and installed.strategy is not strategy:
            raise ValueError(
                f"Room {room_id} holds a strategy already: uninstall_strategy() first"
            )
        if (
            installed is None
            and room_id in self._discussions
            and not self._replaces_discussion(room_id, strategy)
        ):
            raise ValueError(f"Room {room_id} holds a discussion: no other strategy can join it")

    async def _forget_strategy(self, room_id: str) -> None:
        """Take back the strategy recorded for *room_id* when the store has
        just created the room anew.

        The record outlived its room (deleted from the store, or overwritten
        by a create with the same id): what its install set up in this
        process (room hooks, tools and turn runners, the strategy's own
        state) would serve the new room, and the record would refuse it a
        strategy (RFC §19.7). The new room holds none of the old one's
        bindings or metadata, so there are none to take back.
        """
        installed = self._room_strategies.get(room_id)
        if installed is None:
            return
        await installed.strategy.uninstall(self, room_id)  # ty: ignore[invalid-argument-type]
        await self._drop_record(room_id, installed.strategy)

    async def _drop_record(self, room_id: str, strategy: Orchestration | None) -> None:
        """Forget the record of *strategy* in *room_id*, taking back what its
        install set up in this process: its hooks, tools and turn runners.
        The room's bindings and metadata are not this process's to take back
        any more (the room is new, or another process uninstalled it)."""
        installed = self._room_strategies.get(room_id)
        if installed is None or installed.strategy is not strategy:
            return
        installed.attached, installed.keys = [], []
        await self._take_back(room_id, installed)
        del self._room_strategies[room_id]

    async def _install_strategy(
        self, room_id: str, strategy: Orchestration, *, organization_id: str | None
    ) -> None:
        replacing, held = self._claim_install(room_id, strategy, organization_id)
        claimed = self._room_strategies[room_id]
        try:
            # Under the room lock, what changes in the room meanwhile is the install's.
            async with self._lock_manager.locked(room_id):
                before = await self._snapshot(room_id)
                try:
                    if not replacing:
                        # A discussion taking another's place binds its new agents
                        # itself, once it holds the room (RFC §19.7.5 rule 1).
                        await self._attach_agents(room_id, strategy)
                    await strategy.install(self, room_id)  # ty: ignore[invalid-argument-type]
                except BaseException:
                    await self._undo_install(
                        room_id, strategy, organization_id, before, replacing, held
                    )
                    raise
                installed, bound = await self._recorded(room_id, strategy, organization_id, before)
            self._room_strategies[room_id] = (
                installed if held is None else _joined(held, installed, bound)
            )
        finally:
            claimed.ended.set()

    def _claim_install(
        self, room_id: str, strategy: Orchestration, organization_id: str | None
    ) -> tuple[bool, _Installed | None]:
        """Claim *room_id* for an install of *strategy*, before any await: two
        installs at once are one too many. Whether it takes the place of the
        room's discussion (RFC §19.7.5 rule 1), and that discussion's record.

        Raises:
            ValueError: the room holds another strategy, or an install runs.
        """
        held = self._room_strategies.get(room_id)
        replacing = self._replaces_discussion(room_id, strategy)
        if not replacing:
            self._check_claim(room_id, strategy)
            if held is not None:
                raise ValueError(f"Room {room_id} holds this strategy already")
        elif held is not None and held.installing:
            raise ValueError(f"Room {room_id} holds a discussion another install is replacing")
        self._room_strategies[room_id] = _Installed(strategy, organization_id, installing=True)
        return replacing, held if replacing else None

    async def _recorded(
        self, room_id: str, strategy: Orchestration, organization_id: str | None, before: _Snapshot
    ) -> tuple[_Installed, set[str]]:
        """What an install of *strategy* added since *before*, the agents it
        attached stored with the room's discussion when it is one (RFC
        §19.7); with the channels the room binds now."""
        after = await self._snapshot(room_id)
        added = _added(strategy, organization_id, before, after)
        await self._hold_attached(room_id, added.attached)
        return added, after.bindings

    async def _undo_install(
        self,
        room_id: str,
        strategy: Orchestration,
        organization_id: str | None,
        before: _Snapshot,
        replacing: bool,
        held: _Installed | None,
    ) -> None:
        """Take back what a failed install added; the room stays claimed for
        what could not be, so ``uninstall_strategy`` can finish it. A failed
        replacement records what the discussion runs instead."""
        after = await self._snapshot(room_id)
        added = _added(strategy, organization_id, before, after)
        if replacing:
            self._settle_replacement(room_id, strategy, added, held, after.bindings)
            return
        try:
            await self._take_back(room_id, added)
        except Exception:
            self._room_strategies[room_id] = added
            logger.exception("A failed install in room %s was not taken back whole", room_id)
            return
        del self._room_strategies[room_id]

    def _settle_replacement(
        self,
        room_id: str,
        strategy: Orchestration,
        added: _Installed,
        held: _Installed | None,
        bound: set[str],
    ) -> None:
        """Record what a failed replacement leaves the room holding: *strategy*
        once the discussion runs it (installing it again finishes the
        install), else the record *held* before while the discussion still
        runs its strategy (one this process left has none)."""
        if self._runs_discussion(room_id, strategy):
            self._room_strategies[room_id] = added if held is None else _joined(held, added, bound)
        elif held is not None and self._runs_discussion(room_id, held.strategy):
            self._room_strategies[room_id] = held
        else:
            del self._room_strategies[room_id]

    async def _attach_agents(self, room_id: str, strategy: Orchestration) -> None:
        bound = {b.channel_id for b in await self._store.list_bindings(room_id)}
        agents = strategy.agents()
        register_strategy_agents(self, agents)  # ty: ignore[invalid-argument-type]
        for channel_id in map(id_of, agents):
            if channel_id not in bound:
                await self.attach_channel(
                    room_id, channel_id, category=ChannelCategory.INTELLIGENCE
                )

    async def _snapshot(self, room_id: str) -> _Snapshot:
        room = await self._store.get_room(room_id)
        registries = self._registries()
        return _Snapshot(
            hooks=self._room_hooks(room_id),
            tools={cid: reg.room_tool_names(room_id) for cid, reg in registries.items()},
            runners=self._turn_runners.channels(room_id),
            bindings={b.channel_id for b in await self._store.list_bindings(room_id)},
            keys=set(room.metadata) if room is not None else set(),
        )

    def _room_hooks(self, room_id: str) -> set[tuple[bool, str]]:
        """The room hooks *room_id* holds on either side of the turn backend:
        the kit's, and its turn executor's (RFC §10.6.6)."""
        owner = {(False, name) for name in self._hook_engine.room_hook_names(room_id)}
        return owner | {(True, name) for name in self._turns.room_hook_names(room_id)}

    def _registries(self) -> dict[str, ChannelRegistry]:
        """Where an install sets tools up: the owner's store for an
        intelligence channel (RFC §10.6.2), a realtime channel's own registry
        for the voice installs, which never cross the turn backend."""
        own = {
            cid: registry
            for cid, channel in self._channels.items()
            if channel.category is not ChannelCategory.INTELLIGENCE
            and isinstance(registry := getattr(channel, "_registry", None), ChannelRegistry)
        }
        return {**own, **self._owner_tools.registries()}

    async def _take_back(self, room_id: str, installed: _Installed) -> None:
        """Remove what a strategy's install added for *room_id*: everything it
        can, then a failure if anything stays (what was taken back is dropped
        from *installed*, so a retry finishes the rest)."""
        for on_executor, name in installed.hooks:
            if on_executor:
                self._turns.disable_room_hook(room_id, name)
            else:
                self._hook_engine.remove_room_hook(room_id, name)
        installed.hooks = []
        registries = self._registries()
        for cid in {*installed.tools, *installed.runners}:
            registry = registries.get(cid)
            if registry is not None:
                registry.release(room_id, installed.tools.get(cid, ()))
            if cid in installed.runners:
                self._turn_runners.release(room_id, cid)
        installed.tools, installed.runners = {}, []
        stayed = await self._detach_agents(room_id, installed.attached)
        installed.attached = stayed
        if installed.keys:
            await self._store.patch_room_metadata(room_id, dict.fromkeys(installed.keys))
            installed.keys = []
        if stayed:
            raise RuntimeError(f"Room {room_id} still binds {stayed} its strategy attached")

    async def _detach_agents(self, room_id: str, channel_ids: list[str]) -> list[str]:
        """Detach each of *channel_ids* from *room_id*, whatever one of them
        raises; the ones still bound."""
        stayed: list[str] = []
        for cid in channel_ids:
            try:
                await self.detach_channel(room_id, cid)
            except Exception:
                logger.exception("Agent %s was not detached from room %s", cid, room_id)
                stayed.append(cid)
        return stayed


def _joined(held: _Installed, added: _Installed, bound: set[str]) -> _Installed:
    """The record of *added*'s strategy, which took the place of *held*'s:
    what both installs set up in this process, but agents no longer *bound*."""
    tools = dict(held.tools)
    for cid, names in added.tools.items():
        tools[cid] = tools.get(cid, frozenset()) | names
    return _Installed(
        added.strategy,
        added.organization_id,
        hooks=sorted({*held.hooks, *added.hooks}),
        tools=tools,
        runners=sorted({*held.runners, *added.runners}),
        attached=sorted({*held.attached, *added.attached} & bound),
        keys=sorted({*held.keys, *added.keys}),
    )


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
