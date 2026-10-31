"""DiscussionMixin — the kit side of a discussion (RFC §19.7.5).

A discussion takes a room's turns: the strategy (``roomkit.orchestration``)
keeps the speak queue and gives the turns; the kit installs it, plans each
turn as a rerun for one agent, announces the queue's changes and gives the
host ``speak_queue``, ``listen_only`` and ``talk_again``. A room whose
metadata holds a discussion another process installed is followed here too
(rule 16): every context the kit builds for it joins it, so this process asks
no agent at broadcast and queues by the stored configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from roomkit.channels._discussion_turn import DISCUSSION_TURN
from roomkit.channels.declared import register_strategy_agents
from roomkit.core.lanes import DeliveryCascade, anchor_index
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import ChannelCategory, ChannelType, HookTrigger
from roomkit.orchestration.strategies.discussion._config import DiscussionConfig
from roomkit.orchestration.strategies.discussion._room import DiscussionRoom
from roomkit.orchestration.strategies.discussion.strategy import Discussion

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.room import Room
    from roomkit.orchestration.base import Orchestration
    from roomkit.orchestration.strategies.discussion import (
        DispatchDecisionEvent,
        SpeakQueue,
        SpeakQueueEvent,
    )
    from roomkit.store.base import ConversationStore

logger = logging.getLogger("roomkit.framework")

# Voice and realtime channels, which a discussion of this version does not hold.
_SPOKEN = frozenset(
    {
        ChannelType.VOICE,
        ChannelType.REALTIME_VOICE,
        ChannelType.REALTIME_AUDIO_VIDEO,
        ChannelType.AUDIO_VIDEO,
    }
)


class DiscussionMixin(HelpersMixin):
    """Adds ``speak_queue()``, ``listen_only()`` and ``talk_again()`` to RoomKit."""

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _hook_engine: HookEngine
    _channels: dict[str, Channel]
    _discussions: dict[str, Any]
    _installing_discussions: set[str]
    _turn_runners: Any  # TurnRunners, the owner's (RFC §10.6.1)

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # RoomKit._get_router
    get_room: Any  # RoomLifecycleMixin
    _enqueue_exec: Any  # LaneExecutionMixin
    _detach_agents: Any  # StrategyMixin
    _drop_record: Any  # StrategyMixin
    attach_channel: Any  # ChannelOpsMixin
    register_channel: Any  # ChannelOpsMixin

    # -- The host's side --

    def speak_queue(
        self, room_id: str, *, organization_id: str | None = None
    ) -> SpeakQueue | None:
        """The speak queue of the discussion *room_id* holds in this process,
        or None when it holds none (RFC §19.7.5 rule 17). *organization_id*
        scopes the read to one tenant (RFC §17.2): a room of another reads as
        holding none."""
        discussion = self._discussions.get(room_id)
        if discussion is None or not _in_scope(discussion, organization_id):
            return None
        return discussion.view()

    async def listen_only(
        self, room_id: str, channel_ids: Sequence[str], *, organization_id: str | None = None
    ) -> None:
        """Have agents of a discussion only listen (RFC §19.7.5 rule 12).

        An agent that only listens keeps its place in the queue and takes no
        turn an agent's message asks for; a person's message that names it
        gives it one turn. Setting the state on the agent whose turn runs cuts
        that turn, as a ``Cancel`` does; what the turn committed stays.

        Raises:
            ValueError: the room holds no discussion in this process (or in
                *organization_id*'s scope).
        """
        await self._discussion(room_id, organization_id).listen_only(list(channel_ids))

    async def talk_again(
        self, room_id: str, channel_ids: Sequence[str], *, organization_id: str | None = None
    ) -> None:
        """End the listening state of agents of a discussion (RFC §19.7.5 rule 12).

        Raises:
            ValueError: the room holds no discussion in this process (or in
                *organization_id*'s scope).
        """
        await self._discussion(room_id, organization_id).talk_again(list(channel_ids))

    async def _queue_discussion_post(self, room_id: str, event: RoomEvent) -> bool:
        """Queue a task's result to post, *event*, in the discussion *room_id*
        holds (RFC §19.7.5 rule 19); whether it was queued."""
        discussion = self._discussions.get(room_id)
        return discussion is not None and await discussion.queue_post(event)

    def _holds_discussion(self, room_id: str | None) -> bool:
        """Whether *room_id* (any room, for None) holds a discussion."""
        return bool(self._discussions) if room_id is None else room_id in self._discussions

    def _refuse_discussion_binding(
        self, room_id: str, channel: Channel, category: ChannelCategory | None
    ) -> None:
        """Refuse a binding a room's discussion cannot share it with (rule 1).

        Raises:
            ValueError: the room holds a discussion that refuses the binding.
        """
        discussion = self._discussions.get(room_id)
        if discussion is None:
            return
        refusal = discussion_binding_refusal(
            discussion, channel.channel_id, category or channel.category, channel
        )
        if refusal is not None:
            raise ValueError(f"Room {room_id} holds a discussion: {refusal}")

    def _discussion(self, room_id: str, organization_id: str | None = None) -> Any:
        discussion = self._discussions.get(room_id)
        if discussion is None or not _in_scope(discussion, organization_id):
            raise ValueError(f"Room {room_id} holds no discussion")
        return discussion

    # -- Install --

    def _replaces_discussion(self, room_id: str, strategy: Orchestration) -> bool:
        """Whether *strategy* takes the place of the discussion *room_id*
        holds here, installed or followed: a Discussion does (RFC §19.7)."""
        return isinstance(strategy, Discussion) and room_id in self._discussions

    def _stored_with_room(self, kind: Any) -> bool:
        """Whether *kind* is a strategy stored with the room, which another
        process's install may replace: a Discussion (rule 16)."""
        return isinstance(kind, type) and issubclass(kind, Discussion)

    async def _discussion_is_current(
        self, room_id: str, kind: type, *, organization_id: str | None
    ) -> bool:
        """Whether the discussion of *kind* this process installed is the one
        the room stores now: the one this process gives turns by (rule 16).
        One another process uninstalled is forgotten here at once, so
        ``room_strategy()`` tells the host it is gone.

        Raises:
            RoomNotFoundError: the room does not exist in the scope.
        """
        seen = self._discussions.get(room_id)
        installs = seen.installs if seen is not None else 0
        room = await self.get_room(room_id, organization_id=organization_id)
        self._follow_discussion(room)
        held = self._discussions.get(room.id)
        if held is None:
            return False
        if (
            held is seen
            and _uninstalled(held, room)
            and await self._forget_discussion(room.id, held, installs=installs)
        ):
            return False
        return isinstance(held.strategy, kind) and held.drives

    def _runs_discussion(self, room_id: str, strategy: Orchestration) -> bool:
        """Whether the discussion *room_id* holds here runs *strategy*."""
        discussion = self._discussions.get(room_id)
        return discussion is not None and discussion.strategy is strategy

    async def _install_discussion(self, room_id: str, discussion: Any) -> None:
        """Give *room_id* to *discussion*, or put it in place of the discussion
        the room holds, here or stored by another process (rules 1 and 16).

        Raises:
            ValueError: a discussion is being installed in the room, or the room
                holds something the discussion cannot share it with (rule 1).
        """
        room = await self._store.get_room(room_id)
        if room_id in self._installing_discussions:
            raise ValueError(f"Room {room_id} holds a discussion already")
        if room is not None:
            self._follow_discussion(room)
        held = self._discussions.get(room_id)
        # Claimed before the next await: two installs at once are one too many.
        self._installing_discussions.add(room_id)
        try:
            if held is not None and held.driver is not None and _uninstalled(held, room):
                # Gone since this process installed it: left here, never
                # reused. The install takes the place of the discussion the
                # room stores now, another process's, else starts afresh.
                await self._leave_uninstalled(room_id, held)
                held = self._join_stored(room) if room is not None else None
            if held is None:
                await self._start_discussion(room_id, discussion)
            else:
                await self._replace_discussion(room_id, held, discussion)
        finally:
            self._installing_discussions.discard(room_id)

    async def _start_discussion(self, room_id: str, discussion: Any) -> None:
        """Give *room_id*, which holds no discussion, to *discussion*: its
        agents bound, its configuration stored, its hooks and turns started."""
        context = await self._prepare_discussion(room_id, discussion)
        await self._bind_agents(room_id, discussion, context)
        await discussion.load()
        self._discussions[room_id] = discussion
        for hook in discussion.hooks():
            self._hook_engine.add_room_hook(room_id, hook)
        discussion.driver.start()

    async def _replace_discussion(self, room_id: str, held: Any, incoming: Any) -> None:
        """Put *incoming*'s configuration in place of the one *held* routes
        by, keeping the queue (rule 1): the agents that leave and an install
        attached unbound, the new ones bound, this process giving turns."""
        # Before any await: a read begun until now tells nothing of this install.
        held.installs += 1
        await held.read()
        stored = held.shared.config
        leaving = [a for a in stored.attached if a not in incoming.agents] if stored else []
        context = await self._prepare_discussion(room_id, incoming, leaving=leaving)
        await self._detach_all(room_id, leaving, "which left its discussion")
        await held.take_over(incoming)
        await self._bind_agents(room_id, held, context)
        held.drive()

    async def _prepare_discussion(
        self, room_id: str, discussion: Any, *, leaving: Sequence[str] = ()
    ) -> RoomContext:
        """Refuse what the room cannot share with *discussion* (rule 1), the
        agents *leaving* it aside, its agents registered; the room's context."""
        # An agent named by an id the kit does not hold is refused by its id.
        agents = [agent or agent_id for agent_id, agent in discussion.agents.items()]
        register_strategy_agents(self, agents)  # ty: ignore[invalid-argument-type]
        context = await self._build_context(room_id)
        refusal = self._discussion_refusal(room_id, discussion, context, leaving=leaving)
        if refusal is not None:
            raise ValueError(f"Room {room_id} cannot hold a discussion: {refusal}")
        return context

    async def _bind_agents(self, room_id: str, discussion: Any, context: RoomContext) -> None:
        """Bind the agents of *discussion* the room's *context* does not bind."""
        for agent_id in discussion.agents:
            if context.get_binding(agent_id) is None:
                await self.attach_channel(room_id, agent_id, category=ChannelCategory.INTELLIGENCE)

    async def _hold_attached(self, room_id: str, attached: list[str]) -> None:
        """Store *attached*, the agents an install of the discussion *room_id*
        holds here attached, with it, for any process uninstalling it to
        detach them (RFC §19.7). Nothing for a room holding no such discussion."""
        discussion = self._discussions.get(room_id)
        if attached and discussion is not None and discussion.strategy is not None:
            await discussion.hold_attached(attached)

    async def _detach_all(self, room_id: str, channel_ids: list[str], whose: str) -> None:
        """Detach *channel_ids*, the agents a discussion lets go, from *room_id*.

        Raises:
            RuntimeError: one of them is still bound, named with *whose*.
        """
        stayed = await self._detach_agents(room_id, channel_ids)
        if stayed:
            raise RuntimeError(f"Room {room_id} still binds {stayed}, {whose}")

    def _stored_attached(self, room: Room) -> list[str]:
        """The agents installs of the discussion *room* stores attached to it."""
        config = DiscussionConfig.stored(room.metadata)
        return list(config.attached) if config is not None else []

    async def _uninstall_discussion(self, room_id: str) -> None:
        """Give *room_id* back to its policy (rule 1): released first, so what
        comes in meanwhile is the policy's, then the turns stopped and the
        discussion forgotten."""
        discussion = self._discussions.pop(room_id, None)
        if discussion is None:
            return
        discussion.closed = True
        for hook in discussion.hooks():
            self._hook_engine.remove_room_hook(room_id, hook.name)
        await discussion.forget()

    async def _uninstall_stored_discussion(self, room: Room) -> bool:
        """Uninstall the discussion *room* holds with no install of this
        process recorded, one another process stored included (rule 16), the
        agents its installs attached detached; whether it held one.

        Raises:
            RuntimeError: an agent its installs attached is still bound.
        """
        self._follow_discussion(room)
        if room.id not in self._discussions:
            return False
        await self._uninstall_discussion(room.id)
        await self._detach_all(room.id, self._stored_attached(room), "which its installs attached")
        return True

    def _follow_discussion(self, room: Room) -> None:
        """Join, follow or leave the discussion *room* holds as stored (rule
        16): a room another process gave a discussion is followed here, one
        whose discussion another process replaced is routed by the new
        configuration, one whose discussion is gone is left. Run for every
        context the kit builds."""
        held = self._discussions.get(room.id)
        config = DiscussionConfig.stored(room.metadata)
        if held is None and config is not None and room.id not in self._installing_discussions:
            self._join_stored(room)
        elif held is not None and config is not None:
            held.follow(config)
        elif held is not None and held.driver is None and config is None:
            self._leave_discussion(room.id, held)

    def _join_stored(self, room: Room) -> Any:
        """Follow the discussion *room* stores, another process's (rule 16):
        this process routes and queues by it; None when it stores none."""
        config = DiscussionConfig.stored(room.metadata)
        if config is None:
            return None
        follower = DiscussionRoom.following(self, room.id, config)
        follower.organization_id = room.organization_id
        self._discussions[room.id] = follower
        for hook in follower.hooks():
            self._hook_engine.add_room_hook(room.id, hook)
        return follower

    async def _forget_discussion(self, room_id: str, discussion: Any, *, installs: int) -> bool:
        """Another process uninstalled the discussion this one gave turns for,
        as a read begun once *discussion* had taken *installs* installs found:
        leave it here; whether it was left. A read begun before an install of
        this process took its place tells nothing of that install (rule 16)."""
        if discussion.installs != installs or self._discussions.get(room_id) is not discussion:
            return False
        await self._leave_uninstalled(room_id, discussion)
        return True

    async def _leave_uninstalled(self, room_id: str, discussion: Any) -> None:
        """Leave *discussion*, which another process uninstalled, the strategy
        record with it, writing nothing back."""
        if self._discussions.get(room_id) is not discussion:
            return
        # Its queue is gone: a turn cut on the way records nothing, and so
        # takes no room lock its caller may hold (rule 1).
        discussion.forgotten = True
        # The task results this process was to post: the queue is gone, they
        # are posted all the same (rule 19).
        discussion.post_held()
        self._leave_discussion(room_id, discussion)
        await self._drop_record(room_id, discussion.strategy)
        if discussion.driver is not None:
            await discussion.driver.stop()

    def _leave_discussion(self, room_id: str, discussion: Any) -> None:
        discussion.closed = True
        if self._discussions.get(room_id) is discussion:
            del self._discussions[room_id]
        for hook in discussion.hooks():
            self._hook_engine.remove_room_hook(room_id, hook.name)

    async def _stop_discussions(self) -> None:
        """Stop every discussion's turns; their queues stay stored (rule 16)."""
        for discussion in list(self._discussions.values()):
            try:
                await discussion.stop()
            except Exception:
                logger.exception("Discussion of room %s did not stop", discussion.room_id)

    def _discussion_refusal(
        self, room_id: str, discussion: Any, context: RoomContext, *, leaving: Sequence[str] = ()
    ) -> str | None:
        """What the room holds that a discussion cannot share it with (rule 1),
        but for the agents *leaving* the room with the discussion it replaces."""
        for binding in context.bindings:
            if binding.channel_id in leaving:
                continue
            channel = self._channels.get(binding.channel_id)
            refusal = discussion_binding_refusal(
                discussion, binding.channel_id, binding.category, channel
            )
            if refusal is not None:
                return refusal
        for agent in discussion.agents.values():
            if getattr(agent, "thinks_while_listening", False):
                return f"{agent.channel_id} thinks while it listens"
            # A strategy's runner already takes this agent's turns here.
            if self._turn_runners.get(room_id, agent.channel_id) is not None:
                return f"another strategy runs {agent.channel_id}'s turns"
        if self._hook_engine.has_router_hook(room_id):
            return "a router is installed"
        return None

    # -- Its turns --

    async def _plan_discussion_turn(
        self,
        room_id: str,
        answered: RoomEvent,
        agent_id: str,
        context: RoomContext,
        *,
        mark: dict[str, Any],
        max_depth: int,
    ) -> DeliveryCascade | None:
        """Put a discussion's turn in the room's lane: a rerun of *answered*
        for *agent_id* alone, carrying the turn's *mark*, bounded by the
        discussion's own depth limit (rules 9 and 11); its cascade, or None
        when the agent cannot be asked to answer it. Under the room lock, with
        *context* read under it."""
        source = context.get_binding(answered.source.channel_id)
        if source is None:
            return None
        trigger = answered.model_copy(
            update={"metadata": {**(answered.metadata or {}), DISCUSSION_TURN: mark}}
        )
        plan = self._get_router().plan(trigger, source, context)
        plan.targets = [t for t in plan.targets if t.channel_id == agent_id]
        if not plan.targets:
            return None
        plan.rerun = True
        plan.turn_for = agent_id
        plan.max_chain_depth = max_depth
        plan.response_visibility = answered.response_visibility
        cascade = self._cascades.new(room_id, reentry_budget=10 * max_depth)
        cascade.retain()
        self._enqueue_exec(
            room_id, plan, cascade, index=None, after_index=anchor_index(context.room.latest_index)
        )
        return cascade

    async def _fire_speak_queue(self, event: SpeakQueueEvent) -> None:
        """Start the ``ON_SPEAK_QUEUE`` hooks for a change of a speak queue."""
        trigger = HookTrigger.ON_SPEAK_QUEUE
        if not self._hook_engine.has_hooks(trigger):
            return
        context = await self._hook_context(event.room_id, trigger)
        if context is None:
            return
        try:
            self._hook_engine._start_async_hooks(
                event.room_id, trigger, event, context, skip_event_filter=True
            )
        except Exception:
            logger.exception("ON_SPEAK_QUEUE hooks failed in room %s", event.room_id)

    async def _fire_dispatch_decision(self, event: DispatchDecisionEvent) -> None:
        """Start the ``ON_DISPATCH_DECISION`` hooks for a dispatch policy's decision."""
        trigger = HookTrigger.ON_DISPATCH_DECISION
        if not self._hook_engine.has_hooks(trigger):
            return
        context = await self._hook_context(event.room_id, trigger)
        if context is None:
            return
        try:
            self._hook_engine._start_async_hooks(
                event.room_id, trigger, event, context, skip_event_filter=True
            )
        except Exception:
            logger.exception("ON_DISPATCH_DECISION hooks failed in room %s", event.room_id)


def _in_scope(discussion: Any, organization_id: str | None) -> bool:
    return organization_id is None or discussion.organization_id == organization_id


def _uninstalled(discussion: Any, room: Room | None) -> bool:
    """Whether *discussion* is gone: *room*, just read, stores none, or this
    process read it uninstalled before, one installed again since included
    (rule 16)."""
    return discussion.forgotten or room is None or DiscussionConfig.stored(room.metadata) is None


def discussion_binding_refusal(
    discussion: Any, channel_id: str, category: ChannelCategory, channel: Channel | None
) -> str | None:
    """Why a room holding *discussion* cannot bind *channel_id*, if it cannot."""
    if channel is not None and channel.channel_type in _SPOKEN:
        return f"{channel_id} is a voice or realtime channel"
    if category == ChannelCategory.INTELLIGENCE and channel_id not in discussion.agents:
        return f"{channel_id} is an intelligence channel that is not one of its agents"
    return None
