"""DiscussionMixin — the kit side of a discussion (RFC §19.7.5).

A discussion takes a room's turns: the strategy (``roomkit.orchestration``)
keeps the speak queue and gives the turns; the kit installs it, plans each
turn as a rerun for one agent, announces the queue's changes and gives the
host ``speak_queue``, ``listen_only`` and ``talk_again``. The kit knows the
installed discussion only through the methods it calls on it, so the core
imports nothing of the strategy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from roomkit.channels._discussion_turn import DISCUSSION_TURN
from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import ChannelCategory, ChannelType, HookTrigger

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.orchestration.strategies.discussion import SpeakQueue, SpeakQueueEvent
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

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # RoomKit._get_router
    _enqueue_exec: Any  # LaneExecutionMixin
    attach_channel: Any  # ChannelOpsMixin
    register_channel: Any  # ChannelOpsMixin

    # -- The host's side --

    def speak_queue(self, room_id: str) -> SpeakQueue | None:
        """The speak queue of the discussion *room_id* holds in this process,
        or None when it holds none (RFC §19.7.5 rule 17)."""
        discussion = self._discussions.get(room_id)
        return discussion.view() if discussion is not None else None

    async def listen_only(self, room_id: str, channel_ids: Sequence[str]) -> None:
        """Have agents of a discussion only listen (RFC §19.7.5 rule 12).

        An agent that only listens keeps its place in the queue and takes no
        turn an agent's message asks for; a person's message that names it
        gives it one turn. Setting the state on the agent whose turn runs cuts
        that turn, as a ``Cancel`` does; what the turn committed stays.

        Raises:
            ValueError: the room holds no discussion in this process.
        """
        await self._discussion(room_id).listen_only(list(channel_ids))

    async def talk_again(self, room_id: str, channel_ids: Sequence[str]) -> None:
        """End the listening state of agents of a discussion (RFC §19.7.5 rule 12).

        Raises:
            ValueError: the room holds no discussion in this process.
        """
        await self._discussion(room_id).talk_again(list(channel_ids))

    def _holds_discussion(self, room_id: str | None) -> bool:
        """Whether *room_id* (any room, for None) holds a discussion."""
        return bool(self._discussions) if room_id is None else room_id in self._discussions

    def _claim_room_strategy(self, room_id: str) -> None:
        """Refuse another strategy in a room a discussion holds (RFC §19.7.5
        rule 1): one rule decides who speaks, never two.

        Raises:
            ValueError: the room holds a discussion.
        """
        if room_id in self._discussions:
            raise ValueError(f"Room {room_id} holds a discussion: no other strategy can join it")

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

    def _discussion(self, room_id: str) -> Any:
        discussion = self._discussions.get(room_id)
        if discussion is None:
            raise ValueError(f"Room {room_id} holds no discussion")
        return discussion

    # -- Install --

    async def _install_discussion(self, room_id: str, discussion: Any) -> None:
        """Give *room_id* to *discussion*: its agents attached, its hooks
        installed, its stored queue read, its turns started (rule 1).

        Raises:
            ValueError: the room holds a discussion already, or something the
                discussion cannot share it with (rule 1).
        """
        if room_id in self._discussions:
            raise ValueError(f"Room {room_id} holds a discussion already")
        for agent in discussion.agents.values():
            if agent.channel_id not in self._channels:
                self.register_channel(agent)
        context = await self._build_context(room_id)
        refusal = self._discussion_refusal(room_id, discussion, context)
        if refusal is not None:
            raise ValueError(f"Room {room_id} cannot hold a discussion: {refusal}")
        for agent_id in discussion.agents:
            if context.get_binding(agent_id) is None:
                await self.attach_channel(room_id, agent_id, category=ChannelCategory.INTELLIGENCE)
        await discussion.load()
        self._discussions[room_id] = discussion
        for hook in discussion.hooks():
            self._hook_engine.add_room_hook(room_id, hook)
        discussion.driver.start()

    async def _uninstall_discussion(self, room_id: str) -> None:
        """Give *room_id* back to its policy: the turns stopped, the queue
        dropped, the hooks removed (rule 1)."""
        discussion = self._discussions.get(room_id)
        if discussion is None:
            return
        await discussion.stop()
        await discussion.drop()
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
        self, room_id: str, discussion: Any, context: RoomContext
    ) -> str | None:
        """What the room holds that a discussion cannot share it with (rule 1)."""
        for binding in context.bindings:
            channel = self._channels.get(binding.channel_id)
            refusal = discussion_binding_refusal(
                discussion, binding.channel_id, binding.category, channel
            )
            if refusal is not None:
                return refusal
        for agent in discussion.agents.values():
            if getattr(agent, "_thinker", None) is not None:
                return f"{agent.channel_id} thinks while it listens"
            if agent._registry.turn_runner(room_id) is not None:
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
        cascade = DeliveryCascade(room_id, reentry_budget=10 * max_depth)
        cascade.retain()
        self._enqueue_exec(
            room_id, plan, cascade, index=None, after_index=context.room.latest_index
        )
        return cascade

    async def _fire_speak_queue(self, event: SpeakQueueEvent) -> None:
        """Run the ``ON_SPEAK_QUEUE`` hooks for a change of a speak queue."""
        trigger = HookTrigger.ON_SPEAK_QUEUE
        if not self._hook_engine.has_hooks(trigger):
            return
        context = await self._hook_context(event.room_id, trigger)
        if context is None:
            return
        try:
            await self._hook_engine.run_async_hooks(
                event.room_id, trigger, event, context, skip_event_filter=True
            )
        except Exception:
            logger.exception("ON_SPEAK_QUEUE hooks failed in room %s", event.room_id)


def discussion_binding_refusal(
    discussion: Any, channel_id: str, category: ChannelCategory, channel: Channel | None
) -> str | None:
    """Why a room holding *discussion* cannot bind *channel_id*, if it cannot."""
    if channel is not None and channel.channel_type in _SPOKEN:
        return f"{channel_id} is a voice or realtime channel"
    if category == ChannelCategory.INTELLIGENCE and channel_id not in discussion.agents:
        return f"{channel_id} is an intelligence channel that is not one of its agents"
    return None
