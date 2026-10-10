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
from roomkit.core.lanes import DeliveryCascade
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import ChannelCategory, ChannelType, HookTrigger
from roomkit.orchestration.strategies.discussion._config import DiscussionConfig
from roomkit.orchestration.strategies.discussion._room import DiscussionRoom

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.room import Room
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
    _installing_discussions: set[str]

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _get_router: Any  # RoomKit._get_router
    _enqueue_exec: Any  # LaneExecutionMixin
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

    async def _install_discussion(self, room_id: str, discussion: Any) -> None:
        """Give *room_id* to *discussion*: its agents attached, its hooks
        installed, its stored queue read, its turns started (rule 1).

        Raises:
            ValueError: the room holds a discussion already, or something the
                discussion cannot share it with (rule 1).
        """
        held = self._discussions.get(room_id)
        installing = room_id in self._installing_discussions
        if (held is not None and held.driver is not None) or installing:
            raise ValueError(f"Room {room_id} holds a discussion already")
        if held is not None:
            # Followed here until now: installed, this process may give turns.
            self._leave_discussion(room_id, held)
        # Claimed before the first await: two installs at once are one too many.
        self._installing_discussions.add(room_id)
        try:
            await self._prepare_discussion(room_id, discussion)
            await discussion.load()
        finally:
            self._installing_discussions.discard(room_id)
        self._discussions[room_id] = discussion
        for hook in discussion.hooks():
            self._hook_engine.add_room_hook(room_id, hook)
        discussion.driver.start()

    async def _prepare_discussion(self, room_id: str, discussion: Any) -> None:
        """Refuse what the room cannot share with *discussion*, then bind its
        agents (rule 1)."""
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
        await discussion.stop()
        await discussion.reset()

    def _follow_discussion(self, room: Room) -> None:
        """Join, or leave, the discussion *room* holds as stored (rule 16): a
        room another process gave a discussion is followed here, one whose
        discussion is gone is left. Run for every context the kit builds."""
        held = self._discussions.get(room.id)
        config = DiscussionConfig.stored(room.metadata)
        if held is None and config is not None and room.id not in self._installing_discussions:
            follower = DiscussionRoom.following(self, room.id, config)
            follower.organization_id = room.organization_id
            self._discussions[room.id] = follower
            for hook in follower.hooks():
                self._hook_engine.add_room_hook(room.id, hook)
        elif held is not None and held.driver is None and config is None:
            self._leave_discussion(room.id, held)

    async def _forget_discussion(self, room_id: str, discussion: Any) -> None:
        """Another process uninstalled the discussion this one gave turns for:
        leave it here, writing nothing back."""
        if self._discussions.get(room_id) is not discussion:
            return
        self._leave_discussion(room_id, discussion)
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


def _in_scope(discussion: Any, organization_id: str | None) -> bool:
    return organization_id is None or discussion.organization_id == organization_id


def discussion_binding_refusal(
    discussion: Any, channel_id: str, category: ChannelCategory, channel: Channel | None
) -> str | None:
    """Why a room holding *discussion* cannot bind *channel_id*, if it cannot."""
    if channel is not None and channel.channel_type in _SPOKEN:
        return f"{channel_id} is a voice or realtime channel"
    if category == ChannelCategory.INTELLIGENCE and channel_id not in discussion.agents:
        return f"{channel_id} is an intelligence channel that is not one of its agents"
    return None
