"""ChannelStateMixin — what a room keeps for a channel, its per-room verbs,
the release of a declared channel's build, and which executors serve the
room's agents (RFC §10.6.3, §10.6.7, §6.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.channels.declared import DeclaredChannel
from roomkit.core.exceptions import ChannelNotFoundError, ChannelNotRegisteredError
from roomkit.models.enums import ChannelCategory
from roomkit.turns._rooms import room_state_key, stored_room_state
from roomkit.turns.presence import AgentPresence, ExecutorInfo
from roomkit.turns.wire import ChannelCall, ChannelReleased, RoomDetached

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.channel import ChannelBinding
    from roomkit.store.base import ConversationStore
    from roomkit.turns.executor import TurnExecutor
    from roomkit.turns.owner import TurnClient


class ChannelStateMixin:
    """``channel_call``, and the room's record of what it keeps for a channel:
    written from a verb's answer, forgotten when the channel is detached."""

    _store: ConversationStore
    _lock_manager: RoomLockManager
    _channels: dict[str, Channel]
    _turns: TurnClient
    turn_executor: TurnExecutor
    get_room: Any  # see RoomLifecycleMixin

    async def channel_call(
        self,
        channel_id: str,
        room_id: str,
        verb: str,
        *,
        organization_id: str | None = None,
        **arguments: Any,
    ) -> Any:
        """Run a per-room verb of a channel attached to a room, on the
        executor holding the room's state for it (RFC §10.6.3): its result.

        An ACP agent's are its session's: ``set_config_option`` (``config_id``,
        ``value``), ``close_session``, ``session_config`` and
        ``config_options``. What the room keeps for the channel, when the verb
        changes it (the values a host set on the session), is stored with the
        room (RFC §6.4), so the room's next executor finds it::

            await kit.channel_call("claude", room.id, "set_config_option",
                                   config_id="model", value="opus")

        *organization_id* scopes the call to one tenant (RFC §17.2); a room
        belonging to another organization is reported as not found.

        Raises:
            RoomNotFoundError: no room *room_id*.
            ChannelNotRegisteredError: no channel *channel_id*.
            ChannelNotFoundError: the channel is not attached to the room.
            ValueError: a verb the channel does not have.
            TurnLinkError: ``no_executor``, no executor serves the channel.
        """
        room = await self.get_room(room_id, organization_id=organization_id)
        if channel_id not in self._channels:
            raise ChannelNotRegisteredError(f"Channel {channel_id} not registered")
        if await self._store.get_binding(room_id, channel_id) is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not attached to room {room_id}")
        call = ChannelCall(
            channel_id=channel_id,
            room_id=room_id,
            verb=verb,
            arguments=arguments,
            room_state=stored_room_state(room.metadata, channel_id),
        )
        answer = await self._turns.control(call)
        if answer is None:
            return None
        if answer.room_state is not None:
            await self._keep_answered_state(room_id, channel_id, answer.room_state)
        if answer.error is not None:
            raise self._turns.errors.rebuild(answer.error)
        return answer.result

    async def release_channel(self, channel_id: str) -> None:
        """Have every executor holding a build of *channel_id* let it go
        (RFC §10.6.3): each closes it once the turns running on it end, and
        builds it anew for the next turn, in this process or another. For a
        channel declared to the kit whose host changed what it is built from
        (its model, its provider); a channel the kit holds live is its
        registry's, which its host replaces by registering the successor.

        Raises:
            ChannelNotRegisteredError: the kit holds no channel *channel_id*.
        """
        if channel_id not in self._channels:
            raise ChannelNotRegisteredError(f"No channel {channel_id!r} is registered")
        await self._turns.control(ChannelReleased(channel_id=channel_id))

    async def executors(self) -> list[ExecutorInfo]:
        """The executors the kit's turn backend carries turns to, the kit's
        own aside: what each declares, and whether it is online (RFC
        §10.6.7, §10.6.8). A closed executor is not listed."""
        return await self._turns.backend.executors()

    async def agent_presence(
        self, room_id: str, *, organization_id: str | None = None
    ) -> list[AgentPresence]:
        """Each of *room_id*'s agents (its intelligence channels), sorted by
        channel id: online while an executor online covers it in the room,
        declaring it and defining the room's enabled hooks, with the ids of
        those executors (RFC §10.6.7); an agent the kit holds live, while
        the kit's own executor takes its turns there.

        *organization_id* scopes the read to one tenant (RFC §17.2).

        Raises:
            RoomNotFoundError: no room *room_id*, or another organization's.
        """
        await self.get_room(room_id, organization_id=organization_id)
        hooks = self._turns.room_hook_names(room_id)
        infos = await self.executors()
        agents = [
            binding.channel_id
            for binding in await self._store.list_bindings(room_id)
            if binding.category == ChannelCategory.INTELLIGENCE
        ]
        return [self._presence_of(agent, hooks, infos) for agent in sorted(agents)]

    def _presence_of(
        self, channel_id: str, hooks: list[str], infos: list[ExecutorInfo]
    ) -> AgentPresence:
        """Whether *channel_id* can answer in a room with *hooks*: on the
        kit's own executor for a channel it holds live, else on the executors
        of *infos* covering it."""
        channel = self._channels.get(channel_id)
        if channel is not None and not isinstance(channel, DeclaredChannel):
            own = self.turn_executor
            covering = [own.executor_id] if own.takes(channel_id, hooks) else []
        else:
            covering = sorted(i.executor_id for i in infos if i.covers(channel_id, hooks))
        return AgentPresence(channel_id=channel_id, online=bool(covering), executors=covering)

    async def _keep_answered_state(
        self, room_id: str, channel_id: str, state: dict[str, Any]
    ) -> None:
        """Store a verb's *state* while the channel is still attached to the
        room, read under the room lock ``detach_channel`` holds: a detach
        forgets what the room kept for the channel (RFC §10.6.2), and a call
        that ran across it does not write it back."""
        async with self._lock_manager.locked(room_id):
            if await self._store.get_binding(room_id, channel_id) is None:
                return
            await self._store_room_state(room_id, channel_id, state)

    async def _store_room_state(
        self, room_id: str, channel_id: str, state: dict[str, Any]
    ) -> None:
        """Keep *state* with the room as what it holds for *channel_id*
        (empty: nothing)."""
        key = room_state_key(channel_id)
        if state:
            await self._store.patch_room_metadata(room_id, {key: state})
        else:
            await self._store.patch_room_metadata(room_id, {}, unset=[key])

    async def _on_room_detached(self, room_id: str, binding: ChannelBinding) -> None:
        """Let the channel of *binding* let go of the room: an intelligence
        channel on the executors that may run its turns, the room forgetting what it
        kept for the channel even when the executors could not be told (RFC
        §10.6.3, §6.4)."""
        channel_id = binding.channel_id
        if binding.category is not ChannelCategory.INTELLIGENCE:
            channel = self._channels.get(channel_id)
            if channel is not None:
                await channel.on_room_detached(room_id)
            return
        try:
            await self._turns.control(RoomDetached(channel_id=channel_id, room_id=room_id))
        finally:
            room = await self.get_room(room_id)
            if room_state_key(channel_id) in room.metadata:
                await self._store_room_state(room_id, channel_id, {})
