"""ChannelOpsMixin — channel registration, binding, and state management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.core._helpers import HelpersMixin
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
)

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.event_router import EventRouter
    from roomkit.core.locks import RoomLockManager
    from roomkit.store.base import ConversationStore


class ChannelOpsMixin(HelpersMixin):
    """Channel registration, attachment, and binding operations."""

    _store: ConversationStore
    _channels: dict[str, Channel]
    _lock_manager: RoomLockManager
    _event_router: EventRouter | None

    def register_channel(self, channel: Channel) -> None:
        """Register a channel implementation by its ID."""
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.channels.voice import VoiceChannel

        self._channels[channel.channel_id] = channel
        self._event_router = None  # Reset router cache

        # Wire protocol trace framework handler on all channels
        channel._trace_framework_handler = self._on_channel_trace

        # Propagate telemetry to all channels
        channel._telemetry = self._telemetry  # type: ignore[attr-defined]

        # Propagate realtime backend to AI channels for tool call events
        from roomkit.channels.ai import AIChannel

        if isinstance(channel, AIChannel):
            channel._realtime = self._realtime  # type: ignore[attr-defined]

        # Propagate telemetry to channel's sub-providers (AI, STT, TTS, etc.)
        if hasattr(channel, "_propagate_telemetry"):
            channel._propagate_telemetry()

        # Set framework reference on voice channels for inbound routing
        if isinstance(channel, (VoiceChannel, RealtimeVoiceChannel)):
            channel.set_framework(self)  # type: ignore[arg-type]

        # Auto-greet: register global ON_SESSION_STARTED hook for agents
        from roomkit.channels.agent import Agent as AgentChannel

        if isinstance(channel, AgentChannel) and channel.auto_greet and channel.greeting:
            self._register_auto_greet_hook(channel)

    async def attach_channel(
        self,
        room_id: str,
        channel_id: str,
        channel_type: ChannelType | None = None,
        category: ChannelCategory = ChannelCategory.TRANSPORT,
        access: Access = Access.READ_WRITE,
        visibility: str = "all",
        **kwargs: Any,
    ) -> ChannelBinding:
        """Attach a registered channel to a room."""
        from roomkit.core.framework import ChannelNotRegisteredError

        async with self._lock_manager.locked(room_id):
            await self.get_room(room_id)  # type: ignore[attr-defined]
            channel = self._channels.get(channel_id)
            if channel is None:
                raise ChannelNotRegisteredError(f"Channel {channel_id} not registered")
            ct = channel_type or channel.channel_type
            binding = ChannelBinding(
                channel_id=channel_id,
                room_id=room_id,
                channel_type=ct,
                category=category,
                access=access,
                visibility=visibility,
                capabilities=channel.capabilities(),
                **kwargs,
            )
            result = await self._store.add_binding(binding)
            await self._emit_system_event(
                room_id,
                EventType.CHANNEL_ATTACHED,
                code="channel_attached",
                message=f"Channel {channel_id} attached ({access}, {visibility})",
                data={
                    "channel_id": channel_id,
                    "access": str(access),
                    "visibility": visibility,
                },
            )
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_CHANNEL_ATTACHED,
                EventType.CHANNEL_ATTACHED,
                code="channel_attached",
                message=f"Channel {channel_id} attached",
                data={
                    "channel_id": channel_id,
                    "access": str(access),
                    "visibility": visibility,
                },
            )
            await self._emit_framework_event(
                "room_channel_attached",
                room_id=room_id,
                channel_id=channel_id,
                data={"access": str(access), "visibility": visibility},
            )
            # Flush traces that arrived before the room existed
            await self._flush_pending_traces(room_id)
            return result

    async def detach_channel(self, room_id: str, channel_id: str) -> bool:
        """Detach a channel from a room."""
        async with self._lock_manager.locked(room_id):
            removed = await self._store.remove_binding(room_id, channel_id)
            if removed:
                await self._emit_system_event(
                    room_id,
                    EventType.CHANNEL_DETACHED,
                    code="channel_detached",
                    message=f"Channel {channel_id} detached",
                    data={"channel_id": channel_id},
                )
                await self._fire_lifecycle_hook(
                    room_id,
                    HookTrigger.ON_CHANNEL_DETACHED,
                    EventType.CHANNEL_DETACHED,
                    code="channel_detached",
                    message=f"Channel {channel_id} detached",
                    data={"channel_id": channel_id},
                )
                await self._emit_framework_event(
                    "room_channel_detached",
                    room_id=room_id,
                    channel_id=channel_id,
                )
            return removed

    async def mute(self, room_id: str, channel_id: str) -> ChannelBinding:
        """Mute a channel in a room."""
        async with self._lock_manager.locked(room_id):
            binding = await self._get_binding(room_id, channel_id)
            updated = binding.model_copy(update={"muted": True})
            result = await self._store.update_binding(updated)
            self._notify_binding_updated(room_id, channel_id, result)
            await self._emit_system_event(
                room_id,
                EventType.CHANNEL_MUTED,
                code="channel_muted",
                message=f"Channel {channel_id} muted",
                data={"channel_id": channel_id},
            )
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_CHANNEL_MUTED,
                EventType.CHANNEL_MUTED,
                code="channel_muted",
                message=f"Channel {channel_id} muted",
                data={"channel_id": channel_id},
            )
            return result

    async def unmute(self, room_id: str, channel_id: str) -> ChannelBinding:
        """Unmute a channel in a room."""
        async with self._lock_manager.locked(room_id):
            binding = await self._get_binding(room_id, channel_id)
            updated = binding.model_copy(update={"muted": False})
            result = await self._store.update_binding(updated)
            self._notify_binding_updated(room_id, channel_id, result)
            await self._emit_system_event(
                room_id,
                EventType.CHANNEL_UNMUTED,
                code="channel_unmuted",
                message=f"Channel {channel_id} unmuted",
                data={"channel_id": channel_id},
            )
            await self._fire_lifecycle_hook(
                room_id,
                HookTrigger.ON_CHANNEL_UNMUTED,
                EventType.CHANNEL_UNMUTED,
                code="channel_unmuted",
                message=f"Channel {channel_id} unmuted",
                data={"channel_id": channel_id},
            )
            return result

    async def set_visibility(
        self, room_id: str, channel_id: str, visibility: str
    ) -> ChannelBinding:
        """Set visibility for a channel in a room."""
        async with self._lock_manager.locked(room_id):
            binding = await self._get_binding(room_id, channel_id)
            old_visibility = binding.visibility
            updated = binding.model_copy(update={"visibility": visibility})
            result = await self._store.update_binding(updated)
            await self._emit_system_event(
                room_id,
                EventType.CHANNEL_UPDATED,
                code="channel_visibility_changed",
                message=f"Channel {channel_id} visibility: {old_visibility} -> {visibility}",
                data={
                    "channel_id": channel_id,
                    "old_visibility": old_visibility,
                    "visibility": visibility,
                },
            )
            return result

    async def set_access(self, room_id: str, channel_id: str, access: Access) -> ChannelBinding:
        """Set access level for a channel in a room."""
        async with self._lock_manager.locked(room_id):
            binding = await self._get_binding(room_id, channel_id)
            old_access = binding.access
            updated = binding.model_copy(update={"access": access})
            result = await self._store.update_binding(updated)
            self._notify_binding_updated(room_id, channel_id, result)
            await self._emit_system_event(
                room_id,
                EventType.CHANNEL_UPDATED,
                code="channel_access_changed",
                message=f"Channel {channel_id} access: {old_access} -> {access}",
                data={
                    "channel_id": channel_id,
                    "old_access": str(old_access),
                    "access": str(access),
                },
            )
            return result

    async def update_binding_metadata(
        self, room_id: str, channel_id: str, metadata: dict[str, Any]
    ) -> ChannelBinding:
        """Update metadata on a channel binding."""
        async with self._lock_manager.locked(room_id):
            binding = await self._get_binding(room_id, channel_id)
            updated = binding.model_copy(update={"metadata": {**binding.metadata, **metadata}})
            result = await self._store.update_binding(updated)
            await self._emit_system_event(
                room_id,
                EventType.CHANNEL_UPDATED,
                code="channel_metadata_updated",
                message=f"Channel {channel_id} metadata updated",
                data={"channel_id": channel_id, "keys": list(metadata.keys())},
            )
            return result

    def get_channel(self, channel_id: str) -> Channel | None:
        """Get a registered channel by ID."""
        return self._channels.get(channel_id)

    def list_channels(self) -> list[Channel]:
        """List all registered channels."""
        return list(self._channels.values())

    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding:
        """Get a channel binding. Raises ChannelNotFoundError if missing."""
        return await self._get_binding(room_id, channel_id)

    async def list_bindings(self, room_id: str) -> list[ChannelBinding]:
        """List all channel bindings for a room."""
        return await self._store.list_bindings(room_id)

    def _notify_binding_updated(
        self, room_id: str, channel_id: str, binding: ChannelBinding
    ) -> None:
        """Push updated binding state to the channel for audio gating."""
        channel = self._channels.get(channel_id)
        if channel is not None:
            channel.update_binding(room_id, binding)

    async def _get_binding(self, room_id: str, channel_id: str) -> ChannelBinding:
        from roomkit.core.framework import ChannelNotFoundError

        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not in room {room_id}")
        return binding

    def _register_auto_greet_hook(self, agent: Channel) -> None:
        """Register a global ON_SESSION_STARTED hook for agent auto-greeting."""
        from roomkit.channels.agent import Agent as AgentChannel
        from roomkit.core.hooks import HookRegistration

        assert isinstance(agent, AgentChannel)  # noqa: S101
        agent_id = agent.channel_id
        hook_name = f"_agent_auto_greet:{agent_id}"
        kit_ref = self
        # Track rooms already greeted for text channels (no session to dedup on)
        greeted_rooms: set[str] = set()

        async def _auto_greet_handler(event: Any, ctx: Any) -> None:
            room_id = event.room_id

            # Voice path: dedup via session metadata
            if event.session is not None:
                session = event.session
                dedup_key = f"_auto_greeted:{agent_id}"
                if session.metadata.get(dedup_key):
                    return
                session.metadata[dedup_key] = True

                # Check agent is attached to this room
                bindings = await kit_ref._store.list_bindings(room_id)
                if not any(b.channel_id == agent_id for b in bindings):
                    session.metadata.pop(dedup_key, None)
                    return
            else:
                # Text path: dedup via in-memory set keyed by room+agent
                dedup_key = f"{room_id}:{agent_id}"
                if dedup_key in greeted_rooms:
                    return
                greeted_rooms.add(dedup_key)

                # Check agent is attached to this room
                bindings = await kit_ref._store.list_bindings(room_id)
                if not any(b.channel_id == agent_id for b in bindings):
                    greeted_rooms.discard(dedup_key)
                    return

            await kit_ref.send_greeting(room_id, agent_id=agent_id)  # type: ignore[attr-defined]

        self._hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_SESSION_STARTED,
                execution=HookExecution.ASYNC,
                fn=_auto_greet_handler,
                name=hook_name,
            ),
        )
