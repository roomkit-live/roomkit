"""ChannelOpsMixin — channel registration, binding, and state management."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels.base import Channel, FrameworkAwareChannel
from roomkit.channels.declared import DeclaredChannel
from roomkit.core.exceptions import (
    ChannelAlreadyRegisteredError,
    ChannelNotFoundError,
    ChannelNotRegisteredError,
    ConferenceAlreadyAttachedError,
)
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models._visibility import check_channel_id, parse_given_visibility
from roomkit.models.channel import ChannelBinding, ChannelDeclaration
from roomkit.models.enums import (
    Access,
    ChannelCategory,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
    Visibility,
)
from roomkit.turns.wire import RoomAttached

if TYPE_CHECKING:
    from roomkit.channels._author_images import AttachedMedia, MediaResolver
    from roomkit.core.event_router import EventRouter
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.models.room import Room
    from roomkit.realtime.base import RealtimeBackend
    from roomkit.store.base import ConversationStore
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.tools.external import ExternalToolHandler
    from roomkit.turns._registry_tools import OwnerToolStore
    from roomkit.turns.owner import TurnClient

logger = logging.getLogger("roomkit.framework")

# Room-metadata key recording channels whose binding was explicitly revoked
# via detach_channel(). The inbound path's auto-attach consults it so a
# revocation cannot be undone by the next message (RFC §7.5-7). It lives in
# room metadata — not framework memory — because the revocation must survive
# process restarts and be visible to every process sharing the store.
DETACHED_CHANNELS_KEY = "roomkit:detached_channels"


def is_channel_detached(room: Room, channel_id: str) -> bool:
    """Whether *channel_id* was explicitly detached from *room* (RFC §7.5-7)."""
    detached = room.metadata.get(DETACHED_CHANNELS_KEY)
    return isinstance(detached, list) and channel_id in detached


def _named_correspondent(binding: ChannelBinding, channel: Channel) -> ChannelBinding:
    """*binding* naming its correspondent as the router compares senders (RFC §10.4).

    The addresses the host gives, its recipient and its participant id, are
    written as the channel writes an inbound sender (E.164 on a phone
    channel). A binding that names no one but delivers to an address is that
    person's conversation, so it names them: the router then finds the room
    for them and admits no one else.
    """
    update: dict[str, Any] = {}
    recipient = channel.recipient_address(binding)
    if recipient is not None:
        written = channel.recipient_metadata(recipient)
        if any(binding.metadata.get(key) != value for key, value in written.items()):
            update["metadata"] = {**binding.metadata, **written}
    if binding.participant_id is not None:
        named = channel.normalize_address(binding.participant_id)
    else:
        named = None if binding.group else recipient
    if named != binding.participant_id:
        update["participant_id"] = named
    return binding.model_copy(update=update) if update else binding


def _auto_greet_hook_name(agent_id: str) -> str:
    return f"_agent_auto_greet:{agent_id}"


def _refuse_declared_binding(channel: Channel, category: ChannelCategory | None) -> None:
    """A channel the kit holds by its declaration alone takes part as an
    intelligence channel only: nothing here delivers to it (RFC §10.6.2)."""
    if isinstance(channel, DeclaredChannel) and category not in (
        None,
        ChannelCategory.INTELLIGENCE,
    ):
        raise ValueError(
            f"Channel {channel.channel_id!r} is registered by its declaration alone: "
            "it can be attached as an intelligence channel only"
        )


def _refuse_host_as_intelligence(channel: Channel, category: ChannelCategory | None) -> None:
    """A host channel takes part as a transport only: the host's words are a
    party's message, never an agent's output (RFC §10.5)."""
    bound_as = category if category is not None else channel.category
    if channel.channel_type is ChannelType.HOST and bound_as is ChannelCategory.INTELLIGENCE:
        raise ValueError(
            f"Channel {channel.channel_id!r} is a host channel: it can be attached as a "
            "transport only, never as an intelligence channel"
        )


@runtime_checkable
class ChannelOpsHost(Protocol):
    """Contract: capabilities a host class must provide for ChannelOpsMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _channels: Registry of channel-id to :class:`Channel` instances.
        _lock_manager: Per-room lock for serialised mutation.
        _event_router: Cached event router (or ``None`` to rebuild).
        _hook_engine: Engine for hook execution.
        _telemetry: Telemetry / tracing provider.
        _realtime: Ephemeral event backend (typing, presence).
        _owner_tools: The tools orchestration sets up, per channel and room.

    Cross-mixin methods (provided by other mixins in the MRO):
        get_room: From :class:`RoomLifecycleMixin`.
        join: From :class:`VoiceOpsMixin`.
        _set_greeting_gate: From :class:`GreetingMixin`.
        send_greeting: From :class:`GreetingMixin`.
        _clear_greeting_gate: From :class:`GreetingMixin`.
        _force_clear_greeting_gate: From :class:`GreetingMixin`.
        _build_room_tasks_loader: From :class:`DelegationMixin`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _lock_manager: RoomLockManager
    _event_router: EventRouter | None
    _hook_engine: HookEngine
    _telemetry: TelemetryProvider
    _realtime: RealtimeBackend
    _turns: TurnClient
    _owner_tools: OwnerToolStore


class ChannelOpsMixin(HelpersMixin):
    """Channel registration, attachment, and binding operations.

    Host contract: :class:`ChannelOpsHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _turns: TurnClient
    _owner_tools: OwnerToolStore
    _lock_manager: RoomLockManager
    _event_router: EventRouter | None
    _hook_engine: HookEngine
    _telemetry: TelemetryProvider
    _realtime: RealtimeBackend
    _media_resolver: MediaResolver | None
    _attached_media: AttachedMedia | None

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # see ChannelOpsHost
    join: Any  # see ChannelOpsHost
    _set_greeting_gate: Any  # see ChannelOpsHost
    send_greeting: Any  # see ChannelOpsHost
    _clear_greeting_gate: Any  # see ChannelOpsHost
    _force_clear_greeting_gate: Any  # see ChannelOpsHost
    _build_room_tasks_loader: Any  # see ChannelOpsHost
    _refuse_discussion_binding: Any  # see DiscussionMixin
    _on_room_detached: Any  # see ChannelStateMixin
    turn_executor: Any  # see RoomKit
    _kit_turn_hooks: Any  # see RoomKit

    def register_channel(self, channel: Channel | ChannelDeclaration) -> None:
        """Register a channel implementation by its ID.

        A :class:`ChannelDeclaration` registers an intelligence channel the
        kit holds by its declaration alone (RFC §10.6.2), as a
        :class:`DeclaredChannel`: bound, routed and planned for as the live
        channel would be, its turns run on an executor whose host builds it.

        A duplicate ``channel_id`` is refused: silently replacing a live
        channel would leave existing room bindings routing to an object the
        framework no longer delivers to, and the first registration's wiring
        (hooks, telemetry, realtime) would keep firing on the orphan. Call
        :meth:`unregister_channel` first to swap an implementation
        deliberately.

        Raises:
            ChannelAlreadyRegisteredError: a channel with this ID is
                already registered.
            ValueError: a channel id a visibility could not name (a
                keyword, a comma, surrounding whitespace; RFC §7.3), a
                channel of the type ``SYSTEM``, reserved to the framework's
                own records (RFC §5.3), or a
                declaration of a channel that is not an
                intelligence channel, or of one a turn hook registered on
                the kit could fire (its turns run on another executor, RFC
                §10.6.6).
            pydantic.ValidationError: an intelligence channel whose own
                declaration is invalid (an empty id, a negative window).
        """
        if isinstance(channel, ChannelDeclaration):
            channel = DeclaredChannel(channel)
        # A visibility names channels by id, so an id it could not name is
        # refused (RFC §7.3).
        check_channel_id(channel.channel_id)
        if channel.channel_type == ChannelType.SYSTEM:
            # The framework's own records' type, which no channel claims: a
            # SYSTEM source makes the changes reserved to the system (RFC §5.3).
            raise ValueError(
                f"channel {channel.channel_id} claims the reserved type {ChannelType.SYSTEM}"
            )
        if channel.channel_id in self._channels:
            raise ChannelAlreadyRegisteredError(
                f"Channel {channel.channel_id!r} is already registered. "
                f"Call unregister_channel({channel.channel_id!r}) first to "
                f"replace it deliberately."
            )
        if isinstance(channel, DeclaredChannel):
            # Its turns run elsewhere: a turn hook on the kit would not fire.
            self._kit_turn_hooks.refuse_declaration(channel)
        # The kit reads an intelligence channel through its declaration (RFC
        # §10.6.2): one it cannot give is refused before anything is set up.
        declaration = (
            channel.declaration() if channel.category is ChannelCategory.INTELLIGENCE else None
        )
        self._channels[channel.channel_id] = channel
        self._event_router = None  # Reset router cache

        # Wire protocol trace framework handler on all channels
        channel._trace_framework_handler = self._on_channel_trace

        # Propagate telemetry to all channels
        channel._telemetry = self._telemetry  # ty: ignore[invalid-assignment]

        # What an intelligence channel's turns fire the hooks through is the
        # executor's to give (RFC §10.6.2): the kit's own wires the channels the
        # kit registers, as a host's executor wires the ones it builds. A
        # declared channel's executor is a host's.
        if not isinstance(channel, DeclaredChannel):
            self.turn_executor.wire(channel)

        # Propagate telemetry to channel's sub-providers (AI, STT, TTS, etc.)
        if hasattr(channel, "_propagate_telemetry"):
            channel._propagate_telemetry()  # ty: ignore[call-non-callable]

        # Set framework reference on session-based channels for inbound routing.
        # A channel opts in by inheriting FrameworkAwareChannel, so this needs no
        # edit when one is written — and a channel that merely owns a method of
        # that name is not called with an argument it never expected.
        if isinstance(channel, FrameworkAwareChannel):
            # The host is the RoomKit instance; the mixin cannot say so.
            channel.set_framework(self)  # ty: ignore[invalid-argument-type]

        if declaration is not None:
            self._set_up_auto_greet(declaration)

        # RFC §8.2 — the registry changing is an observable framework event.
        self._emit_framework_event_soon(
            "channel_registered",
            channel_id=channel.channel_id,
            data={
                "channel_id": channel.channel_id,
                "channel_type": str(channel.channel_type),
            },
        )

    def _wire_external_tool_handler(self, channel_id: str, handler: ExternalToolHandler) -> None:
        """Inject hook callbacks so external tool calls fire RoomKit tool hooks."""
        self.turn_executor.wiring.wire_external_tool_handler(channel_id, handler)

    def unregister_channel(self, channel_id: str) -> Channel | None:
        """Remove a channel from the registry and return it.

        The inverse of :meth:`register_channel`. The channel is returned
        (not closed) so the caller can ``await channel.close()`` — closing
        is async and owns the session/provider teardown, so it stays
        explicit at the call site. Returns ``None`` for an unknown id.
        Room bindings are untouched; callers tearing down per-session
        channels typically close the room as well.
        """
        channel = self._channels.pop(channel_id, None)
        if channel is None:
            return None
        self._event_router = None  # Reset router cache
        # A channel registered again under the id greets and serves tools by
        # its own setup.
        self._hook_engine.remove_global_hook(_auto_greet_hook_name(channel_id))
        self._owner_tools.forget(channel_id)
        self._emit_framework_event_soon(
            "channel_unregistered",
            channel_id=channel_id,
            data={
                "channel_id": channel_id,
                "channel_type": str(channel.channel_type),
            },
        )
        return channel

    async def attach_channel(
        self,
        room_id: str,
        channel_id: str,
        category: ChannelCategory | None = None,
        access: Access = Access.READ_WRITE,
        visibility: str = Visibility.ALL,
        organization_id: str | None = None,
        **kwargs: Any,
    ) -> ChannelBinding:
        """Attach a registered channel to a room.

        *category* is the channel's own (``channel.category``) unless given:
        an agent attached without one takes part as an intelligence channel,
        a transport as a transport. *organization_id* scopes the operation to
        one tenant (RFC §17.2); a room belonging to another organization is
        reported as not found. An invalid *visibility*, and ``internal``,
        which marks a record the framework wrote, raise ``ValueError`` (RFC
        §7.3).
        """
        visibility = parse_given_visibility(visibility)
        async with self._lock_manager.locked(room_id):
            room = await self.get_room(room_id, organization_id=organization_id)
            # RFC §5.1: the room's due timer transitions apply first.
            room = await self._room_timers.apply_due(room)
            channel = self._channels.get(channel_id)
            if channel is None:
                raise ChannelNotRegisteredError(f"Channel {channel_id} not registered")
            _refuse_declared_binding(channel, category)
            _refuse_host_as_intelligence(channel, category)
            self._refuse_discussion_binding(room_id, channel, category)
            # A conference maps 1:1 to a room (RFC 12.10.1 principle 2, made
            # normative in 12.10.4): a second conference channel is a second
            # bot, a second transcription of every utterance and a second AI
            # voice for the same room. Re-attaching the same channel is an
            # ordinary attach over a live attachment and stays allowed.
            #
            # The reservation outlives the binding (RFC 12.10.4): a detach
            # removes the binding at its start and takes the bot out at its
            # end, so the bindings alone would admit a second conference while
            # the first one's teardown — possibly deferred — still has its bot
            # in the meeting. The other channel's own books are the durable
            # reservation, and this is a refusal rather than a wait because
            # the attach may be issued from inside the very announcement the
            # teardown is deferred behind, where waiting deadlocks.
            if channel.channel_type is ChannelType.CONFERENCE:
                holders = {
                    existing.channel_id
                    for existing in await self._store.list_bindings(room_id)
                    if existing.channel_type is ChannelType.CONFERENCE
                    and existing.channel_id != channel_id
                }
                for other_id, other in self._channels.items():
                    if other_id == channel_id or other.channel_type is not ChannelType.CONFERENCE:
                        continue
                    still_holds = getattr(other, "_holds_conference", None)
                    if still_holds is not None and still_holds(room_id):
                        holders.add(other_id)
                if holders:
                    named = ", ".join(sorted(holders))
                    raise ConferenceAlreadyAttachedError(
                        f"Room {room_id!r} already has conference channel {named!r} on it — "
                        f"attached, or with a teardown still releasing its session — and a "
                        f"conference maps 1:1 to a room (RFC 12.10.4). {channel_id!r} is "
                        f"refused; retry once {named!r} has fully let go."
                    )
            # The channel's own direction unless the attachment names one:
            # it decides whether the channel is delivered to (RFC §4.2).
            kwargs.setdefault("direction", channel.direction)
            binding = ChannelBinding(
                channel_id=channel_id,
                room_id=room_id,
                channel_type=channel.channel_type,
                category=category if category is not None else channel.category,
                access=access,
                visibility=visibility,
                capabilities=channel.capabilities(),
                **kwargs,
            )
            binding = _named_correspondent(binding, channel)
            # Read before it is replaced, so a refusal can put it back. An
            # attach over a live attachment is the case that needs it: the
            # channel is still attached when it refuses, and a room left with
            # no binding at all is one `detach_channel()` no longer reaches.
            previous = await self._store.get_binding(room_id, channel_id)
            result = await self._store.add_binding(binding)
            # An explicit attach is the integrator re-granting access, so the
            # revocation recorded by detach_channel() no longer applies.
            await self._clear_detach_tombstone(room, channel_id)
            # Before the attachment is announced, not after: what the channel
            # establishes here is what the binding claims exists, and a channel
            # that cannot establish it has not been attached. Left to the
            # ON_CHANNEL_ATTACHED hook, the failure would be logged and swallowed
            # (hooks are observation) and the caller would receive a binding to
            # a conference that was never created.
            await self._on_room_attached(room_id, channel, result, previous)
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

        # Auto-start voice session for single-user backends (e.g. LocalAudioBackend).
        # Runs AFTER the lock is released to avoid deadlocks if callbacks
        # re-enter the framework (e.g. hooks calling get_room).
        await self._post_attach(room_id, channel_id, channel, result)
        return result

    async def _on_room_attached(
        self,
        room_id: str,
        channel: Channel,
        binding: ChannelBinding,
        previous: ChannelBinding | None,
    ) -> None:
        """Let the channel establish the attachment, or take the binding back.

        The rollback is what makes propagating safe here. Only the binding has
        been written — no system event, no hook, no framework event — so undoing
        it leaves nothing for an observer to have seen. A few lines further down
        that stops being true: ``CHANNEL_ATTACHED`` is indexed and events do not
        come back, so a failure there would have to be announced as a detach of
        an attachment that never happened.

        Undoing means restoring, not deleting. Attaching over a live attachment
        replaces its binding, and the channel refusing the new one says nothing
        about the old: it is still attached, still collecting, still holding
        whatever it joined. Deleting the row would take the room's only handle
        on it away — ``detach_channel()`` finds nothing to remove and returns
        false, and the attachment it could have torn down runs on unreachable.
        """
        try:
            if binding.category is ChannelCategory.INTELLIGENCE:
                # The executors that may run its turns are told (RFC §10.6.3).
                attached = RoomAttached(
                    channel_id=channel.channel_id, room_id=room_id, binding=binding
                )
                await self._turns.control(attached)
            else:
                await channel.on_room_attached(room_id, binding)
        except Exception:
            await self._undo_refused_binding(room_id, channel.channel_id, previous)
            raise

    async def _undo_refused_binding(
        self, room_id: str, channel_id: str, previous: ChannelBinding | None
    ) -> None:
        """Put back what a channel's refusal replaced, saying so if even that fails.

        Swallowing here is what lets the refusal be the exception the caller
        sees: it is the reason the attachment is being undone, and a store that
        is down as well is a second problem rather than a replacement for the
        first.
        """
        try:
            if previous is None:
                await self._store.remove_binding(room_id, channel_id)
            else:
                await self._store.add_binding(previous)
        except Exception:
            logger.exception(
                "Could not restore the binding of channel %r to room %s after it "
                "refused the attachment; the room may keep a binding to a channel "
                "that is not attached, or have lost the one it had",
                channel_id,
                room_id,
            )

    async def _post_attach(
        self,
        room_id: str,
        channel_id: str,
        channel: Channel,
        binding: ChannelBinding,
    ) -> None:
        """Auto-start voice session when backend declares ``auto_connect``."""
        from roomkit.channels.voice import VoiceChannel

        if not isinstance(channel, VoiceChannel):
            return
        backend = channel._backend
        if backend is None or not backend.auto_connect:
            return

        await self.join(room_id, channel_id)

    async def detach_channel(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> bool:
        """Detach a channel from a room.

        *organization_id* scopes the operation to one tenant (RFC §17.2); a
        room belonging to another organization is reported as not found.
        """
        async with self._lock_manager.locked(room_id):
            # RFC §5.1: the room's due timer transitions apply first.
            await self._room_timers.room_due(room_id, organization_id=organization_id)
            binding = await self._store.get_binding(room_id, channel_id)
            if binding is None:
                return False
            # Record the revocation before the binding goes: the tombstone in
            # room metadata is what keeps the inbound path's auto-attach from
            # undoing an explicit detach (RFC §7.5-7), and it must survive
            # restarts and be visible to every process sharing the store — an
            # in-memory set is neither. Written first so a store failure
            # leaves the binding in place rather than an unrecorded
            # revocation. An explicit attach_channel() clears it —
            # re-granting access is the integrator's call.
            await self._record_detach_tombstone(room_id, channel_id)
            removed = await self._store.remove_binding(room_id, channel_id)
            if removed:
                await self._emit_system_event(
                    room_id,
                    EventType.CHANNEL_DETACHED,
                    code="channel_detached",
                    message=f"Channel {channel_id} detached",
                    data={"channel_id": channel_id},
                )
                # Before the hooks, so an integrator's handler runs after the
                # channel has finished letting go rather than concurrently with
                # it — async hooks of one trigger run together, and the channel's
                # own teardown is not one of the things they should race.
                #
                # Announced whatever the channel makes of it. The binding is
                # already gone and CHANNEL_DETACHED already indexed, so the
                # detach has happened as far as the room is concerned; a channel
                # that raised on its way out changes how *well* it happened, not
                # whether, and observers told nothing would go on believing the
                # channel is attached. The error still reaches the caller, after
                # the room has been told.
                refusal: Exception | None = None
                try:
                    await self._on_room_detached(room_id, binding)
                except Exception as error:
                    refusal = error
                await self._announce_detach(room_id, channel_id, refusal)
                if refusal is not None:
                    raise refusal
            return removed

    async def _record_detach_tombstone(self, room_id: str, channel_id: str) -> None:
        """Persist the revocation of *channel_id*'s binding (RFC §7.5-7).

        Appends the channel to the room's ``DETACHED_CHANNELS_KEY`` metadata
        list via ``patch_room_metadata`` — atomic on persistent backends, so
        two processes detaching different channels do not clobber each other.
        Runs under the room lock like every binding mutation.
        """
        room = await self.get_room(room_id)
        detached = room.metadata.get(DETACHED_CHANNELS_KEY)
        current = list(detached) if isinstance(detached, list) else []
        if channel_id in current:
            return
        current.append(channel_id)
        await self._store.patch_room_metadata(room_id, {DETACHED_CHANNELS_KEY: current})

    async def _clear_detach_tombstone(self, room: Room, channel_id: str) -> None:
        """Drop *channel_id* from the room's detach tombstones, if present.

        The key itself is unset when the last tombstone goes, so a room that
        never saw a detach — or whose detaches were all undone — carries no
        trace of the mechanism in its metadata.
        """
        detached = room.metadata.get(DETACHED_CHANNELS_KEY)
        if not isinstance(detached, list) or channel_id not in detached:
            return
        remaining = [c for c in detached if c != channel_id]
        if remaining:
            await self._store.patch_room_metadata(room.id, {DETACHED_CHANNELS_KEY: remaining})
        else:
            await self._store.patch_room_metadata(room.id, {}, unset=[DETACHED_CHANNELS_KEY])

    async def _announce_detach(
        self, room_id: str, channel_id: str, refusal: Exception | None = None
    ) -> None:
        """Tell the room's observers that a channel has been detached.

        ``refusal`` is what the channel raised on its way out, when it raised
        anything. It is why this can fail quietly: that exception is the one the
        caller is owed, and an announcement that fails as well must not take its
        place. With nothing to displace, a failure here is the caller's to see.
        """
        try:
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
        except Exception:
            if refusal is None:
                raise
            logger.exception(
                "Could not announce the detach of channel %r from room %s, which is "
                "being reported as the channel's own failure to let go",
                channel_id,
                room_id,
            )

    async def mute(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> ChannelBinding:
        """Mute a channel in a room. Muting a muted channel writes nothing and
        fires nothing (RFC §7.2)."""
        async with self._lock_manager.locked(room_id):
            binding = await self._due_binding(room_id, channel_id, organization_id)
            updated = binding.model_copy(update={"muted": True})
            if updated == binding:
                return self._gate_unchanged(room_id, channel_id, binding)
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

    async def unmute(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> ChannelBinding:
        """Unmute a channel in a room. Unmuting a channel that is not muted
        writes nothing and fires nothing (RFC §7.2)."""
        async with self._lock_manager.locked(room_id):
            binding = await self._due_binding(room_id, channel_id, organization_id)
            updated = binding.model_copy(update={"muted": False})
            if updated == binding:
                return self._gate_unchanged(room_id, channel_id, binding)
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

    async def mute_output(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> ChannelBinding:
        """Mute AI/provider output for a channel — audio from the provider
        will not be forwarded to the transport."""
        async with self._lock_manager.locked(room_id):
            binding = await self._due_binding(room_id, channel_id, organization_id)
            updated = binding.model_copy(update={"output_muted": True})
            if updated == binding:
                return self._gate_unchanged(room_id, channel_id, binding)
            result = await self._store.update_binding(updated)
            self._notify_binding_updated(room_id, channel_id, result)
            return result

    async def unmute_output(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> ChannelBinding:
        """Unmute AI/provider output for a channel — audio from the provider
        will be forwarded to the transport."""
        async with self._lock_manager.locked(room_id):
            binding = await self._due_binding(room_id, channel_id, organization_id)
            updated = binding.model_copy(update={"output_muted": False})
            if updated == binding:
                return self._gate_unchanged(room_id, channel_id, binding)
            result = await self._store.update_binding(updated)
            self._notify_binding_updated(room_id, channel_id, result)
            return result

    async def set_visibility(
        self,
        room_id: str,
        channel_id: str,
        visibility: str,
        *,
        organization_id: str | None = None,
    ) -> ChannelBinding:
        """Set visibility for a channel in a room. An invalid *visibility*,
        and ``internal``, which marks a record the framework wrote, raise
        ``ValueError`` (RFC §7.3). The visibility the binding holds already
        writes nothing (RFC §7.2)."""
        visibility = parse_given_visibility(visibility)
        async with self._lock_manager.locked(room_id):
            binding = await self._due_binding(room_id, channel_id, organization_id)
            old_visibility = binding.visibility
            updated = binding.model_copy(update={"visibility": visibility})
            if updated == binding:
                return binding
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

    async def set_access(
        self,
        room_id: str,
        channel_id: str,
        access: Access,
        *,
        organization_id: str | None = None,
    ) -> ChannelBinding:
        """Set access level for a channel in a room. The access the binding
        holds already writes nothing (RFC §7.2)."""
        async with self._lock_manager.locked(room_id):
            binding = await self._due_binding(room_id, channel_id, organization_id)
            old_access = binding.access
            updated = binding.model_copy(update={"access": access})
            if updated == binding:
                return self._gate_unchanged(room_id, channel_id, binding)
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
        self,
        room_id: str,
        channel_id: str,
        metadata: dict[str, Any],
        *,
        organization_id: str | None = None,
    ) -> ChannelBinding:
        """Update metadata on a channel binding, merged into what it holds.
        An update that changes nothing writes nothing (RFC §7.2)."""
        async with self._lock_manager.locked(room_id):
            binding = await self._due_binding(room_id, channel_id, organization_id)
            updated = binding.model_copy(update={"metadata": {**binding.metadata, **metadata}})
            channel = self._channels.get(channel_id)
            if channel is not None:
                updated = _named_correspondent(updated, channel)
            if updated == binding:
                return binding
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

    async def get_binding(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> ChannelBinding:
        """Get a channel binding. Raises ChannelNotFoundError if missing."""
        return await self._get_binding(room_id, channel_id, organization_id=organization_id)

    async def list_bindings(
        self, room_id: str, *, organization_id: str | None = None
    ) -> list[ChannelBinding]:
        """List all channel bindings for a room."""
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        return await self._store.list_bindings(room_id)

    def _gate_unchanged(
        self, room_id: str, channel_id: str, binding: ChannelBinding
    ) -> ChannelBinding:
        """A gate change (mute, access) that changes nothing (RFC §7.2):
        nothing is written and nothing fires, but the channel is told the
        binding as stored, since its cached gate state may have drifted from
        the store (another process wrote it): a repeated mute never leaves
        audio flowing."""
        self._notify_binding_updated(room_id, channel_id, binding)
        return binding

    def _notify_binding_updated(
        self, room_id: str, channel_id: str, binding: ChannelBinding
    ) -> None:
        """Push updated binding state to the channel for audio gating."""
        channel = self._channels.get(channel_id)
        if channel is not None:
            channel.update_binding(room_id, binding)

    async def _due_binding(
        self, room_id: str, channel_id: str, organization_id: str | None
    ) -> ChannelBinding:
        """The binding a binding change starts from, read under the room
        lock once the room's due timer transitions are applied (RFC §5.1)."""
        await self._room_timers.room_due(room_id, organization_id=organization_id)
        return await self._get_binding(room_id, channel_id)

    async def _get_binding(
        self, room_id: str, channel_id: str, *, organization_id: str | None = None
    ) -> ChannelBinding:
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        binding = await self._store.get_binding(room_id, channel_id)
        if binding is None:
            raise ChannelNotFoundError(f"Channel {channel_id} not in room {room_id}")
        return binding

    def _set_up_auto_greet(self, declaration: ChannelDeclaration) -> None:
        """Greet on a session start for an agent that says so, a declared one
        as a live one (RFC §10.6.2, §19.1)."""
        identity = declaration.identity
        if identity is not None and identity.auto_greet and identity.greeting:
            self._register_auto_greet_hook(declaration.channel_id)

    def _register_auto_greet_hook(self, agent_id: str) -> None:
        """Register a global ON_SESSION_STARTED hook for agent auto-greeting."""
        from roomkit.core.hooks import HookRegistration

        hook_name = _auto_greet_hook_name(agent_id)
        kit_ref = self
        # Track rooms already greeted for text channels (no session to dedup on).
        # Bounded to prevent unbounded growth on long-running servers.
        # Uses OrderedDict as LRU — on overflow, evict oldest 10% instead of
        # clearing all history (which would allow duplicate greetings).
        greeted_rooms: OrderedDict[str, None] = OrderedDict()
        greeted_rooms_max = 10_000

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
                if len(greeted_rooms) >= greeted_rooms_max:
                    for _ in range(greeted_rooms_max // 10):
                        if greeted_rooms:
                            greeted_rooms.popitem(last=False)
                greeted_rooms[dedup_key] = None

                # Check agent is attached to this room
                bindings = await kit_ref._store.list_bindings(room_id)
                if not any(b.channel_id == agent_id for b in bindings):
                    greeted_rooms.pop(dedup_key, None)
                    return

            kit_ref._set_greeting_gate(room_id)
            try:
                await kit_ref.send_greeting(
                    room_id,
                    agent_id=agent_id,
                    session=event.session,
                    channel_type=event.channel_type,
                )
                kit_ref._clear_greeting_gate(room_id)
            except Exception:
                kit_ref._force_clear_greeting_gate(room_id)
                raise

        self._hook_engine.register(
            HookRegistration(
                trigger=HookTrigger.ON_SESSION_STARTED,
                execution=HookExecution.ASYNC,
                fn=_auto_greet_handler,
                name=hook_name,
            ),
        )
