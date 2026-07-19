"""EventOpsMixin — direct, host-owned mutation of persisted events (RFC §10.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import HookTrigger
from roomkit.models.event import EventContent, EventSource, RoomEvent

if TYPE_CHECKING:
    from roomkit.core.hooks import HookEngine
    from roomkit.core.locks import RoomLockManager
    from roomkit.store.base import ConversationStore


@runtime_checkable
class EventOpsHost(Protocol):
    """Contract: capabilities a host class must provide for EventOpsMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store holding the events.
        _hook_engine: Hook engine for the mutation triggers.
        _lock_manager: Per-room lock for serialised mutation.
    """

    _store: ConversationStore
    _hook_engine: HookEngine
    _lock_manager: RoomLockManager


class EventOpsMixin(HelpersMixin):
    """Direct update / delete of persisted events.

    The host application owns authorization on this path — unlike inbound
    EDIT/DELETE events (RFC §10.3), no author/identity check runs here. Both
    operations serialise against inbound processing via the room lock and fire
    their mutation trigger (ON_EVENT_UPDATED / ON_EVENT_DELETED) after the
    lock is released (RFC §10.1), so observers such as denormalized-projection
    maintainers see every stored-state change regardless of origin.

    Host contract: :class:`EventOpsHost`.
    """

    _store: ConversationStore
    _hook_engine: HookEngine
    _lock_manager: RoomLockManager

    async def update_event(
        self,
        room_id: str,
        event_id: str,
        *,
        content: EventContent | None = None,
        source: EventSource | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RoomEvent | None:
        """Update a persisted event's content, source, and/or metadata.

        Provided fields replace the stored values wholesale (``metadata`` is
        not merged — pass the full mapping). Unlike the inbound EDIT path,
        no ``edited`` marker is added; the caller controls metadata.

        Fires ON_EVENT_UPDATED with the updated event. Returns the updated
        event, or ``None`` when *event_id* does not exist in *room_id* (no
        hook fires).
        """
        async with self._lock_manager.locked(room_id):
            target = await self._store.get_event(event_id)
            if target is None or target.room_id != room_id:
                return None
            update: dict[str, Any] = {}
            if content is not None:
                update["content"] = content
            if source is not None:
                update["source"] = source
            if metadata is not None:
                update["metadata"] = metadata
            if not update:
                return target
            updated = await self._store.update_event(target.model_copy(update=update))
        context = await self._build_context(room_id)
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_EVENT_UPDATED, updated, context
        )
        return updated

    async def delete_event(
        self, room_id: str, event_id: str, *, cascade_replies: bool = True
    ) -> list[str]:
        """Hard-delete a persisted event, cascading to thread replies by default.

        Fires ON_EVENT_DELETED with the pre-delete snapshot of the root event.
        Returns the deleted event IDs (root first, then replies), or an empty
        list when *event_id* does not exist in *room_id* (no hook fires).
        """
        async with self._lock_manager.locked(room_id):
            target = await self._store.get_event(event_id)
            if target is None or target.room_id != room_id:
                return []
            deleted = await self._store.delete_event(
                room_id, event_id, cascade_replies=cascade_replies
            )
        if deleted:
            context = await self._build_context(room_id)
            await self._hook_engine.run_async_hooks(
                room_id, HookTrigger.ON_EVENT_DELETED, target, context
            )
        return deleted
