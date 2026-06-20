"""Row ↔ model mapping helpers for the PostgreSQL ConversationStore.

Pure functions translating asyncpg rows into RoomKit models (and the
inverse ``_source_extra`` projection). No dependency on store instance
state — split out of ``postgres.py`` to keep the CRUD class focused.
"""

from __future__ import annotations

import json
from typing import Any

from roomkit.models.channel import ChannelBinding
from roomkit.models.event import ChannelData, EventSource, RoomEvent
from roomkit.models.participant import Participant
from roomkit.models.room import Room, RoomTimers
from roomkit.models.task import Observation, Task


def _row_to_room(row: Any) -> Room:
    """Convert a database row to a Room model."""
    return Room(
        id=row["id"],
        organization_id=row["organization_id"],
        status=row["status"],
        event_count=row["event_count"],
        latest_index=row["latest_index"],
        metadata=(
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        ),
        timers=RoomTimers.model_validate(
            json.loads(row["timers"]) if isinstance(row["timers"], str) else row["timers"]
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        closed_at=row["closed_at"],
    )


def _row_to_event(row: Any) -> RoomEvent:
    """Convert a database row to a RoomEvent model."""
    content_raw = row["content"]
    content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
    source_extra = row["source_extra"]
    if isinstance(source_extra, str):
        source_extra = json.loads(source_extra)
    channel_data_raw = row["channel_data"]
    if isinstance(channel_data_raw, str):
        channel_data_raw = json.loads(channel_data_raw)
    metadata_raw = row["metadata"]
    if isinstance(metadata_raw, str):
        metadata_raw = json.loads(metadata_raw)
    return RoomEvent(
        id=row["id"],
        room_id=row["room_id"],
        type=row["type"],
        content=content,
        source=EventSource(
            channel_id=row["source_channel_id"],
            channel_type=row["source_channel_type"],
            direction=row["source_direction"],
            participant_id=row["source_participant_id"],
            provider=row["source_provider"],
            raw_payload=source_extra.get("raw_payload", {}),
            external_id=source_extra.get("external_id"),
            provider_message_id=source_extra.get("provider_message_id"),
        ),
        status=row["status"],
        visibility=row["visibility"],
        response_visibility=row["response_visibility"],
        index=row["index"],
        chain_depth=row["chain_depth"],
        correlation_id=row["correlation_id"],
        parent_event_id=row["parent_event_id"],
        idempotency_key=row["idempotency_key"],
        blocked_by=row["blocked_by"],
        metadata=metadata_raw,
        channel_data=ChannelData.model_validate(channel_data_raw),
        created_at=row["created_at"],
    )


def _row_to_binding(row: Any) -> ChannelBinding:
    """Convert a database row to a ChannelBinding model."""
    caps = row["capabilities"]
    if isinstance(caps, str):
        caps = json.loads(caps)
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return ChannelBinding(
        channel_id=row["channel_id"],
        room_id=row["room_id"],
        channel_type=row["channel_type"],
        category=row["category"],
        direction=row["direction"],
        access=row["access"],
        muted=row["muted"],
        output_muted=row["output_muted"],
        visibility=row["visibility"],
        participant_id=row["participant_id"],
        last_read_index=row["last_read_index"],
        attached_at=row["attached_at"],
        capabilities=caps,
        metadata=meta,
    )


def _row_to_participant(row: Any) -> Participant:
    """Convert a database row to a Participant model."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Participant(
        id=row["id"],
        room_id=row["room_id"],
        channel_id=row["channel_id"],
        display_name=row["display_name"],
        role=row["role"],
        status=row["status"],
        identification=row["identification"],
        identity_id=row["identity_id"],
        external_id=row["external_id"],
        joined_at=row["joined_at"],
        resolved_at=row["resolved_at"],
        resolved_by=row["resolved_by"],
        metadata=meta,
    )


def _row_to_task(row: Any) -> Task:
    """Convert a database row to a Task model."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Task(
        id=row["id"],
        room_id=row["room_id"],
        title=row["title"],
        description=row["description"],
        assigned_to=row["assigned_to"],
        status=row["status"],
        created_at=row["created_at"],
        metadata=meta,
    )


def _row_to_observation(row: Any) -> Observation:
    """Convert a database row to an Observation model."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Observation(
        id=row["id"],
        room_id=row["room_id"],
        channel_id=row["channel_id"],
        content=row["content"],
        category=row["category"],
        confidence=row["confidence"],
        created_at=row["created_at"],
        metadata=meta,
    )


def _source_extra(source: EventSource) -> dict[str, Any]:
    """Extract non-column source fields into a JSONB dict."""
    extra: dict[str, Any] = {}
    if source.raw_payload:
        extra["raw_payload"] = source.raw_payload
    if source.external_id:
        extra["external_id"] = source.external_id
    if source.provider_message_id:
        extra["provider_message_id"] = source.provider_message_id
    return extra
