"""Internal orchestration for room-level media recording."""

from __future__ import annotations

import logging

from roomkit.recorder.base import (
    MediaRecordingHandle,
    MediaRecordingResult,
    RecordingTrack,
    RoomRecorderBinding,
)

logger = logging.getLogger("roomkit.recorder")

# Type alias for clarity: each active binding pairs with its recording handle.
_ActiveBinding = tuple[RoomRecorderBinding, MediaRecordingHandle]


class RoomRecorderManager:
    """Manages room-level media recorders across all rooms.

    Isolates recording orchestration from the framework so that
    ``framework.py`` stays focused on routing and lifecycle.
    """

    def __init__(self) -> None:
        self._registry: dict[str, list[_ActiveBinding]] = {}

    def register(self, room_id: str, bindings: list[RoomRecorderBinding]) -> None:
        """Start recordings for all bindings in a room."""
        active: list[_ActiveBinding] = []
        for binding in bindings:
            if not binding.enabled:
                continue
            handle = binding.recorder.on_recording_start(binding.config)
            handle.room_id = room_id
            active.append((binding, handle))
            logger.info(
                "Room recording started: %s (recorder=%s, room=%s)",
                handle.id,
                binding.recorder.name,
                room_id,
            )
        if active:
            self._registry[room_id] = active

    def on_track_added(self, room_id: str, track: RecordingTrack) -> None:
        """Notify all recorders in a room about a new track."""
        for binding, handle in self._registry.get(room_id, []):
            binding.recorder.on_track_added(handle, track)

    def on_track_removed(self, room_id: str, track: RecordingTrack) -> None:
        """Notify all recorders in a room about a removed track."""
        for binding, handle in self._registry.get(room_id, []):
            binding.recorder.on_track_removed(handle, track)

    def on_data(
        self,
        room_id: str,
        track: RecordingTrack,
        data: bytes,
        timestamp_ms: float,
    ) -> None:
        """Fan out media data to all recorders in a room."""
        for binding, handle in self._registry.get(room_id, []):
            binding.recorder.on_data(handle, track, data, timestamp_ms)

    def stop_room(self, room_id: str) -> list[MediaRecordingResult]:
        """Stop all recordings in a room and return results."""
        results: list[MediaRecordingResult] = []
        for binding, handle in self._registry.pop(room_id, []):
            result = binding.recorder.on_recording_stop(handle)
            results.append(result)
            logger.info(
                "Room recording stopped: %s (%.1fs, %d bytes)",
                result.id,
                result.duration_seconds,
                result.size_bytes,
            )
        return results

    def has_recorders(self, room_id: str) -> bool:
        """Check if a room has active recorders."""
        return room_id in self._registry

    def close(self) -> None:
        """Stop all rooms and close all recorders."""
        seen_recorders: set[int] = set()
        for room_id in list(self._registry):
            for binding, handle in self._registry.pop(room_id, []):
                binding.recorder.on_recording_stop(handle)
                recorder_id = id(binding.recorder)
                if recorder_id not in seen_recorders:
                    seen_recorders.add(recorder_id)
                    binding.recorder.close()
