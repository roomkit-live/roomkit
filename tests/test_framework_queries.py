"""Tests for framework query and management API:
channels, bindings, timeline, tasks, and exports."""

from __future__ import annotations

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import (
    ChannelNotFoundError,
    RoomKit,
)
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventType,
)
from roomkit.models.event import RoomEvent
from roomkit.models.task import Observation, Task

# -- Helpers --


class StubChannel(Channel):
    channel_type = ChannelType.WEBSOCKET
    category = ChannelCategory.TRANSPORT

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def _setup_kit_with_room() -> tuple[RoomKit, str]:
    kit = RoomKit()
    ch = StubChannel("ch1")
    kit.register_channel(ch)
    room = await kit.create_room("r1")
    await kit.attach_channel("r1", "ch1")
    return kit, room.id


# ============================================================================
# E1: Missing Framework Methods (Area 15)
# ============================================================================


class TestGetChannel:
    def test_get_existing_channel(self) -> None:
        kit = RoomKit()
        ch = StubChannel("ch1")
        kit.register_channel(ch)
        assert kit.get_channel("ch1") is ch

    def test_get_missing_channel(self) -> None:
        kit = RoomKit()
        assert kit.get_channel("nope") is None


class TestListChannels:
    def test_empty(self) -> None:
        kit = RoomKit()
        assert kit.list_channels() == []

    def test_returns_all(self) -> None:
        kit = RoomKit()
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        kit.register_channel(ch1)
        kit.register_channel(ch2)
        channels = kit.list_channels()
        assert len(channels) == 2
        ids = {c.channel_id for c in channels}
        assert ids == {"ch1", "ch2"}


class TestGetBinding:
    async def test_get_existing_binding(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        binding = await kit.get_binding(room_id, "ch1")
        assert binding.channel_id == "ch1"
        assert binding.room_id == room_id

    async def test_get_missing_binding_raises(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        with pytest.raises(ChannelNotFoundError):
            await kit.get_binding(room_id, "nope")


class TestListBindings:
    async def test_list_bindings(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        bindings = await kit.list_bindings(room_id)
        assert len(bindings) >= 1
        ids = {b.channel_id for b in bindings}
        assert "ch1" in ids

    async def test_list_bindings_empty_room(self) -> None:
        kit = RoomKit()
        await kit.create_room("r2")
        bindings = await kit.list_bindings("r2")
        assert bindings == []


class TestUpdateBindingMetadata:
    async def test_updates_metadata(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        result = await kit.update_binding_metadata(room_id, "ch1", {"key": "value"})
        assert result.metadata["key"] == "value"

    async def test_merges_metadata(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        await kit.update_binding_metadata(room_id, "ch1", {"a": 1})
        result = await kit.update_binding_metadata(room_id, "ch1", {"b": 2})
        assert result.metadata["a"] == 1
        assert result.metadata["b"] == 2

    async def test_emits_system_event(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        await kit.update_binding_metadata(room_id, "ch1", {"k": "v"})
        events = await kit.store.list_events(room_id)
        update_events = [
            e
            for e in events
            if e.type == EventType.CHANNEL_UPDATED
            and hasattr(e.content, "code")
            and e.content.code == "channel_metadata_updated"
        ]
        assert len(update_events) >= 1


class TestGetTimeline:
    async def test_returns_events(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        # The room should already have system events from attach
        timeline = await kit.get_timeline(room_id)
        assert isinstance(timeline, list)
        assert len(timeline) > 0

    async def test_with_limit(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        timeline = await kit.get_timeline(room_id, limit=1)
        assert len(timeline) <= 1

    async def test_raises_on_missing_room(self) -> None:
        kit = RoomKit()
        from roomkit.core.framework import RoomNotFoundError

        with pytest.raises(RoomNotFoundError):
            await kit.get_timeline("nonexistent")


class TestListTasks:
    async def test_empty(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        tasks = await kit.list_tasks(room_id)
        assert tasks == []

    async def test_returns_tasks(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        task = Task(id="t1", room_id=room_id, title="Test task")
        await kit.store.add_task(task)
        tasks = await kit.list_tasks(room_id)
        assert len(tasks) == 1
        assert tasks[0].id == "t1"

    async def test_filter_by_status(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        t1 = Task(id="t1", room_id=room_id, title="Pending", status="pending")
        t2 = Task(id="t2", room_id=room_id, title="Done", status="completed")
        await kit.store.add_task(t1)
        await kit.store.add_task(t2)
        pending = await kit.list_tasks(room_id, status="pending")
        assert len(pending) == 1
        assert pending[0].id == "t1"


class TestListObservations:
    async def test_empty(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        obs = await kit.list_observations(room_id)
        assert obs == []

    async def test_returns_observations(self) -> None:
        kit, room_id = await _setup_kit_with_room()
        observation = Observation(
            id="o1",
            room_id=room_id,
            channel_id="ch1",
            content="test observation",
            category="test",
        )
        await kit.store.add_observation(observation)
        obs = await kit.list_observations(room_id)
        assert len(obs) == 1
        assert obs[0].id == "o1"


# ============================================================================
# E2: Public API Exports (Area 16)
# ============================================================================


class TestPublicExports:
    def test_core_exports(self) -> None:
        import roomkit

        assert hasattr(roomkit, "RoomKit")
        assert hasattr(roomkit, "RoomNotFoundError")
        assert hasattr(roomkit, "ChannelNotFoundError")
        assert hasattr(roomkit, "ChannelNotRegisteredError")

    def test_routing_exports(self) -> None:
        import roomkit

        assert hasattr(roomkit, "InboundRoomRouter")
        assert hasattr(roomkit, "DefaultInboundRoomRouter")

    def test_channel_exports(self) -> None:
        import roomkit

        for name in [
            "Channel",
            "AIChannel",
            "EmailChannel",
            "SMSChannel",
            "WebSocketChannel",
            "MessengerChannel",
            "HTTPChannel",
            "WhatsAppChannel",
        ]:
            assert hasattr(roomkit, name), f"Missing export: {name}"

    def test_model_exports(self) -> None:
        import roomkit

        for name in [
            "RateLimit",
            "RetryPolicy",
            "Task",
            "Observation",
            "Room",
            "RoomTimers",
            "RoomEvent",
            "ChannelBinding",
            "ChannelCapabilities",
            "ChannelOutput",
            "Participant",
            "Identity",
        ]:
            assert hasattr(roomkit, name), f"Missing export: {name}"

    def test_provider_abc_exports(self) -> None:
        import roomkit

        for name in [
            "AIProvider",
            "EmailProvider",
            "HTTPProvider",
            "MessengerProvider",
            "SMSProvider",
            "WhatsAppProvider",
        ]:
            assert hasattr(roomkit, name), f"Missing export: {name}"

    def test_mock_provider_exports(self) -> None:
        import roomkit

        for name in [
            "MockAIProvider",
            "MockEmailProvider",
            "MockHTTPProvider",
            "MockMessengerProvider",
            "MockSMSProvider",
            "MockWhatsAppProvider",
        ]:
            assert hasattr(roomkit, name), f"Missing export: {name}"

    def test_enum_exports(self) -> None:
        import roomkit

        for name in [
            "Access",
            "ChannelCategory",
            "ChannelDirection",
            "ChannelMediaType",
            "ChannelType",
            "DeliveryMode",
            "EventStatus",
            "EventType",
            "HookExecution",
            "HookTrigger",
            "IdentificationStatus",
            "ParticipantRole",
            "ParticipantStatus",
            "RoomStatus",
        ]:
            assert hasattr(roomkit, name), f"Missing export: {name}"

    def test_all_matches_dir(self) -> None:
        """Every name in __all__ should be importable."""
        import roomkit

        for name in roomkit.__all__:
            assert hasattr(roomkit, name), f"__all__ lists {name} but it's not importable"
