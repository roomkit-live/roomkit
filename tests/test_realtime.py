"""Tests for realtime backend."""

from __future__ import annotations

import asyncio

from roomkit import (
    EphemeralEvent,
    EphemeralEventType,
    InMemoryRealtime,
    RealtimeBackend,
    RoomKit,
)


class TestEphemeralEvent:
    def test_to_dict_and_from_dict(self) -> None:
        event = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_START,
            user_id="user-1",
            channel_id="ch-1",
            data={"key": "value"},
        )
        data = event.to_dict()
        restored = EphemeralEvent.from_dict(data)

        assert restored.id == event.id
        assert restored.room_id == event.room_id
        assert restored.type == event.type
        assert restored.user_id == event.user_id
        assert restored.channel_id == event.channel_id
        assert restored.data == event.data
        assert restored.timestamp == event.timestamp

    def test_default_values(self) -> None:
        event = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.PRESENCE_ONLINE,
            user_id="user-1",
        )
        assert event.id  # auto-generated
        assert event.channel_id is None
        assert event.data == {}
        assert event.timestamp  # auto-generated


class TestInMemoryRealtime:
    async def test_publish_subscribe(self) -> None:
        backend = InMemoryRealtime()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await backend.subscribe("ch", callback)
        event = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_START,
            user_id="user-1",
        )
        await backend.publish("ch", event)

        # Allow task to process
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].id == event.id

        await backend.unsubscribe(sub_id)
        await backend.close()

    async def test_multiple_subscribers(self) -> None:
        backend = InMemoryRealtime()
        received1: list[EphemeralEvent] = []
        received2: list[EphemeralEvent] = []

        async def callback1(event: EphemeralEvent) -> None:
            received1.append(event)

        async def callback2(event: EphemeralEvent) -> None:
            received2.append(event)

        sub1 = await backend.subscribe("ch", callback1)
        sub2 = await backend.subscribe("ch", callback2)

        event = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.PRESENCE_ONLINE,
            user_id="user-1",
        )
        await backend.publish("ch", event)

        await asyncio.sleep(0.01)

        assert len(received1) == 1
        assert len(received2) == 1
        assert received1[0].id == received2[0].id

        await backend.unsubscribe(sub1)
        await backend.unsubscribe(sub2)
        await backend.close()

    async def test_unsubscribe_stops_delivery(self) -> None:
        backend = InMemoryRealtime()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await backend.subscribe("ch", callback)

        # First event should be delivered
        event1 = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_START,
            user_id="user-1",
        )
        await backend.publish("ch", event1)
        await asyncio.sleep(0.01)
        assert len(received) == 1

        # Unsubscribe
        result = await backend.unsubscribe(sub_id)
        assert result is True

        # Second event should not be delivered
        event2 = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_STOP,
            user_id="user-1",
        )
        await backend.publish("ch", event2)
        await asyncio.sleep(0.01)
        assert len(received) == 1  # Still 1

        await backend.close()

    async def test_unsubscribe_nonexistent(self) -> None:
        backend = InMemoryRealtime()
        result = await backend.unsubscribe("nonexistent")
        assert result is False
        await backend.close()

    async def test_room_convenience_methods(self) -> None:
        backend = InMemoryRealtime()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await backend.subscribe_to_room("room-1", callback)

        event = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.READ_RECEIPT,
            user_id="user-1",
            data={"event_id": "evt-123"},
        )
        await backend.publish_to_room("room-1", event)

        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].data["event_id"] == "evt-123"

        await backend.unsubscribe(sub_id)
        await backend.close()

    async def test_queue_overflow_drops_oldest(self) -> None:
        backend = InMemoryRealtime(max_queue_size=2)
        received: list[EphemeralEvent] = []
        gate = asyncio.Event()

        async def slow_callback(event: EphemeralEvent) -> None:
            await gate.wait()
            received.append(event)

        sub_id = await backend.subscribe("ch", slow_callback)

        # Publish 3 events while callback is blocked
        for i in range(3):
            event = EphemeralEvent(
                room_id="room-1",
                type=EphemeralEventType.TYPING_START,
                user_id=f"user-{i}",
            )
            await backend.publish("ch", event)

        # Release gate and let events process
        gate.set()
        await asyncio.sleep(0.05)

        # Only 2 should be received (oldest dropped)
        assert len(received) == 2
        assert received[0].user_id == "user-1"  # user-0 dropped
        assert received[1].user_id == "user-2"

        await backend.unsubscribe(sub_id)
        await backend.close()

    async def test_subscription_count(self) -> None:
        backend = InMemoryRealtime()

        async def callback(event: EphemeralEvent) -> None:
            pass

        assert backend.subscription_count == 0

        sub1 = await backend.subscribe("ch1", callback)
        assert backend.subscription_count == 1

        _ = await backend.subscribe("ch2", callback)
        assert backend.subscription_count == 2

        await backend.unsubscribe(sub1)
        assert backend.subscription_count == 1

        await backend.close()
        assert backend.subscription_count == 0

    async def test_close_stops_all_subscriptions(self) -> None:
        backend = InMemoryRealtime()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        await backend.subscribe("ch", callback)
        await backend.subscribe("ch", callback)
        assert backend.subscription_count == 2

        await backend.close()
        assert backend.subscription_count == 0

        # Publishing after close should not error
        event = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_START,
            user_id="user-1",
        )
        await backend.publish("ch", event)  # Should not raise

    async def test_callback_error_does_not_stop_subscription(self) -> None:
        backend = InMemoryRealtime()
        received: list[EphemeralEvent] = []
        call_count = 0

        async def flaky_callback(event: EphemeralEvent) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated error")
            received.append(event)

        sub_id = await backend.subscribe("ch", flaky_callback)

        # First event will error
        event1 = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_START,
            user_id="user-1",
        )
        await backend.publish("ch", event1)
        await asyncio.sleep(0.01)

        # Second event should still be processed
        event2 = EphemeralEvent(
            room_id="room-1",
            type=EphemeralEventType.TYPING_STOP,
            user_id="user-1",
        )
        await backend.publish("ch", event2)
        await asyncio.sleep(0.01)

        assert call_count == 2
        assert len(received) == 1  # Only second event succeeded

        await backend.unsubscribe(sub_id)
        await backend.close()


class TestRoomKitRealtimeIntegration:
    async def test_publish_typing(self) -> None:
        kit = RoomKit()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await kit.subscribe_room("room-1", callback)

        await kit.publish_typing("room-1", "user-1")
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].type == EphemeralEventType.TYPING_START
        assert received[0].user_id == "user-1"
        assert received[0].room_id == "room-1"

        await kit.unsubscribe_room(sub_id)
        await kit.close()

    async def test_publish_typing_stop(self) -> None:
        kit = RoomKit()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await kit.subscribe_room("room-1", callback)

        await kit.publish_typing("room-1", "user-1", is_typing=False)
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].type == EphemeralEventType.TYPING_STOP

        await kit.unsubscribe_room(sub_id)
        await kit.close()

    async def test_publish_presence(self) -> None:
        kit = RoomKit()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await kit.subscribe_room("room-1", callback)

        await kit.publish_presence("room-1", "user-1", "online")
        await kit.publish_presence("room-1", "user-1", "away")
        await kit.publish_presence("room-1", "user-1", "offline")
        await asyncio.sleep(0.01)

        assert len(received) == 3
        assert received[0].type == EphemeralEventType.PRESENCE_ONLINE
        assert received[1].type == EphemeralEventType.PRESENCE_AWAY
        assert received[2].type == EphemeralEventType.PRESENCE_OFFLINE

        await kit.unsubscribe_room(sub_id)
        await kit.close()

    async def test_publish_presence_custom_status(self) -> None:
        kit = RoomKit()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await kit.subscribe_room("room-1", callback)

        await kit.publish_presence("room-1", "user-1", "busy")
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].type == EphemeralEventType.CUSTOM
        assert received[0].data["status"] == "busy"

        await kit.unsubscribe_room(sub_id)
        await kit.close()

    async def test_publish_read_receipt(self) -> None:
        kit = RoomKit()
        received: list[EphemeralEvent] = []

        async def callback(event: EphemeralEvent) -> None:
            received.append(event)

        sub_id = await kit.subscribe_room("room-1", callback)

        await kit.publish_read_receipt("room-1", "user-1", "evt-123")
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].type == EphemeralEventType.READ_RECEIPT
        assert received[0].user_id == "user-1"
        assert received[0].data["event_id"] == "evt-123"

        await kit.unsubscribe_room(sub_id)
        await kit.close()

    async def test_realtime_property(self) -> None:
        kit = RoomKit()
        assert isinstance(kit.realtime, RealtimeBackend)
        await kit.close()

    async def test_custom_realtime_backend(self) -> None:
        custom_backend = InMemoryRealtime(max_queue_size=50)
        kit = RoomKit(realtime=custom_backend)
        assert kit.realtime is custom_backend
        await kit.close()

    async def test_context_manager_closes_realtime(self) -> None:
        backend = InMemoryRealtime()

        async def callback(event: EphemeralEvent) -> None:
            pass

        async with RoomKit(realtime=backend) as kit:
            await kit.subscribe_room("room-1", callback)
            assert backend.subscription_count == 1

        # After context exit, subscriptions should be cleaned up
        assert backend.subscription_count == 0
