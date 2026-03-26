"""Tests for DeliveryItem model and DeliveryItemStatus."""

from __future__ import annotations

from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus


class TestDeliveryItemStatus:
    def test_values(self) -> None:
        assert DeliveryItemStatus.PENDING == "pending"
        assert DeliveryItemStatus.IN_PROGRESS == "in_progress"
        assert DeliveryItemStatus.DELIVERED == "delivered"
        assert DeliveryItemStatus.FAILED == "failed"
        assert DeliveryItemStatus.DEAD_LETTER == "dead_letter"


class TestDeliveryItem:
    def test_defaults(self) -> None:
        item = DeliveryItem(room_id="r1", content="hello")
        assert item.room_id == "r1"
        assert item.content == "hello"
        assert item.channel_id is None
        assert item.strategy == {"type": "immediate", "params": {}}
        assert item.metadata == {}
        assert item.retry_count == 0
        assert item.max_retries == 3
        assert item.status == DeliveryItemStatus.PENDING
        assert item.worker_id is None
        assert item.error is None
        assert item.id  # auto-generated
        assert item.created_at  # auto-generated

    def test_unique_ids(self) -> None:
        a = DeliveryItem(room_id="r1", content="a")
        b = DeliveryItem(room_id="r1", content="b")
        assert a.id != b.id

    def test_serialization_roundtrip(self) -> None:
        item = DeliveryItem(
            room_id="r1",
            content="hello",
            channel_id="ch1",
            strategy={"type": "wait_for_idle", "params": {"buffer": 2.0}},
            metadata={"key": "value"},
        )
        data = item.model_dump()
        restored = DeliveryItem.model_validate(data)
        assert restored.room_id == item.room_id
        assert restored.content == item.content
        assert restored.channel_id == item.channel_id
        assert restored.strategy == item.strategy
        assert restored.metadata == item.metadata

    def test_json_roundtrip(self) -> None:
        item = DeliveryItem(room_id="r1", content="test")
        json_str = item.model_dump_json()
        restored = DeliveryItem.model_validate_json(json_str)
        assert restored.id == item.id
        assert restored.room_id == item.room_id
