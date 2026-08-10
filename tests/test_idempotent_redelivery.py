"""A redelivered message observes what the first delivery did (RFC §13.4).

"If seen → return the original result without reprocessing." The check honoured
the second half and answered the first with ``blocked=True, reason="duplicate"``
and no event — so a sender retrying because it never saw the first response
learned only that its message had been refused, which is the one thing that had
not happened.
"""

from __future__ import annotations

from roomkit import RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import TextContent
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore
from tests.test_framework import SimpleChannel


async def _kit(store=None) -> RoomKit:  # noqa: ANN001
    kit = RoomKit(store=store) if store is not None else RoomKit()
    kit.register_channel(SimpleChannel("ch-1", ChannelType.SMS))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ch-1")
    return kit


def _message(text: str = "hello", *, key: str | None = "k-1") -> InboundMessage:
    return InboundMessage(
        channel_id="ch-1",
        sender_id="user-1",
        content=TextContent(body=text),
        idempotency_key=key,
    )


class TestTheSecondDeliveryAnswersWithTheFirst:
    async def test_the_duplicate_returns_the_committed_event(self) -> None:
        kit = await _kit()

        first = await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert first.event is not None
        assert second.event is not None
        assert second.event.id == first.event.id
        assert second.event.index == first.event.index

    async def test_the_duplicate_does_not_read_as_a_refusal(self) -> None:
        kit = await _kit()

        await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert second.blocked is False
        assert second.reason is None

    async def test_nothing_is_reprocessed(self) -> None:
        """The MUST that was already honoured stays honoured."""
        kit = await _kit()

        await kit.process_inbound(_message())
        before = len(await kit.get_timeline("r1"))
        await kit.process_inbound(_message("a different body"))

        assert len(await kit.get_timeline("r1")) == before

    async def test_a_distinct_key_is_processed_normally(self) -> None:
        kit = await _kit()

        first = await kit.process_inbound(_message(key="k-1"))
        second = await kit.process_inbound(_message(key="k-2"))

        assert first.event is not None
        assert second.event is not None
        assert second.event.id != first.event.id

    async def test_no_key_means_no_idempotency(self) -> None:
        kit = await _kit()

        first = await kit.process_inbound(_message(key=None))
        second = await kit.process_inbound(_message(key=None))

        assert first.event is not None
        assert second.event is not None
        assert second.event.id != first.event.id


class TestAStoreThatCannotResolveTheKey:
    async def test_falls_back_to_the_blocked_result(self) -> None:
        """``get_event_by_idempotency_key`` defaults to ``None`` on the ABC, so a
        store written before it existed keeps the answer it always gave."""

        class _OlderStore(InMemoryStore):
            async def get_event_by_idempotency_key(self, room_id: str, key: str):  # noqa: ANN202
                return None

        kit = await _kit(store=_OlderStore())

        await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert second.blocked is True
        assert second.reason == "duplicate"
        assert second.event is None


class TestSQLite:
    async def test_the_duplicate_returns_the_committed_event(self, tmp_path) -> None:  # noqa: ANN001
        kit = await _kit(store=SQLiteStore(tmp_path / "rooms.db"))

        first = await kit.process_inbound(_message())
        second = await kit.process_inbound(_message())

        assert first.event is not None
        assert second.event is not None
        assert second.event.id == first.event.id


class TestTheStoreLookup:
    async def test_a_recorded_key_resolves_to_its_event(self) -> None:
        kit = await _kit()
        result = await kit.process_inbound(_message())
        assert result.event is not None

        found = await kit.store.get_event_by_idempotency_key("r1", "k-1")

        assert found is not None
        assert found.id == result.event.id

    async def test_an_unknown_key_resolves_to_nothing(self) -> None:
        kit = await _kit()
        await kit.process_inbound(_message())

        assert await kit.store.get_event_by_idempotency_key("r1", "never-sent") is None

    async def test_a_key_is_scoped_to_its_room(self) -> None:
        kit = await _kit()
        await kit.create_room(room_id="r2")
        await kit.process_inbound(_message())

        assert await kit.store.get_event_by_idempotency_key("r2", "k-1") is None
