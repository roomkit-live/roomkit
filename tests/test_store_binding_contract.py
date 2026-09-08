"""Binding serialization contract shared by all conversation stores."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from roomkit.models.channel import ChannelBinding, RateLimit, RetryPolicy
from roomkit.models.room import Room
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore
from roomkit.store.sqlite import SQLiteStore


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def contract_store(request: pytest.FixtureRequest) -> AsyncIterator[ConversationStore]:
    store: ConversationStore
    if request.param == "memory":
        store = InMemoryStore()
    elif request.param == "sqlite":
        store = SQLiteStore(":memory:")
    else:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            pytest.skip("POSTGRES_DSN not set")
        from roomkit.store.postgres import PostgresStore

        store = PostgresStore(dsn=dsn)
        await store.init(min_size=1, max_size=2)
    try:
        yield store
    finally:
        await store.close()


async def test_binding_round_trip_and_policy_updates(contract_store: ConversationStore) -> None:
    room = await contract_store.create_room(Room(id=uuid4().hex))
    binding = ChannelBinding(
        room_id=room.id,
        channel_id="sms",
        channel_type="sms",
        access="read_only",
        muted=True,
        output_muted=True,
        visibility="private",
        participant_id="alice",
        last_read_index=4,
        metadata={"nested": {"key": [1]}},
        rate_limit=RateLimit(max_per_minute=30),
        retry_policy=RetryPolicy(max_retries=7, base_delay_seconds=0.3),
    )
    try:
        await contract_store.add_binding(binding)
        assert await contract_store.get_binding(room.id, "sms") == binding
        assert await contract_store.list_bindings(room.id) == [binding]
        changed = binding.model_copy(update={"rate_limit": RateLimit(max_per_second=2)})
        await contract_store.add_binding(changed)
        assert await contract_store.get_binding(room.id, "sms") == changed
        changed = changed.model_copy(update={"retry_policy": RetryPolicy(max_retries=2)})
        await contract_store.update_binding(changed)
        assert await contract_store.get_binding(room.id, "sms") == changed
        cleared = changed.model_copy(update={"rate_limit": None, "retry_policy": None})
        await contract_store.update_binding(cleared)
        assert await contract_store.get_binding(room.id, "sms") == cleared
    finally:
        await contract_store.delete_room(room.id)


@pytest.mark.parametrize("newest_first", [False, True])
@pytest.mark.parametrize("cursor", ["after", "before", "offset"])
async def test_pagination_order_contract(
    contract_store: ConversationStore, newest_first: bool, cursor: str
) -> None:
    from tests.conftest import make_event

    room = await contract_store.create_room(Room(id=uuid4().hex))
    try:
        for i in range(6):
            await contract_store.add_event(make_event(room_id=room.id, index=i, body=str(i)))
        if cursor == "after":
            page = await contract_store.list_events(
                room.id, after_index=1, newest_first=newest_first, limit=2, offset=99
            )
            expected = [2, 3]
        elif cursor == "before":
            page = await contract_store.list_events(
                room.id, before_index=4, newest_first=newest_first, limit=2, offset=99
            )
            expected = [2, 3]
        else:
            page = await contract_store.list_events(
                room.id, newest_first=newest_first, limit=2, offset=1
            )
            expected = [3, 4] if newest_first else [1, 2]
        assert [event.index for event in page] == expected
    finally:
        await contract_store.delete_room(room.id)
