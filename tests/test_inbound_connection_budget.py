"""RMK-97 — the inbound path's connection budget, held to a number.

Every pooled connection an inbound message checks out costs a full extra round
trip: asyncpg resets a connection on release (``pg_advisory_unlock_all();
CLOSE ALL; UNLISTEN *; RESET ALL;``). On the measured bench those resets were
~36% of the SQL statements per message — more than any single query.

Why a checkout count and not a throughput number: it is **deterministic**. It
catches a +1 round trip per message the day it lands, where a throughput test
sees nothing under its own ±5% noise. Its blind spot is the mirror image — a
regression that keeps the count and makes each query more expensive (an O(room)
scan, a full row read for a yes/no); that one needs a query plan or a profile.

Requires a real PostgreSQL — the budget is about a pool, and ``InMemoryStore``
has none. Set POSTGRES_DSN to run it:

    POSTGRES_DSN=postgresql://user:pass@localhost/roomkit_test \
    pytest tests/test_inbound_connection_budget.py -v
"""

from __future__ import annotations

import os

import pytest

from roomkit.channels.base import Channel
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from tests.conftest import PoolCheckoutRecorder

POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(POSTGRES_DSN is None, reason="POSTGRES_DSN not set"),
    pytest.mark.asyncio,
]

# Checkouts a single inbound message may cost on the simplest possible room:
# one transport channel, no AI, no identity resolver, no idempotency key.
#
# Measured breakdown at the time of writing (8.00):
#   2  _build_context      — runs twice, unlocked then under the room lock;
#                            each pass groups its reads on one connection. The
#                            locked pass re-reads room/bindings/participants —
#                            the lock exists for that (RFC §10.1 steps 6 and
#                            12) — but carries the history the first pass
#                            deserialised (RMK-105): same checkout, one query
#                            fewer
#   2  lane delivery cursor — read outside the claim, read again under it
#   1  routing             — room_exists + binding_exists, grouped
#   1  the §7.5 source-binding read, after the sync hooks
#   1  the commit transaction
#   1  the cursor advance
#
# Raising this ceiling is a decision, not a merge conflict to resolve: say in
# the commit message which round trip was added and why it could not be grouped.
BUDGET_PER_MESSAGE = 8


class BudgetChannel(Channel):
    """A transport channel that reads no history and answers nothing."""

    channel_type = ChannelType.SMS

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


async def test_an_inbound_message_stays_within_its_connection_budget() -> None:
    from roomkit.store.postgres import PostgresStore

    store = PostgresStore(dsn=POSTGRES_DSN)
    await store.init(min_size=2, max_size=8)
    async with store._pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE rooms, events, bindings, participants, "
            "identities, identity_addresses, tasks, observations, read_markers CASCADE"
        )

    recorder = PoolCheckoutRecorder(store._pool)
    store._ensure_pool = lambda: recorder  # type: ignore[method-assign]

    kit = RoomKit(store=store)
    kit.register_channel(BudgetChannel("sms"))
    room = await kit.create_room(room_id="budget")
    await kit.attach_channel(room.id, "sms")

    async def send(count: int) -> None:
        for i in range(count):
            await kit.process_inbound(
                InboundMessage(
                    channel_id="sms",
                    sender_id="+15550000",
                    content=TextContent(body=f"msg {i}"),
                ),
                room_id=room.id,
            )

    messages = 10
    try:
        # Room creation and first attach are one-off costs, not per-message ones.
        await send(3)
        recorder.reset()

        await send(messages)
        # Read the tally before the teardown below, whose own store calls are
        # not part of what a message costs.
        per_message = recorder.total / messages
        breakdown = recorder.breakdown(per=messages)
    finally:
        await kit.close()
        await store.close()

    assert per_message <= BUDGET_PER_MESSAGE, (
        f"inbound now costs {per_message:.2f} pooled connections per message, "
        f"budget is {BUDGET_PER_MESSAGE}:\n{breakdown}"
    )
