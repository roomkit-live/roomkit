"""Queue acceptance, execution and retries preserve proactive delivery identity."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, Queued, RoomKit
from roomkit.core._reentrance import within_turn
from roomkit.core.delivery import _deliver_to_channel
from roomkit.delivery.base import DeliveryItem, DeliveryItemStatus
from roomkit.delivery.memory import InMemoryDeliveryBackend
from roomkit.delivery.worker import execute_delivery
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelType
from roomkit.models.event import TextContent
from tests.test_addressing import _room
from tests.test_framework import SimpleChannel


async def test_enqueue_replay_preserves_address_and_key() -> None:
    kit, _, agents = await _room("a", "b")
    backend = InMemoryDeliveryBackend(retry_delay=0)
    async with kit:
        # Set after the kit started, so the backend is not started:
        # acceptance alone must not solicit an agent, and this test is the
        # only one dequeuing.
        kit._delivery_backend = backend
        queued = await kit.deliver(
            "room-1",
            "external result",
            addressed_to=["a"],
            idempotency_key="external:1",
        )
        assert queued.status == "queued" and queued.event_id is None
        assert agents["a"].solicited == []
        [item] = await backend.dequeue("worker", timeout=0.1)
        restored = DeliveryItem.model_validate_json(item.model_dump_json())
        assert restored.id == queued.delivery_item_id
        assert restored.addressed_to == ["a"]
        assert restored.idempotency_key == "external:1"
        first = await execute_delivery(kit, restored)
        await backend.nack(item.id, error="ack connection lost")
        [retry] = await backend.dequeue("worker", timeout=0.1)
        second = await execute_delivery(kit, retry)
        await backend.ack(retry.id)
        assert first.status == second.status == "sent"
        assert first.event_id == second.event_id
        assert second.duplicate
        assert agents["a"].solicited == ["external result"]
        assert agents["b"].solicited == []


async def test_worker_retries_unavailable_destination_then_delivers() -> None:
    backend = InMemoryDeliveryBackend(retry_delay=0)
    async with RoomKit() as kit:
        await kit.create_room(room_id="r")
        item = DeliveryItem(room_id="r", content="result", channel_id="text", idempotency_key="k")
        await backend.enqueue(item)
        [first] = await backend.dequeue("worker", timeout=0.1)
        unavailable = await execute_delivery(kit, first)
        assert unavailable.status == "unavailable"
        await backend.nack(first.id, error=unavailable.reason)
        kit.register_channel(SimpleChannel("text"))
        await kit.attach_channel("r", "text")
        [retry] = await backend.dequeue("worker", timeout=0.1)
        assert retry.idempotency_key == "k" and retry.retry_count == 1
        assert retry.outcome.status == "unavailable"
        sent = await execute_delivery(kit, retry)
        await backend.ack(retry.id)
        assert sent.status == "sent" and not sent.duplicate
        assert retry.status == DeliveryItemStatus.DELIVERED


async def test_real_worker_dead_letters_missing_target_and_records_refusal() -> None:
    backend = InMemoryDeliveryBackend(retry_delay=0)
    async with RoomKit(delivery_backend=backend) as kit:
        await kit.create_room(room_id="r")
        item = DeliveryItem(room_id="r", content="result", max_retries=2)
        await backend.enqueue(item)
        await backend.start(kit)
        async with asyncio.timeout(2):
            while not await backend.get_dead_letter_items():
                await asyncio.sleep(0.01)
        [dead] = await backend.get_dead_letter_items()
        assert dead.outcome.status == "unavailable"
        # Two retries after the first attempt: three attempts failed.
        assert dead.retry_count == 3
        await backend.close()

        @kit.hook(HookTrigger.BEFORE_DELIVER)
        async def refuse(event, context):
            return HookResult.block("policy")

        blocked = DeliveryItem(room_id="r", content="refused")
        await backend.enqueue(blocked)
        await backend.start(kit)
        async with asyncio.timeout(2):
            while blocked.status != DeliveryItemStatus.BLOCKED:
                await asyncio.sleep(0.01)
        assert blocked.outcome.status == "blocked"
        assert blocked.outcome.reason == "policy"
        assert blocked.retry_count == 0


async def test_worker_after_hook_sees_rewrite_and_real_outcome() -> None:
    kit, _, agents = await _room("a")
    observed = []

    @kit.hook(HookTrigger.BEFORE_DELIVER)
    async def rewrite(event, context):
        return HookResult.modify(
            event.model_copy(update={"content": TextContent(body="redacted")})
        )

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def record(event, context):
        observed.append(event)

    async with kit:
        result = await execute_delivery(
            kit,
            DeliveryItem(
                room_id="room-1",
                content="secret",
                addressed_to=["a"],
                idempotency_key="k",
            ),
        )
        assert result.status == "sent"
        assert agents["a"].solicited == ["redacted"]
        await kit.hook_engine.drain_observers()
        assert observed[0].content.body == "redacted"
        assert observed[0].metadata["delivery_outcome"]["event_id"] == result.event_id
        assert observed[0].addressed_to == ["a"]


@pytest.mark.parametrize("keyed", [False, True])
async def test_queued_keeps_incompatible_targets_and_keyed_publications_separate(keyed) -> None:
    kit, human, agents = await _room("a", "b")
    human.channel_type = ChannelType.VOICE
    strategy = Queued(buffer=0)
    entered, release = asyncio.Event(), asyncio.Event()

    async def wait(*args):
        entered.set()
        await release.wait()

    async with kit:
        with patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=wait):
            first = asyncio.create_task(
                kit.deliver(
                    "room-1",
                    "first",
                    addressed_to=["a"],
                    strategy=strategy,
                    idempotency_key="k" if keyed else None,
                )
            )
            await entered.wait()
            others = [
                asyncio.create_task(
                    kit.deliver(
                        "room-1",
                        body,
                        addressed_to=[target],
                        strategy=strategy,
                        idempotency_key=key if keyed else None,
                    )
                )
                for body, target, key in [("first", "a", "k"), ("second", "b", "other")]
            ]
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, *others), 2)
        for result in results:
            assert result.status == "sent"
            if result.duplicate:
                # A replayed publication has no delivery of its own to follow
                # (RFC §22.1): its original's turn is the first call's.
                assert result.inbound.delivery is None
            else:
                await result.inbound.delivery.wait()
        assert agents["b"].solicited == ["second"]
        if keyed:
            assert agents["a"].solicited == ["first"]
            assert results[1].duplicate
        else:
            assert agents["a"].solicited == ["first\n\nfirst"]
        assert results[0].event_id == results[1].event_id
        assert results[2].event_id != results[0].event_id


async def test_a_queued_follower_is_delivered_at_its_own_depth() -> None:
    """RMK-983: one caller drains a shared Queued strategy for all; a
    follower's delivery runs as the follower made it, never inside the
    leader's work. An application's message at depth 0 is stored at 0,
    though the leader delivers from work 3 deep."""
    kit, _, _ = await _room("a")
    strategy = Queued(buffer=0)
    entered, release = asyncio.Event(), asyncio.Event()
    wait = _first_wait_held(entered, release)

    async def lead():
        with within_turn(3):
            return await kit.deliver("room-1", "leader", addressed_to=[], strategy=strategy)

    async with kit:
        with patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=wait):
            leader = asyncio.create_task(lead())
            await entered.wait()
            follower = asyncio.create_task(
                kit.deliver("room-1", "app", addressed_to=[], strategy=strategy)
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(asyncio.gather(leader, follower), 2)
        depths = {
            e.content.body: e.chain_depth
            for e in await kit.get_timeline("room-1")
            if isinstance(e.content, TextContent)
        }
    assert depths == {"leader": 3, "app": 0}


async def test_a_queue_drained_under_the_room_lock_lets_its_followers_deliver() -> None:
    """RMK-983: a locked BEFORE_BROADCAST hook delivers through a shared
    Queued strategy, and an application's request for the room, which its
    batch cannot take (another metadata), waits behind it. The hook delivers
    its own batch and hands the drain on: the application delivers once the
    hook's section is over, never waiting on a lock the hook holds."""
    kit, _, _ = await _room("a")
    strategy = Queued(buffer=0)
    entered, release = asyncio.Event(), asyncio.Event()
    wait = _first_wait_held(entered, release)
    got: dict[str, object] = {}

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="lead")
    async def lead(event, ctx):
        if isinstance(event.content, TextContent) and event.content.body == "go":
            got["hook"] = await kit.deliver("room-1", "hook", addressed_to=[], strategy=strategy)
        return HookResult.allow()

    async with kit:
        with patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=wait):
            go = asyncio.create_task(kit.process_inbound(_said("go")))
            await entered.wait()
            app = asyncio.create_task(
                kit.deliver(
                    "room-1", "app", addressed_to=[], strategy=strategy, metadata={"from": "app"}
                )
            )
            await asyncio.sleep(0)
            release.set()
            inbound, delivered = await asyncio.wait_for(asyncio.gather(go, app), 2)
    assert (inbound.blocked, got["hook"].status, delivered.status) == (False, "sent", "sent")


@pytest.mark.parametrize("into", ["its room", "another room"])
@pytest.mark.parametrize("drains", ["another caller", "the kit, the hook's own message"])
async def test_a_caller_holding_the_room_lock_delivers_past_a_drain_under_way(
    drains: str, into: str
) -> None:
    """RMK-997: a locked BEFORE_BROADCAST hook delivers, into its room or
    another one, through a shared Queued strategy while another caller
    drains it: an application whose idle wait is still running, or the
    kit-wide strategy delivering the very message whose pass runs the hook.
    Behind that batch, whose delivery needs the lock the hook holds, it
    waited on itself until ``process_timeout``; it now delivers at once,
    unbatched, and every delivery completes (RFC §9.5.1, §22.2)."""
    kit = RoomKit(process_timeout=0.5, delivery_strategy="queued")
    kit.register_channel(SimpleChannel("human"))
    for room_id in ("room-1", "room-2"):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "human")
    target = "room-1" if into == "its room" else "room-2"
    strategy = Queued(buffer=0) if drains == "another caller" else None
    entered, release = asyncio.Event(), asyncio.Event()
    hooked = asyncio.Event()
    got: dict[str, object] = {}

    @kit.hook(HookTrigger.BEFORE_BROADCAST, name="hooked")
    async def hook(event, ctx):
        if isinstance(event.content, TextContent) and event.content.body == "go":
            got["hook"] = await kit.deliver(
                target, "from the hook", addressed_to=[], strategy=strategy, metadata={"h": 1}
            )
            hooked.set()
        return HookResult.allow()

    async with kit:
        with patch(
            "roomkit.core.delivery._wait_for_voice_idle",
            side_effect=_first_wait_held(entered, release),
        ):
            if drains == "another caller":
                first = asyncio.create_task(
                    kit.deliver("room-1", "app", addressed_to=[], strategy=strategy)
                )
                await entered.wait()
                go = asyncio.create_task(kit.process_inbound(_said("go"), room_id="room-1"))
                await asyncio.wait_for(hooked.wait(), 1)  # the app's wait still runs
                release.set()
            else:
                release.set()
                first = asyncio.create_task(kit.deliver("room-1", "go", addressed_to=[]))
                go = None
            outcomes = await asyncio.wait_for(asyncio.gather(first, *[go] if go else []), 2)
        bodies = [
            e.content.body
            for room_id in ("room-1", "room-2")
            for e in await kit.get_timeline(room_id)
            if isinstance(e.content, TextContent)
        ]
    assert [getattr(o, "status", None) or o.blocked for o in outcomes] == (
        ["sent", False] if go else ["sent"]
    )
    assert (got["hook"].status, sorted(bodies)) == (
        "sent",
        sorted(["app", "go", "from the hook"] if go else ["go", "from the hook"]),
    )


async def test_a_queued_batch_is_delivered_once_its_idle_wait_times_out() -> None:
    """RFC §22.2: a wait for idle longer than ``playback_timeout`` delivers
    the batch anyway, as WaitForIdle's does (RMK-983)."""
    kit, _, _ = await _room("a")
    async with kit:
        with patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=TimeoutError):
            outcome = await kit.deliver(
                "room-1", "late", addressed_to=[], strategy=Queued(buffer=0)
            )
        bodies = [
            e.content.body
            for e in await kit.get_timeline("room-1")
            if isinstance(e.content, TextContent)
        ]
    assert (outcome.status, bodies) == ("sent", ["late"])


@pytest.mark.parametrize(
    ("cut", "expected"),
    [
        ("while it waits for idle", ("sent", None, ["second"])),
        ("while its batch is sent", ("unknown", "strategy_outcome_unknown", [])),
    ],
)
async def test_a_drain_cut_short_hands_the_queue_on(
    cut: str, expected: tuple[str, str | None, list[str]]
) -> None:
    """RFC §22.2: every waiting caller receives its own outcome. The caller
    draining a shared Queued strategy is cancelled; the caller behind it,
    which nothing cancelled, is not. Cut while it waits for idle, the drain
    passes to that caller and its message is delivered; cut while the batch
    carrying both messages is sent, that caller learns its outcome is
    unknown (RMK-983)."""
    kit, _, _ = await _room("a")
    strategy = Queued(buffer=0)
    entered, release, sending = asyncio.Event(), asyncio.Event(), asyncio.Event()
    wait = _first_wait_held(entered, release)

    async def send(ctx, channel_id):
        if ctx.content.startswith("first"):
            sending.set()
            await asyncio.Event().wait()
        return await _deliver_to_channel(ctx, channel_id)

    async with kit:
        with (
            patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=wait),
            patch("roomkit.core.delivery._deliver_to_channel", side_effect=send),
        ):
            first = asyncio.create_task(
                kit.deliver("room-1", "first", addressed_to=[], strategy=strategy)
            )
            await entered.wait()
            second = asyncio.create_task(
                kit.deliver("room-1", "second", addressed_to=[], strategy=strategy)
            )
            await asyncio.sleep(0)
            if cut == "while its batch is sent":
                release.set()
                await sending.wait()
            first.cancel()
            outcome = await asyncio.wait_for(second, 2)
        bodies = [
            e.content.body
            for e in await kit.get_timeline("room-1")
            if isinstance(e.content, TextContent)
        ]
    assert first.cancelled()
    assert (outcome.status, outcome.reason, bodies) == expected


def _first_wait_held(entered: asyncio.Event, release: asyncio.Event):
    """A voice-idle wait whose first call holds until *release*, *entered*
    set as it starts: a drain that waits, others queued behind it."""
    waits = 0

    async def wait(*args):
        nonlocal waits
        waits += 1
        if waits == 1:
            entered.set()
            await release.wait()

    return wait


def _said(body: str):
    return InboundMessage(channel_id="human", sender_id="u", content=TextContent(body=body))


async def test_queued_accepts_reentrant_delivery_from_agent_without_deadlocking() -> None:
    kit, _, agents = await _room("a")
    strategy = Queued(buffer=0)
    original = agents["a"].on_event

    async def followup(event, binding, context):
        if isinstance(event.content, TextContent) and event.content.body == "first":
            await kit.deliver("room-1", "second", addressed_to=[], strategy=strategy)
        return await original(event, binding, context)

    agents["a"].on_event = followup
    async with kit:
        result = await asyncio.wait_for(
            kit.deliver(
                "room-1",
                "first",
                addressed_to=["a"],
                strategy=strategy,
            ),
            2,
        )
        await asyncio.wait_for(result.inbound.delivery.wait(), 2)
        bodies = [
            e.content.body
            for e in await kit.get_timeline("room-1")
            if isinstance(e.content, TextContent)
        ]
        assert bodies == ["first", "second"]
