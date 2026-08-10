"""The §8.2 framework events and §9.2 triggers that were declared but never raised.

RFC §8.2 line 1535: "Implementations MUST emit these events", and §9.2
lists every trigger it declares Implemented. These lock in the emission sites
and firing sites for the ones this framework raises: an event nobody emits and
a trigger nothing fires are both invisible to a host.
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.circuit_breaker import CircuitBreaker
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import RoomStatus
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.models.hook import HookResult
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


def _collect(kit: RoomKit, *types: str) -> list[FrameworkEvent]:
    seen: list[FrameworkEvent] = []
    for event_type in types:

        @kit.on(event_type)
        async def capture(fe: FrameworkEvent) -> None:
            seen.append(fe)

    return seen


class TestChannelRegistryEvents:
    async def test_register_and_unregister_are_observable(self) -> None:
        kit = RoomKit()
        seen = _collect(kit, "channel_registered", "channel_unregistered")

        kit.register_channel(SimpleChannel("sms1"))
        kit.unregister_channel("sms1")
        await asyncio.sleep(0)  # emission is scheduled from sync API
        await asyncio.sleep(0)

        assert [fe.type for fe in seen] == ["channel_registered", "channel_unregistered"]
        assert seen[0].channel_id == "sms1"

    async def test_unknown_channel_emits_nothing(self) -> None:
        kit = RoomKit()
        seen = _collect(kit, "channel_unregistered")
        assert kit.unregister_channel("nope") is None
        await asyncio.sleep(0)
        assert seen == []


class TestCircuitBreakerEvents:
    """RFC §13.1 MUST — "When a circuit breaker opens, implementations MUST
    emit a ``circuit_breaker_opened`` framework event"."""

    def test_breaker_reports_open_and_close_once_per_edge(self) -> None:
        states: list[str] = []
        breaker = CircuitBreaker(failure_threshold=2, on_state_change=states.append)

        breaker.record_failure()
        assert states == []
        breaker.record_failure()
        assert states == ["open"]
        # Still open — no repeat on every subsequent failure.
        breaker.record_failure()
        assert states == ["open"]
        breaker.record_success()
        assert states == ["open", "closed"]
        # Already closed — a success is not an edge.
        breaker.record_success()
        assert states == ["open", "closed"]


class TestHookTimeoutEvent:
    async def test_timed_out_hook_emits_hook_timeout(self) -> None:
        """A timeout is its own condition, distinct from a hook that
        raised."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        seen = _collect(kit, "hook_timeout")

        @kit.hook(HookTrigger.BEFORE_BROADCAST, name="slowpoke", timeout=0.01)
        async def slow(event: RoomEvent, ctx: RoomContext) -> HookResult:
            await asyncio.sleep(1.0)
            return HookResult.allow()

        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi"))
        )

        assert len(seen) == 1
        assert seen[0].data["hook_name"] == "slowpoke"
        assert seen[0].data["timeout"] == 0.01


class TestIdentityResolvedEvent:
    async def test_resolved_identity_is_observable(self) -> None:
        from roomkit.identity.mock import MockIdentityResolver
        from roomkit.models.identity import Identity
        from roomkit.models.identity import Identity as _Identity

        kit = RoomKit(
            identity_resolver=MockIdentityResolver(
                mapping={"user1": _Identity(id="ident-1", display_name="Alice")}
            )
        )
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        seen = _collect(kit, "identity_resolved")

        identity = Identity(id="ident-1", display_name="Alice")
        await kit.store.create_identity(identity)

        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="user1", content=TextContent(body="hi")),
            room_id="r1",
        )

        assert len(seen) == 1
        assert seen[0].data["identity_id"] == "ident-1"


class TestRoomArchival:
    async def test_archive_room_reaches_the_status_and_emits(self) -> None:
        """RFC §5.1 / §8.2 — ``archive_room`` is the one path to the status,
        and the source of the ``room_archived`` event."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        seen = _collect(kit, "room_archived")

        room = await kit.archive_room("r1")
        assert room.status == RoomStatus.ARCHIVED
        assert len(seen) == 1

        # Terminal: it refuses new events like CLOSED does, and reads still work.
        result = await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi")),
            room_id="r1",
        )
        assert result.blocked
        assert await kit.get_timeline("r1") is not None

        # Idempotent.
        await kit.archive_room("r1")
        assert len(seen) == 1


class TestTriggersThatReachHooks:
    """RFC §9.2 — a trigger the enum declares must reach the room's hooks.

    Publishing only an ephemeral or framework event satisfies an observer
    watching the stream and nobody who registered a hook, which is the surface
    the enum advertises.
    """

    async def test_on_ai_thinking_fires(self) -> None:
        """RFC §9.2 — reasoning reaches hooks, not only ephemeral events."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        provider = MockAIProvider(
            ai_responses=[AIResponse(content="done", thinking="weighing the options")]
        )
        kit.register_channel(AIChannel("ai1", provider=provider))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")
        await kit.attach_channel("r1", "ai1")

        thinking: list[object] = []

        @kit.hook(HookTrigger.ON_AI_THINKING, HookExecution.ASYNC)
        async def on_thinking(event, context):  # noqa: ANN001
            thinking.append(event)

        await kit.process_inbound(
            InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi"))
        )
        await asyncio.sleep(0.05)

        assert len(thinking) == 1
        assert thinking[0].room_id == "r1"
        assert thinking[0].channel_id == "ai1"

    async def test_on_status_posted_fires_for_room_scoped_status(self) -> None:
        """RFC §9.2 — a room-scoped status reaches the room's hooks, not
        only the ``status_posted`` framework event."""
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms1"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "sms1")

        posted: list[object] = []

        @kit.hook(HookTrigger.ON_STATUS_POSTED, HookExecution.ASYNC)
        async def on_status(event, context):  # noqa: ANN001
            posted.append(event)

        # The framework subscribes to the bus lazily, on first room activity.
        await kit._ensure_status_bus_subscribed()
        await kit.status_bus.post_async(
            agent_id="agent-1",
            action="working",
            status=StatusLevel.PENDING,
            detail="drafting the answer",
            metadata={"room_id": "r1"},
        )
        await asyncio.sleep(0.05)

        assert len(posted) == 1
        assert posted[0].agent_id == "agent-1"
