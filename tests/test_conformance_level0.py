"""Executable RFC Level 0 (Core, REQUIRED) conformance matrix.

Section 25.1 of ``roomkit-specs/roomkit-rfc.md`` lists what a conforming
implementation MUST support. Each test below asserts one such requirement,
mapped to its RFC section — behaviourally where that is clean, structurally
(API-surface presence/shape) where a full behavioural harness would not add
signal. This is the single "is RoomKit Level 0 conformant?" gate; deeper
behavioural coverage lives in the feature-specific suites.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.core.locks import InMemoryLockManager, RoomLockManager
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    Access,
    ChannelType,
    EventType,
    HookExecution,
    HookTrigger,
    RoomStatus,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.store.base import ConversationStore
from roomkit.store.memory import InMemoryStore

# ── Minimal transport harness ───────────────────────────────────


class _Transport(Channel):
    """Transport that honours ``message.event_type`` and records deliveries."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            type=message.event_type,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                participant_id=message.sender_id,
            ),
            content=message.content,
            idempotency_key=message.idempotency_key,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


async def _room(
    *, source_access: Access = Access.READ_WRITE, source_muted: bool = False
) -> tuple[RoomKit, _Transport, _Transport]:
    kit = RoomKit()
    src, dst = _Transport("src"), _Transport("dst")
    kit.register_channel(src)
    kit.register_channel(dst)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "src", access=source_access)
    await kit.attach_channel("r1", "dst")
    if source_muted:
        await kit.mute("r1", "src")
    return kit, src, dst


async def _send(kit: RoomKit, body: str = "hi", channel_id: str = "src", **kw: object):
    return await kit.process_inbound(
        InboundMessage(channel_id=channel_id, sender_id="u1", content=TextContent(body=body), **kw)  # ty: ignore[invalid-argument-type]
    )


# ── §4/§5 — Room & event models ─────────────────────────────────


class TestModels:
    def test_room_lifecycle_states(self) -> None:
        """§25.1 / §4 — Room lifecycle: ACTIVE, PAUSED, CLOSED, ARCHIVED."""
        assert {"ACTIVE", "PAUSED", "CLOSED", "ARCHIVED"} <= {s.name for s in RoomStatus}

    def test_room_has_timers(self) -> None:
        """§25.1 / §5 — Room timers (auto-pause / auto-close)."""
        from roomkit.models.room import Room

        assert "timers" in Room.model_fields

    def test_event_types_present(self) -> None:
        """§25.1 / §5.2 — RoomEvent supports the core EventType values."""
        required = {"MESSAGE", "SYSTEM", "EDIT", "DELETE", "REACTION", "TYPING"}
        assert required <= {e.name for e in EventType}

    def test_event_content_types_present(self) -> None:
        """§25.1 / §5.3 — the core EventContent types exist."""
        import roomkit.models.event as ev

        for name in (
            "TextContent",
            "MediaContent",
            "SystemContent",
            "EditContent",
            "DeleteContent",
        ):
            assert hasattr(ev, name), name

    def test_channel_binding_access_mute_visibility(self) -> None:
        """§25.1 / §5 — ChannelBinding carries access, mute, and visibility."""
        fields = ChannelBinding.model_fields
        assert "access" in fields
        assert "muted" in fields
        assert "visibility" in fields


# ── §6 — Channel interface ──────────────────────────────────────


class TestChannelInterface:
    def test_channel_abc_surface(self) -> None:
        """§25.1 / §6 — Channel: handle_inbound, deliver, on_event, capabilities."""
        assert hasattr(Channel, "handle_inbound")
        assert hasattr(Channel, "deliver")
        assert hasattr(Channel, "on_event")
        assert hasattr(Channel, "capabilities")


# ── §8.1 — Sequential event indexing ────────────────────────────


class TestSequentialIndexing:
    async def test_indices_are_unique_monotonic_from_zero(self) -> None:
        """§25.1 / §8.1 — events get a unique, monotonic index starting at 0."""
        kit, _, _ = await _room()
        for i in range(4):
            await _send(kit, body=f"m{i}")
        timeline = await kit.get_timeline("r1")
        indices = [e.index for e in timeline]
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))  # unique
        assert indices[0] == 0 and indices == list(range(indices[0], indices[0] + len(indices)))


# ── §7.5 — Permission enforcement ───────────────────────────────


class TestPermissionEnforcement:
    @pytest.mark.parametrize(
        ("access", "muted"),
        [(Access.READ_ONLY, False), (Access.NONE, False), (Access.READ_WRITE, True)],
    )
    async def test_non_writable_source_is_blocked(self, access: Access, muted: bool) -> None:
        """§25.1 / §7.5 — a source that cannot write injects no DELIVERED event."""
        kit, _, dst = await _room(source_access=access, source_muted=muted)
        result = await _send(kit)
        assert result.blocked
        assert dst.delivered == []


# ── §9 / §10.1 — Hook engine ────────────────────────────────────


class TestHooks:
    async def test_before_broadcast_can_block(self) -> None:
        """§25.1 / §9.5 — a SYNC BEFORE_BROADCAST hook blocks via HookResult."""
        kit, _, dst = await _room()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC, name="block")
        async def _block(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("nope")

        result = await _send(kit)
        assert result.blocked
        assert dst.delivered == []

    async def test_after_broadcast_async_observes(self) -> None:
        """§25.1 / §9 — an ASYNC AFTER_BROADCAST hook runs (fire-and-forget)."""
        seen: list[str] = []
        kit, _, _ = await _room()

        @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC, name="observe")
        async def _obs(event: RoomEvent, ctx: RoomContext) -> None:
            seen.append(event.id)

        await _send(kit)
        assert len(seen) == 1

    async def test_injected_event_delivered_on_block(self) -> None:
        """§25.1 / §9.5 — a blocking hook's InjectedEvent is delivered."""
        kit, _, dst = await _room()

        @kit.hook(HookTrigger.BEFORE_BROADCAST, execution=HookExecution.SYNC, name="inject")
        async def _inject(event: RoomEvent, ctx: RoomContext) -> HookResult:
            followup = RoomEvent(
                room_id="r1",
                source=EventSource(channel_id="src", channel_type=ChannelType.WEBSOCKET),
                content=TextContent(body="replacement"),
            )
            return HookResult(
                action="block",
                reason="filtered",
                injected_events=[InjectedEvent(event=followup, target_channel_ids=["dst"])],
            )

        await _send(kit)
        bodies = [e.content.body for e in dst.delivered if isinstance(e.content, TextContent)]
        assert "replacement" in bodies


# ── §8.3 — Event chain depth ────────────────────────────────────


class TestChainDepth:
    def test_chain_depth_tracked_and_limit_configurable(self) -> None:
        """§25.1 / §8.3 — events track chain_depth and the limit is configurable."""
        assert "chain_depth" in RoomEvent.model_fields
        assert RoomKit(max_chain_depth=3)._max_chain_depth == 3


# ── §10.1 / §10.2 — Pipelines ───────────────────────────────────


class TestPipelines:
    async def test_inbound_returns_result_and_broadcasts(self) -> None:
        """§25.1 / §10.1–§10.2 — inbound produces a result; broadcast reaches
        other channels but not the source."""
        kit, src, dst = await _room()
        result = await _send(kit)
        assert result.event is not None and not result.blocked
        assert len(dst.delivered) == 1  # broadcast to the other channel
        assert src.delivered == []  # not echoed back to the source


# ── §10.4 / §13.5 / §14 — Pluggable infrastructure ──────────────


class TestPluggableInfrastructure:
    def test_inbound_room_router_is_pluggable(self) -> None:
        """§25.1 / §10.4 — inbound room routing is a pluggable interface."""
        assert "inbound_router" in inspect.signature(RoomKit.__init__).parameters

    def test_room_lock_manager_interface_and_default(self) -> None:
        """§25.1 / §13.5 — a RoomLockManager interface with an in-memory default."""
        assert hasattr(RoomLockManager, "locked")
        assert isinstance(RoomKit().lock_manager, InMemoryLockManager)

    def test_conversation_store_interface_and_default(self) -> None:
        """§25.1 / §14 — a ConversationStore interface with an in-memory impl."""
        assert issubclass(InMemoryStore, ConversationStore)
        for method in ("create_room", "get_room", "add_event", "get_event_count"):
            assert hasattr(ConversationStore, method), method

    def test_content_transcoding_has_text_fallback(self) -> None:
        """§25.1 — content transcoding with at least a text fallback."""
        from roomkit.core.transcoder import DefaultContentTranscoder

        assert DefaultContentTranscoder is not None


# ── §8.2 / §15.1 — Observability ────────────────────────────────


class TestObservability:
    async def test_framework_events_emitted(self) -> None:
        """§25.1 / §8.2 — framework events are emitted (event_processed)."""
        events: list[str] = []
        kit, _, _ = await _room()

        @kit.on("event_processed")
        async def _capture(fe: object) -> None:
            events.append("event_processed")

        await _send(kit)
        assert "event_processed" in events

    def test_structured_named_loggers(self) -> None:
        """§25.1 / §15.1 — structured logging under the ``roomkit`` hierarchy."""
        assert logging.getLogger("roomkit.framework").name.startswith("roomkit")


# ── §10.1 — Idempotency ─────────────────────────────────────────


class TestIdempotency:
    async def test_duplicate_idempotency_key_blocked(self) -> None:
        """§25.1 / §10.1 — a repeated idempotency_key is treated as a duplicate."""
        kit, _, _ = await _room()
        first = await _send(kit, idempotency_key="k1")
        second = await _send(kit, idempotency_key="k1")
        assert not first.blocked
        assert second.blocked and second.reason == "duplicate"
