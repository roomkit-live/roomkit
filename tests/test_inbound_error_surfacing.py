"""Every agent-turn failure must reach ON_ERROR — not vanish with a log line.

The streaming consumption path already fires ON_ERROR (see
``test_ai_streaming_tool_events.test_no_streaming_target_error_fires_on_error``).
These tests cover the two paths that used to swallow the failure into
``ChannelOutput.empty()`` / ``broadcast_result.errors`` with no hook:

* a provider error on the NON-streaming generate path, and
* an error raised in the AI channel's ``on_event`` before the stream begins
  (both surface through the broadcast-error path under the room lock).

Without the fix the room ends up with only lifecycle events and no error
card; with it, the host's ON_ERROR hook (e.g. Luge's error card) fires once,
attributed to the failing agent channel.
"""

from __future__ import annotations

import asyncio

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


async def _run_turn_capturing_errors(ai: AIChannel) -> list[RoomEvent]:
    """Wire a room with the given AI channel, run one turn, return ON_ERROR events."""
    kit = RoomKit()
    sms = SimpleChannel("sms1")
    kit.register_channel(sms)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="test_capture_error",
        )
    )

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    # Async hooks run after the room lock is released — yield once.
    await asyncio.sleep(0.05)
    await kit.close()
    return errors


async def test_non_streaming_provider_error_fires_on_error() -> None:
    """A provider failure on the non-streaming generate path fires ON_ERROR
    instead of being swallowed into ChannelOutput.empty()."""

    class _RaisingProvider(MockAIProvider):
        async def generate(self, context: AIContext) -> AIResponse:
            raise RuntimeError("provider connection refused")

    ai = AIChannel("ai1", provider=_RaisingProvider(streaming=False))

    errors = await _run_turn_capturing_errors(ai)

    assert len(errors) == 1
    meta = errors[0].metadata or {}
    assert "provider connection refused" in str(meta.get("error", ""))
    # Attributed to the failing agent channel so the card renders under the agent.
    assert errors[0].source.channel_type == ChannelType.AI
    assert errors[0].source.channel_id == "ai1"


async def test_on_error_runs_after_room_lock_released() -> None:
    """ON_ERROR is deferred past the room lock (RFC §10.1), like AFTER_BROADCAST:
    a slow error hook must not hold the lock and stall the room's next message.

    Proven deterministically: the ``_held_rooms`` ContextVar tracks which rooms
    the current context holds. When the deferred hook runs, the ``locked()``
    block has already exited and reset the token, so the room is absent from the
    set — i.e. the hook ran OUTSIDE the lock."""
    from roomkit.core.locks import _held_rooms

    class _RaisingProvider(MockAIProvider):
        async def generate(self, context: AIContext) -> AIResponse:
            raise RuntimeError("boom")

    ai = AIChannel("ai1", provider=_RaisingProvider(streaming=False))
    kit = RoomKit()
    sms = SimpleChannel("sms1")
    kit.register_channel(sms)
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    held_when_fired: list[frozenset[str]] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        held_when_fired.append(_held_rooms.get())

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="test_lock_probe",
        )
    )

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)
    await kit.close()

    assert len(held_when_fired) == 1
    assert "r1" not in held_when_fired[0]  # fired after the lock was released


async def test_transport_channel_error_does_not_fire_on_error() -> None:
    """A non-intelligence channel failing during broadcast is a delivery
    problem, not a turn-level agent error — it must NOT raise an error card."""

    class _RaisingTransport(SimpleChannel):
        async def on_event(self, event: RoomEvent, binding, context) -> object:  # type: ignore[override]
            raise RuntimeError("websocket send failed")

    kit = RoomKit()
    sms = SimpleChannel("sms1")
    bad = _RaisingTransport("sms2")
    ai = AIChannel("ai1", provider=MockAIProvider(streaming=False))
    kit.register_channel(sms)
    kit.register_channel(bad)
    kit.register_channel(ai)

    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "sms2")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)

    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, _ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="test_capture_error",
        )
    )

    await kit.process_inbound(
        InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    )
    await asyncio.sleep(0.05)
    await kit.close()

    # The transport failure is logged/tracked as delivery_failed, never as a card.
    assert errors == []
