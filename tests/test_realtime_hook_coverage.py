"""Realtime provider callbacks reach the hooks the RFC maps them to (§12.5).

Two of the mappings in RFC Section 12.5's callback table were absent. `on_error`
reached a log line and nothing else, so a rate limit or a rejected turn — errors
that leave the session alive and running — were invisible to a host watching the
room. And `ON_REALTIME_TEXT_INJECTED` fired only where an inbound event drove the
injection, never for a caller reaching `inject_text()` directly, which is the
case an audit of "what text reached the model" exists for.
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


async def _channel() -> tuple[RoomKit, RealtimeVoiceChannel]:
    channel = RealtimeVoiceChannel(
        "rt-1", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt-1")
    return kit, channel


class TestProviderErrorsReachOnError:
    async def test_a_recoverable_error_is_announced(self) -> None:
        """The session survives it, which is exactly why nobody sees it."""
        kit, channel = await _channel()
        session = await channel.start_session("r1", "user-1", "fake-ws")

        seen: list[object] = []

        @kit.hook(HookTrigger.ON_ERROR, HookExecution.ASYNC)
        async def on_error(event, ctx) -> None:  # noqa: ANN001
            seen.append(event)

        channel._on_provider_error(session, "rate_limit_exceeded", "slow down")  # noqa: SLF001
        await asyncio.sleep(0.05)

        assert len(seen) == 1
        assert seen[0].metadata["error"] == "slow down"
        assert seen[0].metadata["error_type"] == "rate_limit_exceeded"
        assert seen[0].metadata["error_category"] == "realtime_provider"

    async def test_the_error_names_its_channel(self) -> None:
        kit, channel = await _channel()
        session = await channel.start_session("r1", "user-1", "fake-ws")

        seen: list[object] = []

        @kit.hook(HookTrigger.ON_ERROR, HookExecution.ASYNC)
        async def on_error(event, ctx) -> None:  # noqa: ANN001
            seen.append(event)

        channel._on_provider_error(session, "bad_request", "malformed turn")  # noqa: SLF001
        await asyncio.sleep(0.05)

        assert seen[0].source.channel_id == "rt-1"
        assert seen[0].room_id == "r1"


class TestTextInjectionIsAnnounced:
    async def test_a_direct_inject_text_fires_the_hook(self) -> None:
        kit, channel = await _channel()
        session = await channel.start_session("r1", "user-1", "fake-ws")

        seen: list[object] = []

        @kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, HookExecution.ASYNC)
        async def on_injected(event, ctx) -> None:  # noqa: ANN001
            seen.append(event)

        await channel.inject_text(session, "the order shipped", role="system")
        await asyncio.sleep(0.05)

        assert len(seen) == 1
        assert seen[0].content.body == "the order shipped"
        assert seen[0].metadata["injected_role"] == "system"

    async def test_the_injection_still_reaches_the_provider(self) -> None:
        """The announcement is added to the path, not put in front of it."""
        kit, channel = await _channel()
        session = await channel.start_session("r1", "user-1", "fake-ws")

        await channel.inject_text(session, "hello there")
        await asyncio.sleep(0.05)

        assert "hello there" in [t for _sid, t, _role in channel._provider.injected_texts]  # noqa: SLF001

    async def test_a_failing_hook_does_not_break_the_injection(self) -> None:
        kit, channel = await _channel()
        session = await channel.start_session("r1", "user-1", "fake-ws")

        @kit.hook(HookTrigger.ON_REALTIME_TEXT_INJECTED, HookExecution.ASYNC)
        async def boom(event, ctx) -> None:  # noqa: ANN001
            raise RuntimeError("hook is broken")

        await channel.inject_text(session, "still delivered")
        await asyncio.sleep(0.05)

        assert "still delivered" in [t for _sid, t, _role in channel._provider.injected_texts]  # noqa: SLF001
