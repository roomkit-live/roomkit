"""The public surfaces a host bills a realtime call from (RFC §12.4.2).

``on_usage`` fires on every report; ``VoiceSession.last_usage`` is the snapshot
the last report left behind. Neither asks the host to reach for a private
attribute, and neither lets one failing listener take the session down.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from roomkit.voice.base import VoiceSession
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import until


def _session(sid: str = "s1") -> VoiceSession:
    return VoiceSession(id=sid, room_id="r1", participant_id="p1", channel_id="ch1")


async def _settle() -> None:
    await asyncio.sleep(0.01)


class TestOnUsage:
    async def test_receives_the_totals_and_the_breakdown(self) -> None:
        provider = MockRealtimeProvider()
        session = _session()
        seen: list[dict[str, Any]] = []

        async def record(_s: VoiceSession, usage: dict[str, Any]) -> None:
            seen.append(usage)

        provider.on_usage(record)
        provider._record_usage(
            session,
            23520,
            44,
            details={"prompt_tokens_details": {"AUDIO": 3520, "TEXT": 20000}},
        )
        await _settle()

        assert seen == [
            {
                "input_tokens": 23520,
                "output_tokens": 44,
                "prompt_tokens_details": {"AUDIO": 3520, "TEXT": 20000},
            }
        ]

    async def test_sync_callbacks_are_served_too(self) -> None:
        provider = MockRealtimeProvider()
        session = _session()
        seen: list[dict[str, Any]] = []

        provider.on_usage(lambda _s, usage: seen.append(usage))
        provider._record_usage(session, 10, 2)
        await _settle()

        assert seen == [{"input_tokens": 10, "output_tokens": 2}]

    async def test_every_report_is_delivered(self) -> None:
        """A snapshot only keeps the last one; a host billing a call needs all."""
        provider = MockRealtimeProvider()
        session = _session()
        seen: list[dict[str, Any]] = []

        provider.on_usage(lambda _s, usage: seen.append(usage))
        provider._record_usage(session, 10, 2)
        provider._record_usage(session, 30, 4)
        await _settle()

        assert [u["input_tokens"] for u in seen] == [10, 30]
        assert session.last_usage["input_tokens"] == 30

    async def test_the_payload_is_the_host_s_own_copy(self) -> None:
        provider = MockRealtimeProvider()
        session = _session()
        seen: list[dict[str, Any]] = []

        provider.on_usage(lambda _s, usage: seen.append(usage))
        provider._record_usage(session, 10, 2)
        await _settle()

        seen[0]["input_tokens"] = 999
        assert session.last_usage["input_tokens"] == 10

    async def test_a_failing_listener_does_not_reach_the_others(self) -> None:
        provider = MockRealtimeProvider()
        session = _session()
        seen: list[dict[str, Any]] = []

        def explode(_s: VoiceSession, _usage: dict[str, Any]) -> None:
            raise RuntimeError("bookkeeping is down")

        provider.on_usage(explode)
        provider.on_usage(lambda _s, usage: seen.append(usage))
        provider._record_usage(session, 10, 2)
        await _settle()

        assert seen == [{"input_tokens": 10, "output_tokens": 2}]
        assert session.last_usage == {"input_tokens": 10, "output_tokens": 2}

    async def test_a_slow_listener_does_not_hold_the_recording_path(self) -> None:
        """Recording happens inside a provider's event handler; it returns now."""
        provider = MockRealtimeProvider()
        session = _session()
        released = asyncio.Event()

        async def slow(_s: VoiceSession, _usage: dict[str, Any]) -> None:
            await released.wait()

        provider.on_usage(slow)
        provider._record_usage(session, 10, 2)

        assert session.last_usage["input_tokens"] == 10  # already recorded
        released.set()
        await _settle()

    def test_recording_without_a_loop_still_lands_on_the_session(self) -> None:
        """No loop to schedule on: the snapshot is the surface that remains."""
        provider = MockRealtimeProvider()
        session = _session()
        provider.on_usage(lambda _s, _usage: None)

        provider._record_usage(session, 10, 2)

        assert session.last_usage == {"input_tokens": 10, "output_tokens": 2}


class TestLastUsage:
    def test_is_empty_until_the_first_report(self) -> None:
        assert _session().last_usage == {}

    def test_is_a_snapshot_the_caller_cannot_write_through(self) -> None:
        provider = MockRealtimeProvider()
        session = _session()
        provider._record_usage(session, 10, 2)

        snapshot = session.last_usage
        snapshot["input_tokens"] = 999
        snapshot["invented"] = True

        assert session.last_usage == {"input_tokens": 10, "output_tokens": 2}

    def test_carries_the_provider_s_own_keys(self) -> None:
        provider = MockRealtimeProvider()
        session = _session()
        provider._record_usage(session, 10, 2, details={"cached_content_token_count": 12000})

        assert session.last_usage["cached_content_token_count"] == 12000
        with pytest.raises(KeyError):  # absent means unreported, never zero
            session.last_usage["thoughts_token_count"]


@pytest.mark.parametrize("traced", [False, True], ids=["no telemetry", "telemetry"])
async def test_the_channel_clears_the_snapshot_at_each_turn_s_end(traced: bool) -> None:
    """The realtime channel clears ``last_usage`` when a turn ends, whether or
    not a telemetry provider traces the turn (RFC §12.4.2, RMK-2055): it was
    cleared only where the turn had a span, so turning telemetry on changed
    what a host polling the snapshot read."""
    telemetry = MockTelemetryProvider()
    kit = RoomKit(telemetry=telemetry) if traced else RoomKit()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel("rt", provider=provider, transport=MockRealtimeTransport())
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "p1", "ws")

    await provider.simulate_response_start(session)
    await provider.simulate_usage(session, 10, 5)
    during = session.last_usage
    await provider.simulate_response_end(session)
    try:
        await until(lambda: session.last_usage == {}, timeout=1.0)
    finally:
        await kit.close()

    assert during == {"input_tokens": 10, "output_tokens": 5}
    # The turn's span still reads the snapshot before it is cleared.
    turns = telemetry.get_spans(SpanKind.REALTIME_TURN)
    assert [span.attributes.get(Attr.LLM_INPUT_TOKENS) for span in turns] == (
        [10] if traced else []
    )
