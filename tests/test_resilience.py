"""Tests for resilience: circuit breaker, rate limiter, retry, transcoding, and timers."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from roomkit.channels.base import Channel
from roomkit.core.circuit_breaker import CircuitBreaker
from roomkit.core.event_router import EventRouter
from roomkit.core.rate_limiter import TokenBucketRateLimiter
from roomkit.core.retry import retry_with_backoff
from roomkit.core.transcoder import DefaultContentTranscoder
from roomkit.models.channel import (
    ChannelBinding,
    ChannelCapabilities,
    ChannelOutput,
    RateLimit,
    RetryPolicy,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelMediaType,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    RoomStatus,
)
from roomkit.models.event import (
    AudioContent,
    ChannelData,
    CompositeContent,
    EventSource,
    RoomEvent,
    TextContent,
    VideoContent,
)
from roomkit.models.room import Room
from roomkit.models.task import Observation, Task
from roomkit.providers.ai.base import AIResponse
from tests.conftest import make_binding, make_event

# -- Helpers --


class StubChannel(Channel):
    channel_type = ChannelType.WEBSOCKET
    category = ChannelCategory.TRANSPORT

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.delivered: list[RoomEvent] = []

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered.append(event)
        return ChannelOutput.empty()


class FailingChannel(Channel):
    """Channel that raises on deliver."""

    channel_type = ChannelType.WEBSOCKET
    category = ChannelCategory.TRANSPORT

    def __init__(self, channel_id: str, fail_count: int = 999) -> None:
        super().__init__(channel_id)
        self._fail_count = fail_count
        self._attempt = 0

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self._attempt += 1
        if self._attempt <= self._fail_count:
            raise RuntimeError(f"fail #{self._attempt}")
        return ChannelOutput.empty()


def _make_context(bindings: list[ChannelBinding]) -> RoomContext:
    return RoomContext(room=Room(id="test-room"), bindings=bindings)


# ============================================================================
# D1: Content Transcoding
# ============================================================================


class TestAudioTranscoding:
    async def test_audio_passthrough_when_supported(self) -> None:
        transcoder = DefaultContentTranscoder()
        content = AudioContent(url="https://a.mp3", transcript="hello")
        source = make_binding("ch1", media_types=[ChannelMediaType.AUDIO])
        target = make_binding("ch2", media_types=[ChannelMediaType.AUDIO])
        result = await transcoder.transcode(content, source, target)
        assert isinstance(result, AudioContent)

    async def test_audio_with_transcript_transcodes_to_text(self) -> None:
        transcoder = DefaultContentTranscoder()
        content = AudioContent(url="https://a.mp3", transcript="hello world")
        source = make_binding("ch1", media_types=[ChannelMediaType.AUDIO])
        target = make_binding("ch2", media_types=[ChannelMediaType.TEXT])
        result = await transcoder.transcode(content, source, target)
        assert isinstance(result, TextContent)
        assert result.body == "hello world"

    async def test_audio_without_transcript_uses_url(self) -> None:
        transcoder = DefaultContentTranscoder()
        content = AudioContent(url="https://a.mp3")
        source = make_binding("ch1", media_types=[ChannelMediaType.AUDIO])
        target = make_binding("ch2", media_types=[ChannelMediaType.TEXT])
        result = await transcoder.transcode(content, source, target)
        assert isinstance(result, TextContent)
        assert "https://a.mp3" in result.body


class TestVideoTranscoding:
    async def test_video_passthrough_when_supported(self) -> None:
        transcoder = DefaultContentTranscoder()
        content = VideoContent(url="https://v.mp4")
        source = make_binding("ch1", media_types=[ChannelMediaType.VIDEO])
        target = make_binding("ch2", media_types=[ChannelMediaType.VIDEO])
        result = await transcoder.transcode(content, source, target)
        assert isinstance(result, VideoContent)

    async def test_video_transcodes_to_text(self) -> None:
        transcoder = DefaultContentTranscoder()
        content = VideoContent(url="https://v.mp4")
        source = make_binding("ch1", media_types=[ChannelMediaType.VIDEO])
        target = make_binding("ch2", media_types=[ChannelMediaType.TEXT])
        result = await transcoder.transcode(content, source, target)
        assert isinstance(result, TextContent)
        assert "https://v.mp4" in result.body


class TestCompositeTranscoding:
    async def test_composite_flattens_to_text(self) -> None:
        transcoder = DefaultContentTranscoder()
        content = CompositeContent(parts=[TextContent(body="part1"), TextContent(body="part2")])
        source = make_binding("ch1", media_types=[ChannelMediaType.TEXT])
        target = make_binding("ch2", media_types=[ChannelMediaType.TEXT])
        result = await transcoder.transcode(content, source, target)
        assert isinstance(result, TextContent)
        assert "part1" in result.body
        assert "part2" in result.body

    async def test_composite_with_audio_parts(self) -> None:
        transcoder = DefaultContentTranscoder()
        content = CompositeContent(
            parts=[
                TextContent(body="text part"),
                AudioContent(url="https://a.mp3", transcript="audio text"),
            ]
        )
        source = make_binding("ch1", media_types=[ChannelMediaType.TEXT, ChannelMediaType.AUDIO])
        target = make_binding("ch2", media_types=[ChannelMediaType.TEXT])
        result = await transcoder.transcode(content, source, target)
        # All parts become text, so result should be flattened TextContent
        assert isinstance(result, TextContent)
        assert "text part" in result.body
        assert "audio text" in result.body

    async def test_composite_empty_parts_returns_none(self) -> None:
        """Composite with no transcodable parts returns None.

        Uses model_construct to bypass the non-empty validation so we can
        test the transcoder's handling of an empty-parts edge case.
        """
        transcoder = DefaultContentTranscoder()
        content = CompositeContent.model_construct(type="composite", parts=[])
        source = make_binding("ch1")
        target = make_binding("ch2")
        result = await transcoder.transcode(content, source, target)
        assert result is None


class TestTranscodingFailureInRouter:
    async def test_transcode_failure_skips_delivery(self) -> None:
        """When transcoding returns None, delivery is skipped and error recorded."""
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = EventRouter(channels={"ch1": ch1, "ch2": ch2})

        b1 = ChannelBinding(
            channel_id="ch1",
            room_id="test-room",
            channel_type=ChannelType.EMAIL,
            capabilities=ChannelCapabilities(
                media_types=[ChannelMediaType.TEXT, ChannelMediaType.VIDEO]
            ),
        )
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.SMS,
            capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
        )

        # CompositeContent with empty parts → None from transcoder
        # Use model_construct to bypass validation for this edge-case test
        empty_composite = CompositeContent.model_construct(type="composite", parts=[])
        event = RoomEvent.model_construct(
            id="test-id",
            room_id="test-room",
            type=EventType.MESSAGE,
            source=EventSource(channel_id="ch1", channel_type=ChannelType.EMAIL),
            content=empty_composite,
            status=EventStatus.PENDING,
            visibility="all",
            index=0,
            chain_depth=0,
            metadata={},
            channel_data=ChannelData(),
            delivery_results={},
        )
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert "ch2" in result.errors
        assert result.errors["ch2"] == "transcoding_failed"
        assert len(ch2.delivered) == 0


# ============================================================================
# D2: Provider ABC Enhancements
# ============================================================================


class TestProviderABCEnhancements:
    def test_ai_response_has_tasks_and_observations(self) -> None:
        resp = AIResponse(content="hello")
        assert resp.tasks == []
        assert resp.observations == []

    def test_ai_response_with_tasks(self) -> None:
        task = Task(id="t1", room_id="r1", title="Test task")
        obs = Observation(id="o1", room_id="r1", channel_id="ch1", content="obs")
        resp = AIResponse(content="hello", tasks=[task], observations=[obs])
        assert resp.tasks == [task]
        assert resp.observations == [obs]


# ============================================================================
# D3: Retry
# ============================================================================


class TestRetryWithBackoff:
    async def test_succeeds_on_first_attempt(self) -> None:
        fn = AsyncMock(return_value="ok")
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.001)
        result = await retry_with_backoff(fn, policy)
        assert result == "ok"
        assert fn.call_count == 1

    async def test_succeeds_on_second_attempt(self) -> None:
        fn = AsyncMock(side_effect=[RuntimeError("fail"), "ok"])
        policy = RetryPolicy(max_retries=3, base_delay_seconds=0.001)
        result = await retry_with_backoff(fn, policy)
        assert result == "ok"
        assert fn.call_count == 2

    async def test_exhausted_raises_last_exception(self) -> None:
        fn = AsyncMock(side_effect=RuntimeError("always fails"))
        policy = RetryPolicy(max_retries=2, base_delay_seconds=0.001)
        with pytest.raises(RuntimeError, match="always fails"):
            await retry_with_backoff(fn, policy)
        assert fn.call_count == 3  # 1 initial + 2 retries

    async def test_zero_retries_raises_immediately(self) -> None:
        fn = AsyncMock(side_effect=RuntimeError("fail"))
        policy = RetryPolicy(max_retries=0)
        with pytest.raises(RuntimeError, match="fail"):
            await retry_with_backoff(fn, policy)
        assert fn.call_count == 1


# ============================================================================
# D3: Circuit Breaker
# ============================================================================


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.is_closed
        assert not cb.is_open
        assert cb.allow_request()

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_closed  # not yet
        cb.record_failure()
        assert cb.is_open
        assert not cb.allow_request()

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()  # only 1 now
        assert cb.is_closed

    def test_reset_closes_breaker(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert cb.is_closed

    def test_half_open_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.is_open
        time.sleep(0.02)
        assert cb.is_half_open
        assert cb.allow_request()  # probe allowed

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.is_half_open
        cb.record_success()
        assert cb.is_closed


# ============================================================================
# D3: Rate Limiter
# ============================================================================


class TestTokenBucketRateLimiter:
    def test_allows_within_limit(self) -> None:
        rl = TokenBucketRateLimiter()
        limit = RateLimit(max_per_second=10.0)
        assert rl.acquire("ch1", limit) is True

    def test_blocks_over_limit(self) -> None:
        rl = TokenBucketRateLimiter()
        limit = RateLimit(max_per_second=1.0)
        assert rl.acquire("ch1", limit) is True
        assert rl.acquire("ch1", limit) is False

    def test_infinite_rate_always_allows(self) -> None:
        rl = TokenBucketRateLimiter()
        limit = RateLimit()  # no limits set → infinite rate
        for _ in range(100):
            assert rl.acquire("ch1", limit) is True

    def test_per_minute_rate(self) -> None:
        rl = TokenBucketRateLimiter()
        limit = RateLimit(max_per_minute=60.0)  # = 1/s
        assert rl.acquire("ch1", limit) is True
        assert rl.acquire("ch1", limit) is False

    def test_per_hour_rate(self) -> None:
        rl = TokenBucketRateLimiter()
        limit = RateLimit(max_per_hour=3600.0)  # = 1/s
        assert rl.acquire("ch1", limit) is True
        assert rl.acquire("ch1", limit) is False

    async def test_wait_blocks_until_token(self) -> None:
        rl = TokenBucketRateLimiter()
        limit = RateLimit(max_per_second=100.0)
        # Drain the token
        rl.acquire("ch1", limit)
        # wait should eventually complete (token refills quickly at 100/s)
        await rl.wait("ch1", limit)

    async def test_wait_infinite_returns_immediately(self) -> None:
        rl = TokenBucketRateLimiter()
        limit = RateLimit()
        await rl.wait("ch1", limit)  # should return immediately


# ============================================================================
# D3: Router Resilience Wiring
# ============================================================================


class TestRouterCircuitBreaker:
    async def test_circuit_breaker_open_skips_delivery(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = EventRouter(channels={"ch1": ch1, "ch2": ch2})

        # Trip the breaker manually
        breaker = router._get_breaker("ch2")
        for _ in range(5):
            breaker.record_failure()
        assert breaker.is_open

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert result.errors.get("ch2") == "circuit_breaker_open"
        assert len(ch2.delivered) == 0

    async def test_circuit_breaker_records_success(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = EventRouter(channels={"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        event = make_event(channel_id="ch1")
        await router.broadcast(event, b1, _make_context([b1, b2]))
        breaker = router._get_breaker("ch2")
        assert breaker.is_closed

    async def test_circuit_breaker_records_failure(self) -> None:
        ch1 = StubChannel("ch1")
        ch2 = FailingChannel("ch2")
        router = EventRouter(channels={"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        event = make_event(channel_id="ch1")
        await router.broadcast(event, b1, _make_context([b1, b2]))
        breaker = router._get_breaker("ch2")
        assert breaker._failure_count == 1


class TestRouterRetry:
    async def test_retry_on_delivery_failure(self) -> None:
        """Delivery retries when retry_policy is set and channel fails then succeeds."""
        ch1 = StubChannel("ch1")
        ch2 = FailingChannel("ch2", fail_count=1)  # fails once, then succeeds
        router = EventRouter(channels={"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
            retry_policy=RetryPolicy(max_retries=3, base_delay_seconds=0.001),
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert "ch2" in result.delivery_outputs
        assert "ch2" not in result.errors

    async def test_no_retry_when_policy_absent(self) -> None:
        """Without retry_policy, single failure is recorded as error."""
        ch1 = StubChannel("ch1")
        ch2 = FailingChannel("ch2")
        router = EventRouter(channels={"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert "ch2" in result.errors


class TestRouterRateLimit:
    async def test_rate_limited_delivery_still_succeeds(self) -> None:
        """Rate-limited channel waits for token then delivers."""
        ch1 = StubChannel("ch1")
        ch2 = StubChannel("ch2")
        router = EventRouter(channels={"ch1": ch1, "ch2": ch2})

        b1 = make_binding("ch1")
        b2 = ChannelBinding(
            channel_id="ch2",
            room_id="test-room",
            channel_type=ChannelType.WEBSOCKET,
            category=ChannelCategory.TRANSPORT,
            rate_limit=RateLimit(max_per_second=100.0),
        )

        event = make_event(channel_id="ch1")
        result = await router.broadcast(event, b1, _make_context([b1, b2]))
        assert "ch2" in result.delivery_outputs
        assert len(ch2.delivered) == 1


# ============================================================================
# D4: Room Timers
# ============================================================================


class TestRoomTimers:
    async def _make_kit(self) -> tuple:
        from roomkit.core.framework import RoomKit

        kit = RoomKit()
        return kit

    async def test_active_room_stays_active_within_threshold(self) -> None:
        kit = await self._make_kit()
        room = await kit.create_room("r1")
        room.timers.inactive_after_seconds = 3600
        room.timers.last_activity_at = datetime.now(UTC)
        await kit.store.update_room(room)

        result = await kit.check_room_timers("r1")
        assert result.status == RoomStatus.ACTIVE

    async def test_inactive_room_transitions_to_paused(self) -> None:
        kit = await self._make_kit()
        room = await kit.create_room("r1")
        room.timers.inactive_after_seconds = 60
        room.timers.last_activity_at = datetime.now(UTC) - timedelta(seconds=120)
        await kit.store.update_room(room)

        result = await kit.check_room_timers("r1")
        assert result.status == RoomStatus.PAUSED

    async def test_paused_room_transitions_to_closed(self) -> None:
        kit = await self._make_kit()
        room = await kit.create_room("r1")
        room.status = RoomStatus.PAUSED
        room.timers.closed_after_seconds = 60
        room.timers.last_activity_at = datetime.now(UTC) - timedelta(seconds=120)
        await kit.store.update_room(room)

        result = await kit.check_room_timers("r1")
        assert result.status == RoomStatus.CLOSED
        assert result.closed_at is not None

    async def test_active_room_closes_directly(self) -> None:
        """Active room can close directly if closed_after exceeds threshold."""
        kit = await self._make_kit()
        room = await kit.create_room("r1")
        room.timers.closed_after_seconds = 60
        room.timers.last_activity_at = datetime.now(UTC) - timedelta(seconds=120)
        await kit.store.update_room(room)

        result = await kit.check_room_timers("r1")
        assert result.status == RoomStatus.CLOSED

    async def test_closed_room_stays_closed(self) -> None:
        kit = await self._make_kit()
        room = await kit.create_room("r1")
        room.status = RoomStatus.CLOSED
        room.timers.inactive_after_seconds = 1
        room.timers.last_activity_at = datetime.now(UTC) - timedelta(seconds=3600)
        await kit.store.update_room(room)

        result = await kit.check_room_timers("r1")
        assert result.status == RoomStatus.CLOSED

    async def test_no_timers_configured(self) -> None:
        kit = await self._make_kit()
        await kit.create_room("r1")
        result = await kit.check_room_timers("r1")
        assert result.status == RoomStatus.ACTIVE

    async def test_check_all_timers_returns_transitioned(self) -> None:
        kit = await self._make_kit()

        # r1: should pause
        r1 = await kit.create_room("r1")
        r1.timers.inactive_after_seconds = 60
        r1.timers.last_activity_at = datetime.now(UTC) - timedelta(seconds=120)
        await kit.store.update_room(r1)

        # r2: should stay active
        r2 = await kit.create_room("r2")
        r2.timers.inactive_after_seconds = 3600
        r2.timers.last_activity_at = datetime.now(UTC)
        await kit.store.update_room(r2)

        transitioned = await kit.check_all_timers()
        ids = [r.id for r in transitioned]
        assert "r1" in ids
        assert "r2" not in ids

    async def test_check_all_timers_paused_to_closed(self) -> None:
        kit = await self._make_kit()

        r1 = await kit.create_room("r1")
        r1.status = RoomStatus.PAUSED
        r1.timers.closed_after_seconds = 60
        r1.timers.last_activity_at = datetime.now(UTC) - timedelta(seconds=120)
        await kit.store.update_room(r1)

        transitioned = await kit.check_all_timers()
        assert len(transitioned) == 1
        assert transitioned[0].status == RoomStatus.CLOSED


# ============================================================================
# D4: ON_ROOM_PAUSED hook trigger
# ============================================================================


class TestRoomPausedHookTrigger:
    def test_on_room_paused_exists(self) -> None:
        assert HookTrigger.ON_ROOM_PAUSED == "on_room_paused"


# ============================================================================
# D2: RetryPolicy model
# ============================================================================


class TestRetryPolicyModel:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 60.0
        assert policy.exponential_base == 2.0

    def test_custom_values(self) -> None:
        policy = RetryPolicy(
            max_retries=5,
            base_delay_seconds=0.5,
            max_delay_seconds=30.0,
            exponential_base=3.0,
        )
        assert policy.max_retries == 5

    def test_retry_policy_on_binding(self) -> None:
        binding = ChannelBinding(
            channel_id="ch1",
            room_id="r1",
            channel_type=ChannelType.SMS,
            retry_policy=RetryPolicy(max_retries=2),
        )
        assert binding.retry_policy is not None
        assert binding.retry_policy.max_retries == 2
