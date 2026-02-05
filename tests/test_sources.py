"""Tests for event-driven sources."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from roomkit import (
    BaseSourceProvider,
    InboundMessage,
    RoomKit,
    SourceAlreadyAttachedError,
    SourceHealth,
    SourceNotFoundError,
    SourceProvider,
    SourceStatus,
    TextContent,
)
from roomkit.sources.base import EmitCallback

if TYPE_CHECKING:
    from roomkit.models.delivery import InboundResult


# =============================================================================
# Test Fixtures: Mock Sources
# =============================================================================


class MockSource(SourceProvider):
    """Simple mock source for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self._status = SourceStatus.STOPPED
        self._started = False
        self._stopped = False
        self._emit: EmitCallback | None = None
        self._messages_to_emit: list[InboundMessage] = []
        self._stop_event = asyncio.Event()

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> SourceStatus:
        return self._status

    async def start(self, emit: EmitCallback) -> None:
        self._started = True
        self._emit = emit
        self._status = SourceStatus.CONNECTED
        self._stop_event.clear()

        # Emit any queued messages
        for msg in self._messages_to_emit:
            await emit(msg)

        # Wait until stopped
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stopped = True
        self._status = SourceStatus.STOPPED
        self._stop_event.set()

    def queue_message(self, msg: InboundMessage) -> None:
        """Queue a message to be emitted when started."""
        self._messages_to_emit.append(msg)

    async def emit_message(self, msg: InboundMessage) -> InboundResult:
        """Emit a message directly (must be started)."""
        if self._emit is None:
            raise RuntimeError("Source not started")
        return await self._emit(msg)


class FailingSource(SourceProvider):
    """Source that fails after a delay."""

    def __init__(self, fail_after: float = 0.1, fail_count: int = 1):
        self._fail_after = fail_after
        self._fail_count = fail_count
        self._attempts = 0
        self._status = SourceStatus.STOPPED

    @property
    def name(self) -> str:
        return "failing-source"

    @property
    def status(self) -> SourceStatus:
        return self._status

    async def start(self, emit: EmitCallback) -> None:
        self._attempts += 1
        self._status = SourceStatus.CONNECTED

        await asyncio.sleep(self._fail_after)

        if self._attempts <= self._fail_count:
            self._status = SourceStatus.ERROR
            raise RuntimeError(f"Simulated failure #{self._attempts}")

        # After fail_count, run forever (until stopped)
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._status = SourceStatus.STOPPED
        if hasattr(self, "_stop_event"):
            self._stop_event.set()


class CountingSource(BaseSourceProvider):
    """Source using BaseSourceProvider for testing helpers."""

    def __init__(self, message_count: int = 3):
        super().__init__()
        self._message_count = message_count

    @property
    def name(self) -> str:
        return "counting-source"

    async def start(self, emit: EmitCallback) -> None:
        self._reset_stop()
        self._set_status(SourceStatus.CONNECTING)

        await asyncio.sleep(0.01)
        self._set_status(SourceStatus.CONNECTED)

        for i in range(self._message_count):
            if self._should_stop():
                break

            msg = InboundMessage(
                channel_id="counting",
                sender_id="counter",
                content=TextContent(body=f"Message {i + 1}"),
            )
            await emit(msg)
            self._record_message()

            await asyncio.sleep(0.01)

        # Wait until stopped
        while not self._should_stop():
            await asyncio.sleep(0.01)


# =============================================================================
# SourceStatus Tests
# =============================================================================


class TestSourceStatus:
    def test_status_values(self) -> None:
        assert SourceStatus.STOPPED == "stopped"
        assert SourceStatus.CONNECTING == "connecting"
        assert SourceStatus.CONNECTED == "connected"
        assert SourceStatus.RECONNECTING == "reconnecting"
        assert SourceStatus.ERROR == "error"

    def test_status_is_string_enum(self) -> None:
        assert isinstance(SourceStatus.CONNECTED, str)
        assert SourceStatus.CONNECTED == "connected"


# =============================================================================
# SourceHealth Tests
# =============================================================================


class TestSourceHealth:
    def test_default_health(self) -> None:
        health = SourceHealth()
        assert health.status == SourceStatus.STOPPED
        assert health.connected_at is None
        assert health.last_message_at is None
        assert health.messages_received == 0
        assert health.error is None

    def test_health_with_values(self) -> None:
        now = datetime.now(UTC)
        health = SourceHealth(
            status=SourceStatus.CONNECTED,
            connected_at=now,
            last_message_at=now,
            messages_received=42,
            error=None,
            metadata={"version": "1.0"},
        )
        assert health.status == SourceStatus.CONNECTED
        assert health.connected_at == now
        assert health.messages_received == 42
        assert health.metadata == {"version": "1.0"}

    def test_health_with_error(self) -> None:
        health = SourceHealth(
            status=SourceStatus.ERROR,
            error="Connection refused",
        )
        assert health.status == SourceStatus.ERROR
        assert health.error == "Connection refused"


# =============================================================================
# SourceProvider ABC Tests
# =============================================================================


class TestSourceProviderABC:
    def test_default_status(self) -> None:
        source = MockSource()
        assert source.status == SourceStatus.STOPPED

    async def test_default_healthcheck(self) -> None:
        source = MockSource()
        health = await source.healthcheck()
        assert isinstance(health, SourceHealth)
        assert health.status == SourceStatus.STOPPED


# =============================================================================
# BaseSourceProvider Tests
# =============================================================================


class TestBaseSourceProvider:
    def test_initial_state(self) -> None:
        source = CountingSource()
        assert source.status == SourceStatus.STOPPED

    async def test_healthcheck_tracks_state(self) -> None:
        source = CountingSource(message_count=2)

        # Before start
        health = await source.healthcheck()
        assert health.status == SourceStatus.STOPPED
        assert health.messages_received == 0

        # Start in background
        task = asyncio.create_task(source.start(lambda m: asyncio.sleep(0)))

        # Wait for messages to be emitted
        await asyncio.sleep(0.1)

        health = await source.healthcheck()
        assert health.status == SourceStatus.CONNECTED
        assert health.messages_received == 2
        assert health.connected_at is not None
        assert health.last_message_at is not None

        # Stop
        await source.stop()
        await task

        health = await source.healthcheck()
        assert health.status == SourceStatus.STOPPED

    def test_set_status_clears_error_on_connected(self) -> None:
        source = CountingSource()
        source._set_status(SourceStatus.ERROR, "test error")
        assert source._error == "test error"

        source._set_status(SourceStatus.CONNECTED)
        assert source._error is None
        assert source._connected_at is not None


# =============================================================================
# RoomKit Integration Tests
# =============================================================================


class TestAttachSource:
    async def test_attach_source_basic(self) -> None:
        kit = RoomKit()
        source = MockSource("test-source")

        await kit.attach_source("test-channel", source)

        assert "test-channel" in kit._sources
        assert kit._sources["test-channel"] is source

        # Cleanup
        await kit.close()

    async def test_attach_source_starts_source(self) -> None:
        kit = RoomKit()
        source = MockSource()

        await kit.attach_source("ch1", source)

        # Give it time to start
        await asyncio.sleep(0.05)

        assert source._started is True
        assert source.status == SourceStatus.CONNECTED

        await kit.close()

    async def test_attach_source_already_attached_raises(self) -> None:
        kit = RoomKit()
        source1 = MockSource("source1")
        source2 = MockSource("source2")

        await kit.attach_source("ch1", source1)

        with pytest.raises(SourceAlreadyAttachedError):
            await kit.attach_source("ch1", source2)

        await kit.close()

    async def test_attach_multiple_sources(self) -> None:
        kit = RoomKit()
        source1 = MockSource("source1")
        source2 = MockSource("source2")

        await kit.attach_source("ch1", source1)
        await kit.attach_source("ch2", source2)

        assert len(kit._sources) == 2
        assert "ch1" in kit._sources
        assert "ch2" in kit._sources

        await kit.close()


class TestDetachSource:
    async def test_detach_source_basic(self) -> None:
        kit = RoomKit()
        source = MockSource()

        await kit.attach_source("ch1", source)
        await asyncio.sleep(0.05)

        await kit.detach_source("ch1")

        assert "ch1" not in kit._sources
        assert source._stopped is True

    async def test_detach_source_not_found_raises(self) -> None:
        kit = RoomKit()

        with pytest.raises(SourceNotFoundError):
            await kit.detach_source("nonexistent")

    async def test_detach_source_stops_task(self) -> None:
        kit = RoomKit()
        source = MockSource()

        await kit.attach_source("ch1", source)
        task = kit._source_tasks["ch1"]
        assert not task.done()

        await kit.detach_source("ch1")

        # Task should be cancelled
        assert task.done()


class TestKitSourceHealth:
    async def test_source_health_returns_health(self) -> None:
        kit = RoomKit()
        source = MockSource()

        await kit.attach_source("ch1", source)
        await asyncio.sleep(0.05)

        health = await kit.source_health("ch1")
        assert health is not None
        assert health.status == SourceStatus.CONNECTED

        await kit.close()

    async def test_source_health_not_found_returns_none(self) -> None:
        kit = RoomKit()

        health = await kit.source_health("nonexistent")
        assert health is None


class TestListSources:
    async def test_list_sources_empty(self) -> None:
        kit = RoomKit()
        sources = kit.list_sources()
        assert sources == {}

    async def test_list_sources_with_sources(self) -> None:
        kit = RoomKit()
        source1 = MockSource()
        source2 = MockSource()

        await kit.attach_source("ch1", source1)
        await kit.attach_source("ch2", source2)
        await asyncio.sleep(0.05)

        sources = kit.list_sources()
        assert len(sources) == 2
        assert sources["ch1"] == SourceStatus.CONNECTED
        assert sources["ch2"] == SourceStatus.CONNECTED

        await kit.close()


class TestSourceClose:
    async def test_close_stops_all_sources(self) -> None:
        kit = RoomKit()
        source1 = MockSource("s1")
        source2 = MockSource("s2")

        await kit.attach_source("ch1", source1)
        await kit.attach_source("ch2", source2)
        await asyncio.sleep(0.05)

        await kit.close()

        assert source1._stopped is True
        assert source2._stopped is True
        assert len(kit._sources) == 0

    async def test_context_manager_closes_sources(self) -> None:
        source = MockSource()

        async with RoomKit() as kit:
            await kit.attach_source("ch1", source)
            await asyncio.sleep(0.05)

        assert source._stopped is True


class TestSourceAutoRestart:
    async def test_auto_restart_on_failure(self) -> None:
        kit = RoomKit()
        source = FailingSource(fail_after=0.05, fail_count=2)

        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.05,
        )

        # Wait for failures and restarts
        await asyncio.sleep(0.5)

        # Should have attempted 3 times (2 failures + 1 success)
        assert source._attempts >= 3

        await kit.close()

    async def test_no_restart_when_disabled(self) -> None:
        kit = RoomKit()
        source = FailingSource(fail_after=0.05, fail_count=10)

        await kit.attach_source("ch1", source, auto_restart=False)

        # Wait for failure
        await asyncio.sleep(0.2)

        # Should have only attempted once
        assert source._attempts == 1

        # Task should be done (failed)
        assert kit._source_tasks["ch1"].done()

        await kit.close()


class TestSourceMessageEmission:
    async def test_emitted_messages_reach_process_inbound(self) -> None:
        from roomkit import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        kit = RoomKit()
        source = MockSource()
        received_events: list = []

        # Register the channel
        provider = MockSMSProvider()
        channel = SMSChannel("ch1", provider=provider)
        kit.register_channel(channel)

        # Create a room and bind the channel
        await kit.create_room("test-room")
        await kit.attach_channel(
            room_id="test-room",
            channel_id="ch1",
            metadata={"phone_number": "+15551234567"},
        )

        # Track inbound messages via hook
        @kit.hook(
            trigger="before_broadcast",
            execution="sync",
        )
        async def track_message(event, context):
            received_events.append(event)
            from roomkit.models.hook import HookResult

            return HookResult.allow()

        # Queue a message
        msg = InboundMessage(
            channel_id="ch1",
            sender_id="+15559876543",
            content=TextContent(body="Hello from source"),
        )
        source.queue_message(msg)

        await kit.attach_source("ch1", source)
        await asyncio.sleep(0.1)

        # Message should have been processed
        assert len(received_events) >= 1
        assert received_events[0].content.body == "Hello from source"

        await kit.close()


# =============================================================================
# Framework Events Tests
# =============================================================================


class TestSourceFrameworkEvents:
    async def test_source_attached_event(self) -> None:
        kit = RoomKit()
        source = MockSource("my-source")
        events: list[dict] = []

        @kit.on("source_attached")
        async def on_attached(event):
            events.append({"type": "attached", "data": event.data})

        await kit.attach_source("ch1", source)

        assert len(events) == 1
        assert events[0]["data"]["source_name"] == "my-source"

        await kit.close()

    async def test_source_detached_event(self) -> None:
        kit = RoomKit()
        source = MockSource("my-source")
        events: list[dict] = []

        @kit.on("source_detached")
        async def on_detached(event):
            events.append({"type": "detached", "data": event.data})

        await kit.attach_source("ch1", source)
        await asyncio.sleep(0.05)
        await kit.detach_source("ch1")

        assert len(events) == 1
        assert events[0]["data"]["source_name"] == "my-source"

    async def test_source_error_event(self) -> None:
        kit = RoomKit()
        source = FailingSource(fail_after=0.01, fail_count=1)
        events: list[dict] = []

        @kit.on("source_error")
        async def on_error(event):
            events.append({"type": "error", "data": event.data})

        await kit.attach_source("ch1", source, auto_restart=False)
        await asyncio.sleep(0.1)

        assert len(events) >= 1
        assert "error" in events[0]["data"]

        await kit.close()


# =============================================================================
# Edge Cases
# =============================================================================


class TestSourceEdgeCases:
    async def test_source_name_used_in_task_name(self) -> None:
        kit = RoomKit()
        source = MockSource("unique-name")

        await kit.attach_source("ch1", source)

        task = kit._source_tasks["ch1"]
        assert "source:ch1" in task.get_name()

        await kit.close()

    async def test_detach_during_start(self) -> None:
        """Detaching while source is starting should work gracefully."""
        kit = RoomKit()

        class SlowStartSource(SourceProvider):
            def __init__(self):
                self._status = SourceStatus.STOPPED

            @property
            def name(self) -> str:
                return "slow"

            @property
            def status(self) -> SourceStatus:
                return self._status

            async def start(self, emit) -> None:
                self._status = SourceStatus.CONNECTING
                await asyncio.sleep(1.0)  # Slow start
                self._status = SourceStatus.CONNECTED

            async def stop(self) -> None:
                self._status = SourceStatus.STOPPED

        source = SlowStartSource()
        await kit.attach_source("ch1", source)

        # Detach immediately (while still connecting)
        await kit.detach_source("ch1")

        assert "ch1" not in kit._sources

    async def test_multiple_attach_detach_cycles(self) -> None:
        """Source can be reattached after detachment."""
        kit = RoomKit()

        for i in range(3):
            source = MockSource(f"source-{i}")
            await kit.attach_source("ch1", source)
            await asyncio.sleep(0.02)

            assert source._started is True

            await kit.detach_source("ch1")
            assert source._stopped is True

        await kit.close()


# =============================================================================
# Exponential Backoff Tests
# =============================================================================


class TestSourceExponentialBackoff:
    async def test_backoff_doubles_on_each_failure(self) -> None:
        """Restart delay should double on each consecutive failure."""
        kit = RoomKit()
        restart_times: list[float] = []
        start_time = asyncio.get_event_loop().time()

        class TimingSource(SourceProvider):
            def __init__(self):
                self._attempts = 0
                self._status = SourceStatus.STOPPED

            @property
            def name(self) -> str:
                return "timing"

            @property
            def status(self) -> SourceStatus:
                return self._status

            async def start(self, emit) -> None:
                self._attempts += 1
                restart_times.append(asyncio.get_event_loop().time() - start_time)
                self._status = SourceStatus.CONNECTED
                raise RuntimeError(f"Failure {self._attempts}")

            async def stop(self) -> None:
                self._status = SourceStatus.STOPPED

        source = TimingSource()
        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.05,  # Start at 50ms
            max_restart_delay=1.0,
            max_restart_attempts=4,
        )

        # Wait for all attempts to complete
        await asyncio.sleep(0.8)

        # Should have 4 attempts (or close to it)
        assert source._attempts >= 3

        # Verify delays increased (with tolerance for timing)
        # Attempt 1: immediate
        # Attempt 2: after ~0.05s
        # Attempt 3: after ~0.05 + 0.1 = 0.15s
        # Attempt 4: after ~0.05 + 0.1 + 0.2 = 0.35s
        if len(restart_times) >= 3:
            delay_1_to_2 = restart_times[1] - restart_times[0]
            delay_2_to_3 = restart_times[2] - restart_times[1]
            # Second delay should be roughly double the first (with tolerance)
            assert delay_2_to_3 > delay_1_to_2 * 1.5

        await kit.close()

    async def test_backoff_caps_at_max_restart_delay(self) -> None:
        """Backoff should not exceed max_restart_delay."""
        kit = RoomKit()
        delays: list[float] = []
        last_time = [asyncio.get_event_loop().time()]

        class DelayTrackingSource(SourceProvider):
            def __init__(self):
                self._attempts = 0
                self._status = SourceStatus.STOPPED

            @property
            def name(self) -> str:
                return "delay-tracking"

            @property
            def status(self) -> SourceStatus:
                return self._status

            async def start(self, emit) -> None:
                self._attempts += 1
                now = asyncio.get_event_loop().time()
                if self._attempts > 1:
                    delays.append(now - last_time[0])
                last_time[0] = now
                self._status = SourceStatus.CONNECTED
                raise RuntimeError(f"Failure {self._attempts}")

            async def stop(self) -> None:
                self._status = SourceStatus.STOPPED

        source = DelayTrackingSource()
        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.05,
            max_restart_delay=0.1,  # Cap at 100ms
            max_restart_attempts=5,
        )

        await asyncio.sleep(0.6)

        # All delays after the first few should be capped at ~0.1s
        if len(delays) >= 3:
            # Later delays should not exceed max_restart_delay (with tolerance)
            for delay in delays[2:]:
                assert delay < 0.15  # 0.1 + tolerance

        await kit.close()


# =============================================================================
# Max Restart Attempts Tests
# =============================================================================


class TestSourceMaxRestartAttempts:
    async def test_source_exhausted_after_max_attempts(self) -> None:
        """Source should stop retrying after max_restart_attempts."""
        kit = RoomKit()
        source = FailingSource(fail_after=0.01, fail_count=100)  # Always fails

        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.02,
            max_restart_attempts=3,
        )

        # Wait for exhaustion
        await asyncio.sleep(0.5)

        # Should have stopped at max attempts
        assert source._attempts == 3

        # Task should be done (exhausted, not running)
        assert kit._source_tasks["ch1"].done()

        await kit.close()

    async def test_source_exhausted_event_emitted(self) -> None:
        """source_exhausted framework event should be emitted."""
        kit = RoomKit()
        source = FailingSource(fail_after=0.01, fail_count=100)
        exhausted_events: list = []

        @kit.on("source_exhausted")
        async def on_exhausted(event):
            exhausted_events.append(event)

        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.02,
            max_restart_attempts=2,
        )

        await asyncio.sleep(0.3)

        assert len(exhausted_events) == 1
        assert exhausted_events[0].data["source_name"] == "failing-source"
        assert exhausted_events[0].data["attempts"] == 2
        assert "last_error" in exhausted_events[0].data

        await kit.close()

    async def test_unlimited_retries_when_max_is_none(self) -> None:
        """With max_restart_attempts=None, retries should be unlimited."""
        kit = RoomKit()
        source = FailingSource(fail_after=0.01, fail_count=5)  # Fails 5 times

        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.02,
            max_restart_attempts=None,  # Unlimited
        )

        await asyncio.sleep(0.5)

        # Should have retried past 5 failures and succeeded
        assert source._attempts >= 5

        await kit.close()


# =============================================================================
# Backpressure Tests
# =============================================================================


class TestSourceBackpressure:
    async def test_max_concurrent_emits_limits_concurrency(self) -> None:
        """max_concurrent_emits should limit concurrent emit() calls."""
        kit = RoomKit()
        concurrent_count = [0]
        max_concurrent_seen = [0]

        # Register a channel for the source
        from roomkit import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        provider = MockSMSProvider()
        channel = SMSChannel("ch1", provider=provider)
        kit.register_channel(channel)

        await kit.create_room("test-room")
        await kit.attach_channel("test-room", "ch1", metadata={"phone_number": "+1555"})

        # Add a slow hook to simulate processing time
        @kit.hook(trigger="before_broadcast", execution="sync")
        async def slow_hook(event, context):
            concurrent_count[0] += 1
            max_concurrent_seen[0] = max(max_concurrent_seen[0], concurrent_count[0])
            await asyncio.sleep(0.05)  # Slow processing
            concurrent_count[0] -= 1
            from roomkit.models.hook import HookResult

            return HookResult.allow()

        class BurstSource(BaseSourceProvider):
            def __init__(self):
                super().__init__()
                self._emitted = 0

            @property
            def name(self) -> str:
                return "burst"

            async def start(self, emit) -> None:
                self._reset_stop()
                self._set_status(SourceStatus.CONNECTED)

                # Emit many messages rapidly
                tasks = []
                for i in range(10):
                    msg = InboundMessage(
                        channel_id="ch1",
                        sender_id="+15559876543",
                        content=TextContent(body=f"Msg {i}"),
                    )
                    tasks.append(asyncio.create_task(emit(msg)))
                    self._emitted += 1

                await asyncio.gather(*tasks)

                while not self._should_stop():
                    await asyncio.sleep(0.01)

        source = BurstSource()
        await kit.attach_source(
            "ch1",
            source,
            max_concurrent_emits=3,  # Limit to 3 concurrent
        )

        await asyncio.sleep(0.5)

        # Max concurrent should be limited to 3
        assert max_concurrent_seen[0] <= 3

        await kit.close()

    async def test_semaphore_not_created_when_none(self) -> None:
        """With max_concurrent_emits=None, no semaphore should be applied."""
        # This test verifies that the unlimited path works correctly.
        # Note: process_inbound has internal room-level locking, so actual
        # message processing is serialized. The semaphore limits how many
        # emit() calls can be pending simultaneously.
        kit = RoomKit()
        emit_count = [0]

        from roomkit import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        provider = MockSMSProvider()
        channel = SMSChannel("ch1", provider=provider)
        kit.register_channel(channel)

        await kit.create_room("test-room")
        await kit.attach_channel("test-room", "ch1", metadata={"phone_number": "+1555"})

        @kit.hook(trigger="before_broadcast", execution="sync")
        async def count_hook(event, context):
            emit_count[0] += 1
            from roomkit.models.hook import HookResult

            return HookResult.allow()

        class QuickSource(BaseSourceProvider):
            @property
            def name(self) -> str:
                return "quick"

            async def start(self, emit) -> None:
                self._reset_stop()
                self._set_status(SourceStatus.CONNECTED)

                # Emit multiple messages
                for i in range(5):
                    msg = InboundMessage(
                        channel_id="ch1",
                        sender_id="+15559876543",
                        content=TextContent(body=f"Msg {i}"),
                    )
                    await emit(msg)

                while not self._should_stop():
                    await asyncio.sleep(0.01)

        source = QuickSource()
        await kit.attach_source(
            "ch1",
            source,
            max_concurrent_emits=None,  # Unlimited
        )

        await asyncio.sleep(0.3)

        # All messages should have been processed
        assert emit_count[0] == 5

        await kit.close()

    async def test_default_max_concurrent_emits_is_10(self) -> None:
        """Default max_concurrent_emits should be 10."""
        kit = RoomKit()
        max_concurrent_seen = [0]
        concurrent_count = [0]

        from roomkit import SMSChannel
        from roomkit.providers.sms.mock import MockSMSProvider

        provider = MockSMSProvider()
        channel = SMSChannel("ch1", provider=provider)
        kit.register_channel(channel)

        await kit.create_room("test-room")
        await kit.attach_channel("test-room", "ch1", metadata={"phone_number": "+1555"})

        @kit.hook(trigger="before_broadcast", execution="sync")
        async def slow_hook(event, context):
            concurrent_count[0] += 1
            max_concurrent_seen[0] = max(max_concurrent_seen[0], concurrent_count[0])
            await asyncio.sleep(0.03)
            concurrent_count[0] -= 1
            from roomkit.models.hook import HookResult

            return HookResult.allow()

        class BurstSource(BaseSourceProvider):
            @property
            def name(self) -> str:
                return "burst"

            async def start(self, emit) -> None:
                self._reset_stop()
                self._set_status(SourceStatus.CONNECTED)

                tasks = []
                for i in range(20):  # More than default limit
                    msg = InboundMessage(
                        channel_id="ch1",
                        sender_id="+15559876543",
                        content=TextContent(body=f"Msg {i}"),
                    )
                    tasks.append(asyncio.create_task(emit(msg)))

                await asyncio.gather(*tasks)

                while not self._should_stop():
                    await asyncio.sleep(0.01)

        source = BurstSource()
        # Don't pass max_concurrent_emits - use default
        await kit.attach_source("ch1", source)

        await asyncio.sleep(0.8)

        # Default limit is 10
        assert max_concurrent_seen[0] <= 10

        await kit.close()
