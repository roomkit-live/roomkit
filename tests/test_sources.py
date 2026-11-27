"""Tests for event-driven sources."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from roomkit import (
    InboundMessage,
    RoomKit,
    SourceAlreadyAttachedError,
    SourceNotFoundError,
    TextContent,
)
from roomkit.core.mixins import source_ops
from roomkit.sources.base import (
    BaseSourceProvider,
    EmitCallback,
    SourceHealth,
    SourceProvider,
    SourceStatus,
)
from tests.conference.test_conference_realtime import until

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


class _Clock:
    """source_ops' asyncio with its restart sleeps recorded, not waited out:
    the delays a restart waited are what the backoff is."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __getattr__(self, name: str) -> object:
        return getattr(asyncio, name)

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)
        await asyncio.sleep(0)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(source_ops, "asyncio", clock)
    return clock


class _Entered:
    """A kit whose inbound processing holds every message until released,
    counting how many an attached source has in flight at once."""

    def __init__(self, kit: RoomKit) -> None:
        self.now = self.most = 0
        self.release = asyncio.Event()
        kit.process_inbound = self._process  # type: ignore[method-assign]

    async def _process(self, message: InboundMessage) -> None:
        self.now += 1
        self.most = max(self.most, self.now)
        await self.release.wait()
        self.now -= 1


class _GatedSource(BaseSourceProvider):
    """Connects once its gate opens, then waits to be stopped; ``reconnect``
    drops the connection and makes it again."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    @property
    def name(self) -> str:
        return "gated"

    async def start(self, emit: EmitCallback) -> None:
        self._set_status(SourceStatus.CONNECTING)
        await self.gate.wait()
        self._set_status(SourceStatus.CONNECTED)
        await self._stop_event.wait()

    def reconnect(self) -> None:
        self._set_status(SourceStatus.RECONNECTING)
        self._set_status(SourceStatus.CONNECTED)


class _Burst(BaseSourceProvider):
    """Emits *count* messages at once, then waits to be stopped."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self._count = count
        self.emitted = asyncio.Event()

    @property
    def name(self) -> str:
        return "burst"

    async def start(self, emit: EmitCallback) -> None:
        self._reset_stop()
        self._set_status(SourceStatus.CONNECTED)
        tasks = [
            asyncio.create_task(
                emit(
                    InboundMessage(
                        channel_id="ch1",
                        sender_id="+15559876543",
                        content=TextContent(body=f"Msg {i}"),
                    )
                )
            )
            for i in range(self._count)
        ]
        await asyncio.gather(*tasks)
        self.emitted.set()
        while not self._should_stop():
            await asyncio.sleep(0.01)


async def _in_flight_at_most(**attach: object) -> int:
    """How many of a burst of 20 messages a source had in flight at once."""
    kit = RoomKit()
    entered = _Entered(kit)
    source = _Burst(20)
    await kit.attach_source("ch1", source, **attach)  # type: ignore[arg-type]
    # Every emit has started or waits on the limit once the loop goes idle.
    await until(lambda: entered.now > 0)
    for _ in range(20):
        await asyncio.sleep(0)
    most = entered.most
    entered.release.set()
    await asyncio.wait_for(source.emitted.wait(), 5)
    await kit.close()
    return most


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


# =============================================================================
# SourceProvider ABC Tests
# =============================================================================


class TestSourceProviderABC:
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

        async def received() -> int:
            return (await source.healthcheck()).messages_received

        for _ in range(500):  # both messages emitted
            if await received() == 2:
                break
            await asyncio.sleep(0.005)

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
        await until(lambda: source._started)

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
        await until(lambda: source._started)

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
        await until(lambda: source._started)

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
        await until(lambda: source1._started and source2._started)

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
        await until(lambda: source1._started and source2._started)

        await kit.close()

        assert source1._stopped is True
        assert source2._stopped is True
        assert len(kit._sources) == 0

    async def test_context_manager_closes_sources(self) -> None:
        source = MockSource()

        async with RoomKit() as kit:
            await kit.attach_source("ch1", source)
            await until(lambda: source._started)

        assert source._stopped is True


class TestSourceAutoRestart:
    async def test_auto_restart_on_failure(self, clock: _Clock) -> None:
        kit = RoomKit()
        source = FailingSource(fail_after=0, fail_count=2)

        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.05,
        )

        # Wait for the failures and the restarts: on a condition, since a fixed
        # sleep leaves too few attempts when the machine is loaded.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while source._attempts < 3 and loop.time() < deadline:
            await asyncio.sleep(0.01)

        # Should have attempted 3 times (2 failures + 1 success)
        assert source._attempts >= 3

        await kit.close()

    async def test_no_restart_when_disabled(self) -> None:
        kit = RoomKit()
        source = FailingSource(fail_after=0, fail_count=10)

        await kit.attach_source("ch1", source, auto_restart=False)

        # The runner ends on the failure: no restart can come after it.
        await asyncio.wait({kit._source_tasks["ch1"]}, timeout=5)

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
        await until(lambda: bool(received_events))

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

    async def test_source_connected_once_the_source_connects(self) -> None:
        # RFC §8.2: source_connected is the source connected, not attached.
        # A source that never connects is never announced connected; one
        # that connects again after a drop is announced each time.
        kit = RoomKit()
        source = _GatedSource()
        events: list[tuple[str, str | None, dict]] = []

        for name in ("source_attached", "source_connected"):

            @kit.on(name)
            async def record(event) -> None:
                events.append((event.type, event.channel_id, event.data))

        try:
            await kit.attach_source("ch1", source)
            await until(lambda: source.status is SourceStatus.CONNECTING)
            await asyncio.sleep(0)
            assert [name for name, _, _ in events] == ["source_attached"]

            source.gate.set()
            await until(lambda: len(events) == 2)
            source.reconnect()
            await until(lambda: len(events) == 3)
            connected = ("source_connected", "ch1", {"source_name": "gated"})
            assert events[1:] == [connected, connected]
        finally:
            await kit.close()

    async def test_a_source_on_the_bare_interface_is_never_announced_connected(self) -> None:
        # It reports no change of its status: whether it connected is unknown.
        kit = RoomKit()
        source = MockSource("bare")
        events: list[str] = []

        @kit.on("source_connected")
        async def record(event) -> None:
            events.append(event.type)

        try:
            await kit.attach_source("ch1", source)
            await until(lambda: source._started)
            await asyncio.sleep(0)
            assert events == []
        finally:
            await kit.close()

    async def test_source_detached_event(self) -> None:
        kit = RoomKit()
        source = MockSource("my-source")
        events: list[dict] = []

        @kit.on("source_detached")
        async def on_detached(event):
            events.append({"type": "detached", "data": event.data})

        await kit.attach_source("ch1", source)
        await until(lambda: source._started)
        await kit.detach_source("ch1")

        assert len(events) == 1
        assert events[0]["data"]["source_name"] == "my-source"

    async def test_source_error_event(self) -> None:
        kit = RoomKit()
        source = FailingSource(fail_after=0, fail_count=1)
        events: list[dict] = []

        @kit.on("source_error")
        async def on_error(event):
            events.append({"type": "error", "data": event.data})

        await kit.attach_source("ch1", source, auto_restart=False)
        # The runner ends on the failure, its error event emitted first.
        await asyncio.wait({kit._source_tasks["ch1"]}, timeout=5)

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
            await until(lambda source=source: source._started)

            assert source._started is True

            await kit.detach_source("ch1")
            assert source._stopped is True

        await kit.close()


# =============================================================================
# Exponential Backoff Tests
# =============================================================================


class _AlwaysFails(SourceProvider):
    """A source whose every start fails at once."""

    def __init__(self) -> None:
        self.attempts = 0

    @property
    def name(self) -> str:
        return "always-fails"

    @property
    def status(self) -> SourceStatus:
        return SourceStatus.ERROR

    async def start(self, emit: EmitCallback) -> None:
        self.attempts += 1
        raise RuntimeError(f"Failure {self.attempts}")

    async def stop(self) -> None:
        pass


class TestSourceExponentialBackoff:
    async def test_backoff_doubles_on_each_failure(self, clock: _Clock) -> None:
        """Restart delay should double on each consecutive failure."""
        kit = RoomKit()
        source = _AlwaysFails()
        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.05,
            max_restart_delay=1.0,
            max_restart_attempts=4,
        )
        await asyncio.wait({kit._source_tasks["ch1"]}, timeout=5)  # exhausted

        assert source.attempts == 4
        assert clock.delays == [0.05, 0.1, 0.2]
        await kit.close()

    async def test_backoff_caps_at_max_restart_delay(self, clock: _Clock) -> None:
        """Backoff should not exceed max_restart_delay."""
        kit = RoomKit()
        await kit.attach_source(
            "ch1",
            _AlwaysFails(),
            auto_restart=True,
            restart_delay=0.05,
            max_restart_delay=0.1,
            max_restart_attempts=5,
        )
        await asyncio.wait({kit._source_tasks["ch1"]}, timeout=5)

        assert clock.delays == [0.05, 0.1, 0.1, 0.1]
        await kit.close()


# =============================================================================
# Max Restart Attempts Tests
# =============================================================================


class TestSourceMaxRestartAttempts:
    async def test_source_exhausted_after_max_attempts(self, clock: _Clock) -> None:
        """Source should stop retrying after max_restart_attempts."""
        kit = RoomKit()
        source = FailingSource(fail_after=0, fail_count=100)  # Always fails

        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.02,
            max_restart_attempts=3,
        )
        await asyncio.wait({kit._source_tasks["ch1"]}, timeout=5)

        # Stopped at max attempts, the runner done (exhausted, not running)
        assert source._attempts == 3
        assert kit._source_tasks["ch1"].done()

        await kit.close()

    async def test_source_exhausted_event_emitted(self, clock: _Clock) -> None:
        """source_exhausted framework event should be emitted."""
        kit = RoomKit()
        source = FailingSource(fail_after=0, fail_count=100)
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
        await asyncio.wait({kit._source_tasks["ch1"]}, timeout=5)

        assert len(exhausted_events) == 1
        assert exhausted_events[0].data["source_name"] == "failing-source"
        assert exhausted_events[0].data["attempts"] == 2
        assert "last_error" in exhausted_events[0].data

        await kit.close()

    async def test_unlimited_retries_when_max_is_none(self, clock: _Clock) -> None:
        """With max_restart_attempts=None, retries should be unlimited."""
        kit = RoomKit()
        source = FailingSource(fail_after=0, fail_count=5)  # Fails 5 times

        await kit.attach_source(
            "ch1",
            source,
            auto_restart=True,
            restart_delay=0.02,
            max_restart_attempts=None,  # Unlimited
        )
        # Past its 5 failures, the sixth start runs until stopped.
        await until(lambda: source._attempts == 6)

        assert len(clock.delays) == 5
        await kit.close()


# =============================================================================
# Backpressure Tests
# =============================================================================


class TestSourceBackpressure:
    async def test_max_concurrent_emits_limits_concurrency(self) -> None:
        """max_concurrent_emits should limit concurrent emit() calls."""
        assert await _in_flight_at_most(max_concurrent_emits=3) == 3

    async def test_semaphore_not_created_when_none(self) -> None:
        """With max_concurrent_emits=None, no emit waits on a limit."""
        assert await _in_flight_at_most(max_concurrent_emits=None) == 20

    async def test_default_max_concurrent_emits_is_10(self) -> None:
        """Default max_concurrent_emits should be 10."""
        assert await _in_flight_at_most() == 10
