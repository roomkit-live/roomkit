"""Tests for StatusBus integration with the RoomKit framework."""

from __future__ import annotations

from roomkit import RoomKit
from roomkit.orchestration.status_bus import InMemoryStatusBackend, StatusBus, StatusLevel


async def test_status_bus_available_by_default() -> None:
    """kit.status_bus is always available with a default InMemoryStatusBackend."""
    async with RoomKit() as kit:
        assert isinstance(kit.status_bus, StatusBus)


async def test_status_bus_custom_instance() -> None:
    """Custom StatusBus passed to constructor is used as-is."""
    custom = StatusBus(backend=InMemoryStatusBackend(max_entries=10))
    async with RoomKit(status_bus=custom) as kit:
        assert kit.status_bus is custom


async def test_framework_event_on_post_async() -> None:
    """kit.on('status_posted') fires when status_bus.post_async() is called."""
    received: list[dict] = []

    kit = RoomKit()

    @kit.on("status_posted")
    async def handler(event):  # type: ignore[no-untyped-def]
        received.append(event.data)

    async with kit:
        await kit.status_bus.post_async("exec", "search", StatusLevel.OK, detail="found 3")

    assert len(received) == 1
    assert received[0]["agent_id"] == "exec"
    assert received[0]["action"] == "search"
    assert received[0]["status"] == "ok"
    assert received[0]["detail"] == "found 3"


async def test_framework_event_on_post_sync() -> None:
    """kit.on('status_posted') fires for sync post() too (via scheduled task)."""
    import asyncio

    received: list[dict] = []

    kit = RoomKit()

    @kit.on("status_posted")
    async def handler(event):  # type: ignore[no-untyped-def]
        received.append(event.data)

    async with kit:
        kit.status_bus.post("voice", "greet", "info", detail="hello")
        # sync post schedules an asyncio task — give it time to complete
        await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0]["agent_id"] == "voice"


async def test_close_closes_status_bus() -> None:
    """kit.close() closes the underlying status bus."""
    kit = RoomKit()
    backend = kit.status_bus._backend

    closed = False
    original_close = backend.close

    async def tracking_close() -> None:
        nonlocal closed
        closed = True
        await original_close()

    backend.close = tracking_close  # type: ignore[assignment]
    await kit.close()
    assert closed


async def test_context_manager_subscribes() -> None:
    """Using RoomKit as async context manager wires the status bus subscription."""
    received: list[dict] = []

    kit = RoomKit()

    @kit.on("status_posted")
    async def handler(event):  # type: ignore[no-untyped-def]
        received.append(event.data)

    async with kit:
        await kit.status_bus.post_async("a", "act", StatusLevel.OK)
        assert len(received) == 1
