"""Tests for the StatusBus, StatusEntry, StatusLevel, and InMemoryStatusBackend."""

from __future__ import annotations

import asyncio

from roomkit.orchestration.status_bus import (
    InMemoryStatusBackend,
    StatusBus,
    StatusEntry,
    StatusLevel,
)

# ---------------------------------------------------------------------------
# StatusLevel enum
# ---------------------------------------------------------------------------


def test_status_level_values() -> None:
    assert StatusLevel.OK == "ok"
    assert StatusLevel.FAILED == "failed"
    assert StatusLevel.PENDING == "pending"
    assert StatusLevel.INFO == "info"
    assert StatusLevel.COMPLETED == "completed"


def test_status_level_from_string() -> None:
    assert StatusLevel("ok") is StatusLevel.OK
    assert StatusLevel("failed") is StatusLevel.FAILED


# ---------------------------------------------------------------------------
# StatusEntry model
# ---------------------------------------------------------------------------


def test_status_entry_model_dump() -> None:
    entry = StatusEntry(
        ts="2026-01-01T00:00:00",
        agent_id="a1",
        action="search",
        status=StatusLevel.OK,
        detail="done",
        metadata={"key": "val"},
    )
    d = entry.model_dump()
    assert d["status"] == "ok"
    assert d["agent_id"] == "a1"
    assert d["metadata"] == {"key": "val"}


def test_status_entry_model_validate() -> None:
    data = {
        "ts": "2026-01-01T00:00:00",
        "agent_id": "a1",
        "action": "search",
        "status": "ok",
        "detail": "",
        "metadata": {},
    }
    entry = StatusEntry.model_validate(data)
    assert entry.status == StatusLevel.OK
    assert entry.agent_id == "a1"


def test_status_entry_defaults() -> None:
    entry = StatusEntry(
        ts="t", agent_id="a", action="x", status=StatusLevel.INFO,
    )
    assert entry.detail == ""
    assert entry.metadata == {}


# ---------------------------------------------------------------------------
# InMemoryStatusBackend
# ---------------------------------------------------------------------------


async def test_backend_publish_and_recent() -> None:
    backend = InMemoryStatusBackend()
    entry = StatusEntry(
        ts="t1", agent_id="a", action="act", status=StatusLevel.OK,
    )
    await backend.publish(entry)
    recent = await backend.recent(10)
    assert len(recent) == 1
    assert recent[0].agent_id == "a"


async def test_backend_recent_filters() -> None:
    backend = InMemoryStatusBackend()
    for i, status in enumerate([StatusLevel.OK, StatusLevel.FAILED, StatusLevel.OK]):
        await backend.publish(StatusEntry(
            ts=f"t{i}", agent_id="a" if i < 2 else "b",
            action="act", status=status,
        ))
    assert len(await backend.recent(10, agent_id="a")) == 2
    assert len(await backend.recent(10, status="ok")) == 2
    assert len(await backend.recent(10, agent_id="b", status="ok")) == 1


async def test_backend_eviction() -> None:
    backend = InMemoryStatusBackend(max_entries=3)
    for i in range(5):
        await backend.publish(StatusEntry(
            ts=f"t{i}", agent_id="a", action="act", status=StatusLevel.OK,
        ))
    entries = await backend.recent(10)
    assert len(entries) == 3
    assert entries[0].ts == "t2"


async def test_backend_subscribe_receives_entries() -> None:
    backend = InMemoryStatusBackend()
    received: list[StatusEntry] = []

    async def cb(entry: StatusEntry) -> None:
        received.append(entry)

    await backend.subscribe(cb)
    await backend.publish(StatusEntry(
        ts="t", agent_id="a", action="act", status=StatusLevel.OK,
    ))
    assert len(received) == 1


async def test_backend_persistence(tmp_path: object) -> None:
    import json
    from pathlib import Path

    p = Path(str(tmp_path)) / "status.jsonl"
    backend = InMemoryStatusBackend(persist_path=p)
    await backend.publish(StatusEntry(
        ts="t", agent_id="a", action="act", status=StatusLevel.OK,
    ))
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# StatusBus
# ---------------------------------------------------------------------------


async def test_bus_post_async() -> None:
    bus = StatusBus()
    entry = await bus.post_async("agent", "act", StatusLevel.OK, detail="hi")
    assert entry.status == StatusLevel.OK
    recent = await bus.recent(10)
    assert len(recent) == 1


async def test_bus_post_sync_schedules() -> None:
    bus = StatusBus()
    entry = bus.post("agent", "act", "ok")
    assert entry.status == StatusLevel.OK
    # Let the scheduled task complete
    await asyncio.sleep(0.05)
    recent = await bus.recent(10)
    assert len(recent) == 1


async def test_bus_subscribe_sync_callback() -> None:
    bus = StatusBus()
    received: list[StatusEntry] = []

    def sync_cb(entry: StatusEntry) -> None:
        received.append(entry)

    await bus.subscribe(sync_cb)
    await bus.post_async("a", "act", StatusLevel.OK)
    assert len(received) == 1


async def test_bus_subscribe_async_callback() -> None:
    bus = StatusBus()
    received: list[StatusEntry] = []

    async def async_cb(entry: StatusEntry) -> None:
        received.append(entry)

    await bus.subscribe(async_cb)
    await bus.post_async("a", "act", StatusLevel.OK)
    assert len(received) == 1


async def test_bus_print_summary(capsys: object) -> None:
    bus = StatusBus()
    await bus.post_async("a", "search", StatusLevel.OK, detail="found 3")
    await bus.post_async("b", "fail_op", StatusLevel.FAILED)
    await bus.print_summary()
    # capsys is pytest's capsys fixture
    captured = capsys.readouterr()  # type: ignore[union-attr]
    assert "Status Log" in captured.out
    assert "search" in captured.out
    assert "2 entries" in captured.out


async def test_bus_print_summary_empty(capsys: object) -> None:
    bus = StatusBus()
    await bus.print_summary()
    captured = capsys.readouterr()  # type: ignore[union-attr]
    assert "No status entries" in captured.out


async def test_bus_post_accepts_string_status() -> None:
    """post() and post_async() accept plain strings that get coerced to StatusLevel."""
    bus = StatusBus()
    entry = await bus.post_async("a", "act", "failed")
    assert entry.status == StatusLevel.FAILED
