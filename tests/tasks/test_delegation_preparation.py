"""A delegation interrupted before announcement leaves no child task pending."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest

import roomkit.core.framework as framework
from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.models.room import Room
from roomkit.tasks.status import room_task_lines
from roomkit.telemetry.mock import MockTelemetryProvider
from tests.conftest import make_event
from tests.tasks.test_cancelled_delegation import _delegation_spans, _kit


@dataclass
class _Preparation:
    kit: RoomKit
    telemetry: MockTelemetryProvider
    completed: list[Any]
    announced: list[str]
    callbacks: list[Any]
    call: asyncio.Task[Any]
    release: asyncio.Event
    fails: asyncio.Event

    async def stop(self, cut: str) -> None:
        if cut == "cancelled":
            self.fails.clear()
            self.call.cancel()
            self.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(self.call, 5)
        else:
            self.release.set()
            with pytest.raises(RuntimeError, match="preparation failed"):
                await asyncio.wait_for(self.call, 5)


@asynccontextmanager
async def _preparing(
    monkeypatch: pytest.MonkeyPatch, *, wait: bool = True, boundary: str = "before_announce"
) -> AsyncIterator[_Preparation]:
    kit, telemetry, completed = await _kit()
    reached, release = asyncio.Event(), asyncio.Event()
    fails = asyncio.Event()
    fails.set()
    announced: list[str] = []
    callbacks: list[Any] = []
    create, load = kit.store.create_room, kit.store.load_room_context

    async def stop_here(at: str) -> None:
        if boundary != at:
            return
        reached.set()
        await release.wait()
        if fails.is_set():
            raise RuntimeError("preparation failed")

    async def create_room(room: Room) -> Room:
        await stop_here("before_create")
        created = await create(room)
        await stop_here("after_create")
        return created

    async def load_context(room_id: str) -> Any:
        if room_id == "p":
            await stop_here("before_announce")
        return await load(room_id)

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def delegated(event: Any, ctx: Any) -> None:
        announced.append(event.room_id)

    async def on_complete(result: Any) -> None:
        callbacks.append(result)

    monkeypatch.setattr(kit.store, "create_room", create_room)
    monkeypatch.setattr(kit.store, "load_room_context", load_context)
    call = asyncio.create_task(
        kit.delegate("p", "worker", "go", wait=wait, on_complete=on_complete)
    )
    try:
        await asyncio.wait_for(reached.wait(), 5)
        yield _Preparation(kit, telemetry, completed, announced, callbacks, call, release, fails)
    finally:
        release.set()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        await kit.close()


@pytest.mark.parametrize("wait", [True, False], ids=["inline", "background"])
@pytest.mark.parametrize("cut", ["cancelled", "failed"])
@pytest.mark.parametrize("boundary", ["before_create", "after_create", "before_announce"])
async def test_interrupted_preparation_keeps_no_child_task_pending(
    monkeypatch: pytest.MonkeyPatch, wait: bool, cut: str, boundary: str
) -> None:
    """Include a committed create not yet returned; no task was announced."""
    async with _preparing(monkeypatch, wait=wait, boundary=boundary) as preparing:
        await preparing.stop(cut)
        kit = preparing.kit
        children = [room for room in await kit.store.list_rooms() if room.id != "p"]
        lines = await room_task_lines(kit.status_bus, "p", 50)
        await kit.close()
        assert preparing.announced == preparing.completed == preparing.callbacks == lines == []
        mode = "inline" if wait else "background"
        assert _delegation_spans(preparing.telemetry) == [
            (f"delegation.{mode}", "error" if cut == "failed" else "cancelled")
        ]
        if boundary == "before_create" and cut == "failed":
            assert children == []
        else:
            assert len(children) == 1
            child = children[0]
            assert child.metadata["parent_room_id"] == "p"
            # Attaching the worker records its lifecycle event, not a turn.
            assert await kit.store.get_event_count(child.id) == int(boundary == "before_announce")
            assert child.metadata["task_status"] == cut


@pytest.mark.parametrize("cut", ["cancelled", "failed"])
async def test_preparation_preserves_the_child_record(
    monkeypatch: pytest.MonkeyPatch, cut: str
) -> None:
    async with _preparing(monkeypatch) as preparing:
        kit = preparing.kit
        child = next(room for room in await kit.store.list_rooms() if room.id != "p")
        await kit.store.patch_room_metadata(child.id, {"audit": {"keep": True}})
        await kit.commit_event(
            child.id, make_event(room_id=child.id, body="Keep this preparation record.")
        )
        before = await kit.get_room(child.id)
        bindings = await kit.store.list_bindings(child.id)
        events = await kit.store.list_events(child.id)
        await preparing.stop(cut)
        after = await kit.get_room(child.id)
        assert after == before.model_copy(
            update={
                "metadata": {**before.metadata, "task_status": cut, "task_result": None},
                "updated_at": after.updated_at,
            }
        )
        assert after.updated_at >= before.updated_at
        assert await kit.store.list_bindings(child.id) == bindings
        assert await kit.store.list_events(child.id) == events


@pytest.mark.parametrize("closing", [False, True], ids=["caller", "close"])
async def test_preparation_end_survives_repeated_cancellation(
    monkeypatch: pytest.MonkeyPatch, closing: bool
) -> None:
    async with _preparing(monkeypatch) as preparing:
        kit = preparing.kit
        patch = kit.store.patch_room_metadata
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_patch(room_id: str, metadata: dict[str, Any], **kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return await patch(room_id, metadata, **kwargs)

        monkeypatch.setattr(kit.store, "patch_room_metadata", held_patch)
        ending = asyncio.create_task(preparing.stop("cancelled"))
        close = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            for _ in range(2):
                preparing.call.cancel()
                await asyncio.sleep(0)
                assert not preparing.call.done()
            if closing:
                close = asyncio.create_task(kit.close())
                async with asyncio.timeout(5):
                    while not kit._closed:
                        await asyncio.sleep(0)
                assert not close.done()
            release.set()
            await asyncio.wait_for(ending, 5)
            if close is not None:
                await asyncio.wait_for(close, 5)
            children = [room for room in await kit.store.list_rooms() if room.id != "p"]
            assert [room.metadata["task_status"] for room in children] == ["cancelled"]
            assert preparing.announced == preparing.completed == preparing.callbacks == []
        finally:
            release.set()
            await asyncio.gather(
                ending, *([close] if close is not None else []), return_exceptions=True
            )


@pytest.mark.parametrize("cut", ["cancelled", "failed"])
@pytest.mark.parametrize("operation", ["get_room", "patch_room_metadata"])
async def test_preparation_cleanup_failure_keeps_the_original_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, cut: str, operation: str
) -> None:
    async with _preparing(monkeypatch) as preparing:
        kit = preparing.kit
        original = getattr(kit.store, operation)

        async def unavailable(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("cleanup unavailable")

        monkeypatch.setattr(kit.store, operation, unavailable)
        await preparing.stop(cut)
        monkeypatch.setattr(kit.store, operation, original)
        assert "failed to update child room metadata" in caplog.text
        assert "cleanup unavailable" in caplog.text
        child = next(room for room in await kit.store.list_rooms() if room.id != "p")
        assert child.metadata["task_status"] == "pending"
        assert preparing.announced == preparing.completed == preparing.callbacks == []
        assert _delegation_spans(preparing.telemetry) == [
            ("delegation.inline", "error" if cut == "failed" else "cancelled")
        ]


async def test_failed_creation_never_marks_another_tasks_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit, _, _ = await _kit()
    create = kit.store.create_room
    metadata = {"task_id": "another-task", "task_status": "pending", "keep": True}

    async def refused(room: Room) -> Room:
        await create(room.model_copy(update={"metadata": metadata}))
        raise RuntimeError("creation refused")

    monkeypatch.setattr(kit.store, "create_room", refused)
    try:
        with pytest.raises(RuntimeError, match="creation refused"):
            await kit.delegate("p", "worker", "go")
        child = next(room for room in await kit.store.list_rooms() if room.id != "p")
        assert child.metadata == metadata
    finally:
        await kit.close()


@pytest.mark.parametrize("cut", ["cancelled", "failed"])
async def test_close_past_grace_preserves_original_preparation_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, cut: str
) -> None:
    monkeypatch.setattr(framework, "CLOSE_GRACE_S", 0.01)
    async with _preparing(monkeypatch) as preparing:
        entered = asyncio.Event()

        async def held_patch(*args: Any, **kwargs: Any) -> None:
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(preparing.kit.store, "patch_room_metadata", held_patch)
        ending = asyncio.create_task(preparing.stop(cut))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(preparing.kit.close(), 5)
            await asyncio.wait_for(ending, 5)
            assert "close cut the unannounced child's terminal write" in caplog.text
        finally:
            await asyncio.gather(ending, return_exceptions=True)
