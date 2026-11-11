"""A scenario waits for publications whose permission read is still pending."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from roomkit.channels._mark_copies import compile_mark_patterns
from roomkit.models.channel import ChannelBinding
from roomkit.store.sqlite import SQLiteStore
from tests.conformance._harness import Harness


class HeldPublicationStore(SQLiteStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.publisher_reads = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def get_binding(self, room_id: str, channel_id: str) -> ChannelBinding | None:
        task = asyncio.current_task()
        if task is not None and task.get_coro().__qualname__ == "TurnPublisher._run":
            self.publisher_reads += 1
            if self.publisher_reads == 2:
                self.entered.set()
                await self.release.wait()
        return await super().get_binding(room_id, channel_id)


async def test_idle_waits_for_a_terminal_publication_reading_sqlite(tmp_path: Path) -> None:
    """The terminal frame remains observable even when SQLite answers after the turn."""
    path = (
        Path(__file__).parent
        / "scenarios/tool-composition/provider-failure-closes-composition.json"
    )
    scenario = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
    await compile_mark_patterns()
    store = HeldPublicationStore(tmp_path / "store.db")
    reader = SQLiteStore(tmp_path / "store.db")
    harness = Harness(scenario, store, observation_store=reader)
    waiting: asyncio.Task[None] | None = None
    try:
        await harness._setup()
        harness._recording = True
        await harness._step(scenario["steps"][0], "steps[0]")
        await asyncio.wait_for(store.entered.wait(), 2)
        waiting = asyncio.create_task(harness.idle())
        # Let the old three-tick idle test finish. The controlled read stays
        # blocked independently of elapsed time or the machine's load.
        for _ in range(30):
            await asyncio.sleep(0)
        assert not waiting.done(), "idle returned while a publication still awaited SQLite"
        store.release.set()
        await asyncio.wait_for(waiting, 2)
        await harness._observe()
        assert not harness._problems, harness._problems
    finally:
        store.release.set()
        if waiting is not None:
            await asyncio.wait_for(waiting, 2)
        await harness.kit.close()
        await reader.close()
