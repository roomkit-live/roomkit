"""Every memory decorator must preserve its inner provider's contract."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from roomkit.memory.base import MemoryProvider
from roomkit.memory.budget_aware import BudgetAwareMemory
from roomkit.memory.compacting import CompactingMemory
from roomkit.memory.mock import MockMemoryProvider
from roomkit.memory.retrieval import RetrievalMemory
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.memory.summarizing import SummarizingMemory
from roomkit.orchestration import HandoffMemoryProvider
from roomkit.providers.ai.base import AIMessage
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event


@pytest.fixture(params=["budget", "compacting", "summarizing", "retrieval"])
def wrap(request: pytest.FixtureRequest) -> Callable[[MemoryProvider], MemoryProvider]:
    def decorate(inner: MemoryProvider) -> MemoryProvider:
        if request.param == "budget":
            return BudgetAwareMemory(inner, 1000)
        if request.param == "compacting":
            return CompactingMemory(inner, MockAIProvider(), 1000)
        if request.param == "summarizing":
            return SummarizingMemory(inner, MockAIProvider(), 1000)
        return RetrievalMemory([], inner)

    return decorate


async def test_lifecycle_reaches_inner(wrap: Callable[[MemoryProvider], MemoryProvider]) -> None:
    inner = MockMemoryProvider()
    memory = wrap(wrap(inner))
    event = make_event(body="remember")
    await memory.ingest("r1", event, channel_id="ai")
    await memory.clear("r1")
    await memory.close()
    assert len(inner.ingest_calls) == 1
    call = inner.ingest_calls[0]
    assert (call.room_id, call.event, call.channel_id) == ("r1", event, "ai")
    assert inner.clear_calls == ["r1"]
    assert inner.closed


def test_nested_wrappers_preserve_history_window(
    wrap: Callable[[MemoryProvider], MemoryProvider],
) -> None:
    memory = wrap(wrap(SlidingWindowMemory(50)))
    assert memory.recent_events_window == 50


class _Compacts(MockMemoryProvider):
    """A memory that compacts every history to its last message."""

    def __init__(self) -> None:
        super().__init__()
        self.compactions: list[tuple[str, list[AIMessage], dict[str, int], str | None]] = []

    async def compact(
        self,
        room_id: str,
        history: list[AIMessage],
        *,
        usage: dict[str, int],
        channel_id: str | None = None,
    ) -> list[AIMessage] | None:
        self.compactions.append((room_id, history, usage, channel_id))
        return history[-1:]


@pytest.mark.parametrize("kind", ["budget", "compacting", "retrieval", "handoff"])
async def test_compaction_reaches_inner(kind: str) -> None:
    """A decorator hands a compaction between two rounds to the memory it
    wraps; the summarizing memory compacts with its own tier instead."""
    inner = _Compacts()
    memory: MemoryProvider
    if kind == "budget":
        memory = BudgetAwareMemory(inner, 1000)
    elif kind == "compacting":
        memory = CompactingMemory(inner, MockAIProvider(), 1000)
    elif kind == "retrieval":
        memory = RetrievalMemory([], inner)
    else:
        memory = HandoffMemoryProvider(inner)
    history = [AIMessage(role="user", content="a"), AIMessage(role="assistant", content="b")]
    usage = {"input_tokens": 5}
    assert await memory.compact("r1", history, usage=usage, channel_id="ai") == history[-1:]
    assert inner.compactions == [("r1", history, usage, "ai")]
