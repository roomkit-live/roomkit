"""An executor in a process of its own, for tests/test_turns_redis.py.

``python tests/redis_turn_executor.py <redis url> <key prefix> <executor id>
<sqlite path>`` serves the agent ``worker`` on a RedisTurnBackend, reading
the rooms in the SQLite store the room owner shares, prints ``ready``, and
prints ``streaming`` once a turn's model began. The model's script is the
user's text: ``hold`` waits on, ``stall`` blocks the event loop for three
seconds between its two words, anything else answers at once.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator

from roomkit.channels.agent import Agent
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.turns import TurnExecutor, TurnTimeouts
from roomkit.turns.redis import RedisTurnBackend

TIMEOUTS = TurnTimeouts(heartbeat_interval=0.2, heartbeat_timeout=1.0, accept=3.0, cancel=1.0)


class _Scripted(MockAIProvider):
    def __init__(self, executor_id: str) -> None:
        super().__init__(streaming=True)
        self._executor_id = executor_id

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="unused")

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        said = context.messages[-1].content if context.messages else ""
        text = said if isinstance(said, str) else str(said)
        print("streaming", flush=True)
        yield StreamTextDelta(text="Hello ")
        if "hold" in text:
            await asyncio.sleep(60)
        if "stall" in text:
            time.sleep(3)  # noqa: ASYNC251 - the turn's CPU: the event loop stops
        yield StreamTextDelta(text=f"from {self._executor_id}.")
        yield StreamDone(finish_reason="stop")


async def main(url: str, prefix: str, executor_id: str, database: str) -> None:
    backend = RedisTurnBackend(url, key_prefix=prefix, timeouts=TIMEOUTS)
    executor = TurnExecutor(
        lambda channel_id: Agent(channel_id, provider=_Scripted(executor_id)),
        hosts={"worker"},
        executor_id=executor_id,
        timeouts=TIMEOUTS,
    )
    await backend.serve(executor)
    print("ready", flush=True)
    try:
        await asyncio.to_thread(sys.stdin.readline)
    finally:
        await backend.close()
        await executor.close()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:5]))
