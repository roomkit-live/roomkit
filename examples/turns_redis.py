"""An agent's turns run in another process, carried by Redis (RFC §10.6).

The room's owner (this process) holds the agent by its declaration alone. An
executor in a second process (this script, run as ``executor``) builds the
agent and runs its turns. Redis carries everything between them; the store stays
with the room:

1. The owner starts an executor process and waits until it serves.
2. A person asks a question. The turn crosses Redis to the executor, which
   runs the agent; the answer comes back, and the owner stores and shows it.
3. The person asks for a long answer, and the executor process is killed
   while it streams. Within the heartbeat timeout the owner fails the turn
   with ``executor_lost``; what was shown stays, and the room goes on.
4. A new executor process starts and takes the room's next turn.

With ANTHROPIC_API_KEY set the agent is Claude Haiku; without, a scripted
model streams its answers, so the example runs without keys.

Requirements:
    pip install roomkit[redis]          (and roomkit[anthropic] for Claude)
    a Redis server: docker run --rm -d -p 6379:6379 redis:7, or set REDIS_URL

Run with:
    uv run python examples/turns_redis.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    AgentIdentity,
    Channel,
    ChannelBinding,
    ChannelCategory,
    ChannelDeclaration,
    ChannelOutput,
    ChannelType,
    EventSource,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
)
from roomkit.providers.ai import (
    AIContext,
    AIProvider,
    AIResponse,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.turns import TurnExecutor, TurnTimeouts

try:
    from redis.asyncio import from_url
    from redis.exceptions import RedisError

    from roomkit import RedisTurnBackend
except ImportError:
    print("Error: install the redis extra: pip install roomkit[redis]")
    sys.exit(1)

logger = setup_logging("example.turns_redis")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
ROOM = "support"
# Both processes keep the same bounds: short ones, so the loss shows quickly.
TIMEOUTS = TurnTimeouts(heartbeat_interval=0.5, heartbeat_timeout=2.0, accept=5.0, cancel=2.0)
LONG_ANSWER = (
    "Here is the whole story of the invoice, line by line, with every charge "
    "explained and every tax spelled out, slowly enough to read along."
)


# -- The executor's process: where the agent's live objects are --------------------


class ScriptedModel(AIProvider):
    """A model that answers a question at once, and a long answer slowly."""

    def __init__(self, executor_id: str) -> None:
        self._executor_id = executor_id

    @property
    def model_name(self) -> str:
        return "scripted"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_streaming(self) -> bool:
        return True

    async def generate(self, context: AIContext) -> AIResponse:
        return AIResponse(content="unused")

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        asked = str(context.messages[-1].content) if context.messages else ""
        if "long" not in asked:
            yield StreamTextDelta(text=f"Hello, this is {self._executor_id}. Your plan renews ")
            yield StreamTextDelta(text="on the first of the month.")
        else:
            for word in LONG_ANSWER.split(" "):
                yield StreamTextDelta(text=word + " ")
                await asyncio.sleep(0.3)
        yield StreamDone(finish_reason="stop")


def build_agent(executor_id: str) -> Any:
    """The executor's factory: the agent, with its live provider."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        provider: AIProvider = AnthropicAIProvider(
            AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-haiku-4-5")
        )
    else:
        provider = ScriptedModel(executor_id)

    def build(channel_id: str) -> Channel | None:
        if channel_id != "assistant":
            return None
        return Agent(
            "assistant",
            provider=provider,
            system_prompt="You answer billing questions in plain prose.",
        )

    return build


async def run_executor(url: str, prefix: str, executor_id: str) -> None:
    """Serve the agent ``assistant`` until the owner says stop (or kills us)."""
    backend = RedisTurnBackend(url, key_prefix=prefix, timeouts=TIMEOUTS)
    executor = TurnExecutor(
        build_agent(executor_id),
        hosts={"assistant"},
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


# -- The room owner's process -----------------------------------------------------


class Screen(Channel):
    """A web chat: prints the agent's answer as it streams."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.words = 0
        self.streaming = asyncio.Event()

    def watch(self) -> None:
        """Set ``streaming`` once the next answer showed a few words."""
        self.words = 0
        self.streaming.clear()

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(channel_id=self.channel_id, channel_type=self.channel_type),
            content=message.content,
        )

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        print("  assistant: ", end="", flush=True)
        try:
            async for chunk in text_stream:
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    self.words += len(chunk.split())
                    if self.words >= 5:
                        self.streaming.set()
        finally:
            print(flush=True)
        return ChannelOutput.empty()


class ExecutorProcess:
    """This script run as ``executor``, in a process of its own."""

    def __init__(self, process: asyncio.subprocess.Process, executor_id: str) -> None:
        self.process = process
        self.executor_id = executor_id

    @classmethod
    async def start(cls, prefix: str, executor_id: str) -> ExecutorProcess:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            "executor",
            REDIS_URL,
            prefix,
            executor_id,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None
        line = await asyncio.wait_for(process.stdout.readline(), 30)
        if line.strip() != b"ready":
            raise RuntimeError(f"executor {executor_id} did not start")
        logger.info("Executor %s serves in process %d", executor_id, process.pid)
        return cls(process, executor_id)

    async def stop(self) -> None:
        if self.process.returncode is None:
            assert self.process.stdin is not None
            self.process.stdin.write(b"stop\n")
            await self.process.stdin.drain()
            await self.process.wait()

    async def kill(self) -> None:
        self.process.kill()
        await self.process.wait()
        logger.info("Executor %s killed", self.executor_id)


async def check_redis() -> None:
    """Exit with a clear message when Redis cannot be reached."""
    client = from_url(REDIS_URL)
    try:
        await client.ping()
    except (RedisError, OSError) as exc:
        where = urlsplit(REDIS_URL)  # host and port only: a URL may carry a password
        print(
            f"Error: cannot reach Redis at {where.hostname}:{where.port or 6379} ({exc}).\n"
            "Start one with: docker run --rm -d -p 6379:6379 redis:7, or set REDIS_URL."
        )
        sys.exit(1)
    finally:
        await client.aclose()


async def ask(kit: RoomKit, text: str) -> Any:
    print(f"  person:    {text}")
    message = InboundMessage(channel_id="web", sender_id="marie", content=TextContent(body=text))
    return await kit.process_inbound(message, room_id=ROOM)


async def run_owner() -> None:
    await check_redis()
    prefix = f"roomkit:example:{uuid4().hex[:8]}"
    kit = RoomKit(
        turn_backend=RedisTurnBackend(REDIS_URL, key_prefix=prefix, timeouts=TIMEOUTS),
    )
    screen = Screen("web")
    kit.register_channel(screen)
    kit.register_channel(
        ChannelDeclaration(channel_id="assistant", identity=AgentIdentity(name="Assistant"))
    )
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "web")
    await kit.attach_channel(ROOM, "assistant", category=ChannelCategory.INTELLIGENCE)
    executor = await ExecutorProcess.start(prefix, "executor-1")
    try:
        logger.info("1. A question, answered in the executor's process")
        await ask(kit, "When does my plan renew?")

        logger.info("2. A long answer, and the executor killed while it streams")
        screen.watch()
        turn = asyncio.ensure_future(ask(kit, "Give me the long version, please."))
        await asyncio.wait_for(screen.streaming.wait(), 30)
        await executor.kill()
        result = await turn
        logger.info("The turn failed: %r", result.error)

        logger.info("3. A new executor takes the room's next turn")
        executor = await ExecutorProcess.start(prefix, "executor-2")
        await ask(kit, "When does my plan renew, again?")
    finally:
        await executor.stop()
        await kit.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "executor":
        asyncio.run(run_executor(*sys.argv[2:5]))
    else:
        asyncio.run(run_owner())
