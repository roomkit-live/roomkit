"""One trace across processes: an agent's turn runs elsewhere (RFC §10.6.9).

The room's owner holds the agent by its declaration alone; an executor builds
the agent and runs its turns, the turns carried by Redis. Here both sides are
two backends of this one script, which cross Redis exactly as two processes
do, each with its own OpenTelemetryProvider (its own tracer). The turn's
request carries the W3C trace context of the span it runs under, so the
executor's spans (the generation) continue the owner's trace: the script
prints each side's spans and the one trace id they share.

Requirements:
    pip install roomkit[redis,opentelemetry]
    a Redis server: docker run --rm -d -p 6379:6379 redis:7, or set REDIS_URL

Run with:
    uv run python examples/telemetry_otel_turns.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    Agent,
    AgentIdentity,
    ChannelCategory,
    ChannelDeclaration,
    InboundMessage,
    RedisTurnBackend,
    RoomKit,
    TextContent,
    TurnExecutor,
    WebSocketChannel,
)
from roomkit.providers.ai import MockAIProvider

logger = setup_logging("telemetry_otel_turns")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


def otel(exporter: Any, service: str) -> Any:
    """One process's provider: a tracer of its own, the shared exporter."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from roomkit.telemetry import OpenTelemetryProvider

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetryProvider(tracer_provider=tracer_provider, service_name=service)


async def main() -> None:
    try:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except ImportError:
        logger.error("Install the OpenTelemetry SDK: pip install roomkit[opentelemetry]")
        return
    exporter = InMemorySpanExporter()
    prefix = f"example:trace:{uuid4().hex[:8]}"
    await run(exporter, prefix)


async def run(exporter: Any, prefix: str) -> None:
    """The owner and the executor, one turn, then each side's spans."""
    # The owner: the room, the agent by its declaration, its own telemetry.
    kit = RoomKit(
        turn_backend=RedisTurnBackend(REDIS_URL, key_prefix=prefix),
        telemetry=otel(exporter, "owner"),
    )
    kit.register_channel(WebSocketChannel("ws-user"))
    kit.register_channel(ChannelDeclaration(channel_id="assistant", identity=AgentIdentity()))
    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "ws-user")
    await kit.attach_channel("support", "assistant", category=ChannelCategory.INTELLIGENCE)

    # The executor: builds the agent, runs its turns, its own telemetry.
    backend = RedisTurnBackend(REDIS_URL, key_prefix=prefix)
    executor = TurnExecutor(
        lambda channel_id: Agent(
            channel_id, provider=MockAIProvider(responses=["Your plan renews on the 1st."])
        ),
        hosts={"assistant"},
        telemetry=otel(exporter, "executor"),
    )
    await backend.serve(executor)

    result = await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user-1",
            content=TextContent(body="When does my plan renew?"),
        ),
        room_id="support",
    )
    await kit.close()
    await backend.close()
    await executor.close()

    for span in exporter.get_finished_spans():
        logger.info(
            "%-8s %-36s trace %032x",
            span.instrumentation_scope.name,
            span.name,
            span.context.trace_id,
        )
    traces = {span.context.trace_id for span in exporter.get_finished_spans()}
    outcome = "answered" if result.error is None else f"failed ({result.error})"
    logger.info("The turn %s; its spans make %d trace across both sides", outcome, len(traces))


if __name__ == "__main__":
    asyncio.run(main())
