"""External tool handler — control and observe provider-executed tools.

Demonstrates how to use ExternalToolHandler to intercept tool calls
executed by an external AI provider (e.g., a Claude Code sandbox).
The handler applies a ToolPolicy to approve or deny tools, and fires
BEFORE_TOOL_USE / ON_TOOL_CALL hooks for observability.

This example uses PolicyExternalToolHandler (auto-approve with policy)
and shows:
  - Denying dangerous tools (Bash) via ToolPolicy
  - Observing tool results via ON_TOOL_CALL hook
  - Gating tools via BEFORE_TOOL_USE hook
  - The handler lifecycle (start/stop)

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/external_tool_handler.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging

from roomkit import (
    AIChannel,
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.external import PolicyExternalToolHandler
from roomkit.tools.policy import ToolPolicy


async def main() -> None:
    setup_logging()

    # --- Mock AI that calls tools ---
    responses = [
        AIResponse(
            content="Let me check that.",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=[
                AIToolCall(id="tc-1", name="Read", arguments={"path": "/etc/hostname"}),
            ],
        ),
        AIResponse(
            content="The hostname is roomkit-dev.",
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 10},
        ),
    ]
    provider = MockAIProvider(ai_responses=responses)

    # --- Tool handler that resolves tools locally ---
    async def tool_handler(name: str, args: dict[str, Any]) -> str:
        if name == "Read":
            return f"Contents of {args.get('path', '?')}: roomkit-dev"
        return f"Unknown tool: {name}"

    # --- External tool handler with policy ---
    # Deny "Bash" and "Write", allow everything else
    handler = PolicyExternalToolHandler(
        policy=ToolPolicy(deny=["Bash", "Write"]),
    )

    ai = AIChannel(
        "ai-agent",
        provider=provider,
        system_prompt="You are a helpful assistant with file access.",
        tool_handler=tool_handler,
        tools=[
            AITool(name="Read", description="Read a file.", parameters={}),
            AITool(name="Bash", description="Run a command.", parameters={}),
        ],
        external_tool_handler=handler,
    )

    kit = RoomKit()
    kit.register_channel(ai)

    # --- Hooks for observability ---
    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="audit-gate")
    async def audit_gate(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        print(f"  [BEFORE_TOOL_USE] Tool: {event.name}, Args: {event.arguments}")
        # Allow all — the PolicyExternalToolHandler handles deny logic
        return HookResult(action="allow")

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="observe")
    async def observe(event: ToolCallEvent, ctx: RoomContext) -> HookResult:
        result_preview = (event.result or "")[:80]
        print(f"  [ON_TOOL_CALL] Tool: {event.name}, Result: {result_preview}")
        return HookResult(action="allow")

    # --- Create room and run ---
    room = await kit.create_room(room_id="ext-tool-room")
    await kit.attach_channel(room.id, "ai-agent", category=ChannelCategory.INTELLIGENCE)

    # Start the external handler
    await handler.start()

    print("--- Testing allowed tool (Read) ---")
    decision = await handler.process_tool_call("Read", {"path": "/etc/hostname"})
    print(f"  Decision: approved={decision.approved}")

    print("\n--- Testing denied tool (Bash) ---")
    decision = await handler.process_tool_call("Bash", {"command": "rm -rf /"})
    print(f"  Decision: approved={decision.approved}, reason={decision.reason}")

    print("\n--- Sending message to trigger AI tool use ---")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ai-agent",
            sender_id="user-1",
            content=TextContent(body="What's the hostname?"),
        )
    )
    await asyncio.sleep(0.3)

    # Show stored events
    events = await kit.store.list_events(room.id)
    for ev in events:
        if isinstance(ev.content, TextContent):
            print(f"  [{ev.source.channel_id}] {ev.content.body[:80]}")

    # Cleanup
    await handler.stop()
    await kit.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
