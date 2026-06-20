"""Tool Search — progressive tool disclosure for large tool catalogues.

When an agent has dozens of tools, sending every schema to the model on
every turn burns context and makes smaller models hallucinate tool names.
Tool Search hides the catalogue behind two discovery tools and lets the
model reveal only what it needs:

- ``find_tools(query)`` — search the catalogue by natural language; the
  matches become directly invocable for the rest of the turn.
- ``list_tools(category)`` — list the catalogue (name + short description).

The model first sees only ``find_tools``/``list_tools`` plus a small
pinned set. When it calls ``find_tools``, the matched tools appear on the
NEXT tool-loop round — the text loop re-sends its (re-filtered) tool list
every round, so no provider reconfigure is needed (this is what makes Tool
Search work on any text/HTTP provider, not just realtime voice).

Activation is automatic (``tool_search=None``) when the deferrable tools would
exceed ``tool_search_threshold_pct`` % of the model's context window (default
10%), self-tuning to model size; it falls back to a ``tool_search_threshold``
tool count when the window is unknown. Pass ``tool_search=True``/``False`` to
force it on/off (this example forces ``True`` for determinism).

This example uses a scripted MockAIProvider so it runs without API keys and
prints the visible tool surface at each round.

Run with:
    uv run python examples/ai_tool_search.py
"""

from __future__ import annotations

import asyncio
import json

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider

# --- A large tool catalogue: a few meaningful tools + noise to cross the
# 20-tool auto-activation threshold. In a real agent these come from MCP. ---
REAL_TOOLS = [
    {
        "name": "send_sms",
        "description": "Send an SMS text message to a phone number.",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    },
    {
        "name": "lookup_contact",
        "description": "Look up a contact's phone number by name.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "get_help",
        "description": "Show what this assistant can do.",
        "parameters": {"type": "object", "properties": {}},
    },
]
NOISE_TOOLS = [
    {"name": f"widget_{i}", "description": f"Operate widget number {i}."} for i in range(25)
]
CATALOGUE = REAL_TOOLS + NOISE_TOOLS


async def tool_handler(name: str, arguments: dict) -> str:
    """Execute the (revealed) business tools. find_tools/list_tools are
    channel-managed and never reach here."""
    if name == "send_sms":
        return json.dumps({"status": "sent", "to": arguments.get("to")})
    return json.dumps({"ok": True, "tool": name})


def _visible(call) -> list[str]:
    # Each round's ``tools`` is a fresh list (the loop re-filters via
    # model_copy), so this is a faithful snapshot of what the model saw.
    return [t.name for t in call.tools]


def _find_tools_result(call) -> dict:
    """The find_tools payload from a call's tool-result messages.

    (``messages`` is appended in place across rounds, so we pick the result
    that carries ``matches`` rather than relying on position.)
    """
    for msg in call.messages:
        if msg.role != "tool":
            continue
        for part in msg.content:
            data = json.loads(part.result)
            if "matches" in data:
                return data
    return {"matches": []}


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    # Scripted model: search → call the revealed tool → answer.
    ai = AIChannel(
        "ai-assistant",
        provider=MockAIProvider(
            ai_responses=[
                # Round 0: the model can't see send_sms yet — it searches.
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        AIToolCall(
                            id="c1",
                            name="find_tools",
                            arguments={"query": "send a text message to a contact"},
                        )
                    ],
                ),
                # Round 1: send_sms is now revealed → call it directly.
                AIResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        AIToolCall(
                            id="c2",
                            name="send_sms",
                            arguments={"to": "+15145550123", "body": "On my way!"},
                        )
                    ],
                ),
                # Round 2: final answer.
                AIResponse(content="Done — I sent your text. ✅", finish_reason="stop"),
            ]
        ),
        system_prompt="You are a helpful assistant.",
        tool_handler=tool_handler,
        # Forced on for a deterministic demo. In production leave tool_search
        # at its default (None = auto): it self-enables when the deferrable
        # tools would exceed ~10% of the model's context window
        # (tool_search_threshold_pct), falling back to a tool count when the
        # window is unknown. Pin one tool so it stays visible without a search.
        tool_search=True,
        tool_search_pinned=["get_help"],
    )
    kit.register_channel(ws)
    kit.register_channel(ai)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    await kit.create_room(room_id="sms-room")
    await kit.attach_channel("sms-room", "ws-user")
    await kit.attach_channel(
        "sms-room",
        "ai-assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": CATALOGUE},
    )

    print(f"Catalogue: {len(CATALOGUE)} tools — Tool Search on (auto in prod: % of window)\n")

    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user",
            content=TextContent(body="Text Alex that I'm on my way"),
        )
    )

    provider: MockAIProvider = ai._provider  # type: ignore[assignment]

    # Round 0 — the model sees only the discovery tools + pinned.
    print("Round 0 — visible to the model:")
    print(f"  {_visible(provider.calls[0])}")
    print("  → send_sms is hidden; the model must find it.\n")

    # find_tools result (returned to the model before round 1).
    matches = _find_tools_result(provider.calls[-1])
    print("find_tools('send a text message to a contact') returned:")
    print(f"  matches: {[m['name'] for m in matches['matches']]}\n")

    # Round 1 — the matched tool is now invocable.
    print("Round 1 — visible to the model:")
    print(f"  {_visible(provider.calls[1])}")
    print("  → send_sms revealed; widget_* noise stays hidden.\n")

    for ev in inbox:
        body = getattr(ev.content, "body", None)
        if ev.source.channel_id == "ai-assistant" and body:
            print(f"AI replied: {body}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
