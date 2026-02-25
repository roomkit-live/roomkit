"""Channel-aware content adaptation with visibility routing.

Demonstrates how the same conversation can deliver different content to
different channels using visibility on ChannelBinding — zero custom code.

A code reviewer produces detailed output (file names, line numbers, code
snippets) visible ONLY on a text dashboard. A voice mediator speaks a
short conversational summary to the developer's ear. Same room, same
conversation, different content per channel.

Architecture:

    Developer speaks → "voice-dev" channel → transcript → Router
                                                            │
                                                  ┌─────────┴─────────┐
                                                  │                   │
                                            agent-mediator      agent-reviewer
                                            visibility="all"    visibility="ws-dashboard"
                                                  │                   │
                                             summary text       detailed report
                                                  │                   │
                                        ┌─────────┴─────┐      ┌─────┴───────────┐
                                        │               │      │                 │
                                  voice-dev       ws-dashboard  ws-dashboard   voice-dev
                                  ✓ RECEIVES      ✓ RECEIVES   ✓ RECEIVES    ✗ FILTERED

In production, "voice-dev" would be a VoiceChannel with STT/TTS (or a
RealtimeVoiceChannel with Gemini Live / OpenAI Realtime). The visibility
pattern works identically regardless of transport type.

Run with:
    uv run python examples/orchestration_voice_mediator.py
"""

from __future__ import annotations

import asyncio
import logging

# Suppress chain-depth warnings from AI-to-AI reentry (expected in multi-agent setups)
logging.getLogger("roomkit").setLevel(logging.ERROR)

from roomkit import (
    Agent,
    ChannelCategory,
    ConversationPipeline,
    ConversationState,
    HandoffMemoryProvider,
    InboundMessage,
    PipelineStage,
    RoomKit,
    SlidingWindowMemory,
    TextContent,
    WebSocketChannel,
    get_conversation_state,
    set_conversation_state,
)
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.mock import MockAIProvider

# --- Pipeline definition ----------------------------------------------------

pipeline = ConversationPipeline(
    stages=[
        PipelineStage(phase="intake", agent_id="agent-mediator", next="review"),
        PipelineStage(phase="review", agent_id="agent-reviewer", next="synthesis"),
        PipelineStage(phase="synthesis", agent_id="agent-mediator", next=None),
    ],
)


# --- Helpers -----------------------------------------------------------------


def collect(inbox: list[RoomEvent], agent_id: str, start: int = 0) -> RoomEvent | None:
    """Find the first event from a specific agent after `start` index."""
    for event in inbox[start:]:
        if event.source.channel_id == agent_id:
            return event
    return None


# --- Main --------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    # --- Transport channels --------------------------------------------------
    # Text dashboard (developer's screen — sees everything)
    ws_dashboard = WebSocketChannel("ws-dashboard")
    text_events: list[RoomEvent] = []

    async def on_text(_conn: str, event: RoomEvent) -> None:
        text_events.append(event)

    ws_dashboard.register_connection("dev-screen", on_text)
    kit.register_channel(ws_dashboard)

    # Simulated voice channel
    # In production: VoiceChannel with STT/TTS, or RealtimeVoiceChannel.
    # Visibility filtering works identically regardless of transport type.
    ws_voice = WebSocketChannel("voice-dev")
    voice_events: list[RoomEvent] = []

    async def on_voice(_conn: str, event: RoomEvent) -> None:
        voice_events.append(event)

    ws_voice.register_connection("dev-earpiece", on_voice)
    kit.register_channel(ws_voice)

    # --- AI agents -----------------------------------------------------------
    mediator = Agent(
        "agent-mediator",
        provider=MockAIProvider(
            responses=[
                "OK, reviewing PR 100 now.",
                (
                    "Two critical issues and three warnings. "
                    "Don't merge yet. Details are on your screen."
                ),
            ]
        ),
        role="Voice mediator",
        description="Speaks concise summaries to the developer",
        system_prompt=(
            "You are a VOICE interface for a developer. "
            "NEVER say file names, line numbers, or code. "
            "Give verdict, severity count, and tell them to check their screen. "
            "Max 30 words."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=20)),
    )

    reviewer = Agent(
        "agent-reviewer",
        provider=MockAIProvider(
            responses=[
                "## PR #100 Review\n\n"
                "### Critical\n"
                "1. SQL injection in `webhook_store.py:145`\n"
                "2. Missing rate limit on `/api/webhooks/retry`\n\n"
                "### Warnings\n"
                "3. Unused import `os` in `utils.py:2`\n"
                "4. Test coverage dropped 87% -> 72%\n"
                "5. Inconsistent error format in `handlers.py`\n\n"
                "**Verdict: REQUEST CHANGES**",
            ]
        ),
        role="Code reviewer",
        description="Produces detailed code reviews with file names and line numbers",
        system_prompt=(
            "You produce detailed code reviews with file names, line numbers, "
            "and code snippets. Be thorough."
        ),
        memory=HandoffMemoryProvider(SlidingWindowMemory(max_events=20)),
    )

    for ch in [mediator, reviewer]:
        kit.register_channel(ch)

    # --- Orchestration -------------------------------------------------------
    router, handler = pipeline.install(kit, [mediator, reviewer])

    # --- Room setup ----------------------------------------------------------
    await kit.create_room(room_id="code-review")

    # Both transport channels
    await kit.attach_channel("code-review", "ws-dashboard")
    await kit.attach_channel("code-review", "voice-dev")

    # Mediator: summary goes to ALL channels (voice + text)
    await kit.attach_channel(
        "code-review",
        "agent-mediator",
        category=ChannelCategory.INTELLIGENCE,
        visibility="all",
    )

    # Reviewer: detail goes ONLY to text dashboard
    await kit.attach_channel(
        "code-review",
        "agent-reviewer",
        category=ChannelCategory.INTELLIGENCE,
        visibility="ws-dashboard",
    )

    # Initialize state to intake (mediator handles first)
    room = await kit.get_room("code-review")
    room = set_conversation_state(
        room, ConversationState(phase="intake", active_agent_id="agent-mediator")
    )
    await kit.store.update_room(room)

    # --- Simulate conversation -----------------------------------------------

    # 1. Developer asks for a review (mediator acknowledges)
    print("=== Phase: Intake ===")
    print('  Developer: "Review PR 100 please."')
    mark = len(text_events)
    await kit.process_inbound(
        InboundMessage(
            channel_id="voice-dev",
            sender_id="developer",
            content=TextContent(body="Review PR 100 please."),
        )
    )
    reply = collect(text_events, "agent-mediator", mark)
    print(f"  Mediator: {reply.content.body}")  # type: ignore[union-attr]

    # 2. Handoff: mediator -> reviewer
    print("\n=== Handoff: mediator -> reviewer ===")
    result = await handler.handle(
        room_id="code-review",
        calling_agent_id="agent-mediator",
        arguments={
            "target": "agent-reviewer",
            "reason": "Code review requested",
            "summary": "Developer wants PR #100 reviewed.",
        },
    )
    print(f"  Accepted: {result.accepted}, Phase: {result.new_phase}")

    # 3. Developer prompts the review (reviewer produces detailed output)
    print("\n=== Phase: Review ===")
    print('  Developer: "Go ahead."')
    await kit.process_inbound(
        InboundMessage(
            channel_id="voice-dev",
            sender_id="developer",
            content=TextContent(body="Go ahead."),
        )
    )

    # 4. Handoff: reviewer -> mediator for voice synthesis
    print("\n=== Handoff: reviewer -> mediator (synthesis) ===")
    result = await handler.handle(
        room_id="code-review",
        calling_agent_id="agent-reviewer",
        arguments={
            "target": "agent-mediator",
            "reason": "Review complete, summarize for voice",
            "summary": (
                "2 critical (SQL injection, missing rate limit), "
                "3 warnings. Verdict: request changes."
            ),
        },
    )
    print(f"  Accepted: {result.accepted}, Phase: {result.new_phase}")

    # 5. Developer asks for the summary (mediator speaks)
    print("\n=== Phase: Synthesis ===")
    print('  Developer: "What\'s the verdict?"')
    await kit.process_inbound(
        InboundMessage(
            channel_id="voice-dev",
            sender_id="developer",
            content=TextContent(body="What's the verdict?"),
        )
    )

    # --- Results: prove visibility filtering works ---------------------------

    print("\n" + "=" * 60)
    print("=== What each channel received ===")
    print(f"  Text dashboard: {len(text_events)} events (detailed review + summaries)")
    print(f"  Voice channel:  {len(voice_events)} events (summaries only)")

    print("\n--- Text dashboard saw ---")
    for e in text_events:
        agent = e.source.channel_id
        preview = e.content.body[:80].replace("\n", " ")  # type: ignore[union-attr]
        print(f"  [{agent}] {preview}...")

    print("\n--- Voice channel heard ---")
    for e in voice_events:
        agent = e.source.channel_id
        print(f"  [{agent}] {e.content.body}")  # type: ignore[union-attr]

    # Verify the key insight
    reviewer_on_voice = [e for e in voice_events if e.source.channel_id == "agent-reviewer"]
    reviewer_on_text = [e for e in text_events if e.source.channel_id == "agent-reviewer"]
    print(f"\n  Reviewer events on voice: {len(reviewer_on_voice)} (filtered out)")
    print(f"  Reviewer events on text:  {len(reviewer_on_text)} (detailed report)")

    # Phase history
    print("\n=== Phase History ===")
    room = await kit.get_room("code-review")
    state = get_conversation_state(room)
    for t in state.phase_history:
        print(f"  {t.from_phase} -> {t.to_phase} ({t.reason})")

    # Cleanup
    for ch in [mediator, reviewer]:
        await ch.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
