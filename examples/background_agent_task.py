"""Background agent delegation via kit.delegate().

Demonstrates the first-class delegation API:

1. User talks to a voice agent on a call
2. User asks "Review the last PR and email me the summary"
3. Voice agent calls ``kit.delegate()`` → child room created automatically
4. The child room shares the parent's EmailChannel
5. PR reviewer works in the background — its own event history
6. Voice conversation continues uninterrupted
7. When the child room completes, result flows back via system prompt injection
8. Voice agent tells the user the result

Key concept:
    A background task IS a child room.  ``kit.delegate()`` handles the
    boilerplate: child room creation, channel sharing, agent execution,
    result routing, and hook firing — all in one call.

Run with:
    uv run python examples/background_agent_task.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookExecution,
    HookTrigger,
    RoomKit,
    TextContent,
    VoiceChannel,
)
from roomkit.channels import EmailChannel
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.event import SystemContent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.email.mock import MockEmailProvider
from roomkit.voice.audio_frame import AudioFrame
from roomkit.voice.backends.mock import MockVoiceBackend
from roomkit.voice.pipeline import (
    AudioPipelineConfig,
    MockVADProvider,
    VADEvent,
    VADEventType,
)
from roomkit.voice.stt.mock import MockSTTProvider
from roomkit.voice.tts.mock import MockTTSProvider

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
# Suppress noisy voice pipeline errors from mock VAD audio frames
logging.getLogger("roomkit.voice").setLevel(logging.CRITICAL)
logger = logging.getLogger("example.background_task")


async def main() -> None:
    kit = RoomKit()

    # ── Providers ─────────────────────────────────────────────────────

    backend = MockVoiceBackend()

    vad = MockVADProvider(
        events=[
            # Turn 1 — user asks for PR review
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.95),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio-1", duration_ms=2000.0),
            # Turn 2 — user chats while task runs
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.93),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio-2", duration_ms=1500.0),
            # Turn 3 — agent delivers the result
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.94),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=b"audio-3", duration_ms=1000.0),
        ]
    )

    stt = MockSTTProvider(
        transcripts=[
            "Can you review the latest PR on roomkit and email me the summary?",
            "Sure, while we wait, what's on my calendar today?",
            "Great, thanks for the update!",
        ]
    )
    tts = MockTTSProvider()

    # Voice assistant — front-facing, talks to the user
    voice_ai = MockAIProvider(
        responses=[
            (
                "I'll review the latest PR on roomkit for you right away. "
                "I'm delegating this to the PR reviewer — you'll have the "
                "summary shortly. What else can I help with?"
            ),
            ("You have a team standup at 10am and a 1-on-1 with Sarah at 2pm. Anything else?"),
            (
                "Great news — the PR review just came back! "
                "PR #42 adds a TaskExecutor ABC with InMemory implementation: "
                "340 additions, 45 deletions across 8 files, 12 unit tests. "
                "Assessment: clean implementation, ready to merge. "
                "I've emailed you the full summary."
            ),
        ]
    )

    # PR reviewer — background agent, works in a child room
    pr_reviewer_ai = MockAIProvider(
        responses=[
            (
                "## PR #42: Add background task executor\n\n"
                "**Author:** quintana | **Files:** 8 | **+340 / -45**\n\n"
                "### Summary\n"
                "Adds `TaskExecutor` ABC with `InMemoryTaskExecutor`. "
                "Introduces child-room pattern for background agent work. "
                "Includes 12 unit tests covering lifecycle, cancellation, errors.\n\n"
                "### Assessment\n"
                "Clean implementation following RoomKit patterns. Good coverage. "
                "Ready to merge."
            ),
        ]
    )

    email_provider = MockEmailProvider()

    # ── Channels ──────────────────────────────────────────────────────

    voice = VoiceChannel(
        "voice-call",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
    )

    voice_agent = AIChannel(
        "voice-assistant",
        provider=voice_ai,
        system_prompt=(
            "You are a helpful voice assistant. "
            "You can delegate complex tasks to background agents. "
            "Keep chatting with the user while tasks run."
        ),
    )

    pr_reviewer = Agent(
        "pr-reviewer",
        provider=pr_reviewer_ai,
        role="PR Reviewer",
        description="Analyzes GitHub pull requests and produces summaries.",
        scope="Read GitHub PRs, analyze changes, produce assessments.",
    )

    email = EmailChannel(
        "email-out",
        provider=email_provider,
        from_address="assistant@company.com",
    )

    for ch in [voice, voice_agent, pr_reviewer, email]:
        kit.register_channel(ch)

    # ── Hooks ─────────────────────────────────────────────────────────

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event, ctx):
        logger.info("[hook] Task delegated: %s", event.metadata.get("task_id"))

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event, ctx):
        logger.info("[hook] Task completed: %s", event.metadata.get("task_id"))

    # ── Parent room ───────────────────────────────────────────────────

    await kit.create_room(room_id="call-room")
    await kit.attach_channel("call-room", "voice-call")
    await kit.attach_channel(
        "call-room",
        "voice-assistant",
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "system_prompt": (
                "You are a helpful voice assistant. "
                "You can delegate complex tasks to background agents."
            ),
        },
    )
    await kit.attach_channel(
        "call-room",
        "email-out",
        metadata={
            "from_": "assistant@company.com",
            "email_address": "quintana@company.com",
        },
    )

    # ── Voice session ─────────────────────────────────────────────────

    session = await backend.connect("call-room", "user-1", "voice-call")
    binding = ChannelBinding(
        room_id="call-room",
        channel_id="voice-call",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "call-room", binding)

    audio_data = b"\x00" * 640  # 20ms of 16kHz 16-bit mono PCM

    async def voice_turn() -> None:
        for _ in range(3):
            frame = AudioFrame(data=audio_data, sample_rate=16000)
            await backend.simulate_audio_received(session, frame)
        await asyncio.sleep(0.1)

    # ══════════════════════════════════════════════════════════════════
    #  SCENARIO
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  BACKGROUND AGENT DELEGATION (kit.delegate)")
    print("=" * 70)

    # ── Turn 1: User asks for PR review ───────────────────────────────

    print("\n--- Turn 1: User asks for PR review ---")
    await voice_turn()

    # ── Delegate to background agent ──────────────────────────────────
    #
    # One call replaces ~60 lines of manual boilerplate:
    # child room creation, agent attachment, channel sharing,
    # event injection, result collection, and parent notification.

    print("\n--- Delegating PR review (kit.delegate) ---")

    task = await kit.delegate(
        room_id="call-room",
        agent_id="pr-reviewer",
        task=(
            "Review the latest PR on the 'roomkit' repository. Produce a summary with assessment."
        ),
        context={
            "requester": "quintana",
            "email": "quintana@company.com",
        },
        share_channels=["email-out"],
        notify="voice-assistant",
    )

    print(f"  Task ID: {task.id}")
    print(f"  Child room: {task.child_room_id}")

    # ── Voice turn 2 + wait for result IN PARALLEL ────────────────────

    print("\n--- Turn 2 + Background task (PARALLEL) ---")

    # Voice conversation continues — user asks about calendar
    await voice_turn()

    # Wait for the background agent to finish
    result = await task.wait(timeout=5.0)
    print(f"\n[result] Status: {result.status}")
    print(f"[result] Duration: {result.duration_ms:.0f}ms")
    if result.output:
        print(f"[result] Preview: {result.output[:80]}...")

    # ── Turn 3: Voice agent delivers the result ───────────────────────

    print("\n--- Turn 3: Agent delivers the background result ---")
    await voice_turn()

    # ══════════════════════════════════════════════════════════════════
    #  TIMELINES
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  PARENT ROOM TIMELINE (call-room)")
    print("=" * 70)

    events = await kit.store.list_events("call-room")
    for ev in events:
        if isinstance(ev.content, TextContent):
            src = ev.source.channel_type or "?"
            print(f"  [{src:>12}] {ev.content.body[:85]}")
        elif isinstance(ev.content, SystemContent):
            print(f"  [      system] {ev.content.body[:85]}")

    print(f"\n  Total events: {len(events)}")

    print("\n" + "=" * 70)
    print(f"  CHILD ROOM TIMELINE ({task.child_room_id})")
    print("=" * 70)

    child_events = await kit.store.list_events(task.child_room_id)
    for ev in child_events:
        if isinstance(ev.content, TextContent):
            src = ev.source.channel_type or "?"
            print(f"  [{src:>12}] {ev.content.body[:85]}")
        elif isinstance(ev.content, SystemContent):
            print(f"  [      system] {ev.content.body[:85]}")

    print(f"\n  Total events: {len(child_events)}")

    # ── Verify child room state ───────────────────────────────────────

    child_room = await kit.get_room(task.child_room_id)
    print(f"\n  Child room status:  {child_room.metadata.get('task_status')}")
    print(f"  Child room parent:  {child_room.metadata.get('parent_room_id')}")
    print(f"  Child room agent:   {child_room.metadata.get('task_agent_id')}")

    # ── Verify notify binding was updated ─────────────────────────────

    voice_binding = await kit.store.get_binding("call-room", "voice-assistant")
    prompt = voice_binding.metadata.get("system_prompt", "") if voice_binding else ""
    print(f"\n  Voice agent prompt updated: {'BACKGROUND TASK COMPLETED' in prompt}")

    await kit.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
