"""Background agent delegation via kit.delegate().

Demonstrates the first-class delegation API:

1. User talks to a voice agent on a call (mock backend, VAD, STT and TTS)
2. User asks for a review of the latest PR
3. The script calls ``kit.delegate()`` → child room created automatically
4. The child room shares the parent's EmailChannel: what is broadcast there
   reaches it — the task brief; the reviewer's answer is collected for the
   hand-back and stored, not broadcast
5. PR reviewer works in the background — its own event history
6. Voice conversation continues uninterrupted (turn 2 while the task runs): the
   user asks how the review is going, and the voice agent looks it up with its
   ``task_status`` tool, which reads the task on ``kit.status_bus``
7. When the child room completes, the result is handed back to the voice agent
   as an instruction through ``kit.deliver()`` (strategy and delivery hooks
   apply), its metadata naming the task (``task_id``, ``agent_id``,
   ``task_status``)
8. Voice agent tells the user the result, the user thanks it (turn 3)

Key concept:
    A background task IS a child room.  ``kit.delegate()`` handles the
    boilerplate: child room creation, channel sharing, agent execution,
    result routing, and hook firing — all in one call.

Run with:
    uv run python examples/background_agent_task.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import (
    ChannelCategory,
    HookExecution,
    HookResult,
    HookTrigger,
    RoomKit,
    TextContent,
    VoiceChannel,
)
from roomkit.channels import EmailChannel
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.enums import EventType
from roomkit.models.event import SystemContent
from roomkit.providers.ai import AIContext, AIResponse
from roomkit.providers.ai.base import AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.email.mock import MockEmailProvider
from roomkit.tasks import TASK_STATUS_TOOL, TaskStatusTool
from roomkit.voice import VoiceCapability
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

logger = setup_logging("example.background_task")

# What the mock VAD hands the STT at each end of speech: 100 ms of 16 kHz
# 16-bit mono PCM silence (an even byte count, as PCM requires).
_SPEECH = b"\x00\x00" * 1600

# How long the mock PR reviewer takes: long enough for the user's second
# turn to happen while it works.
_REVIEW_SECONDS = 1.0


class _SlowReviewerAI(MockAIProvider):
    """A scripted reviewer that takes a while, like a real review would."""

    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(_REVIEW_SECONDS)
        return await super().generate(context)


async def _show_hand_back(kit: RoomKit, voice_ai: MockAIProvider) -> None:
    """Show that the notified agent was told the result, its prompt untouched.

    The result is an instruction addressed to it: it answers at once, and its
    own prompt is left as configured (RFC §23.3).
    """
    voice_binding = await kit.store.get_binding("call-room", "voice-assistant")
    prompt = voice_binding.metadata.get("system_prompt", "") if voice_binding else ""
    print(f"\n  Voice agent prompt holds no task result: {'PR #42' not in prompt}")
    told = any("Background task from" in str(call.messages[-1].content) for call in voice_ai.calls)
    print(f"  Voice agent was told the result: {told}")


async def main() -> None:
    kit = RoomKit()

    # ── Providers ─────────────────────────────────────────────────────

    # The caller's device cancels its own echo, as a browser or a phone does:
    # the user may answer as soon as the agent stops.  Without it, the channel
    # treats speech heard in the 2 s after each reply as echo and drops it.
    backend = MockVoiceBackend(capabilities=VoiceCapability.NATIVE_AEC)

    vad = MockVADProvider(
        events=[
            # Turn 1 — user asks for PR review
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.95),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=_SPEECH, duration_ms=2000.0),
            # Turn 2 — user chats while task runs
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.93),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=_SPEECH, duration_ms=1500.0),
            # Turn 3 — user thanks the agent once it has relayed the result
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.94),
            None,
            VADEvent(type=VADEventType.SPEECH_END, audio_bytes=_SPEECH, duration_ms=1000.0),
        ]
    )

    stt = MockSTTProvider(
        transcripts=[
            "Can you review the latest PR on roomkit for me?",
            "How is the PR review going?",
            "Great, thanks for the update!",
        ]
    )
    tts = MockTTSProvider()

    # Voice assistant — front-facing, talks to the user. On turn 2 it calls
    # its task_status tool, then answers from what the tool said.
    voice_ai = MockAIProvider(
        ai_responses=[
            AIResponse(
                content=(
                    "I'll review the latest PR on roomkit for you right away. "
                    "I'm delegating this to the PR reviewer — you'll have the "
                    "summary shortly. What else can I help with?"
                )
            ),
            AIResponse(
                content="",
                tool_calls=[AIToolCall(id="status-1", name=TASK_STATUS_TOOL, arguments={})],
            ),
            AIResponse(content="The reviewer is still on it. I'll tell you as soon as it's done."),
            # Answer to the hand-back instruction, not to a user turn
            AIResponse(
                content=(
                    "Great news — the PR review just came back! "
                    "PR #42 adds a TaskExecutor ABC with InMemory implementation: "
                    "340 additions, 45 deletions across 8 files, 12 unit tests. "
                    "Assessment: clean implementation, ready to merge."
                )
            ),
            AIResponse(content="You're welcome! Talk to you later."),
        ]
    )

    # PR reviewer — background agent, works in a child room
    pr_reviewer_ai = _SlowReviewerAI(
        responses=[
            (
                "## PR #42: Add background task executor\n\n"
                "**Author:** alex | **Files:** 8 | **+340 / -45**\n\n"
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
            "Keep chatting with the user while tasks run, and check on them "
            f"with {TASK_STATUS_TOOL}."
        ),
        # Reads the room's tasks on kit.status_bus (RFC §23.4).
        tools=[TaskStatusTool(kit)],
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

    # The hand-back is an instruction whose metadata names its task.
    @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def on_hand_back(event, ctx):
        meta = event.metadata
        if "task_id" in meta:
            logger.info(
                "[hook] Hand-back of %s from %s: %s",
                meta["task_id"],
                meta["agent_id"],
                meta["task_status"],
            )
        return HookResult.allow()

    # ── Parent room ───────────────────────────────────────────────────

    # The call stays between the caller and the voice agent (visibility), so
    # the email channel only carries what is broadcast in the child room.
    await kit.create_room(room_id="call-room")
    await kit.attach_channel("call-room", "voice-call", visibility="voice-assistant")
    await kit.attach_channel(
        "call-room",
        "voice-assistant",
        category=ChannelCategory.INTELLIGENCE,
        visibility="voice-call",
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
            "email_address": "alex@company.com",
        },
    )

    # ── Voice session ─────────────────────────────────────────────────

    session = await backend.connect("call-room", "user-1", "voice-call")
    await kit.join("call-room", "voice-call", session=session)

    audio_data = b"\x00" * 640  # 20ms of 16kHz 16-bit mono PCM

    async def voice_turn() -> None:
        """One user utterance (three frames: start, speech, end) and its reply."""
        for _ in range(3):
            frame = AudioFrame(data=audio_data, sample_rate=16000)
            await backend.simulate_audio_received(session, frame)
        await asyncio.sleep(0.1)
        await voice.wait_playback_done("call-room")

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
            "requester": "alex",
            "email": "alex@company.com",
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
    print(f"  Task status after turn 2: {task.status}")
    result = await task.wait(timeout=5.0)
    print(f"\n[result] Status: {result.status}")
    print(f"[result] Duration: {result.duration_ms:.0f}ms")
    if result.output:
        print(f"[result] Preview: {result.output[:80]}...")

    # The hand-back makes the voice agent speak the result unprompted.
    await asyncio.sleep(0.1)
    await voice.wait_playback_done("call-room")

    # ── Turn 3: User thanks the agent ─────────────────────────────────

    print("\n--- Turn 3: User thanks the agent ---")
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
            print(f"  [{src:>12}] {ev.content.body[:85].replace(chr(10), ' ')}")
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
            print(f"  [{src:>12}] {ev.content.body[:85].replace(chr(10), ' ')}")
        elif isinstance(ev.content, SystemContent):
            print(f"  [      system] {ev.content.body[:85]}")

    print(f"\n  Total events: {len(child_events)}")

    # ── Verify child room state ───────────────────────────────────────

    child_room = await kit.get_room(task.child_room_id)
    print(f"\n  Child room status:  {child_room.metadata.get('task_status')}")
    print(f"  Child room parent:  {child_room.metadata.get('parent_room_id')}")
    print(f"  Child room agent:   {child_room.metadata.get('task_agent_id')}")

    await _show_hand_back(kit, voice_ai)
    answers = [
        part.result
        for call in voice_ai.calls
        for message in call.messages
        if message.role == "tool" and isinstance(message.content, list)
        for part in message.content
        if getattr(part, "name", None) == TASK_STATUS_TOOL
    ]
    print(f"  task_status answered the voice agent: {answers[0][:110] if answers else None}")

    print("\n" + "=" * 70)
    print("  STATUS BUS (kit.status_bus)")
    print("=" * 70)
    print(await kit.status_bus.recent_text(10))

    print(f"\n  STT transcriptions: {len(stt.calls)} of 3 user turns")
    print(f"  Lines the voice agent spoke (TTS): {len(tts.calls)}")
    # The shared email channel received the child room's broadcast: the brief.
    print(f"  Emails sent through the shared channel: {len(email_provider.sent)}")
    for sent in email_provider.sent:
        body = sent["event"].content.body.replace("\n", " ")
        print(f"    to {sent['to']}: {body[:60]}...")

    await kit.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
