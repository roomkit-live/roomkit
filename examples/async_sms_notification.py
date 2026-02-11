"""Async SMS notification during a voice conversation.

Demonstrates cross-channel coordination: an SMS arrives while the user
is talking to an AI agent via voice.  Instead of interrupting the call,
the message is silently queued and the AI mentions it at the next
natural moment — e.g. "By the way, you received an SMS from Mom …"

Key mechanism:
    1. A BEFORE_BROADCAST hook intercepts SMS events and blocks broadcast.
    2. The pending message is stored and the AI's system prompt is updated
       via the store.
    3. On the next voice turn the AI naturally surfaces the SMS.

Run with:
    uv run python examples/async_sms_notification.py
"""

from __future__ import annotations

import asyncio
import logging

from roomkit import (
    ChannelBinding,
    ChannelCategory,
    ChannelType,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    VoiceChannel,
)
from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.sms.mock import MockSMSProvider
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

logging.basicConfig(level=logging.INFO)

# Pending SMS messages queued for the AI to mention
pending_sms: list[dict[str, str]] = []


async def main() -> None:
    kit = RoomKit()

    # -- Providers ---------------------------------------------------------

    backend = MockVoiceBackend()

    # Four voice turns: each is SPEECH_START → (silence) → SPEECH_END
    vad = MockVADProvider(
        events=[
            # Turn 1 — weather question
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.95),
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"audio-turn-1",
                duration_ms=1500.0,
            ),
            # Turn 2 — AI will mention the SMS
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.93),
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"audio-turn-2",
                duration_ms=1200.0,
            ),
            # Turn 3 — user says yes, read it
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.94),
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"audio-turn-3",
                duration_ms=900.0,
            ),
            # Turn 4 — user dictates reply
            VADEvent(type=VADEventType.SPEECH_START, confidence=0.96),
            None,
            VADEvent(
                type=VADEventType.SPEECH_END,
                audio_bytes=b"audio-turn-4",
                duration_ms=2000.0,
            ),
        ]
    )

    stt = MockSTTProvider(
        transcripts=[
            "What's the weather looking like tomorrow?",
            "Sure, what's up?",
            "Yes please, read it to me.",
            "Yes, tell her I won't forget, I'll be there for sure.",
        ]
    )
    tts = MockTTSProvider()

    ai_provider = MockAIProvider(
        responses=[
            "Tomorrow looks sunny with a high of 75°F and a gentle breeze. "
            "Perfect day to be outside!",
            "By the way, you just received an SMS from Mom. Would you like me to read it for you?",
            'She says: "Don\'t forget dinner tonight at 7!" Would you like to send a reply?',
            "Done! I've sent your reply to Mom.",
        ]
    )

    sms_provider = MockSMSProvider()

    # -- Channels ----------------------------------------------------------

    voice = VoiceChannel(
        "voice-call",
        stt=stt,
        tts=tts,
        backend=backend,
        pipeline=AudioPipelineConfig(vad=vad),
    )

    sms = SMSChannel("sms-mobile", provider=sms_provider)

    ai = AIChannel(
        "ai-assistant",
        provider=ai_provider,
        system_prompt="You are a helpful voice assistant.",
    )

    for ch in [voice, sms, ai]:
        kit.register_channel(ch)

    # -- Room --------------------------------------------------------------

    await kit.create_room(room_id="call-room")
    await kit.attach_channel("call-room", "voice-call")
    await kit.attach_channel("call-room", "sms-mobile")
    await kit.attach_channel("call-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    # -- Hook: intercept SMS, queue for AI ---------------------------------

    @kit.hook(
        HookTrigger.BEFORE_BROADCAST,
        name="sms_intercept",
        channel_ids={"sms-mobile"},
    )
    async def intercept_sms(event: RoomEvent, ctx: RoomContext) -> HookResult:
        """Block SMS broadcast and queue the message for the AI to mention."""
        if not isinstance(event.content, TextContent):
            return HookResult.allow()

        sender = event.source.participant_id or "Unknown"
        body = event.content.body

        pending_sms.append({"sender": sender, "body": body})
        print(f'\n[hook] SMS intercepted from {sender}: "{body}"')
        print("[hook] Queued for AI to mention on next voice turn")

        # Update the AI binding's system_prompt so it knows about the SMS.
        # We go through the store directly because the hook already runs
        # inside the room lock (update_binding_metadata would deadlock).
        sms_info = "\n".join(f'- From {m["sender"]}: "{m["body"]}"' for m in pending_sms)
        ai_binding = await kit.store.get_binding("call-room", "ai-assistant")
        updated = ai_binding.model_copy(
            update={
                "metadata": {
                    **ai_binding.metadata,
                    "system_prompt": (
                        "You are a helpful voice assistant. "
                        "IMPORTANT: The user has pending SMS messages. "
                        "Mention them naturally after answering "
                        "the current question.\n"
                        f"Pending messages:\n{sms_info}"
                    ),
                }
            }
        )
        await kit.store.update_binding(updated)

        return HookResult.block(reason="SMS queued for AI notification")

    # -- Voice session -----------------------------------------------------

    session = await backend.connect("call-room", "user-1", "voice-call")
    binding = ChannelBinding(
        room_id="call-room",
        channel_id="voice-call",
        channel_type=ChannelType.VOICE,
    )
    voice.bind_session(session, "call-room", binding)

    # 640 bytes = 20 ms of 16 kHz 16-bit mono PCM (standard frame size)
    audio_data = b"\x00" * 640

    async def voice_turn() -> None:
        """Simulate one voice turn (3 audio frames → VAD → STT → AI)."""
        for _ in range(3):
            frame = AudioFrame(data=audio_data, sample_rate=16000)
            await backend.simulate_audio_received(session, frame)
        await asyncio.sleep(0.1)

    # -- Phase 1: Normal voice conversation --------------------------------

    print("\n--- Phase 1: User asks about the weather ---")
    await voice_turn()

    # -- Phase 2: SMS arrives mid-conversation -----------------------------

    print("\n--- Phase 2: SMS arrives (blocked from broadcast, queued) ---")
    result = await kit.process_inbound(
        InboundMessage(
            channel_id="sms-mobile",
            sender_id="Mom (+1 555-987-6543)",
            content=TextContent(body="Don't forget dinner tonight at 7!"),
        )
    )
    print(f"[result] SMS broadcast blocked: {result.blocked}")

    # -- Phase 3: AI mentions the SMS --------------------------------------

    print("\n--- Phase 3: User speaks, AI mentions the SMS ---")
    await voice_turn()

    # -- Phase 4: AI reads the SMS -----------------------------------------

    print("\n--- Phase 4: User asks to read it, AI reads the SMS ---")
    await voice_turn()

    # -- Phase 5: User replies, AI sends it --------------------------------

    print("\n--- Phase 5: User dictates reply, AI sends SMS ---")
    await voice_turn()

    # -- Conversation timeline ---------------------------------------------

    events = await kit.store.list_events("call-room")
    msg_events = [e for e in events if e.type.value == "message"]
    print(f"\n--- Conversation Timeline ({len(msg_events)} events) ---")
    for ev in msg_events:
        if isinstance(ev.content, TextContent):
            tag = ev.source.channel_type.value
            suffix = f"  [BLOCKED: {ev.blocked_by}]" if ev.blocked_by else ""
            print(f"  [{tag:>5}] {ev.content.body}{suffix}")

    await kit.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
