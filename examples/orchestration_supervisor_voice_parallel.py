"""Voice parallel analysis workflow with Grok Realtime.

Same orchestration pattern as the CLI parallel example but using local
microphone/speakers with xAI Grok Realtime for voice.

The supervisor speaks via Grok speech-to-speech. Workers (technical +
business analysts) run as text agents with Anthropic Haiku in child
rooms. When the user asks for analysis, a hook triggers delegation
and injects results back into the voice session.

    Human (voice) → Grok → [Technical | Business] → Grok → Human (voice)

Requirements:
    pip install roomkit[local-audio] websockets aec-audio-processing

Run with:
    XAI_API_KEY=xai-... ANTHROPIC_API_KEY=sk-... \\
        uv run python examples/orchestration_supervisor_voice_parallel.py

Environment variables:
    XAI_API_KEY         (required) xAI API key
    ANTHROPIC_API_KEY   (required) Anthropic API key for workers
    XAI_MODEL           Grok model (default: grok-2-audio)
    XAI_VOICE           Voice preset: eve | ara | rex | sal | leo (default: eve)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from shared.env import require_env

from roomkit import (
    Agent,
    HookExecution,
    HookTrigger,
    RealtimeVoiceChannel,
    RoomKit,
    WaitForIdle,
)
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.anthropic.ai import AnthropicAIProvider
from roomkit.providers.anthropic.config import AnthropicConfig
from roomkit.providers.xai.config import XAIRealtimeConfig
from roomkit.providers.xai.realtime import XAIRealtimeProvider
from roomkit.voice.backends.local import LocalAudioBackend
from roomkit.voice.pipeline.aec.webrtc import WebRTCAECProvider
from roomkit.voice.pipeline.config import AudioPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("voice_parallel")
logging.getLogger("roomkit").setLevel(logging.WARNING)
logging.getLogger("roomkit.tasks").setLevel(logging.DEBUG)


async def main() -> None:
    env = require_env("XAI_API_KEY", "ANTHROPIC_API_KEY")

    # --- Voice transport (local mic/speakers) --------------------------------

    sample_rate = 24000
    aec = WebRTCAECProvider(sample_rate=sample_rate, enable_ns=True)
    pipeline = AudioPipelineConfig(aec=aec)

    transport = LocalAudioBackend(
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        block_duration_ms=20,
        mute_mic_during_playback=False,
        aec=aec,
        pipeline=pipeline,
    )

    # --- Grok Realtime provider (speech-to-speech) ---------------------------

    xai_config = XAIRealtimeConfig(
        api_key=env["XAI_API_KEY"],
        model=os.environ.get("XAI_MODEL", "grok-2-audio"),
        voice=os.environ.get("XAI_VOICE", "eve"),
    )
    xai_provider = XAIRealtimeProvider(xai_config)

    # --- Workers (text agents, run in child rooms) ---------------------------

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    technical = Agent(
        "agent-technical",
        provider=AnthropicAIProvider(haiku_config),
        role="Technical analyst",
        system_prompt=(
            "You are a technical analyst. Analyze the given topic: "
            "architecture, implementation, scalability, trade-offs. "
            "Be concise (3-4 points)."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    business = Agent(
        "agent-business",
        provider=AnthropicAIProvider(haiku_config),
        role="Business analyst",
        system_prompt=(
            "You are a business analyst. Analyze the given topic: "
            "market impact, competitive positioning, revenue, strategy. "
            "Be concise (3-4 points)."
        ),
        memory=SlidingWindowMemory(max_events=50),
    )

    # --- Kit setup (no Supervisor orchestration — hook-based) ----------------

    kit = RoomKit(delivery_strategy=WaitForIdle(buffer=15.0))

    # Register workers on the kit (not attached to room)
    kit.register_channel(technical)
    kit.register_channel(business)

    # Voice channel
    voice = RealtimeVoiceChannel(
        "voice",
        provider=xai_provider,
        transport=transport,
        system_prompt=(
            "You are a project supervisor. When the user asks for analysis, "
            "tell them you're dispatching your analysts and the results will "
            "be ready shortly. You can continue chatting while they work."
        ),
        voice=xai_config.voice,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
    )
    kit.register_channel(voice)

    # --- Parallel delegation via hooks ---------------------------------------
    #
    # When Grok transcribes the user's speech, we check if it's an analysis
    # request and trigger parallel delegation. Results are injected back
    # into the voice session via kit.deliver().

    analysis_keywords = {"analyze", "analyse", "analysis", "research", "investigate"}

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def log_transcription(event: RoomEvent, ctx: RoomContext) -> None:
        """Log all transcriptions so the conversation is visible."""
        text = getattr(event, "text", "")
        role = getattr(event, "role", "?")
        if text.strip():
            label = "You" if role == "user" else "Grok"
            print(f"\n\033[{'33' if role == 'user' else '36'}m{label}:\033[0m {text}")

    async def _extract_topic(text: str) -> str:
        """Use Haiku to extract the topic from conversational speech."""
        from roomkit.providers.ai.base import AIContext, AIMessage
        from roomkit.providers.anthropic.ai import AnthropicAIProvider

        provider = AnthropicAIProvider(haiku_config)
        try:
            response = await provider.generate(
                AIContext(
                    system_prompt=(
                        "Extract the core topic from the user's request. "
                        "Output only the topic name, nothing else."
                    ),
                    messages=[AIMessage(role="user", content=text)],
                )
            )
            return response.content.strip()
        finally:
            await provider.close()

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def on_transcription(event: RoomEvent, ctx: RoomContext) -> None:
        # ON_TRANSCRIPTION receives RealtimeTranscriptionEvent, not RoomEvent
        text = getattr(event, "text", "").lower()
        role = getattr(event, "role", "user")
        if role != "user" or not any(kw in text for kw in analysis_keywords):
            return

        session = getattr(event, "session", None)
        room_id = session.room_id if session else "voice-room"
        logger.info("[hook] Analysis request detected: %s", text[:80])

        # Extract clean topic from conversational speech
        topic = await _extract_topic(text)
        logger.info("[hook] Extracted topic: %s", topic)

        # Run both workers in parallel with the clean topic
        async def delegate_one(agent: Agent) -> dict[str, str]:
            delegated = await kit.delegate(
                room_id,
                agent.channel_id,
                topic,
                wait=True,
            )
            output = ""
            if delegated.result:
                output = delegated.result.output or delegated.result.error or ""
            return {"worker": agent.channel_id, "output": output}

        results = await asyncio.gather(
            delegate_one(technical),
            delegate_one(business),
        )

        # Log worker outputs
        for r in results:
            worker = r["worker"].removeprefix("agent-")
            preview = r["output"][:120] + "..." if len(r["output"]) > 120 else r["output"]
            logger.info("[result] %s: %s", worker, preview)

        # Format and deliver results back to the voice session
        parts = []
        for r in results:
            worker = r["worker"].removeprefix("agent-")
            parts.append(f"{worker}: {r['output']}")

        combined = "\n\n".join(parts)
        logger.info("[hook] Both analysts completed, delivering results")

        await kit.deliver(
            room_id,
            f"Analysis results are ready. Here's what the analysts found:\n\n{combined}",
            channel_id="voice",
        )

    # --- Observability hooks -------------------------------------------------

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        task_id = event.metadata.get("task_id", "?")
        logger.info("[delegated] %s (task %s)", agent, task_id)

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        status = event.metadata.get("task_status", "?")
        duration = event.metadata.get("duration_ms", 0)
        logger.info("[completed] %s (%s, %.0fms)", agent, status, duration)

    # --- Room + session ------------------------------------------------------

    await kit.create_room(room_id="voice-room")
    await kit.attach_channel("voice-room", "voice")

    session = await voice.start_session(
        "voice-room",
        "local-user",
        connection=None,
    )

    logger.info("Voice session started (voice=%s, model=%s)", xai_config.voice, xai_config.model)
    logger.info("Speak into your microphone! Ask to analyze any topic.")
    logger.info("Press Ctrl+C to stop.\n")

    # --- Keep running until Ctrl+C -------------------------------------------

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    logger.info("\nStopping...")
    await voice.end_session(session)
    await kit.close()
    logger.info("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
