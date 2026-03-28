"""Voice parallel analysis workflow with Grok Realtime.

Same orchestration pattern as the CLI parallel example but using local
microphone/speakers with xAI Grok Realtime for voice.

The framework handles everything:
- Intent detection (is this an analysis request?)
- Topic extraction (what to analyze?)
- Parallel worker delegation (technical + business)
- Result delivery (wait for Grok to stop speaking, then inject)

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import require_env, run_until_stopped, setup_console, setup_logging

from roomkit import (
    Agent,
    HookExecution,
    HookTrigger,
    RealtimeVoiceChannel,
    RoomKit,
    Supervisor,
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

logger = setup_logging("voice_parallel")
logging.getLogger("roomkit").setLevel(logging.WARNING)
logging.getLogger("roomkit.tasks").setLevel(logging.DEBUG)
logging.getLogger("roomkit.orchestration.strategies.supervisor").setLevel(logging.DEBUG)


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
    )

    # --- Grok Realtime provider (speech-to-speech) ---------------------------

    xai_config = XAIRealtimeConfig(
        api_key=env["XAI_API_KEY"],
        model=os.environ.get("XAI_MODEL", "grok-2-audio"),
        voice=os.environ.get("XAI_VOICE", "eve"),
    )
    xai_provider = XAIRealtimeProvider(xai_config)

    # --- Agents --------------------------------------------------------------

    haiku_config = AnthropicConfig(
        api_key=env["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    # Supervisor — config-only (Grok handles voice)
    supervisor = Agent(
        "agent-supervisor",
        role="Project supervisor",
        system_prompt="You coordinate analysis. Present results to the user.",
        voice="eve",
    )

    # Workers — text agents in child rooms
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

    # --- Supervisor orchestration (async delivery) ---------------------------
    #
    # auto_delegate=True + async_delivery=True:
    # - Framework installs ON_TRANSCRIPTION hook automatically
    # - Intent detection via first worker's provider (Haiku)
    # - Topic extraction via first worker's provider (Haiku)
    # - Workers run in background as asyncio task
    # - Results delivered via kit.deliver(WaitForIdle)
    # - Conversation continues uninterrupted

    kit = RoomKit(
        delivery_strategy=WaitForIdle(buffer=15.0),
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[technical, business],
            strategy="parallel",
            auto_delegate=True,
            async_delivery=True,
        ),
    )

    # --- Voice channel -------------------------------------------------------

    voice = RealtimeVoiceChannel(
        "voice",
        provider=xai_provider,
        transport=transport,
        system_prompt=supervisor._system_prompt or "",
        voice=xai_config.voice,
        input_sample_rate=sample_rate,
        output_sample_rate=sample_rate,
        pipeline=pipeline,
    )
    kit.register_channel(voice)

    # --- Observability hooks (optional — just for logging) -------------------

    @kit.hook(HookTrigger.ON_TRANSCRIPTION, execution=HookExecution.ASYNC)
    async def log_transcription(event: RoomEvent, ctx: RoomContext) -> None:
        text = getattr(event, "text", "")
        role = getattr(event, "role", "?")
        if text.strip():
            label = "You" if role == "user" else "Grok"
            print(f"\n\033[{'33' if role == 'user' else '36'}m{label}:\033[0m {text}")

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        logger.info("[delegated] %s", agent)

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event: RoomEvent, ctx: RoomContext) -> None:
        agent = event.metadata.get("agent_id", "?")
        duration = event.metadata.get("duration_ms", 0)
        logger.info("[completed] %s (%.0fms)", agent, duration)

    @kit.hook(HookTrigger.BEFORE_DELIVER, execution=HookExecution.ASYNC)
    async def before_deliver(event: RoomEvent, ctx: RoomContext) -> None:
        logger.info("[before_deliver] strategy=%s", event.metadata.get("strategy"))

    @kit.hook(HookTrigger.AFTER_DELIVER, execution=HookExecution.ASYNC)
    async def after_deliver(event: RoomEvent, ctx: RoomContext) -> None:
        error = event.metadata.get("error")
        logger.info("[after_deliver] %s", f"error={error}" if error else "delivered")

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

    async def _cleanup() -> None:
        await voice.end_session(session)

    console_cleanup = setup_console(kit)

    async def _full_cleanup() -> None:
        if console_cleanup:
            await console_cleanup()
        await _cleanup()

    await run_until_stopped(kit, cleanup=_full_cleanup)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
