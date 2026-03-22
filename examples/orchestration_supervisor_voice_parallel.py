"""Voice parallel analysis workflow with Grok Realtime.

Same orchestration as ``orchestration_supervisor_parallel_tasks.py`` but
using local microphone/speakers with xAI Grok Realtime instead of CLI.

The supervisor speaks to the user via Grok speech-to-speech. Workers
(technical + business analysts) run as text agents with Anthropic Haiku
in child rooms. Results are presented via voice.

    Human (voice) → Supervisor (Grok) → [Technical | Business] → Supervisor → Human (voice)

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
import signal

from shared.env import require_env

from roomkit import (
    Agent,
    HookExecution,
    HookTrigger,
    RealtimeVoiceChannel,
    RoomKit,
    Supervisor,
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

    # --- Grok Realtime provider (speech-to-speech for supervisor) -------------

    xai_config = XAIRealtimeConfig(
        api_key=env["XAI_API_KEY"],
        model=env.get("XAI_MODEL", "grok-2-audio"),
        voice="eve",
    )
    xai_provider = XAIRealtimeProvider(xai_config)

    # --- Agents --------------------------------------------------------------

    # Supervisor — config-only agent (no AI provider, Grok handles speech)
    supervisor = Agent(
        "agent-supervisor",
        role="Project supervisor",
        system_prompt="You coordinate analysis. Present a combined summary to the user.",
        voice="eve",
    )

    # Workers — text agents with Anthropic Haiku (run in child rooms)
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

    # --- Supervisor setup ----------------------------------------------------

    kit = RoomKit(
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=[technical, business],
            strategy="parallel",
            auto_delegate=True,
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
    )
    kit.register_channel(voice)

    # --- Observability hooks -------------------------------------------------

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
    async def on_delegated(event: RoomEvent, ctx: RoomContext) -> None:
        m = event.metadata
        agent = m.get("agent_id", "?")
        task_id = m.get("task_id", "?")
        task_input = m.get("task_input", "")
        preview = task_input[:80] + "..." if len(task_input) > 80 else task_input
        logger.info("[delegated] %s (task %s) input: %s", agent, task_id, preview)

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_completed(event: RoomEvent, ctx: RoomContext) -> None:
        m = event.metadata
        agent = m.get("agent_id", "?")
        status = m.get("task_status", "?")
        duration = m.get("duration_ms", 0)
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
