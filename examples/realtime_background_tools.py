"""Watch Gemini 3.8 Live Extended Thinking keep talking while a tool runs.

    GEMINI_API_KEY=... uv run --extra realtime-gemini \
        python examples/realtime_background_tools.py

Through Gemini 3.1 a tool call froze the conversation: the model asked, waited
for the result, then spoke. From 3.8 the call runs in the background and the
model keeps talking, so it can say "let me check that" and carry on while the
work finishes. RoomKit declares the tools ``NON_BLOCKING`` and schedules the
response with ``WHEN_IDLE`` so the result lands between sentences rather than
cutting one in half.

The tool here sleeps on purpose. What the run proves is the overlap: assistant
speech observed while the call is still outstanding. It also shows the other
half of the 3.8 contract, ``interaction_status``: several turns arrive inside
one request and the response ends once, on IDLE, not on each of them.

Uses the voice testing backend, so it needs no microphone. Input is injected
text, so this exercises the Live tool protocol, not speech recognition.

Environment variables:
    GEMINI_API_KEY         (required) Gemini API key
    GEMINI_MODEL           default: gemini-3.8-live-extended-thinking
    GEMINI_THINKING_LEVEL  low | medium | high (default: low)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import json
import os
import tempfile
import time
from typing import Any

from shared import IncomingScenarioBackend, require_env, setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.base import VoiceSession

logger = setup_logging("roomkit.examples.realtime_background_tools")

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-live-extended-thinking")
# Required by the extended-thinking model, which closes the session rather
# than choosing for you. RoomKit falls back to LOW when nothing is named; the
# example says it out loud because that is the knob this model is about.
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "low")
TOOL_SECONDS = 6.0


def inventory_tool() -> dict[str, Any]:
    return {
        "name": "check_inventory",
        "description": (
            "Look up warehouse stock for one item. Slow: takes several seconds. "
            "Tell the caller you are checking, then keep talking while it runs."
        ),
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
    }


async def run(api_key: str, output: Path) -> dict[str, Any]:
    """Run one bounded session and keep the timeline whatever happens."""
    provider = GeminiLiveProvider(api_key=api_key, model=MODEL)
    backend = IncomingScenarioBackend(capture_sample_rate=24000)
    kit = RoomKit()

    started = time.monotonic()
    timeline: list[dict[str, Any]] = []
    tool_running = asyncio.Event()
    tool_done = asyncio.Event()
    response_ends = 0
    speech_during_tool = 0

    def mark(event: str, **fields: Any) -> None:
        timeline.append({"at": round(time.monotonic() - started, 2), "event": event, **fields})

    async def on_transcription(session: VoiceSession, text: str, role: str, final: bool) -> None:
        nonlocal speech_during_tool
        if role != "assistant" or not text.strip():
            return
        overlapped = tool_running.is_set() and not tool_done.is_set()
        if overlapped:
            speech_during_tool += 1
        mark("assistant_speech", text=text.strip()[:80], while_tool_running=overlapped)

    async def on_response_end(session: VoiceSession) -> None:
        nonlocal response_ends
        response_ends += 1
        mark("response_end", count=response_ends)

    provider.on_transcription(on_transcription)
    provider.on_response_end(on_response_end)

    async def handler(name: str, args: dict[str, Any]) -> dict[str, Any]:
        mark("tool_call_received", name=name, arguments=args)
        tool_running.set()
        try:
            await asyncio.sleep(TOOL_SECONDS)
        finally:
            tool_done.set()
        mark("tool_result_returned", name=name)
        return {"item": args.get("item"), "in_stock": 42, "warehouse": "Lyon"}

    channel = RealtimeVoiceChannel(
        "background-tools",
        provider=provider,
        transport=backend,
        tools=[inventory_tool()],
        tool_handler=handler,
        # A call is cut after 10 s by default; this tool may take longer.
        tool_timeouts={"check_inventory": 30},
        input_sample_rate=16000,
        output_sample_rate=24000,
        system_prompt=(
            "You are a warehouse assistant on a phone call. When you need a tool, "
            "say out loud that you are checking, then keep the caller company "
            "while the lookup runs. Never go silent."
        ),
    )
    kit.register_channel(channel)

    session = None
    report: dict[str, Any] = {"model": provider.model_name, "tool_seconds": TOOL_SECONDS}
    try:
        async with asyncio.timeout(120):
            room = await kit.create_room()
            await kit.attach_channel(room.id, channel.channel_id)
            # provider_config rides the session metadata, as the other
            # realtime examples do; the channel constructor takes none.
            session = await channel.start_session(
                room.id,
                "caller",
                object(),
                metadata={"provider_config": {"thinking_level": THINKING_LEVEL}},
            )
            mark("session_started")
            await channel.inject_text(
                session,
                "How many units of the blue widget do we have in stock?",
                role="user",
            )
            # Bounded on purpose: if the model never calls the tool there is
            # still a report to write, and a bare wait would turn that into a
            # timeout traceback instead of an answer.
            try:
                await asyncio.wait_for(tool_done.wait(), timeout=45)
            except TimeoutError:
                mark("tool_never_called")
            else:
                # Leave the model room to deliver the scheduled result.
                await asyncio.sleep(8)
    finally:
        report.update(
            {
                "timeline": timeline,
                "response_ends": response_ends,
                "assistant_turns_during_tool": speech_during_tool,
                # The point of the example: the model kept talking while the
                # call ran, and the request ended once rather than per turn.
                "kept_talking_during_the_call": speech_during_tool > 0,
            }
        )
        output.mkdir(parents=True, exist_ok=True)
        if session is not None and backend.captured(session).data:
            backend.write_capture(session, output / "bot.wav")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        await kit.close()
    return report


async def main() -> None:
    env = require_env("GEMINI_API_KEY")
    output = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="realtime-background-tools-"))
    report = await run(env["GEMINI_API_KEY"], output)
    logger.info(
        "Spoke during the call: %s (%d assistant turns) | response_end fired %d time(s)",
        report["kept_talking_during_the_call"],
        report["assistant_turns_during_tool"],
        report["response_ends"],
    )
    logger.info("Report: %s", output / "report.json")


if __name__ == "__main__":
    asyncio.run(main())
