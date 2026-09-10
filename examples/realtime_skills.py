"""Load a canonical skill on demand over one bounded Gemini Live connection.

    GEMINI_API_KEY=... uv run --extra realtime-gemini python examples/realtime_skills.py

Text input and captured audio exercise skill delivery, not microphone recognition.
The example uses the bundled code-review skill without changing its instructions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from realtime_tool_search import IncomingScenarioBackend, write_evidence

from roomkit import HookTrigger, RealtimeVoiceChannel, RoomKit, VoiceTrace
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.skills import SkillRegistry

logger = logging.getLogger("roomkit.examples.realtime_skills")


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY to run the live example")
    registry = SkillRegistry()
    await asyncio.to_thread(registry.discover, Path(__file__).parent / "skills")
    output = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="realtime-skills-"))
    provider = GeminiLiveProvider(api_key=api_key, model="gemini-3.1-flash-live-preview")
    backend = IncomingScenarioBackend(capture_sample_rate=24000)
    kit = RoomKit()
    trace = VoiceTrace(kit, triggers=[HookTrigger.ON_TOOL_CALL])
    finished = asyncio.Event()
    errors: list[dict[str, str]] = []
    spoken: list[str] = []

    def transcribed(session: Any, text: str, role: str, final: bool) -> None:
        if role == "assistant" and final:
            spoken.append(text)

    def ended(session: Any) -> None:
        if spoken:
            finished.set()

    def failed(session: Any, code: str, message: str) -> None:
        errors.append({"code": code, "message": message})
        finished.set()

    provider.on_transcription(transcribed)
    provider.on_response_end(ended)
    provider.on_error(failed)
    channel = RealtimeVoiceChannel(
        "skills",
        provider=provider,
        transport=backend,
        skills=registry,
        skill_delivery_mode="on_demand",
        input_sample_rate=16000,
        output_sample_rate=24000,
        system_prompt="Review the supplied example without running code or changing files.",
    )
    kit.register_channel(channel)
    session = None
    report: dict[str, Any] = {"model": provider.model_name}
    try:
        async with asyncio.timeout(90):
            room = await kit.create_room()
            await kit.attach_channel(room.id, channel.channel_id)
            session = await channel.start_session(room.id, "caller", object())
            await provider.inject_text(
                session,
                "Activate code-review and review this one-line Python example: "
                "def lookup(db, name): return db.execute(f\"SELECT * FROM users WHERE name='{name}'\") "
                "Follow the skill, read its style guide, and give a short spoken review.",
                role="user",
            )
            await finished.wait()
    finally:
        report.update(
            errors=errors,
            spoken=spoken,
            turn_finished=finished.is_set(),
            calls=[
                {
                    "name": e.payload.name,
                    "arguments": e.payload.arguments,
                    "result": json.loads(e.payload.result),
                }
                for e in trace.entries()
            ],
        )
        try:
            await asyncio.to_thread(write_evidence, output, report, backend, session)
        finally:
            trace.close()
            await kit.close()
        logger.info("Skill delivery evidence: %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
