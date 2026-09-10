"""Exercise fixed-declaration Tool Search with Gemini Live and fictional data.

    GEMINI_API_KEY=... uv run --extra realtime-gemini python examples/realtime_tool_search.py

Uses the voice testing backend and trace; writes the observed calls and captured
speech into a temporary directory. No business integration is contacted. Input
is injected text, so this verifies the Live tool protocol, not speech recognition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from roomkit import HookTrigger, RealtimeVoiceChannel, RoomKit, ScenarioVoiceBackend, VoiceTrace
from roomkit.providers.gemini.realtime import GeminiLiveProvider
from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.examples.realtime_tool_search")


class IncomingScenarioBackend(ScenarioVoiceBackend):
    """Accept an in-process connection while retaining the bench's WAV capture."""

    async def accept(self, session: VoiceSession, connection: Any) -> None:
        self._sessions[session.id] = session


def catalogue() -> list[dict[str, Any]]:
    """Independent domains plus enough filler to require catalogue discovery."""
    tools = []
    for name, description in [
        ("calendar", "Read appointments on the calendar. Action list returns today's meetings."),
        ("projects", "Read project information. Action list returns active projects."),
    ]:
        tools.append(
            {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string", "enum": ["list"]}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
            }
        )
    tools.extend(
        {
            "name": f"inventory_item_{i}",
            "description": f"Read warehouse inventory item {i}.",
            "parameters": {"type": "object", "properties": {}},
        }
        for i in range(110)
    )
    return tools


def write_evidence(
    output: Path,
    report: dict[str, Any],
    backend: IncomingScenarioBackend,
    session: VoiceSession | None,
) -> None:
    """Persist the report and speech capture from a worker thread."""
    output.mkdir(parents=True, exist_ok=True)
    if session is not None and backend.captured(session).data:
        backend.write_capture(session, output / "bot.wav")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


async def run(api_key: str, output: Path) -> dict[str, Any]:
    """Run one bounded session and retain evidence even when it fails."""
    provider = GeminiLiveProvider(api_key=api_key, model="gemini-3.1-flash-live-preview")
    backend = IncomingScenarioBackend(capture_sample_rate=24000)
    kit = RoomKit()
    trace = VoiceTrace(kit, triggers=[HookTrigger.BEFORE_TOOL_USE, HookTrigger.ON_TOOL_CALL])
    calls: list[dict[str, Any]] = []
    completed = asyncio.Event()
    response_ended = asyncio.Event()
    connect_count, reconfigure_count, socket_count = 0, 0, 0
    declared: list[str] = []
    original_connect, original_reconfigure = provider.connect, provider.reconfigure
    original_socket_connect = provider._client.aio.live.connect

    def socket_connect(*args: Any, **kwargs: Any) -> Any:
        nonlocal socket_count
        socket_count += 1
        return original_socket_connect(*args, **kwargs)

    async def connect(*args: Any, **kwargs: Any) -> None:
        nonlocal connect_count
        connect_count += 1
        declared.extend(t["name"] for t in kwargs.get("tools", []))
        await original_connect(*args, **kwargs)

    async def reconfigure(*args: Any, **kwargs: Any) -> None:
        nonlocal reconfigure_count
        reconfigure_count += 1
        await original_reconfigure(*args, **kwargs)

    provider.connect = connect  # type: ignore[method-assign]
    provider.reconfigure = reconfigure  # type: ignore[method-assign]
    provider._client.aio.live.connect = socket_connect

    async def response_end(session: Any) -> None:
        if completed.is_set():
            response_ended.set()

    provider.on_response_end(response_end)

    async def handler(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in {"calendar", "projects"} or args != {"action": "list"}:
            return {"error": "Use calendar or projects with action=list"}
        result = {"items": ["Demo meeting at 09:00"] if name == "calendar" else ["Demo project"]}
        calls.append({"name": name, "arguments": args, "result": result})
        if {c["name"] for c in calls} == {"calendar", "projects"}:
            completed.set()
        return result

    channel = RealtimeVoiceChannel(
        "tool-search",
        provider=provider,
        transport=backend,
        tools=catalogue(),
        tool_handler=handler,
        input_sample_rate=16000,
        output_sample_rate=24000,
        system_prompt="You are testing fictional data. Use tools and report only their results.",
    )
    kit.register_channel(channel)
    session = None
    report: dict[str, Any] = {"model": provider.model_name, "catalogue_size": len(catalogue())}
    try:
        async with asyncio.timeout(90):
            room = await kit.create_room()
            await kit.attach_channel(room.id, channel.channel_id)
            session = await channel.start_session(room.id, "caller", object())
            await provider.inject_text(
                session,
                "Read today's calendar appointments, then list the active projects. "
                "Use both tools and summarize their results in one short sentence.",
                role="user",
            )
            await completed.wait()
            await response_ended.wait()
        report["passed"] = (
            connect_count == 1
            and socket_count == 1
            and reconfigure_count == 0
            and set(declared) == {"find_tools", "list_tools", "call_tool"}
        )
        if not report["passed"]:
            raise AssertionError("The tool protocol reconnected or reconfigured")
    except Exception as exc:
        report["passed"] = False
        report["failure"] = type(exc).__name__
        raise
    finally:
        report.update(
            {
                "connect_count": connect_count,
                "reconfigure_count": reconfigure_count,
                "socket_count": socket_count,
                "declared_tools": declared,
                "business_calls": calls,
                "trace": [
                    {
                        "trigger": e.trigger.value,
                        "name": e.payload.name,
                        "call_id": e.payload.tool_call_id,
                        "arguments": e.payload.arguments,
                        "result": e.payload.result,
                    }
                    for e in trace.entries()
                ],
            }
        )
        try:
            await asyncio.to_thread(write_evidence, output, report, backend, session)
        finally:
            trace.close()
            await kit.close()
    return report


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY to run the live example")
    output = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="realtime-tool-search-"))
    try:
        report = await run(api_key, output)
        logger.info("Passed: %s; report: %s", report["passed"], output / "report.json")
    finally:
        logger.info("Evidence: %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
