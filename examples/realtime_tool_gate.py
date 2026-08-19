"""RoomKit — one pre-execution gate for every realtime tool call.

A realtime tool call arrives one of three ways, and all three pass the same
gate: through the function calling API, spoken as text (``call:name{args}``)
when the model skips the API, or aimed at the channel's own infrastructure
tools. This example drives all three against a mock provider — no API key, no
audio device — and prints what each guard let through.

The tool here moves money, so the difference is easy to see: a denied call must
not run, not merely have its result hidden. It is also gated behind a skill, so
the last two rounds show the same call refused and then allowed.

Run with:
    uv run python examples/realtime_tool_gate.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import setup_logging

from roomkit import HookExecution, HookResult, HookTrigger, RealtimeVoiceChannel, RoomKit
from roomkit.skills.registry import SkillRegistry
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("realtime_tool_gate")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "wire_transfer",
        "description": "Send money to an account",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer"},
                "to": {"type": "string"},
            },
            "required": ["amount", "to"],
            "additionalProperties": False,
        },
    }
]

APPROVAL_THRESHOLD = 1000

transfers: list[dict[str, Any]] = []


async def bank(name: str, arguments: dict[str, Any]) -> str:
    """The side effect the gate exists to prevent."""
    transfers.append(dict(arguments))
    return json.dumps({"status": "sent", **arguments})


def _skills(tmp: Path) -> SkillRegistry:
    """One skill, gating the money tool until it is activated."""
    skill_dir = tmp / "treasury"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: treasury\ndescription: Moving money\n"
        "allowed_tools: wire_transfer\n---\n"
        "Confirm the beneficiary out loud before transferring.",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    registry.discover(tmp)
    return registry


async def main() -> None:
    provider = MockRealtimeProvider(model="gemini-3.1-flash-live-preview")
    with TemporaryDirectory() as tmp:
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=TOOLS,
            tool_handler=bank,
            skills=_skills(Path(tmp)),
            # tool_recovery defaults to True: a spoken call is recovered and gated.
        )
        await run(provider, channel)


async def run(provider: MockRealtimeProvider, channel: RealtimeVoiceChannel) -> None:
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, "voice")

    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="compliance")
    async def compliance(event: Any, ctx: Any) -> HookResult:
        amount = event.arguments.get("amount", 0)
        if event.name == "wire_transfer" and amount > APPROVAL_THRESHOLD:
            return HookResult.block(f"transfers above {APPROVAL_THRESHOLD} need human approval")
        return HookResult.allow()

    session = await channel.start_session(room.id, "caller-1", connection=None)

    async def report(label: str, drive: Any) -> None:
        """Run one call and print what came back, by whichever road it took.

        An issued call is answered with a tool result; a spoken one is answered
        with injected context, because the model has no pending call to close.
        """
        submitted, injected = len(provider.tool_results), len(provider.injected_texts)
        print(f"\n{label}")
        await drive
        await asyncio.sleep(0.1)
        if len(provider.tool_results) > submitted:
            answer = f"tool result   : {provider.tool_results[-1][2]}"
        elif len(provider.injected_texts) > injected:
            answer = f"injected text : {provider.injected_texts[-1][1]}"
        else:
            answer = "(nothing returned)"
        print(f"   transfers made : {transfers}")
        print(f"   {answer}")

    def issued(call_id: str, arguments: dict[str, Any]) -> Any:
        """The model uses the function calling API, as it should."""
        return provider.simulate_tool_call(session, call_id, "wire_transfer", arguments)

    def spoken(text: str) -> Any:
        """The model says the call instead of issuing it."""
        return provider.simulate_transcription(session, text, "assistant", True)

    await report(
        "1. issued through the API — the skill gating this tool is idle",
        issued("c1", {"amount": 100, "to": "acct-9"}),
    )
    await report(
        "2. spoken as text — a different road, the same guard",
        spoken("Bien sur. call:wire_transfer{amount:100,to:acct-9}"),
    )
    await report(
        "3. spoken with an argument the tool never declared — named first",
        spoken("call:wire_transfer{amount:100,to:acct-9,memo:rent}"),
    )
    await report(
        "4. the channel's own tool, which the gate sees but never gates",
        provider.simulate_tool_call(session, "c2", "activate_skill", {"name": "treasury"}),
    )
    await report(
        "5. issued, skill now active, above the approval threshold",
        issued("c3", {"amount": 5000, "to": "acct-9"}),
    )
    await report(
        "6. issued, skill active, within policy",
        issued("c4", {"amount": 100, "to": "acct-9"}),
    )

    print(f"\nOnly the compliant call moved money: {transfers}")
    await channel.end_session(session)
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
