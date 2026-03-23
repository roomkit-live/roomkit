"""AI Eviction — automatic pagination of large tool results.

When a tool returns more tokens than ``evict_threshold_tokens``, AIChannel
stores the full output and replaces it with a preview. The AI can then
paginate through the full result using the ``_read_tool_result`` tool.

Run with:
    ANTHROPIC_API_KEY=sk-... uv run python examples/ai_eviction.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.env import require_env
from shared.hooks import log_tool_call

from roomkit import (
    ChannelCategory,
    CLIChannel,
    HookTrigger,
    RoomKit,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig


class GenerateReport:
    """Tool that returns a large CSV report, triggering eviction."""

    @property
    def definition(self) -> dict[str, Any]:
        return {
            "name": "generate_report",
            "description": "Generate a customer report as CSV. Returns ~200 rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Department to filter by (optional)",
                    },
                },
            },
        }

    async def handler(self, _name: str, arguments: dict[str, Any]) -> str:
        dept_filter = arguments.get("department", "").lower()
        departments = {0: "engineering", 1: "sales", 2: "marketing"}
        header = "id,name,email,department,revenue,signup_date"
        rows = []
        for i in range(1, 201):
            dept = departments[i % 3]
            if dept_filter and dept != dept_filter:
                continue
            rows.append(
                f"{i},Customer {i},customer{i}@example.com,"
                f"{dept},{i * 150 + 50},2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            )
        return header + "\n" + "\n".join(rows)


async def main() -> None:
    env = require_env("ANTHROPIC_API_KEY")

    kit = RoomKit()

    cli = CLIChannel("cli")
    ai = AIChannel(
        "ai-assistant",
        provider=AnthropicAIProvider(AnthropicConfig(api_key=env["ANTHROPIC_API_KEY"])),
        system_prompt=(
            "You are a data analyst assistant.\n"
            "You can generate customer reports with the generate_report tool. "
            "When a result is too large, it will be stored with a preview — "
            "use pagination to read through the full data and analyze it."
        ),
        tools=[GenerateReport()],
        evict_threshold_tokens=3000,  # Low threshold so the 200-row CSV triggers eviction
    )

    kit.register_channel(cli)
    kit.register_channel(ai)

    @kit.hook(HookTrigger.ON_TOOL_CALL)
    async def show_tool_call(event, _ctx):
        return log_tool_call(
            event, tool_names=["generate_report", "read_stored_result"], label="tool"
        )

    await kit.create_room(room_id="eviction-room")
    await kit.attach_channel("eviction-room", "cli")
    await kit.attach_channel(
        "eviction-room", "ai-assistant", category=ChannelCategory.INTELLIGENCE
    )

    await cli.run(
        kit,
        room_id="eviction-room",
        welcome=(
            "\nEviction demo — large tool results are stored and paginated.\n"
            'Try: "Generate a customer report and analyze revenue by department"\n'
        ),
    )

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
