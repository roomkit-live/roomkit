"""RoomKit -- a tool policy that names one MCP server, with two servers composed.

Two MCP servers both serve ``delete_records``. A model that names MCP tools the
way prompts written for Claude Code do calls ``mcp__<server>__<tool>``. Each
provider is given ``label=``, the host's name for its server: its tool handler
serves only the aliases naming that server, so the alias reaches the server it
names whatever the order of the chain (RFC §21.2).

A policy judges an alias under both its names. To keep a server out, deny its
aliases (``deny=["mcp__archive__*"]``); to admit one server only, allow its
aliases and its tools' names (``allow=["mcp__crm__*", "delete_records"]``).
The first session declares the tools under their aliases alone, so a bare
``delete_records`` is refused as undeclared. The second declares the CRM's
tools under their bare names: a bare call runs on the first server of the
chain the policy does not keep out, the CRM, the archive being kept out by
its aliases (RFC §21.2). A provider without
``label=`` serves an alias whatever server it names: a policy that names a
server cannot hold a call to it.

    RealtimeVoiceChannel (tools=the aliases, tool_policy=deny mcp__archive__*)
        └── compose_tool_handlers
              ├── MCPToolProvider(label="archive") ── stdio ── records_server.py archive
              └── MCPToolProvider(label="crm")     ── stdio ── records_server.py crm

Runs on the mock realtime provider, so it needs no API key: the model's tool
calls are simulated.

Requirements:
    pip install roomkit[mcp]

Run:
    uv run python examples/mcp_server_labels.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import RealtimeVoiceChannel, RoomKit, ToolPolicy
from roomkit.tools import MCPToolProvider, compose_tool_handlers
from roomkit.voice.realtime import MockRealtimeProvider, MockRealtimeTransport

logger = setup_logging("mcp_server_labels")

RECORDS_SERVER = str(Path(__file__).resolve().parent / "mcp_servers" / "records_server.py")

CALLS = ["mcp__crm__delete_records", "mcp__archive__delete_records", "delete_records"]

RESULT_WAIT_SECONDS = 5.0


def records(label: str) -> MCPToolProvider:
    """The records server for *label*, started as a subprocess and named by it."""
    return MCPToolProvider.from_command(sys.executable, [RECORDS_SERVER, label], label=label)


def aliases(provider: MCPToolProvider, label: str) -> list[dict[str, Any]]:
    """The provider's tools, declared under their aliases alone."""
    return [
        {
            "name": f"mcp__{label}__{tool.name}",
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in provider.get_tools()
    ]


async def result_of(provider: MockRealtimeProvider, call: int) -> str:
    """What the model reads for its *call*-th tool call (counted from 1)."""
    deadline = asyncio.get_running_loop().time() + RESULT_WAIT_SECONDS
    while len(provider.tool_results) < call:
        if asyncio.get_running_loop().time() > deadline:
            return f"(no result within {RESULT_WAIT_SECONDS:g} s)"
        await asyncio.sleep(0.05)
    return provider.tool_results[call - 1][2]


async def session_calls(
    channel_id: str, tools: list[Any], crm: MCPToolProvider, archive: MCPToolProvider
) -> None:
    """A voice session declaring *tools*, calling each of CALLS once."""
    provider = MockRealtimeProvider()
    voice = RealtimeVoiceChannel(
        channel_id,
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        # The archive first: without labels, every alias would reach it.
        tool_handler=compose_tool_handlers(archive.as_tool_handler(), crm.as_tool_handler()),
        tool_policy=ToolPolicy(deny=["mcp__archive__*"]),  # keep the archive out
    )
    kit = RoomKit()
    kit.register_channel(voice)
    await kit.create_room(room_id="records")
    await kit.attach_channel("records", channel_id)
    session = await voice.start_session("records", "operator", "ws")

    for call, name in enumerate(CALLS, start=1):
        await provider.simulate_tool_call(session, f"call-{call}", name, {"query": "2019"})
        logger.info("%s: %s -> %s", channel_id, name, await result_of(provider, call))
    await kit.close()


async def main() -> None:
    async with records("crm") as crm, records("archive") as archive:
        # The aliases alone: the bare name is not declared.
        await session_calls(
            "aliases", [*aliases(crm, "crm"), *aliases(archive, "archive")], crm, archive
        )
        # The CRM's tools under their bare names: the bare call runs on the
        # CRM, the first server of the chain the policy admits.
        await session_calls("bare", crm.get_tools_as_dicts(), crm, archive)


if __name__ == "__main__":
    asyncio.run(main())
