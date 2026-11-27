"""The largest SSE event an MCP connection reads (RMK-663).

Streamable HTTP reads an event of up to 4 MiB, roomkit-rs's bound on any MCP
message; the legacy HTTP+SSE transport keeps the SDK's 1 MiB,
which its client does not let a host set.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.tools import MCPToolProvider
from tests.conformance._mcp_peer import MCPPeer
from tests.mcp_legacy_peer import LegacySSEPeer

pytest.importorskip("mcp.client.sse")

_MIB = 1024 * 1024


def _script(size: int) -> dict[str, Any]:
    """One tool call whose text result makes an SSE event of about *size* bytes."""
    result = {"content": [{"type": "text", "text": "x" * size}]}
    call = {"method": "tools/call", "params": {"name": "big", "arguments": {}}, "result": result}
    return {"framing": "sse", "replies": [call]}


async def test_streamable_http_reads_an_event_past_the_sdk_default() -> None:
    async with MCPPeer(_script(2 * _MIB)) as peer:
        provider = MCPToolProvider(peer.url, discover=False)
        async with provider:
            result = await provider.call_tool_result("big", {}, timeout=3)
    assert len(result.content[0].text) == 2 * _MIB
    assert peer.problems == []


@pytest.mark.parametrize(
    ("peer_type", "transport", "limit"),
    [
        (MCPPeer, "streamable_http", 4 * _MIB),
        (LegacySSEPeer, "sse", _MIB),
    ],
    ids=["streamable-http-4-mib", "legacy-sse-1-mib"],
)
async def test_an_event_past_the_bound_fails_the_connection(
    peer_type: type[MCPPeer], transport: str, limit: int
) -> None:
    async with peer_type(_script(limit)) as peer:
        provider = MCPToolProvider(peer.url, transport=transport, discover=False)
        with pytest.raises(BaseExceptionGroup) as raised:
            async with provider:
                await provider.call_tool_result("big", {}, timeout=3)
    assert f"exceeded the {limit} byte limit" in str(_leaf(raised.value))


def _leaf(error: BaseException) -> BaseException:
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error
