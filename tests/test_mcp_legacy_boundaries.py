"""Legacy SSE endpoint and redirect boundaries exercised through the public provider."""

from __future__ import annotations

import asyncio

import httpx2
import pytest

from roomkit.tools import MCPToolProvider
from tests.mcp_legacy_peer import LegacySSEPeer

pytest.importorskip("mcp.client.sse")


class RedirectPeer(LegacySSEPeer):
    def __init__(self, verb: str, status: int, *, outside: str | None = None) -> None:
        super().__init__({"replies": []})
        self.verb = verb
        self.status = status
        self.outside = outside

    def _endpoint(self) -> str:
        return f"/messages?session={self.connections}"

    def _post_target(self) -> str:
        return "/accepted" if self.verb == "POST" else super()._post_target()

    async def _redirect(self, writer: asyncio.StreamWriter, verb: str, target: str) -> bool:
        if verb != self.verb or target == "/accepted":
            return False
        location = self.outside or "/accepted"
        writer.write(
            f"HTTP/1.1 {self.status} Redirect\r\nLocation: {location}\r\n"
            "Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
        )
        await writer.drain()
        return True


@pytest.mark.parametrize(
    ("verb", "status"),
    [("GET", s) for s in [301, 302, 303, 307, 308]] + [("POST", 307), ("POST", 308)],
)
async def test_legacy_sse_follows_a_same_origin_redirect_that_keeps_the_method(
    verb: str, status: int
) -> None:
    peer = RedirectPeer(verb, status)
    async with peer:
        provider = MCPToolProvider(
            peer.url, transport="sse", discover=False, headers={"X-Capture": "kept"}
        )
        async with provider:
            assert provider.connected
        await asyncio.wait_for(peer.get_closed.wait(), 3)
        assert (verb, "/accepted") in peer.targets
        assert all(headers.get("x-capture") == "kept" for headers in peer.headers)
        assert peer.requests == [{"method": "initialize"}]
    assert peer.problems == []


@pytest.mark.parametrize(("verb", "status"), [("GET", 307), ("POST", 307), ("POST", 302)])
async def test_legacy_sse_does_not_forward_a_redirect_outside_its_origin_or_change_a_post(
    verb: str, status: int
) -> None:
    async with LegacySSEPeer({"replies": []}) as outside:
        peer = RedirectPeer(verb, status, outside=outside.url if status == 307 else None)
        async with peer:
            provider = MCPToolProvider(
                peer.url, transport="sse", discover=False, connect_timeout=0.05
            )
            with pytest.raises(
                httpx2.HTTPStatusError if verb == "GET" else TimeoutError
            ) as raised:
                await provider.__aenter__()
            if verb == "GET":
                assert isinstance(raised.value, httpx2.HTTPStatusError)
                assert raised.value.response.status_code == status
            assert not provider.connected
            assert peer.targets[0] == ("GET", "/mcp")
            if verb == "POST":
                await asyncio.wait_for(peer.get_closed.wait(), 3)
            assert not outside.targets
            assert not any(target == "/accepted" for _, target in peer.targets)
            assert peer.requests == []
        assert peer.problems == []
    assert outside.problems == []


class EndpointPeer(LegacySSEPeer):
    def __init__(self, kind: str) -> None:
        super().__init__({"replies": []})
        self.kind = kind

    def _endpoint(self) -> str:
        target = super()._endpoint()
        if self.kind == "relative":
            return target.removeprefix("/")
        absolute = self.url.removesuffix("/mcp") + target
        if self.kind == "host-case":
            return absolute.replace("localhost", "LOCALHOST")
        if self.kind == "userinfo":
            return absolute.replace("http://", "http://user@")
        return absolute


@pytest.mark.parametrize("kind", ["relative", "absolute", "host-case", "userinfo"])
async def test_legacy_endpoint_announcement_compares_the_configured_raw_netloc(kind: str) -> None:
    peer = EndpointPeer(kind)
    async with peer:
        if kind == "host-case":
            peer.url = peer.url.replace("127.0.0.1", "localhost")
        provider = MCPToolProvider(
            peer.url,
            transport="sse",
            discover=False,
            connect_timeout=0.05 if kind in {"host-case", "userinfo"} else 3,
        )
        if kind in {"relative", "absolute"}:
            async with provider:
                assert provider.connected
            assert peer.requests == [{"method": "initialize"}]
        else:
            with pytest.raises(TimeoutError, match="within 0.05 s"):
                await provider.__aenter__()
            assert peer.targets == [("GET", "/mcp")]
            assert peer.requests == []
        assert not provider.connected
        await asyncio.wait_for(peer.get_closed.wait(), 3)
    assert peer.problems == []
