"""The skills methods over a server's raw requests (RFC §21.2, §24.7), against
an in-memory server and against a real MCP server reached over stdio."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from roomkit.skills.errors import SkillDiscoveryError, SkillValidationError, SkillVerificationError
from roomkit.skills.mcp import SKILLS_EXTENSION, parse_entry, verify_skill_resource
from roomkit.skills.mcp._wire import (
    MAX_LIST_PAGES,
    declared_settings,
    get_skill,
    list_skills,
    read_bytes,
)
from roomkit.tools.mcp import MCPToolProvider
from tests.mcp_skill_fakes import FakeSkillServer, ServedSkill, skill_md


def _skills(count: int) -> list[ServedSkill]:
    return [
        ServedSkill(f"s{i}", {"SKILL.md": skill_md(f"s{i}", f"Skill {i}")}) for i in range(count)
    ]


class TestListing:
    async def test_reads_every_page(self) -> None:
        server = FakeSkillServer(_skills(5), page_size=2)
        entries = await list_skills(server, "fake")
        assert [e["uri"] for e in entries] == [f"skill://s{i}/SKILL.md" for i in range(5)]
        assert [m for m, _ in server.requests] == ["skills/list"] * 3
        assert server.requests[1][1] == {"cursor": "2"}

    async def test_reads_no_file(self) -> None:
        server = FakeSkillServer(_skills(3))
        await list_skills(server, "fake")
        assert server.reads == []

    async def test_keeps_an_entry_seen_on_two_pages_once(self) -> None:
        class Repeating(FakeSkillServer):
            async def request(self, method: str, params: Any = None) -> dict[str, Any]:
                page = await super().request(method, params)
                if params and params.get("cursor") == "1":
                    page["skills"].insert(0, self.skills[0].entry())
                return page

        server = Repeating(_skills(2), page_size=1)
        assert [e["uri"] for e in await list_skills(server, "fake")] == [
            "skill://s0/SKILL.md",
            "skill://s1/SKILL.md",
        ]

    async def test_keeps_an_entry_it_cannot_read(self) -> None:
        class Odd(FakeSkillServer):
            async def request(self, method: str, params: Any = None) -> dict[str, Any]:
                return {"skills": ["not an entry", {"frontmatter": {}}]}

        assert await list_skills(Odd(), "fake") == ["not an entry", {"frontmatter": {}}]

    async def test_a_page_without_a_list_is_refused(self) -> None:
        class Broken(FakeSkillServer):
            async def request(self, method: str, params: Any = None) -> dict[str, Any]:
                return {"skills": "none"}

        with pytest.raises(SkillDiscoveryError, match="without a list"):
            await list_skills(Broken(), "fake")

    @pytest.mark.parametrize("page", [{"skills": [], "nextCursor": "x"}, {"skills": "repeat"}])
    async def test_a_listing_that_pages_without_end_stops(
        self, page: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        skills = _skills(1)

        class Endless(FakeSkillServer):
            async def request(self, method: str, params: Any = None) -> dict[str, Any]:
                self.requests.append((method, dict(params or {})))
                if page["skills"] == "repeat":
                    return {"skills": [skills[0].entry()], "nextCursor": "same"}
                return dict(page)

        server = Endless()
        entries = await list_skills(server, "fake-target")
        assert len(server.requests) <= 2
        assert "fake-target" in caplog.text
        assert len(entries) == (1 if page["skills"] == "repeat" else 0)

    async def test_a_listing_is_bounded_in_pages(self) -> None:
        class Growing(FakeSkillServer):
            async def request(self, method: str, params: Any = None) -> dict[str, Any]:
                n = len(self.requests)
                self.requests.append((method, {}))
                skill = ServedSkill(f"s{n}", {"SKILL.md": skill_md(f"s{n}", "d")})
                return {"skills": [skill.entry()], "nextCursor": str(n + 1)}

        server = Growing()
        await list_skills(server, "fake")
        assert len(server.requests) == MAX_LIST_PAGES

    async def test_a_legacy_result_without_cache_fields_is_complete(self) -> None:
        # What huggingface.co/mcp answers over a 2025-11-25 session: no ttlMs,
        # cacheScope nor resultType.
        server = FakeSkillServer(_skills(1))
        assert len(await list_skills(server, "fake")) == 1


class TestDeclaration:
    def test_the_settings_a_server_declared(self) -> None:
        assert declared_settings(FakeSkillServer()) == {}
        assert declared_settings(FakeSkillServer(declares=False)) is None


class TestGetSkill:
    async def test_returns_the_entry_of_an_unlisted_skill(self) -> None:
        skills = _skills(2)
        server = FakeSkillServer(skills, listed=set())
        entry = await get_skill(server, skills[1].uri)
        assert entry.name == "s1"
        assert server.reads == []

    async def test_an_entry_for_another_skill_is_refused(self) -> None:
        skills = _skills(2)
        server = FakeSkillServer(skills, entry_overrides={skills[0].uri: skills[1].entry()})
        with pytest.raises(SkillValidationError, match="answered with skill://s1/SKILL.md"):
            await get_skill(server, skills[0].uri)


class TestReadBytes:
    async def test_a_text_is_its_utf8(self) -> None:
        skill = ServedSkill("s", {"SKILL.md": skill_md("s", "Résumé, ça va")})
        server = FakeSkillServer([skill])
        content = await read_bytes(server, skill.uri)
        verify_skill_resource(parse_entry(skill.entry()).resource(skill.uri), content)  # type: ignore[arg-type]

    async def test_a_blob_is_decoded(self) -> None:
        skill = ServedSkill("s", {"SKILL.md": skill_md("s", "d"), "logo.png": b"\x89PNG\x00"})
        server = FakeSkillServer([skill])
        assert await read_bytes(server, skill.file_uri("logo.png")) == b"\x89PNG\x00"

    async def test_content_for_another_uri_is_none(self) -> None:
        class Elsewhere(FakeSkillServer):
            async def read_resource(self, uri: str) -> dict[str, Any]:
                return {"contents": [{"uri": "skill://other/SKILL.md", "text": "x"}]}

        with pytest.raises(SkillVerificationError, match="no content"):
            await read_bytes(Elsewhere(), "skill://s/SKILL.md")

    async def test_a_blob_that_does_not_decode_fails(self) -> None:
        class Garbled(FakeSkillServer):
            async def read_resource(self, uri: str) -> dict[str, Any]:
                return {"contents": [{"uri": uri, "blob": "not base64!"}]}

        with pytest.raises(SkillVerificationError, match="does not decode"):
            await read_bytes(Garbled(), "skill://s/SKILL.md")


# A server written against the wire, not an SDK: mcp 1.x's server cannot serve
# a method it does not model, and the client side is what is under test.
_SERVER = textwrap.dedent(
    """\
    import hashlib, json, sys

    SKILL = "---\\nname: refunds\\ndescription: Process refund requests\\n---\\n\\nBe kind.\\n"
    URI = "skill://acme/refunds/SKILL.md"
    DECLARE = sys.argv[1] == "declare"

    def entry():
        raw = SKILL.encode()
        return {
            "uri": URI,
            "frontmatter": {"name": "refunds", "description": "Process refund requests"},
            "resources": [
                {
                    "uri": URI,
                    "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }
            ],
        }

    def answer(message):
        method, params = message["method"], message.get("params") or {}
        if method == "initialize":
            capabilities = {"resources": {}}
            if DECLARE:
                skills = {"directoryRead": True}
                capabilities["extensions"] = {"io.modelcontextprotocol/skills": skills}
            return {
                "protocolVersion": params["protocolVersion"],
                "capabilities": capabilities,
                "serverInfo": {"name": "skills-test", "version": "0"},
            }
        if method == "skills/list":
            return {"skills": [entry()]}
        if method == "skills/get":
            if params.get("uri") == URI:
                return {"skill": entry()}
            raise LookupError(f"No skill is served at {params.get('uri')}")
        if method == "resources/read" and params.get("uri") == URI:
            return {"contents": [{"uri": URI, "mimeType": "text/markdown", "text": SKILL}]}
        raise LookupError(f"unknown {method}")

    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": message["id"], "result": answer(message)}
        except LookupError as exc:
            error = {"code": -32602, "message": str(exc)}
            reply = {"jsonrpc": "2.0", "id": message["id"], "error": error}
        print(json.dumps(reply), flush=True)
    """
)


@pytest.fixture
def server_script(tmp_path: Path) -> str:
    path = tmp_path / "skills_server.py"
    path.write_text(_SERVER)
    return str(path)


class TestAgainstARealServer:
    async def test_the_provider_carries_what_the_skills_need(self, server_script: str) -> None:
        provider = MCPToolProvider.from_command(
            sys.executable, [server_script, "declare"], discover=False
        )
        async with provider as server:
            assert server.server_extensions == {SKILLS_EXTENSION: {"directoryRead": True}}
            entries = await list_skills(server, "stdio")
            entry = parse_entry(entries[0])
            assert entry.name == "refunds"
            assert (await get_skill(server, entry.uri)).fingerprint == entry.fingerprint
            content = await read_bytes(server, entry.uri)
            verify_skill_resource(entry.resource(entry.uri), content)  # type: ignore[arg-type]

    async def test_an_unknown_skill_is_the_servers_error(self, server_script: str) -> None:
        mcp_error = pytest.importorskip("mcp.shared.exceptions").MCPError
        provider = MCPToolProvider.from_command(
            sys.executable, [server_script, "declare"], discover=False
        )
        async with provider as server:
            with pytest.raises(mcp_error, match="No skill is served"):
                await get_skill(server, "skill://acme/chargebacks/SKILL.md")

    async def test_a_server_that_declares_nothing_has_no_extensions(
        self, server_script: str
    ) -> None:
        provider = MCPToolProvider.from_command(
            sys.executable, [server_script, "plain"], discover=False
        )
        async with provider as server:
            assert server.server_extensions == {}
            assert declared_settings(server) is None

    def test_the_extensions_are_read_only_while_connected(self) -> None:
        with pytest.raises(RuntimeError, match="not connected"):
            _ = MCPToolProvider.from_url("http://localhost:1/mcp").server_extensions
