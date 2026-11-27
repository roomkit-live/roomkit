"""The skills methods of the MCP Skills extension over a server's raw requests.

Temporary. The MCP Python SDK carries the same operations from the release
that ships modelcontextprotocol/python-sdk#3485 (``mcp.client.extensions.skills``);
once RoomKit requires that release, this module gives way to it, and the names
here follow the SDK's so the swap is a replacement. Its ``list_skills`` is not
the one to take: it refuses a whole page for one entry it rejects, where a
registry keeps the server's other skills (RFC §24.7). The listing is therefore
read raw here, each entry left to :func:`~roomkit.skills.mcp.entry.parse_entry`.
"""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Mapping
from typing import Any

from roomkit.skills.errors import SkillDiscoveryError, SkillValidationError, SkillVerificationError
from roomkit.skills.mcp.entry import SKILLS_EXTENSION, SkillEntry, parse_entry
from roomkit.skills.mcp.server import MCPSkillServer

logger = logging.getLogger("roomkit.skills.mcp")

MAX_LIST_PAGES = 1000
"""Pages of ``skills/list`` read at most: the bound a tool listing has
(``roomkit.tools.mcp``), far above any real catalogue."""


def declared_settings(server: MCPSkillServer) -> Mapping[str, Any] | None:
    """The settings the server declared the extension with, or ``None`` when
    it did not declare it: a client asks it nothing then."""
    return server.server_extensions.get(SKILLS_EXTENSION)


async def list_skills(server: MCPSkillServer, target: str) -> list[Any]:
    """Every entry the server lists, raw, following its cursor across pages.

    An entry a changing catalogue lists on two pages is kept once, where it was
    first seen. The reading stops where it stands, with a warning naming
    *target*, when the server pages on without bringing anything new (an empty
    page that still carries a cursor, a cursor handed out twice) or past
    :data:`MAX_LIST_PAGES`. A result written for an earlier protocol revision
    may lack ``ttlMs``, ``cacheScope`` and ``resultType``: none is read.

    Raises:
        SkillDiscoveryError: A page whose ``skills`` is not a list.
    """
    entries: list[Any] = []
    seen_uris: set[str] = set()
    seen_cursors: set[str] = set()
    params: dict[str, Any] = {}
    for _ in range(MAX_LIST_PAGES):
        page = await server.request("skills/list", params)
        skills = page.get("skills")
        if not isinstance(skills, list):
            raise SkillDiscoveryError(f"MCP server {target} answered skills/list without a list")
        for entry in skills:
            uri = entry.get("uri") if isinstance(entry, Mapping) else None
            if isinstance(uri, str):
                if uri in seen_uris:
                    continue
                seen_uris.add(uri)
            entries.append(entry)
        cursor = page.get("nextCursor")
        if not isinstance(cursor, str) or not cursor:
            return entries
        if not skills or cursor in seen_cursors:
            break
        seen_cursors.add(cursor)
        params = {"cursor": cursor}
    logger.warning(
        "MCP server %s pages skills/list without end; keeping the %d entries read",
        target,
        len(entries),
    )
    return entries


async def get_skill(server: MCPSkillServer, uri: str) -> SkillEntry:
    """The server's current entry for the skill whose ``SKILL.md`` is at *uri*.

    Raises:
        SkillValidationError: The entry is one the extension does not allow,
            or names another skill.
        MCPError: The server refused (``-32602`` for a skill it does not serve).
    """
    result = await server.request("skills/get", {"uri": uri})
    entry = parse_entry(result.get("skill"))
    if entry.uri != uri:
        raise SkillValidationError(f"{uri}: the server answered with {entry.uri}")
    return entry


async def read_bytes(server: MCPSkillServer, uri: str) -> bytes:
    """The bytes of the resource at *uri*: a text's UTF-8 encoding, a blob decoded.

    Raises:
        SkillVerificationError: The result holds no content for *uri*, or a
            blob that does not decode.
    """
    result = await server.read_resource(uri)
    for item in _contents(result):
        if item.get("uri") != uri:
            continue
        text, blob = item.get("text"), item.get("blob")
        if isinstance(text, str):
            return text.encode("utf-8")
        if isinstance(blob, str):
            try:
                return base64.b64decode(blob, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise SkillVerificationError(f"{uri}: the blob does not decode") from exc
    raise SkillVerificationError(f"{uri}: the server returned no content for it")


def _contents(result: Any) -> list[Mapping[str, Any]]:
    """The ``contents`` of a ``resources/read`` result, in its wire form, whether
    the SDK modelled it or it arrived as a mapping."""
    dump = getattr(result, "model_dump", None)
    data = dump(by_alias=True, mode="json") if callable(dump) else result
    contents = data.get("contents") if isinstance(data, Mapping) else None
    if not isinstance(contents, list):
        return []
    return [item for item in contents if isinstance(item, Mapping)]
