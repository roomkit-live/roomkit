"""The skills MCP servers serve, as a registry holds them (RFC §24.7).

The catalogue keeps, for each server a host registered, the host's label for
it and the entries it last gave; a :class:`~roomkit.skills.SkillRegistry`
keeps the names, the manifest and the skills loaded. Loading a skill reads its
``SKILL.md`` then, and not before, and checks it against the entry; reading
one of its files reads only a file the held entry lists, checked the same way.
What the model is told of a failure names the skill only: a server's text
reaches it as a skill's content or not at all.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from roomkit._text import is_label
from roomkit.skills.errors import (
    SkillDiscoveryError,
    SkillError,
    SkillPathError,
    SkillValidationError,
    SkillVerificationError,
)
from roomkit.skills.mcp._wire import declared_settings, get_skill, list_skills, read_bytes
from roomkit.skills.mcp.approval import ApproveExecution
from roomkit.skills.mcp.entry import MAX_TOTAL_SIZE, SkillEntry, parse_entry
from roomkit.skills.mcp.server import MCPSkillServer
from roomkit.skills.mcp.skill import MCPSkill
from roomkit.skills.mcp.verify import verify_frontmatter, verify_skill_resource
from roomkit.skills.models import SkillMetadata
from roomkit.skills.parser import frontmatter_text

logger = logging.getLogger("roomkit.skills.mcp")

_PATH = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?[A-Za-z0-9._~@+-]+(?:/[A-Za-z0-9._~@+-]+)*$")
"""A skill's path as a registry names it: the characters of an identifier and
``/``, the scheme kept when it is not ``skill://``; a path holding anything
else (a space, a quote, a line break) cannot be a name the model reads."""

_KNOWN_KEYS = frozenset({"name", "description", "license", "compatibility", "allowed_tools"})


class SkillChangedError(SkillVerificationError):
    """A file of an active served skill failed verification: the skill is
    retired, its entry read again, and it must be activated again."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Skill {name!r} changed on its server or could not be verified: it is no "
            "longer active. Activate it again to load its current version."
        )
        self.name = name


@dataclass(frozen=True)
class ServedOrigin:
    """A server a host registered skills from, under the host's label for it."""

    label: str
    server: MCPSkillServer
    allow_dynamic: bool
    approve_execution: ApproveExecution | None
    execution_tools: tuple[str, ...]


@dataclass(frozen=True)
class Listed:
    """One entry of a listing as the registry records it: available with its
    metadata, or unavailable with the reason the model is given."""

    name: str
    entry: SkillEntry | None
    metadata: SkillMetadata | None
    reason: str | None


def served_name(label: str, entry: SkillEntry) -> str | None:
    """``<label>:<path>``, the skill's name in a registry, or ``None`` when its
    path holds characters a name cannot."""
    return f"{label}:{entry.path}" if _PATH.fullmatch(entry.path) else None


def served_metadata(name: str, entry: SkillEntry) -> SkillMetadata:
    """The metadata of a served skill. ``allowed_tools`` is never read: what a
    served skill's frontmatter would grant is not granted (RFC §24.7)."""
    frontmatter = entry.frontmatter
    return SkillMetadata(
        name=name,
        description=entry.description,
        license=frontmatter_text(frontmatter.get("license")) or None,
        compatibility=frontmatter_text(frontmatter.get("compatibility")) or None,
        allowed_tools=None,
        extra_metadata={
            k: frontmatter_text(v) for k, v in frontmatter.items() if k not in _KNOWN_KEYS
        },
    )


def _unavailable_reason(entry: SkillEntry, origin: ServedOrigin) -> str | None:
    """Why a well-formed entry is not loaded here, or ``None``."""
    if entry.dynamic and not origin.allow_dynamic:
        return (
            "its server generates its content, so nothing of it can be verified, "
            "and this host does not take such skills"
        )
    exceeded = entry.limit_exceeded()
    if exceeded is not None:
        return f"it is past the limits of a skill served over MCP: {exceeded}"
    return None


class MCPSkillCatalog:
    """The servers a registry took skills from, and the entries they gave."""

    def __init__(self) -> None:
        self._origins: dict[str, ServedOrigin] = {}
        self._entries: dict[str, SkillEntry] = {}
        self._labels: dict[str, str] = {}
        # Failed file checks per name: each retires the skill loaded before it
        # in every registry sharing this catalog (RFC §24.7).
        self._retirements: dict[str, int] = {}

    def origin(self, label: str) -> ServedOrigin | None:
        """The server registered under *label*."""
        return self._origins.get(label)

    def label_of(self, name: str) -> str | None:
        """The label of the server that serves *name*, or ``None`` for a name
        no registered server could serve."""
        label = self._labels.get(name)
        if label is not None:
            return label
        head, sep, _ = name.partition(":")
        return head if sep and head in self._origins else None

    def entry(self, name: str) -> SkillEntry | None:
        """The entry the server last gave for *name*."""
        return self._entries.get(name)

    def refusal(self, name: str) -> str | None:
        """Why the entry held for *name* is not one this host takes, or
        ``None``. An entry read again after a failed check is held to the
        listing's rules, as one a registry sharing this catalog took when it
        registered the server again (RFC §24.7)."""
        entry, label = self._entries.get(name), self._labels.get(name)
        if entry is None or label is None:
            return None
        return _unavailable_reason(entry, self._origins[label])

    def is_retired(self, skill: MCPSkill) -> bool:
        """Whether a failed check retired *skill*'s name since it was loaded,
        in any registry sharing this catalog."""
        return skill.retirements < self._retirements.get(skill.name, 0)

    @property
    def has_servers(self) -> bool:
        return bool(self._origins)

    def execution_tools(self) -> set[str]:
        """The patterns of the tools the hosts named as running code, every
        registered server's (RFC §24.7)."""
        return {pattern for origin in self._origins.values() for pattern in origin.execution_tools}

    def names_of(self, label: str) -> list[str]:
        """The names the server under *label* was last listed with."""
        return [name for name, held in self._labels.items() if held == label]

    async def read_listing(
        self,
        server: MCPSkillServer,
        label: str,
        *,
        allow_dynamic: bool,
        approve_execution: ApproveExecution | None,
        execution_tools: Sequence[str],
    ) -> tuple[ServedOrigin, list[Listed]]:
        """List the skills of *server* for *label*, nothing recorded yet.

        Raises:
            SkillValidationError: The label is not an identifier, or another
                server holds it.
            SkillDiscoveryError: The server does not declare the extension, or
                its listing cannot be read.
        """
        if not is_label(label):
            raise SkillValidationError(f"Label {label!r} is not an identifier")
        held = self._origins.get(label)
        if held is not None and held.server is not server:
            raise SkillValidationError(f"Label {label!r} already names another MCP server")
        if declared_settings(server) is None:
            raise SkillDiscoveryError(
                f"MCP server {label!r} does not declare the skills extension"
            )
        origin = ServedOrigin(
            label=label,
            server=server,
            allow_dynamic=allow_dynamic,
            approve_execution=approve_execution,
            execution_tools=tuple(execution_tools),
        )
        raw = await list_skills(server, label)
        return origin, [self._listed(origin, index, item) for index, item in enumerate(raw)]

    def _listed(self, origin: ServedOrigin, index: int, raw: object) -> Listed:
        """One raw entry, read on its own: what the registry records for it."""
        try:
            entry = parse_entry(raw)
        except SkillValidationError as exc:
            logger.warning("MCP server %r lists a malformed entry: %s", origin.label, exc)
            return Listed(
                f"{origin.label}:#{index}", None, None, "its server's entry is malformed"
            )
        name = served_name(origin.label, entry)
        if name is None:
            logger.warning(
                "MCP server %r lists %s, whose path no name can hold", origin.label, entry.uri
            )
            return Listed(
                f"{origin.label}:#{index}", None, None, "its path holds characters a name cannot"
            )
        reason = _unavailable_reason(entry, origin)
        if reason is not None:
            return Listed(name, entry, None, reason)
        return Listed(name, entry, served_metadata(name, entry), None)

    def record(self, origin: ServedOrigin, listed: Sequence[Listed]) -> None:
        """Keep *origin* and the entries of *listed*, in place of the ones the
        server gave before."""
        for name in self.names_of(origin.label):
            self._labels.pop(name, None)
            self._entries.pop(name, None)
        self._origins[origin.label] = origin
        for item in listed:
            self._labels[item.name] = origin.label
            if item.entry is not None:
                self._entries[item.name] = item.entry

    async def read_unlisted(self, name: str) -> Listed:
        """The entry of *name*, a skill its server's listing did not hold, asked
        of that server with ``skills/get`` (a partial listing proves no skill
        missing).

        Raises:
            SkillError: No registered server is labelled as *name* says, or
                the server serves no such skill.
        """
        label, _, path = name.partition(":")
        origin = self._origins.get(label)
        if origin is None or not path or not _PATH.fullmatch(path):
            raise SkillValidationError(f"Skill {name!r} not found")
        uri = f"{path}/SKILL.md" if "://" in path else f"skill://{path}/SKILL.md"
        entry = await self._get(origin, name, uri)
        if served_name(label, entry) != name:
            raise SkillValidationError(f"Skill {name!r} not found")
        reason = _unavailable_reason(entry, origin)
        return Listed(name, entry, None if reason else served_metadata(name, entry), reason)

    def record_unlisted(self, item: Listed) -> None:
        """Install a fetched entry only after its registry projection fits."""
        self._labels[item.name] = item.name.partition(":")[0]
        if item.entry is not None:
            self._entries[item.name] = item.entry

    async def _get(self, origin: ServedOrigin, name: str, uri: str) -> SkillEntry:
        """The server's current entry for *uri*, any failure a skill error that
        names the skill only."""
        try:
            return await get_skill(origin.server, uri)
        except SkillValidationError as exc:
            logger.warning("MCP server %r answers skills/get badly: %s", origin.label, exc)
            raise SkillValidationError(f"Skill {name!r}: its server's entry is malformed") from exc
        except Exception as exc:
            logger.info("MCP server %r has no skill at %s: %s", origin.label, uri, exc)
            raise SkillDiscoveryError(f"Skill {name!r} not found on its server") from exc

    async def load(self, name: str, metadata: SkillMetadata) -> MCPSkill:
        """Read and verify the ``SKILL.md`` of *name*, and the skill it holds.

        Raises:
            SkillVerificationError: The content does not match the entry; the
                entry is read again, so a later load takes the current one.
            SkillDiscoveryError: The server cannot be read.
        """
        entry = self._entries[name]
        origin = self._origins[self._labels[name]]
        resource = entry.resource(entry.uri)
        try:
            content = await self._read(origin, name, entry.uri)
            if resource is not None:
                verify_skill_resource(resource, content)
            elif len(content) > MAX_TOTAL_SIZE:
                raise SkillVerificationError(f"{entry.uri}: past {MAX_TOTAL_SIZE} bytes")
            instructions = verify_frontmatter(entry, _text(entry.uri, content))
        except SkillVerificationError as exc:
            logger.warning("MCP server %r: %s", origin.label, exc)
            await self.refresh(name)
            raise SkillVerificationError(
                f"Skill {name!r} changed on its server or could not be verified"
            ) from exc
        return MCPSkill(
            metadata=metadata,
            instructions=instructions,
            path=Path(os.devnull),
            origin=origin.label,
            entry=entry,
            files={entry.uri: content},
            read_total=len(content),
            retirements=self._retirements.get(name, 0),
        )

    async def read_file(self, skill: MCPSkill, filename: str) -> str:
        """The text of *filename*, a file of *skill* its held entry lists.

        Raises:
            SkillPathError: The name is not one of the skill's files; nothing
                is read.
            SkillChangedError: The file failed verification: the skill is to
                be retired, and its entry was read again.
        """
        entry = skill.held
        uri = entry.resolve(filename)
        resource = entry.resource(uri)
        if resource is None and not entry.dynamic:
            raise SkillPathError(f"{filename!r} is not a file of skill {skill.name!r}")
        content = skill.files.get(uri)
        if content is None:
            origin = self._origins[skill.origin]
            try:
                content = await self._read(origin, skill.name, uri)
                if resource is not None:
                    verify_skill_resource(resource, content)
                elif skill.read_total + len(content) > MAX_TOTAL_SIZE:
                    raise SkillVerificationError(f"{uri}: past {MAX_TOTAL_SIZE} bytes in all")
            except SkillVerificationError as exc:
                logger.warning("MCP server %r: %s", skill.origin, exc)
                self._retirements[skill.name] = self._retirements.get(skill.name, 0) + 1
                await self.refresh(skill.name)
                raise SkillChangedError(skill.name) from exc
            skill.files[uri] = content
            skill.read_total += len(content)
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return f"[binary file of {len(content)} bytes, not shown]"

    async def refresh(self, name: str) -> None:
        """Read the entry of *name* again, so a later load takes the current
        one. A server that cannot answer leaves the old entry in place."""
        label = self._labels.get(name)
        entry = self._entries.get(name)
        if label is None or entry is None:
            return
        try:
            self._entries[name] = await self._get(self._origins[label], name, entry.uri)
        except SkillError:
            return

    async def _read(self, origin: ServedOrigin, name: str, uri: str) -> bytes:
        """The bytes at *uri* on *origin*'s server; a server that cannot answer
        is a skill that could not be verified."""
        try:
            return await read_bytes(origin.server, uri)
        except SkillVerificationError:
            raise
        except Exception as exc:
            raise SkillVerificationError(f"{uri}: the server did not serve it ({exc})") from exc


def _text(uri: str, content: bytes) -> str:
    """A ``SKILL.md``'s bytes as text."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillVerificationError(f"{uri}: not UTF-8 text") from exc
