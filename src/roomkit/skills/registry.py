"""Skill registry for discovering and managing Agent Skills."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Container, Iterable, Mapping, Sequence
from html import escape
from pathlib import Path

from roomkit._text import quoted
from roomkit.skills.errors import (
    SkillDiscoveryError,
    SkillManifestTooLargeError,
    SkillParseError,
    SkillValidationError,
)
from roomkit.skills.mcp.approval import ApproveExecution
from roomkit.skills.mcp.catalog import Listed, MCPSkillCatalog, ServedOrigin, SkillChangedError
from roomkit.skills.mcp.server import MCPSkillServer
from roomkit.skills.mcp.skill import MCPSkill
from roomkit.skills.models import (
    RequiresMatch,
    Skill,
    SkillMetadata,
    bracketed_pattern_errors,
    serves_exactly,
)
from roomkit.skills.parser import (
    find_skill_md,
    parse_skill,
    parse_skill_metadata,
)

logger = logging.getLogger("roomkit.skills")

_QUOTED_MAX = 1100
_MANIFEST_MAX = 65_536
"""The most characters the skills' manifest takes, which every system
prompt carries (RFC §24.1)."""
"""Characters of a served skill's description, license or compatibility the
manifest quotes: a description holds 1024 at most."""


class SkillRegistry:
    """Discover, load, and manage Agent Skills.

    Lightweight metadata is parsed on discover/register. Full skill
    instructions are loaded lazily on first ``get_skill()`` call and
    cached for subsequent access.
    """

    def __init__(self, *, requires_match: RequiresMatch = serves_exactly) -> None:
        """*requires_match* says whether a skill's ``requires`` name is served
        by the tools a conversation offers (RFC §24.3): an exact tool name by
        default, a host's own reading otherwise (a hub served by any tool
        named ``<hub>_*``, say). Every door checks ``requires`` through it."""
        self.requires_match: RequiresMatch = requires_match
        self._metadata: dict[str, SkillMetadata] = {}
        self._skills: dict[str, Skill] = {}
        self._paths: dict[str, Path] = {}
        self._unavailable: dict[str, str] = {}
        # The patterns a skill marked unavailable gated: it can never be
        # activated here, so what it gates stays closed (RFC §24.2).
        self._closed_gates: dict[str, list[str]] = {}
        self._unlisted: set[str] = set()
        # The skills MCP servers serve: shared by a registry's copies, which
        # name a subset of them (RFC §24.7).
        self._mcp = MCPSkillCatalog()
        # The servers this registry took whole: only there may a name the
        # listing did not hold be asked of its server. A copy narrowed to
        # some names reaches those names and no other (RFC §24.3).
        self._owned_labels: set[str] = set()
        # Served names a copy was made without, with their marks: never asked
        # of their server as names the listing did not hold either.
        self._withheld: set[str] = set()

    def discover(self, *directories: str | Path, strict: bool = True) -> int:
        """Scan directories for subdirectories containing SKILL.md.

        A malformed skill is a deployment error, not a runtime condition, so
        the default is to stop. Skipping it instead removes the skill from the
        catalogue while the agent keeps answering: the model is never told the
        capability is missing, and neither is anyone reading the conversation.
        The failure surfaces hours later as an agent that quietly cannot do
        something it was configured to do.

        Discovery commits only once every candidate has parsed, so a strict
        failure leaves the registry exactly as it was rather than half filled.

        Args:
            directories: Directories to scan. Only immediate subdirectories
                containing a SKILL.md are considered.
            strict: Stop on the first unreadable directory or invalid skill.
                Set False to log and skip each instead — appropriate when
                skills come from a source you do not control.

        Returns:
            The number of skills registered.

        Raises:
            SkillDiscoveryError: In strict mode, when a directory cannot be
                scanned or a candidate escapes it through a symlink.
            SkillParseError: In strict mode, when a SKILL.md cannot be parsed.
            SkillValidationError: In strict mode, when metadata is invalid.
            SkillManifestTooLargeError: The skills' manifest would pass
                65,536 characters; nothing is registered.
        """
        parsed: list[tuple[Path, SkillMetadata]] = []
        for skill_dir in self._candidates(directories, strict=strict):
            try:
                parsed.append((skill_dir, parse_skill_metadata(skill_dir)))
            except (SkillParseError, SkillValidationError) as exc:
                if strict:
                    raise
                logger.warning("Skipping invalid skill %s: %s", skill_dir.name, exc)
        self._check_manifest([metadata for _, metadata in parsed])
        for skill_dir, metadata in parsed:
            self._commit(skill_dir, metadata)
        return len(parsed)

    def _candidates(self, directories: tuple[str | Path, ...], *, strict: bool) -> list[Path]:
        """Skill directories found under ``directories``, in a stable order."""
        candidates: list[Path] = []
        for directory in directories:
            try:
                dir_path = Path(directory).resolve()
            except (OSError, RuntimeError) as exc:
                message = f"Skill directory cannot be resolved: {directory}"
                if strict:
                    raise SkillDiscoveryError(message) from exc
                logger.warning("%s: %s", message, exc)
                continue
            if not dir_path.is_dir():
                if strict:
                    raise SkillDiscoveryError(f"Skill directory not found: {dir_path}")
                logger.warning("Skill directory not found: %s", dir_path)
                continue
            try:
                children = sorted(dir_path.iterdir())
            except (OSError, RuntimeError) as exc:
                message = f"Skill directory cannot be scanned: {dir_path}"
                if strict:
                    raise SkillDiscoveryError(message) from exc
                logger.warning("%s: %s", message, exc)
                continue

            for child in children:
                try:
                    if not child.is_dir():
                        continue
                    resolved_child = child.resolve()
                    if dir_path not in resolved_child.parents:
                        raise SkillDiscoveryError(
                            f"Skill candidate escapes discovery directory: {child}"
                        )
                    if find_skill_md(resolved_child) is not None:
                        candidates.append(resolved_child)
                except SkillDiscoveryError:
                    if strict:
                        raise
                    logger.warning("Skipping skill candidate outside %s: %s", dir_path, child)
                except (OSError, RuntimeError) as exc:
                    message = f"Skill candidate cannot be inspected: {child}"
                    if strict:
                        raise SkillDiscoveryError(message) from exc
                    logger.warning("%s: %s", message, exc)
        return candidates

    def register(self, skill_dir: str | Path) -> SkillMetadata:
        """Register a single skill directory.

        Parses frontmatter only (lightweight). Replaces any existing
        skill with the same name.

        Raises:
            SkillParseError: If SKILL.md cannot be found or parsed.
            SkillValidationError: If metadata fails validation.
            SkillManifestTooLargeError: The skills' manifest would pass
                65,536 characters; nothing is registered.
        """
        try:
            skill_path = Path(skill_dir).resolve()
        except (OSError, RuntimeError) as exc:
            raise SkillDiscoveryError(f"Skill directory cannot be resolved: {skill_dir}") from exc
        metadata = parse_skill_metadata(skill_path)
        self._check_manifest([metadata])
        self._commit(skill_path, metadata)
        return metadata

    def _commit(
        self, skill_path: Path, metadata: SkillMetadata, *, log_level: int = logging.INFO
    ) -> None:
        """Record a parsed skill, replacing any earlier one of the same name."""
        self._metadata[metadata.name] = metadata
        self._paths[metadata.name] = skill_path
        # Invalidate cached full skill if re-registering
        self._skills.pop(metadata.name, None)
        # Registering makes the skill usable again — drop any stale mark
        self._unavailable.pop(metadata.name, None)
        self._closed_gates.pop(metadata.name, None)
        self._unlisted.discard(metadata.name)
        logger.log(log_level, "Registered skill: %s", metadata.name)

    def add(self, skill: Skill) -> None:
        """Register a skill built in memory (a store, a marketplace), replacing
        any earlier one of the same name (RFC §24.3).

        Its ``path`` is where its scripts and references are read from. Adding
        it clears its marks, as registering a directory does. Logged at debug:
        a store adds its whole catalogue at once, and says so itself.

        Raises:
            SkillValidationError: The name holds ``:``, which only a skill an
                MCP server serves is named with (RFC §24.7), or a tool pattern
                it gates or requires holds a bracket (RFC §21.1).
            SkillManifestTooLargeError: The skills' manifest would pass
                65,536 characters; nothing is registered.
        """
        if ":" in skill.metadata.name:
            raise SkillValidationError(
                f"Skill name {skill.metadata.name!r} holds ':', which names a served skill"
            )
        fields = {**skill.metadata.extra_metadata, "allowed_tools": skill.metadata.allowed_tools}
        if errors := bracketed_pattern_errors(fields):
            raise SkillValidationError(f"Skill {skill.metadata.name!r}: {'; '.join(errors)}")
        self._check_manifest([skill.metadata])
        self._commit(skill.path, skill.metadata, log_level=logging.DEBUG)
        self._skills[skill.metadata.name] = skill

    def copy(self, names: Iterable[str] | None = None, *, marks: bool = True) -> SkillRegistry:
        """A registry of *names*, every registered skill when ``None`` (RFC §24.3).

        It finds every skill the source finds, one discovered but not yet
        loaded included: each name keeps its path and, once loaded, its skill.
        With *marks*, the copy keeps the source's unlisted marks and the
        unavailable reasons of the names it copies; without, every skill it
        copies is available and listed (a hand-picked set, say), and a name
        the source marks unavailable is not copied at all, nor asked of its
        server.
        """
        wanted = set(self._metadata) | set(self._unavailable) if names is None else set(names)
        copied = SkillRegistry(requires_match=self.requires_match)
        copied._mcp = self._mcp
        copied._owned_labels = set(self._owned_labels) if names is None else set()
        copied._withheld = set(self._withheld)
        if not marks:
            copied._withheld |= {name for name in self._unavailable if self.is_served(name)}
        for name, metadata in self._metadata.items():
            if name not in wanted:
                continue
            copied._metadata[name] = metadata
            if name in self._paths:
                copied._paths[name] = self._paths[name]
            if name in self._skills:
                copied._skills[name] = self._skills[name]
        if marks:
            copied._unlisted = self._unlisted & set(copied._metadata)
            copied._unavailable = {
                name: reason for name, reason in self._unavailable.items() if name in wanted
            }
            copied._closed_gates = {
                name: gates for name, gates in self._closed_gates.items() if name in wanted
            }
        return copied

    async def add_mcp_server(
        self,
        server: MCPSkillServer,
        label: str,
        *,
        allow_dynamic: bool = False,
        approve_execution: ApproveExecution | None = None,
        execution_tools: Sequence[str] = (),
    ) -> int:
        """Register the skills an MCP server serves, under *label* (RFC §24.7).

        Only the listing is read: a skill's ``SKILL.md`` is fetched when it is
        activated, and checked then against the entry the listing gave. Each
        skill is named ``<label>:<path>`` (``hf:hf-cli``). An entry the
        extension does not allow, one past its limits, and a dynamic one
        unless *allow_dynamic* are registered as unavailable with their
        reason, the server's other skills staying usable. Registering a server
        again replaces its entries: a skill it no longer offers, or offers in
        a form this host does not take, retires; one it still offers keeps,
        while loaded, the entry it was loaded from.

        Args:
            server: The connected server
                (:class:`~roomkit.tools.mcp.MCPToolProvider`).
            label: The host's name for the server: an identifier, never the
                name the server gives itself.
            allow_dynamic: Take skills whose content the server generates,
                which nothing can verify.
            approve_execution: Asked before a tool that runs code is served
                while one of this server's skills is active; without it, such
                a call is refused.
            execution_tools: Patterns of the host's own tools that run code,
                gated like the sandbox's.

        Returns:
            How many of the server's skills are available.

        Raises:
            SkillValidationError: *label* is not an identifier, or names
                another server already.
            SkillDiscoveryError: The server does not declare the extension,
                or its listing cannot be read.
        """
        origin, listed = await self._mcp.read_listing(
            server,
            label,
            allow_dynamic=allow_dynamic,
            approve_execution=approve_execution,
            execution_tools=execution_tools,
        )
        offered = {item.name for item in listed if item.metadata is not None}
        for name in self._mcp.names_of(label):
            self._metadata.pop(name, None)
            self._unavailable.pop(name, None)
            self._unlisted.discard(name)
            if name not in offered:
                # No longer offered, or not in a form this host takes: it
                # retires, served nowhere any more (RFC §24.7).
                self._skills.pop(name, None)
        self._mcp.record(origin, listed)
        self._owned_labels.add(label)
        for item in listed:
            self._record_served(item)
        available = sum(1 for item in listed if item.metadata is not None)
        logger.info(
            "Registered %d skills from MCP server %r (%d unavailable)",
            available,
            label,
            len(listed) - available,
        )
        return available

    def _record_served(self, item: Listed) -> None:
        """Record one entry of a server: available, or unavailable with its
        reason, and then served nowhere, a skill loaded before included."""
        if item.metadata is not None:
            self._metadata[item.name] = item.metadata
            self._unavailable.pop(item.name, None)
        else:
            self._metadata.pop(item.name, None)
            self._skills.pop(item.name, None)
            self._unavailable[item.name] = item.reason or "unavailable"

    async def load_skill(self, name: str) -> Skill | None:
        """The skill *name*, loaded: what an activation reads (RFC §24.4).

        A host's own skill is parsed as :meth:`get_skill` parses it. A skill
        an MCP server serves is read from its server now, and checked against
        its entry, unless it is loaded already: it then keeps the entry it was
        loaded from (RFC §24.7). A ``<label>:<path>`` the listing did not
        hold, under a label this registry registered (not a copy narrowed to
        some names), is asked of its server with ``skills/get`` and kept out
        of the manifest. A skill marked unavailable is never loaded, one
        loaded before included, nor one the catalog this registry shares no
        longer holds, or holds in a form this host does not take.

        Raises:
            SkillError: A served skill that is not found on its server, or
                whose content does not match its entry; the message names
                the skill only.
        """
        if self._mcp.label_of(name) is None:
            return await asyncio.to_thread(self.get_skill, name)
        self._follow_catalog(name)
        if name in self._unavailable:
            return None
        metadata = self._metadata.get(name)
        loaded = self._skills.get(name)
        if metadata is not None and loaded is not None:
            return loaded
        if metadata is None:
            metadata = await self._fetch_unlisted(name)
            if metadata is None:
                return None
        skill = await self._mcp.load(name, metadata)
        self._skills[name] = skill
        return skill

    def _follow_catalog(self, name: str) -> None:
        """Bring this registry's record of served *name* in line with the
        catalog it shares: a registry sharing it may have registered the
        server again without the skill, or in a form this host does not
        take, or retired it on a failed file check; and an entry read again
        after a failed check is held to the listing's rules (RFC §24.7)."""
        if name not in self._metadata:
            return
        if self._mcp.entry(name) is None:
            self._metadata.pop(name, None)
            self._skills.pop(name, None)
            self._unlisted.discard(name)
            return
        reason = self._mcp.refusal(name)
        if reason is not None:
            self._record_served(Listed(name, None, None, reason))
            return
        loaded = self._skills.get(name)
        if isinstance(loaded, MCPSkill) and self._mcp.is_retired(loaded):
            self._skills.pop(name, None)

    async def _fetch_unlisted(self, name: str) -> SkillMetadata | None:
        """Ask the server for *name*, a skill its listing did not hold: only
        under a label this registry took whole, and never a name a copy was
        made without. ``None`` when the entry is one this host does not take.

        Raises:
            SkillError: Not found here or on its server.
        """
        if self._mcp.label_of(name) not in self._owned_labels or name in self._withheld:
            raise SkillValidationError(f"Skill {name!r} not found")
        listed = await self._mcp.fetch_unlisted(name)
        self._record_served(listed)
        if listed.metadata is not None:
            self._unlisted.add(name)
        return listed.metadata

    async def read_file(self, name: str, filename: str) -> str:
        """The text of *filename*, a file of skill *name* (RFC §24.6, §24.7).

        A host's own skill reads a reference by name, confined to its
        ``references/`` directory. A served skill reads a path inside the
        skill (``templates/invoice.md``) that the entry it was loaded from
        lists, from its own server, checked against that entry; it must be
        loaded (activated) first.

        Raises:
            KeyError: No skill *name* is loaded or found here.
            SkillPathError: The name escapes the skill, or is not one of a
                served skill's files (nothing is read then).
            FileNotFoundError: A host's skill has no such reference.
            SkillChangedError: A served skill's file failed verification,
                or the catalog this registry shares no longer holds the
                skill in a form this host takes: the skill is retired here
                and must be activated again.
        """
        if self._mcp.label_of(name) is None:
            skill = await asyncio.to_thread(self.get_skill, name)
            if skill is None:
                raise KeyError(name)
            return await asyncio.to_thread(skill.read_reference, filename)
        served = self._skills.get(name)
        if not isinstance(served, MCPSkill):
            raise KeyError(name)
        self._follow_catalog(name)
        if name not in self._skills:
            raise SkillChangedError(name)
        try:
            return await self._mcp.read_file(served, filename)
        except SkillChangedError:
            self._skills.pop(name, None)
            raise

    def is_served(self, name: str) -> bool:
        """Whether *name* is, or could be, a skill an MCP server serves."""
        return self._mcp.label_of(name) is not None

    def origin(self, name: str) -> ServedOrigin | None:
        """The server that serves *name*, or ``None`` for a host's own skill."""
        label = self._mcp.label_of(name)
        return None if label is None else self._mcp.origin(label)

    @property
    def has_served(self) -> bool:
        """Whether a server's skills were registered here."""
        return self._mcp.has_servers

    @property
    def execution_tool_patterns(self) -> frozenset[str]:
        """The patterns of the host's tools that run code, named when its
        servers were registered (``execution_tools``): gated like the sandbox's
        while a served skill is active (RFC §24.7)."""
        return frozenset(self._mcp.execution_tools())

    def get_metadata(self, name: str) -> SkillMetadata | None:
        """Get metadata for a skill by name."""
        return self._metadata.get(name)

    def get_skill(self, name: str) -> Skill | None:
        """Get full skill (with instructions), loading lazily if needed."""
        if self._mcp.label_of(name) is not None:
            self._follow_catalog(name)
        if name in self._skills:
            return self._skills[name]

        if name not in self._paths:
            return None

        try:
            skill = parse_skill(self._paths[name])
            self._skills[name] = skill
            return skill
        except (SkillParseError, SkillValidationError) as exc:
            logger.error("Failed to load skill %s: %s", name, exc)
            return None

    def mark_unavailable(self, name: str, reason: str) -> None:
        """Record a skill that exists but cannot be used in this context.

        The skill leaves the available set entirely — ``skill_names``,
        ``all_metadata()`` and ``get_skill()`` no longer see it — and only
        surfaces through ``unavailable_skills``, the prompt manifest and
        the skill-tool error paths, so callers can say WHY it is missing
        instead of leaving a silent gap (e.g. a ``requires`` gate dropping
        a skill whose tools are not granted in this execution context).

        What it gated stays closed: no activation can open it any more
        (:meth:`gated_tool_names`).
        """
        metadata = self._metadata.pop(name, None)
        if metadata is not None and metadata.gated_tool_names:
            self._closed_gates[name] = metadata.gated_tool_names
        self._skills.pop(name, None)
        self._paths.pop(name, None)
        self._unavailable[name] = reason

    def gated_tool_names(self, activated: Container[str] = ()) -> set[str]:
        """The tool patterns skills keep closed (RFC §24.2): those of every
        registered skill not in *activated*, and the closed ones
        (:meth:`closed_tool_names`). Both doors read skill gating through
        this one rule."""
        gated = {
            pattern
            for metadata in self._metadata.values()
            if metadata.name not in activated
            for pattern in metadata.gated_tool_names
        }
        return gated | self.closed_tool_names(activated)

    def closed_tool_names(self, activated: Container[str] = ()) -> set[str]:
        """The tool patterns only skills marked unavailable gate, which
        nothing can open (RFC §24.2). A pattern a skill in *activated* gates is
        that skill's to open: an unavailable skill gating the same pattern does
        not hold it closed."""
        opened = {
            pattern
            for metadata in self._metadata.values()
            if metadata.name in activated
            for pattern in metadata.gated_tool_names
        }
        return {
            pattern
            for gates in self._closed_gates.values()
            for pattern in gates
            if pattern not in opened
        }

    def unopenable_tool_names(self) -> set[str]:
        """The closed patterns no available skill gates as well: no
        activation can open them (RFC §24.2)."""
        return self.closed_tool_names(self._metadata)

    def mark_unlisted(self, name: str) -> None:
        """Keep *name* activatable but out of the prompt manifest.

        The third visibility state, between available and unavailable: the
        skill stays registered — ``activate_skill``, ``get_skill()`` and
        ``skill_names`` still see it — but ``to_prompt_xml()`` and
        ``listed_names`` do not. For catalogues where advertising every
        entry would drown the ones that matter: a host can keep quiet about
        a skill while any path that names it (a recommender nudge, a user
        asking for it) still activates it, which is what lets it earn its
        listing back.

        Unknown names are ignored — there is nothing to hide.
        """
        if name in self._metadata:
            self._unlisted.add(name)

    @property
    def listed_names(self) -> list[str]:
        """Names of registered skills that the prompt manifest shows."""
        return [name for name in self._metadata if name not in self._unlisted]

    def get_unavailable_reason(self, name: str) -> str | None:
        """Reason a skill is unavailable in this context, or None."""
        return self._unavailable.get(name)

    @property
    def unavailable_skills(self) -> dict[str, str]:
        """Mapping of unavailable skill name -> reason."""
        return dict(self._unavailable)

    def all_metadata(self) -> list[SkillMetadata]:
        """Return metadata for all registered skills."""
        return list(self._metadata.values())

    @property
    def skill_names(self) -> list[str]:
        """Names of all registered skills."""
        return list(self._metadata.keys())

    @property
    def skill_count(self) -> int:
        """Number of registered skills."""
        return len(self._metadata)

    @property
    def has_entries(self) -> bool:
        """Whether the model has anything to be told: a skill it can activate,
        or one marked unavailable, whose reason it gives instead of guessing."""
        return bool(self._metadata or self._unavailable)

    def to_prompt_xml(self) -> str:
        """Generate spec-compliant <available_skills> XML block.

        Skills marked unavailable are listed in a separate
        ``<unavailable_skills>`` block with their reason, so the model
        can explain the gap instead of guessing at a name that will
        answer "not found". Skills marked unlisted are simply absent —
        activatable, but not advertised. Content is HTML-escaped to
        prevent injection.
        """
        listed = [meta for meta in self._metadata.values() if meta.name not in self._unlisted]
        return self._manifest(listed, self._unavailable)

    def _manifest(self, listed: list[SkillMetadata], unavailable: Mapping[str, str]) -> str:
        """The manifest of *listed* skills and *unavailable* ones."""
        if not listed and not unavailable:
            return ""

        lines: list[str] = []
        if listed:
            lines.append("<available_skills>")
            for meta in listed:
                lines.extend(self._manifest_lines(meta))
            lines.append("</available_skills>")
        if unavailable:
            lines.append("<unavailable_skills>")
            for name, reason in unavailable.items():
                lines.append(f'  <skill name="{escape(name)}">')
                lines.append(f"    <reason>{escape(reason)}</reason>")
                lines.append("  </skill>")
            lines.append("</unavailable_skills>")
        return "\n".join(lines)

    def _check_manifest(self, registered: list[SkillMetadata]) -> None:
        """Refuse to register *registered* when the manifest would pass its
        bound once they are, each clearing its marks as a registration does
        (RFC §24.1): nothing is registered then.

        Raises:
            SkillManifestTooLargeError: The manifest would pass
                65,536 characters.
        """
        names = {meta.name for meta in registered}
        metadata = {**self._metadata, **{meta.name: meta for meta in registered}}
        listed = [
            meta
            for meta in metadata.values()
            if meta.name in names or meta.name not in self._unlisted
        ]
        unavailable = {n: r for n, r in self._unavailable.items() if n not in names}
        characters = len(self._manifest(listed, unavailable))
        if characters > _MANIFEST_MAX:
            raise SkillManifestTooLargeError(characters, _MANIFEST_MAX)

    def _manifest_lines(self, meta: SkillMetadata) -> list[str]:
        """One skill of the manifest. A served skill names its origin, and what
        its server wrote is quoted as a text from outside (RFC §6.4, §24.7)."""
        label = self._mcp.label_of(meta.name)
        if label is None:
            opening = f'  <skill name="{escape(meta.name)}">'
            fields = [meta.description, meta.license, meta.compatibility]
        else:
            opening = f'  <skill name="{escape(meta.name)}" origin="{escape(label)}">'
            fields = [
                quoted(value, _QUOTED_MAX) if value else None
                for value in (meta.description, meta.license, meta.compatibility)
            ]
        lines = [opening]
        for tag, value in zip(("description", "license", "compatibility"), fields, strict=True):
            if value:
                # A quote made every double quote mark a single one: element
                # text needs no quote escaped, only its brackets and ampersands.
                text = escape(value) if label is None else escape(value, quote=False)
                lines.append(f"    <{tag}>{text}</{tag}>")
        lines.append("  </skill>")
        return lines
