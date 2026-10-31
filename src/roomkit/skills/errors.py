"""Exceptions raised by the skills framework.

All framework-defined skill failures derive from :class:`SkillError`, so a
caller can catch malformed metadata, discovery and containment errors as one
condition. Ordinary filesystem outcomes that are part of a method's public
contract (for example ``FileNotFoundError`` from ``read_reference``) retain
their standard Python types.
"""

from __future__ import annotations


class SkillError(Exception):
    """Base class for framework-defined skill failures."""


class SkillParseError(SkillError):
    """Failed to parse SKILL.md content."""


class SkillValidationError(SkillError):
    """SKILL.md metadata failed validation."""


class SkillDiscoveryError(SkillError):
    """A directory handed to ``SkillRegistry.discover`` cannot be scanned."""


class SkillVerificationError(SkillError):
    """Content an MCP server served for a skill does not match its entry: a
    length, a digest, a frontmatter, or a file the entry does not list
    (RFC §24.7). The content is not used."""


class SkillManifestTooLargeError(SkillError):
    """The skills' manifest, which every system prompt carries, would be
    longer than its bound (RFC §24.1): *characters* past *limit*."""

    def __init__(self, characters: int, limit: int) -> None:
        super().__init__(f"the skills' manifest would take {characters} characters, past {limit}")
        self.characters = characters
        self.limit = limit


class SkillPathError(SkillError, ValueError):
    """A skill-relative path escapes the directory it must stay inside.

    Also a :class:`ValueError` because ``Skill.read_reference`` raised one
    before this class existed, and integrators catch it.
    """
