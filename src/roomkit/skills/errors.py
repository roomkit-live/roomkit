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


class SkillPathError(SkillError, ValueError):
    """A skill-relative path escapes the directory it must stay inside.

    Also a :class:`ValueError` because ``Skill.read_reference`` raised one
    before this class existed, and integrators catch it.
    """
