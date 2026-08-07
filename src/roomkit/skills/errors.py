"""Exceptions raised by the skills framework.

All skill failures derive from :class:`SkillError`, so a caller that wants to
treat "something is wrong with the skills on disk" as one condition can catch a
single type. The specific classes stay meaningful on their own — a deployment
that fails to start wants to read *which* kind of wrong it was.
"""

from __future__ import annotations


class SkillError(Exception):
    """Base class for every skill failure."""


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
