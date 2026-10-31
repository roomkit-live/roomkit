"""Agent Skills integration for RoomKit."""

from roomkit.skills.errors import (
    SkillDiscoveryError,
    SkillError,
    SkillManifestTooLargeError,
    SkillParseError,
    SkillPathError,
    SkillValidationError,
    SkillVerificationError,
)
from roomkit.skills.executor import ScriptExecutor
from roomkit.skills.mcp import (
    MCPSkill,
    MCPSkillServer,
    SkillChangedError,
    SkillEntry,
    SkillExecutionRequest,
    SkillResource,
)
from roomkit.skills.models import (
    RequiresMatch,
    ScriptResult,
    Skill,
    SkillMetadata,
    missing_required_tools,
    parse_tool_names,
    serves_exactly,
)
from roomkit.skills.parser import parse_frontmatter
from roomkit.skills.paths import safe_join_filename
from roomkit.skills.registry import SkillRegistry

__all__ = [
    "MCPSkill",
    "MCPSkillServer",
    "RequiresMatch",
    "ScriptExecutor",
    "ScriptResult",
    "Skill",
    "SkillChangedError",
    "SkillDiscoveryError",
    "SkillEntry",
    "SkillError",
    "SkillManifestTooLargeError",
    "SkillExecutionRequest",
    "SkillMetadata",
    "SkillParseError",
    "SkillPathError",
    "SkillRegistry",
    "SkillResource",
    "SkillValidationError",
    "SkillVerificationError",
    "missing_required_tools",
    "parse_frontmatter",
    "parse_tool_names",
    "safe_join_filename",
    "serves_exactly",
]
