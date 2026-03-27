"""Shared constants for Agent Skills integration across channel types."""

from __future__ import annotations

from typing import Any

# Tool name constants
TOOL_ACTIVATE_SKILL = "activate_skill"
TOOL_READ_REFERENCE = "read_skill_reference"
TOOL_RUN_SCRIPT = "run_skill_script"

SKILL_INFRA_TOOL_NAMES: frozenset[str] = frozenset(
    {TOOL_ACTIVATE_SKILL, TOOL_READ_REFERENCE, TOOL_RUN_SCRIPT}
)

SKILLS_PREAMBLE = (
    "You have access to Agent Skills — specialized knowledge packages. "
    "Use the activate_skill tool to load a skill's full instructions before "
    "using it. Available skills are listed below."
)

SKILLS_NO_SCRIPTS_NOTE = " Note: Script execution is not available in this environment."

# Shared tool schemas — used by both AIChannel (_ai_tools.py) and
# RealtimeVoiceChannel (_realtime_skills.py) to avoid duplication.

ACTIVATE_SKILL_SCHEMA: dict[str, Any] = {
    "name": TOOL_ACTIVATE_SKILL,
    "description": (
        "Activate a skill to get its full instructions, available scripts, and reference files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the skill to activate.",
            },
        },
        "required": ["name"],
    },
}

READ_REFERENCE_SCHEMA: dict[str, Any] = {
    "name": TOOL_READ_REFERENCE,
    "description": "Read a reference file from a skill.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of the skill.",
            },
            "filename": {
                "type": "string",
                "description": "Reference filename to read.",
            },
        },
        "required": ["skill_name", "filename"],
    },
}

RUN_SCRIPT_SCHEMA: dict[str, Any] = {
    "name": TOOL_RUN_SCRIPT,
    "description": "Run a script from a skill's scripts/ directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "Name of the skill.",
            },
            "script_name": {
                "type": "string",
                "description": "Script filename to run.",
            },
            "arguments": {
                "type": "object",
                "description": "Optional key-value arguments.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["skill_name", "script_name"],
    },
}
