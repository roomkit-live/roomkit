"""Shared skill tool handlers used by both AIChannel and RealtimeVoiceChannel."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.skills.errors import SkillPathError

if TYPE_CHECKING:
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.models import Skill
    from roomkit.skills.registry import SkillRegistry

logger = logging.getLogger("roomkit.channels.skills")


def missing_skill_error(skills: SkillRegistry, skill_name: str) -> str:
    """Error line for a skill ``get_skill()`` could not return.

    A skill marked unavailable gets its reason (e.g. a required tool is
    not granted in this context) instead of a misleading "not found".
    """
    reason = skills.get_unavailable_reason(skill_name)
    if reason is not None:
        return f"Skill {skill_name!r} is unavailable in this context: {reason}"
    return f"Skill {skill_name!r} not found"


def activation_ack(skill: Skill, note: str, *, already_active: bool = False) -> str:
    """Acknowledge an activation without re-sending the skill's body.

    The body reaches the model through the system prompt instead — folded into
    ``system_instruction`` on realtime channels, into the turn's system prompt on
    the text channel. What the tool still owes the model is the activation itself
    plus the skill's file inventory, which is what this returns.
    """
    payload: dict[str, Any] = {"ok": True, "name": skill.name, "_note": note}
    if already_active:
        payload["already_active"] = True
    scripts = skill.list_scripts()
    if scripts:
        payload["scripts"] = scripts
    refs = skill.list_references()
    if refs:
        payload["references"] = refs
    return json.dumps(payload)


async def handle_activate_skill(
    arguments: dict[str, Any],
    skills: SkillRegistry,
) -> tuple[str, str]:
    """Load a skill and return its full instructions.

    Returns:
        A tuple of ``(json_result, skill_name)`` so callers can track
        activation in their own state store.
    """
    skill_name = arguments.get("name", "")
    skill = skills.get_skill(skill_name)
    if skill is None:
        error_json = json.dumps(
            {
                "error": missing_skill_error(skills, skill_name),
                "available_skills": skills.skill_names,
            }
        )
        return error_json, skill_name

    payload: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
    }
    scripts = skill.list_scripts()
    if scripts:
        payload["scripts"] = scripts
    refs = skill.list_references()
    if refs:
        payload["references"] = refs
    return json.dumps(payload), skill_name


async def handle_read_reference(
    arguments: dict[str, Any],
    skills: SkillRegistry,
) -> str:
    """Read a reference file from a skill."""
    skill_name = arguments.get("skill_name", "")
    filename = arguments.get("filename", "")
    skill = skills.get_skill(skill_name)
    if skill is None:
        return json.dumps({"error": missing_skill_error(skills, skill_name)})

    try:
        content = skill.read_reference(filename)
        return json.dumps({"filename": filename, "content": content})
    except (ValueError, FileNotFoundError) as exc:
        return json.dumps({"error": str(exc)})


async def handle_run_script(
    arguments: dict[str, Any],
    skills: SkillRegistry,
    script_executor: ScriptExecutor | None,
) -> str:
    """Execute a script via the configured ScriptExecutor."""
    skill_name = arguments.get("skill_name", "")
    script_name = arguments.get("script_name", "")
    script_args = arguments.get("arguments")
    if not script_executor:
        return json.dumps({"error": "Script execution is not available"})

    skill = skills.get_skill(skill_name)
    if skill is None:
        return json.dumps({"error": missing_skill_error(skills, skill_name)})

    # Which file runs is the framework's call; how it runs is the executor's.
    # Resolving here means a name that escapes the skill never reaches an
    # integrator's executor, rather than relying on each one to re-check it.
    try:
        skill.resolve_script(script_name)
    except (SkillPathError, FileNotFoundError) as exc:
        return json.dumps({"error": str(exc)})

    try:
        result = await script_executor.execute(skill, script_name, arguments=script_args)
        return result.model_dump_json()
    except Exception as exc:
        logger.exception("Script execution failed: %s/%s", skill_name, script_name)
        return json.dumps({"error": f"Script execution failed: {exc}"})
