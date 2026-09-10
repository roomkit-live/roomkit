"""Skill support for RealtimeVoiceChannel.

Handles skill tool definitions, prompt injection, per-session activation
tracking, and tool gating for realtime voice sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
    SKILL_INFRA_TOOL_NAMES,
    SKILLS_INLINE_PREAMBLE,
    SKILLS_NO_SCRIPTS_NOTE,
    SKILLS_PREAMBLE,
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._skill_handlers import (
    activation_ack,
    activation_content,
    handle_read_reference,
    handle_run_script,
    missing_skill_error,
)
from roomkit.channels._tool_search_constants import TOOL_SEARCH_INFRA_TOOL_NAMES
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import matches_any_pattern

if TYPE_CHECKING:
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.models import Skill

logger = logging.getLogger("roomkit.channels.realtime_voice")


SkillDeliveryMode = str
"""``inline_full`` preloads bodies; ``on_demand`` loads only activated skills.

On reconfigurable providers, bodies enter system instructions. Fixed providers
receive the full body in the activation result and must preserve that context.
"""


class RealtimeSkillSupport:
    """Skill delivery and gates scoped to one live conversation.

    Preparing an activation never opens tools. The channel commits it only
    after the provider has accepted its instructions and configuration.
    """

    def __init__(
        self,
        skills: SkillRegistry,
        script_executor: ScriptExecutor | None = None,
        *,
        delivery_mode: SkillDeliveryMode = "on_demand",
        reconfigure_capable: bool = True,
    ) -> None:
        if delivery_mode not in {"inline_full", "on_demand"}:
            raise ValueError(f"Unknown skill delivery mode: {delivery_mode}")
        self._skills = skills
        self._reconfigure_capable = reconfigure_capable
        self._script_executor = script_executor
        self._delivery_mode: SkillDeliveryMode = delivery_mode
        # session_id -> set of activated skill names
        self._activated_skills: dict[str, set[str]] = {}
        # session_id -> ordered list of (skill_name, instructions) tuples
        # for skills activated so far in this session. Concatenated into
        # the system_instruction on the next reconfigure_session call.
        # Order matches activation sequence so a chained flow can layer
        # later skills on top of earlier ones.
        self._activated_bodies: dict[str, list[tuple[str, str]]] = {}

    @property
    def delivery_mode(self) -> SkillDeliveryMode:
        return self._delivery_mode

    # -- Tool definitions (as dicts, the format provider.connect expects) --

    def skill_tool_dicts(self) -> list[dict[str, Any]]:
        """Return skill infrastructure tool definitions as plain dicts."""
        tools: list[dict[str, Any]] = [ACTIVATE_SKILL_SCHEMA, READ_REFERENCE_SCHEMA]
        if self._script_executor:
            tools.append(RUN_SCRIPT_SCHEMA)
        return tools

    # -- System prompt injection --

    def inject_skills_prompt(self, system_prompt: str | None) -> str:
        """Append skills preamble + available-skills XML to the prompt.

        In ``inline_full`` mode every available skill's full body is
        included verbatim so the model has the binding rules in
        attention from the first token. In ``on_demand`` mode only
        skill metadata is included; bodies arrive later via
        ``provider.reconfigure`` or its tool result after the model calls
        ``activate_skill``.
        """
        if self._delivery_mode == "inline_full":
            preamble = SKILLS_INLINE_PREAMBLE
        else:
            preamble = SKILLS_PREAMBLE
        if not self._script_executor:
            preamble += SKILLS_NO_SCRIPTS_NOTE
        skills_xml = self._skills.to_prompt_xml()
        skill_block = f"\n\n{preamble}\n\n{skills_xml}"

        if self._delivery_mode == "inline_full":
            bodies_block = self._render_all_skill_bodies()
            if bodies_block:
                skill_block += f"\n\n{bodies_block}"

        return (system_prompt or "") + skill_block

    def _render_all_skill_bodies(self) -> str | None:
        """Render every skill's body as a block of binding-rule sections.

        Used by ``inline_full`` mode at session start. Skills without
        instructions are skipped (the metadata XML already advertised
        them; nothing actionable to add).
        """
        sections: list[str] = []
        for meta in self._skills.all_metadata():
            skill = self._skills.get_skill(meta.name)
            body = getattr(skill, "instructions", None) if skill else None
            if not body or not body.strip():
                continue
            sections.append(f"## Skill: {meta.name}\n{body.strip()}")
        if not sections:
            return None
        return "# Loaded skill instructions (binding rules)\n\n" + "\n\n".join(sections)

    # -- Per-session activation tracking --

    def init_session(self, session_id: str) -> None:
        """Initialize activation state for a new session."""
        self._activated_skills[session_id] = set()
        self._activated_bodies[session_id] = []

    def cleanup_session(self, session_id: str) -> None:
        """Remove activation state when a session ends."""
        self._activated_skills.pop(session_id, None)
        self._activated_bodies.pop(session_id, None)

    def activated_skills_prompt(self, session_id: str, pending: Skill | None = None) -> str | None:
        """Return concatenated bodies of skills activated in this session.

        Used by the channel's tool dispatcher: after activate_skill
        runs we call ``provider.reconfigure(system_prompt=base + this)``
        so the skill content lives as binding rules in
        ``system_instruction`` rather than as a giant tool result that
        derails realtime function calling.

        Returns ``None`` when no skills have been activated yet so the
        caller can decide whether a reconfigure is even needed.
        """
        bodies = list(self._activated_bodies.get(session_id) or [])
        if (
            pending
            and self._delivery_mode == "on_demand"
            and pending.name not in self._activated_skills.get(session_id, set())
        ):
            bodies.append((pending.name, pending.instructions))
        if not bodies:
            return None
        sections = [
            f"## Active skill: {name}\n{instructions.strip()}"
            for name, instructions in bodies
            if instructions and instructions.strip()
        ]
        return "\n\n".join(sections) if sections else None

    # -- Tool dispatch --

    def is_skill_tool(self, name: str) -> bool:
        """Return True if *name* is a skill infrastructure tool."""
        return name in SKILL_INFRA_TOOL_NAMES

    async def handle_tool_call(self, name: str, arguments: dict[str, Any], session_id: str) -> str:
        """Dispatch a skill tool call and return the JSON result string."""
        if name == TOOL_ACTIVATE_SKILL:
            return await self._handle_activate_skill(arguments, session_id)
        if name == TOOL_READ_REFERENCE:
            return await self._handle_read_reference(arguments)
        if name == TOOL_RUN_SCRIPT:
            return await self._handle_run_script(arguments)
        return json.dumps({"error": f"Unknown skill tool: {name}"})

    # -- Tool gating --

    def _gated_tool_names(self, session_id: str, pending: Skill | None = None) -> set[str]:
        """Collect tool names gated by skills not yet activated in this session."""
        activated = self._activated_skills.get(session_id, set())
        gated: set[str] = set()
        for meta in self._skills.all_metadata():
            if meta.name in activated or (pending is not None and meta.name == pending.name):
                continue
            gated.update(meta.gated_tool_names)
        return gated

    def is_gated(self, name: str, session_id: str, gated: set[str] | None = None) -> bool:
        """Whether *name* is gated by a skill this session has not activated.

        Hiding a tool from the catalogue is not enforcement: a model that saw
        the name before the skill was deactivated — or that read it in a
        transcript — can still call it. Callers ask this at execution time as
        well as at listing time.

        Infrastructure tools are never gated: skill tools are how a skill gets
        activated, and the Tool Search tools are how a gated name is found in
        the first place. Gating them would leave the model told to activate a
        skill it has no way left to name.

        *gated* lets a caller filtering a whole catalogue compute the gated set
        once instead of once per tool.
        """
        if name in SKILL_INFRA_TOOL_NAMES or name in TOOL_SEARCH_INFRA_TOOL_NAMES:
            return False
        if gated is None:
            gated = self._gated_tool_names(session_id)
        return bool(gated) and matches_any_pattern(name, gated)

    def get_visible_tools(
        self, all_tools: list[dict[str, Any]], session_id: str, pending: Skill | None = None
    ) -> list[dict[str, Any]]:
        """Filter tool list, removing gated tools but keeping infra tools."""
        gated = self._gated_tool_names(session_id, pending)
        if not gated:
            return all_tools
        return [
            t for t in all_tools if not self.is_gated(str(t.get("name", "")), session_id, gated)
        ]

    def newly_visible_after_activation(
        self,
        all_tools: list[dict[str, Any]],
        session_id: str,
        skill_name: str,
    ) -> list[dict[str, Any]] | None:
        """Return updated tool list if activation revealed new tools, else None."""
        meta = self._skills.get_metadata(skill_name)
        if not meta or not meta.gated_tool_names:
            return None
        # Re-filter with the now-activated skill
        return self.get_visible_tools(all_tools, session_id)

    # -- Internal handlers --

    @property
    def uses_tool_result(self) -> bool:
        """Whether the provider must retain dynamically delivered bodies."""
        return self._delivery_mode == "on_demand" and not self._reconfigure_capable

    def commit_activation(self, session_id: str, skill: Skill) -> None:
        """Open gates only after delivery, never resurrecting a closed session."""
        activated = self._activated_skills.get(session_id)
        if activated is not None and skill.name not in activated:
            activated.add(skill.name)
            if self._delivery_mode == "on_demand":
                self._activated_bodies[session_id].append((skill.name, skill.instructions))

    async def prepare_activation(
        self, arguments: dict[str, Any], session_id: str, tools: list[dict[str, Any]]
    ) -> tuple[str, Skill | None]:
        """Build an immutable delivery candidate from the authorized catalogue."""
        skill_name = arguments.get("name", "")
        skill = await asyncio.to_thread(self._skills.get_skill, skill_name)
        if skill is None:
            return json.dumps(
                {
                    "error": missing_skill_error(self._skills, skill_name),
                    "available_skills": self._skills.skill_names,
                }
            ), None
        catalogue = {tool["name"]: tool for tool in tools}
        missing = [name for name in skill.metadata.required_tool_names if name not in catalogue]
        if missing:
            return json.dumps(
                {"error": f"Required tools not available: {', '.join(missing)}"}
            ), None

        if self.uses_tool_result:
            result = await asyncio.to_thread(activation_content, skill)
            payload = json.loads(result)
            payload["ok"] = True
            payload["_note"] = (
                "Follow these complete skill instructions for this session. "
                "Use the required tool schemas below; tool names and actions are distinct."
            )
            payload["required_tools"] = [
                catalogue[name] for name in skill.metadata.required_tool_names
            ]
            if skill_name in self._activated_skills.get(session_id, set()):
                payload["already_active"] = True
            return json.dumps(payload), skill

        note = (
            "The skill instructions are already loaded in your system rules. Follow them."
            if self._delivery_mode == "inline_full"
            else "Loading the skill instructions into your system rules before continuing."
        )
        result = await asyncio.to_thread(
            activation_ack,
            skill,
            note,
            already_active=skill_name in self._activated_skills.get(session_id, set()),
        )
        return result, skill

    async def _handle_activate_skill(self, arguments: dict[str, Any], session_id: str) -> str:
        """Prepare a result; the channel owns delivery and activation commit."""
        result, _ = await self.prepare_activation(arguments, session_id, [])
        return result

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        return await handle_read_reference(arguments, self._skills)

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        return await handle_run_script(arguments, self._skills, self._script_executor)
